#!/usr/bin/env python3
"""中国人口结构分析器：新生儿统计 + 老龄化统计。

数据源为世界银行开放 API（api.worldbank.org），其中国人口数据以国家统计局与
联合国 WPP 为底稿，免鉴权、JSON、可回溯至 1960 年。选它而非统计局官网是因为
data.stats.gov.cn 的 easyquery 接口不返回 JSON（需要会话/被拦），akshare 也没有
现成的人口接口，东财数据中心亦无人口报表。

口径提示：世界银行的 65 岁以上占比采用联合国口径，与统计局自行公布的数值存在
零点几个百分点的差异，判断趋势无碍，但不要与统计局公报逐位对照。

出生人口不是世界银行直接发布的字段，由「出生率(‰) × 总人口 ÷ 1000」推算。
2024 年据此得约 955 万，与统计局公布的 954 万一致，可作为口径正确性的交叉验证。
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Any

_WB_URL = ("https://api.worldbank.org/v2/country/CHN/indicator/{code}"
           "?format=json&per_page=100")

# 世界银行指标代码 → 内部字段名
_INDICATORS: dict[str, str] = {
    "SP.DYN.CBRT.IN": "birth_rate",        # 出生率，‰
    "SP.POP.TOTL": "population",           # 总人口
    "SP.DYN.TFRT.IN": "fertility",         # 总和生育率
    "SP.POP.65UP.TO.ZS": "elderly_share",  # 65 岁以上占比，%
    "SP.POP.DPND.OL": "old_dependency",    # 老年抚养比，%
    "SP.POP.0014.TO.ZS": "child_share",    # 0-14 岁占比，%
}

# 联合国通行的老龄化阶段门槛（按 65 岁以上人口占比）
_AGING_STAGES: list[tuple[float, str, str, str]] = [
    (7.0, "老龄化社会", "#f59e0b",
     "65 岁以上占比超过 7%，进入老龄化社会，劳动力供给开始见顶。"),
    (14.0, "深度老龄化社会", "#ef4444",
     "65 岁以上占比超过 14%，养老与医疗支出压力显著上升，储蓄率与地产需求承压。"),
    (20.0, "超老龄化社会", "#7c3aed",
     "65 岁以上占比超过 20%，抚养负担沉重，经济增速中枢与资产定价逻辑发生结构性改变。"),
]

_REPLACEMENT_FERTILITY = 2.1  # 世代更替水平

# ── 国家统计局公布的出生人口与出生率（历年国民经济和社会发展统计公报）──
#
# 为何要内嵌：统计局 easyquery 接口对外返回 403（WAF 拦截），akshare 无人口接口，
# 东财数据中心无人口报表，没有可用的实时权威源。
#
# 为何不直接用世界银行推算值：世行/联合国口径在 2011-2014、2016-2017 这几年显著
# 高于统计局公布值（2012 年推算 1973 万 vs 公布 1635 万，差 21%），据此得出的
# "近20年峰值在2012年"与公众认知的 2016 年峰值矛盾。其余 14 年两者几乎一致。
# 两个口径都会在页面上给出，分歧年份单独标注。
#
# 维护：每年 1 月统计局发布上一年公报后追加一行。若世界银行数据出现比本表更晚的
# 年份，analyze_births 会输出 nbs_stale 提示。
_NBS_LAST_YEAR = 2024
_NBS_BIRTHS_WAN: dict[int, float] = {
    2005: 1617, 2006: 1584, 2007: 1594, 2008: 1608, 2009: 1591,
    2010: 1592, 2011: 1604, 2012: 1635, 2013: 1640, 2014: 1687,
    2015: 1655, 2016: 1786, 2017: 1723, 2018: 1523, 2019: 1465,
    2020: 1202, 2021: 1062, 2022: 956, 2023: 902, 2024: 954,
}
_NBS_BIRTH_RATE: dict[int, float] = {
    2005: 12.40, 2006: 12.09, 2007: 12.10, 2008: 12.14, 2009: 11.95,
    2010: 11.90, 2011: 11.93, 2012: 12.10, 2013: 12.08, 2014: 12.37,
    2015: 12.07, 2016: 12.95, 2017: 12.43, 2018: 10.94, 2019: 10.48,
    2020: 8.52, 2021: 7.52, 2022: 6.77, 2023: 6.39, 2024: 6.77,
}

# 两口径差异超过该比例即视为分歧，单独列出
_DIVERGENCE_PCT = 2.0

_CACHE: dict[str, Any] | None = None
_CACHE_TS = 0.0
_CACHE_TTL = 24 * 3600  # 年度数据，一天一次足够


def _curl_json(url: str, timeout: int = 30) -> Any | None:
    """用 curl --noproxy 直连取 JSON。

    与 stock_server._fetch_fred_index 同样的理由：本机 Clash 代理环境下
    requests 会 ReadTimeout 或撞上中间人证书，curl 直连最稳。
    """
    try:
        proc = subprocess.run(
            ["curl", "-s", "--noproxy", "*", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def _fetch_indicator(code: str) -> dict[int, float]:
    """取单个世界银行指标，返回 {年份: 数值}（按年份升序使用）。"""
    payload = _curl_json(_WB_URL.format(code=code))
    if not payload or len(payload) < 2 or not payload[1]:
        return {}
    out: dict[int, float] = {}
    for row in payload[1]:
        val = row.get("value")
        if val is None:
            continue
        try:
            out[int(row["date"])] = float(val)
        except (ValueError, TypeError, KeyError):
            continue
    return out


def _fetch_all(force: bool = False) -> dict[str, dict[int, float]]:
    """取齐全部指标，带进程内缓存（年度数据，TTL 一天）。"""
    global _CACHE, _CACHE_TS
    now = time.time()
    if not force and _CACHE is not None and now - _CACHE_TS < _CACHE_TTL:
        return _CACHE
    series: dict[str, dict[int, float]] = {}
    for code, name in _INDICATORS.items():
        data = _fetch_indicator(code)
        if not data:
            print(f"[demographics] {name} ({code}) 取数失败或为空")
        series[name] = data
    if any(series.values()):
        _CACHE = series
        _CACHE_TS = now
    return series


def _tail_years(data: dict[int, float], years: int) -> list[tuple[int, float]]:
    """取最近 years 年，按年份升序。"""
    if not data:
        return []
    ordered = sorted(data.items())
    return ordered[-years:]


def wb_derived_births(series: dict[str, dict[int, float]]) -> dict[int, float]:
    """世行口径推算的出生人口（万人）= 出生率(‰) × 总人口 ÷ 1000 ÷ 10000。"""
    rate = series.get("birth_rate") or {}
    pop = series.get("population") or {}
    return {
        y: round(rate[y] * pop[y] / 1000 / 10000, 1)
        for y in sorted(set(rate) & set(pop))
    }


def analyze_births(series: dict[str, dict[int, float]], years: int = 20) -> dict[str, Any]:
    """新生儿统计。主序列取统计局公布值，世行口径作对照并标出分歧年份。"""
    fert = series.get("fertility") or {}
    wb_births = wb_derived_births(series)

    if not _NBS_BIRTHS_WAN:
        return {"error": "出生人口数据不可用"}

    win = _tail_years(_NBS_BIRTHS_WAN, years)
    y_latest, v_latest = win[-1]
    peak_year, peak_val = max(win, key=lambda kv: kv[1])
    trough_year, trough_val = min(win, key=lambda kv: kv[1])

    prev = dict(win).get(y_latest - 1)
    yoy = round((v_latest / prev - 1) * 100, 1) if prev else None
    from_peak = round((v_latest / peak_val - 1) * 100, 1) if peak_val else None

    rate_win = _tail_years(_NBS_BIRTH_RATE, years)
    fert_win = _tail_years(fert, years)
    fert_latest = fert_win[-1] if fert_win else None

    # 两口径分歧：同年相差超过阈值的列出来，避免读者以为只有一套数
    divergence = []
    for y, v in win:
        w = wb_births.get(y)
        if w is None or not v:
            continue
        diff = (w / v - 1) * 100
        if abs(diff) >= _DIVERGENCE_PCT:
            divergence.append({"year": y, "nbs": v, "wb": w, "diff_pct": round(diff, 1)})

    result: dict[str, Any] = {
        "years": years,
        "range": f"{win[0][0]}~{y_latest}",
        "latest_year": y_latest,
        "latest_births_wan": v_latest,
        "yoy_pct": yoy,
        "peak_year": peak_year,
        "peak_births_wan": peak_val,
        "from_peak_pct": from_peak,
        "trough_year": trough_year,
        "trough_births_wan": trough_val,
        "latest_birth_rate": rate_win[-1][1] if rate_win else None,
        "latest_birth_rate_year": rate_win[-1][0] if rate_win else None,
        "divergence": divergence,
        "divergence_threshold_pct": _DIVERGENCE_PCT,
        "history": {
            "years": [y for y, _ in win],
            "births_wan": [v for _, v in win],
            "births_wan_wb": [wb_births.get(y) for y, _ in win],
            "birth_rate": [dict(rate_win).get(y) for y, _ in win],
            "fertility": [dict(fert_win).get(y) for y, _ in win],
        },
    }

    # 内嵌表是否落后于实时数据源
    wb_last = max(wb_births) if wb_births else None
    if wb_last and wb_last > _NBS_LAST_YEAR:
        result["nbs_stale"] = (
            f"统计局内嵌数据截至 {_NBS_LAST_YEAR} 年，而世界银行已有 {wb_last} 年数据，"
            f"请补录 {_NBS_LAST_YEAR + 1} 年及以后的统计公报数值"
        )
    if fert_latest:
        fy, fv = fert_latest
        result.update(
            latest_fertility=fv,
            latest_fertility_year=fy,
            replacement_level=_REPLACEMENT_FERTILITY,
            fertility_gap_pct=round((fv / _REPLACEMENT_FERTILITY - 1) * 100, 1),
        )

    parts = [f"{y_latest} 年出生人口 {v_latest:.0f} 万"]
    if yoy is not None:
        parts.append(f"同比{'增长' if yoy >= 0 else '下降'}{abs(yoy):.1f}%")
    if from_peak is not None and from_peak < 0:
        parts.append(f"较 {peak_year} 年峰值（{peak_val:.0f} 万）下降 {abs(from_peak):.0f}%")
    if fert_latest:
        parts.append(f"总和生育率 {fert_latest[1]:.2f}，仅为世代更替水平 2.1 的"
                     f"{fert_latest[1] / _REPLACEMENT_FERTILITY * 100:.0f}%")
    result["summary"] = "；".join(parts) + "。"
    return result


def analyze_aging(series: dict[str, dict[int, float]], years: int = 20) -> dict[str, Any]:
    """老龄化统计：65 岁以上占比、老年抚养比、少儿占比，并判定所处阶段。"""
    elderly = series.get("elderly_share") or {}
    dep = series.get("old_dependency") or {}
    child = series.get("child_share") or {}

    if not elderly:
        return {"error": "65 岁以上人口占比数据不可用"}

    win = _tail_years(elderly, years)
    y_latest, v_latest = win[-1]

    # 当前所处阶段：取已跨过的最高门槛
    stage = None
    for thr, name, color, desc in _AGING_STAGES:
        if v_latest >= thr:
            stage = {"threshold": thr, "name": name, "color": color, "desc": desc}
    if stage is None:
        stage = {"threshold": 0.0, "name": "尚未进入老龄化社会", "color": "#22c55e",
                 "desc": "65 岁以上占比低于 7%。"}

    # 下一门槛与距离
    nxt = next((s for s in _AGING_STAGES if v_latest < s[0]), None)
    next_stage = None
    if nxt:
        thr, name, color, desc = nxt
        next_stage = {"threshold": thr, "name": name,
                      "gap_pct_points": round(thr - v_latest, 2)}

    # 各门槛的跨越年份（从完整历史里找首次达到的年份）
    crossings = []
    full = sorted(elderly.items())
    for thr, name, _color, _desc in _AGING_STAGES:
        hit = next((y for y, v in full if v >= thr), None)
        crossings.append({"threshold": thr, "name": name, "year": hit})

    dep_win = _tail_years(dep, years)
    child_win = _tail_years(child, years)

    result: dict[str, Any] = {
        "years": years,
        "range": f"{win[0][0]}~{y_latest}",
        "latest_year": y_latest,
        "elderly_share": round(v_latest, 2),
        "stage": stage,
        "next_stage": next_stage,
        "crossings": crossings,
        "latest_old_dependency": round(dep_win[-1][1], 2) if dep_win else None,
        "latest_child_share": round(child_win[-1][1], 2) if child_win else None,
        "thresholds": [{"threshold": t, "name": n} for t, n, _c, _d in _AGING_STAGES],
        "history": {
            "years": [y for y, _ in win],
            "elderly_share": [round(v, 2) for _, v in win],
            "old_dependency": [dict(dep_win).get(y) for y, _ in win],
            "child_share": [dict(child_win).get(y) for y, _ in win],
        },
    }

    # 20 年间的变化幅度
    if len(win) >= 2:
        y0, v0 = win[0]
        result["change_pct_points"] = round(v_latest - v0, 2)
        result["start_year"] = y0
        result["start_elderly_share"] = round(v0, 2)

    parts = [f"{y_latest} 年 65 岁以上占比 {v_latest:.2f}%，处于「{stage['name']}」"]
    if result.get("change_pct_points") is not None:
        parts.append(f"较 {result['start_year']} 年的 {result['start_elderly_share']}% "
                     f"上升 {result['change_pct_points']} 个百分点")
    if next_stage:
        parts.append(f"距「{next_stage['name']}」门槛（{next_stage['threshold']}%）"
                     f"还差 {next_stage['gap_pct_points']} 个百分点")
    if result.get("latest_child_share") is not None:
        parts.append(f"0-14 岁占比 {result['latest_child_share']}%")
    result["summary"] = "；".join(parts) + "。"
    return result


def analyze_demographics(years: int = 20, force: bool = False) -> dict[str, Any]:
    """新生儿 + 老龄化的合并结果，供 /api/demographics 使用。"""
    series = _fetch_all(force=force)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "世界银行开放数据（api.worldbank.org），底稿为国家统计局与联合国 WPP",
        "source_note": (
            "出生人口由「出生率(‰) × 总人口 ÷ 1000」推算——世界银行不直接发布该字段；"
            "2024 年据此得约 955 万，与统计局公布的 954 万一致。"
            "65 岁以上占比采用联合国口径，与统计局自行公布的数值有零点几个百分点差异，"
            "看趋势无碍，不要与统计局公报逐位对照。"
        ),
        "births": analyze_births(series, years),
        "aging": analyze_aging(series, years),
    }
