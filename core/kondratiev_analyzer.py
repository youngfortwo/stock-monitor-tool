#!/usr/bin/env python3
"""康波周期（Kondratiev Wave）分析器。

基于周金涛（中信建投）康波理论框架，以 ~54 年为一个完整康波周期，
每个周期分为四阶段（回升→繁荣→滞胀→萧条），每阶段约 13-15 年。

历史康波划分（参考周金涛《涛动周期论》及后续研究者修正）：
  第1轮 1784-1842  蒸汽机/纺织
  第2轮 1842-1896  铁路/钢铁
  第3轮 1896-1948  电力/化工
  第4轮 1948-1982  汽车/石油
  第5轮 1982-2015  信息技术/通信
  第6轮 2015-      新能源/AI/生物技术

本模块：
  1. 根据当前日期定位所处康波轮次与阶段
  2. 拉取宏观指标（CPI/PPI/GDP/M1M2）验证阶段判定
  3. 输出阶段特征、资产配置建议、历史类比
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Any

# ── 康波历史划分（start_year, end_year, 技术革命, 四阶段边界）──
# 每阶段 (start_year, end_year, phase_key, phase_name)
_KONDRATIEV_CYCLES: list[dict[str, Any]] = [
    {
        "cycle": 1,
        "start": 1784, "end": 1842,
        "tech": "蒸汽机 / 纺织工业",
        "phases": [
            (1784, 1798, "spring", "回升"),
            (1798, 1812, "summer", "繁荣"),
            (1812, 1827, "autumn", "滞胀"),
            (1827, 1842, "winter", "萧条"),
        ],
    },
    {
        "cycle": 2,
        "start": 1842, "end": 1896,
        "tech": "铁路 / 钢铁",
        "phases": [
            (1842, 1856, "spring", "回升"),
            (1856, 1870, "summer", "繁荣"),
            (1870, 1883, "autumn", "滞胀"),
            (1883, 1896, "winter", "萧条"),
        ],
    },
    {
        "cycle": 3,
        "start": 1896, "end": 1948,
        "tech": "电力 / 化工",
        "phases": [
            (1896, 1910, "spring", "回升"),
            (1910, 1924, "summer", "繁荣"),
            (1924, 1936, "autumn", "滞胀"),
            (1936, 1948, "winter", "萧条"),
        ],
    },
    {
        "cycle": 4,
        "start": 1948, "end": 1982,
        "tech": "汽车 / 石油",
        "phases": [
            (1948, 1962, "spring", "回升"),
            (1962, 1973, "summer", "繁荣"),
            (1973, 1978, "autumn", "滞胀"),
            (1978, 1982, "winter", "萧条"),
        ],
    },
    {
        "cycle": 5,
        "start": 1982, "end": 2015,
        "tech": "信息技术 / 通信",
        "phases": [
            (1982, 1991, "spring", "回升"),
            (1991, 2000, "summer", "繁荣"),
            (2000, 2008, "autumn", "滞胀"),
            (2008, 2015, "winter", "萧条"),
        ],
    },
    {
        "cycle": 6,
        "start": 2015, "end": 2070,
        "tech": "新能源 / AI / 生物技术",
        "phases": [
            (2015, 2019, "spring", "回升"),
            (2019, 2023, "summer", "繁荣"),
            (2023, 2030, "autumn", "滞胀"),
            (2030, 2040, "winter", "萧条"),
        ],
    },
]

# ── 各阶段特征与资产配置建议 ──
_PHASE_INFO: dict[str, dict[str, str]] = {
    "spring": {
        "name": "回升",
        "season": "春",
        "color": "#22c55e",
        "description": "新技术萌芽，经济从萧条中复苏，流动性宽松，风险偏好回升。",
        "features": "利率低位、股市估值修复、新兴产业（本轮：新能源/AI）开始受到关注。",
        "assets": "股票（成长股）> 债券 > 大宗商品 > 现金",
        "risk": "低→中",
        "analogy": "类比 1982-1991（IT 萌芽）或 2015-2019（新能源/AI 起步）",
    },
    "summer": {
        "name": "繁荣",
        "season": "夏",
        "color": "#f59e0b",
        "description": "新技术全面扩散，经济过热，资产泡沫形成，通胀温和上行。",
        "features": "股市牛市、杠杆上升、大宗商品走强、央行开始收紧。",
        "assets": "大宗商品 > 股票 > 房地产 > 债券",
        "risk": "中→高",
        "analogy": "类比 1991-2000（互联网泡沫）或 2019-2021（全球大放水后资产泡沫）",
    },
    "autumn": {
        "name": "滞胀",
        "season": "秋",
        "color": "#ef4444",
        "description": "经济增长放缓但通胀仍高（或通缩压力），资产价格见顶回落，政策两难。",
        "features": "股债双杀风险、大宗商品见顶、现金为王、防御性行业跑赢。",
        "assets": "现金 > 黄金 > 债券 > 股票（防御）> 大宗商品",
        "risk": "高",
        "analogy": "类比 2000-2008（科网泡沫破裂+次贷危机）或 2022-2027（高利率+经济放缓）",
    },
    "winter": {
        "name": "萧条",
        "season": "冬",
        "color": "#6366f1",
        "description": "经济衰退、通缩、资产价格深度回调，为下一轮康波回升积蓄能量。",
        "features": "利率降至极低、股市深度低估、优质资产被错杀、新技术方向开始酝酿。",
        "assets": "现金 > 债券（长债）> 黄金 > 股票（左侧布局）",
        "risk": "极高（但底部机会最大）",
        "analogy": "类比 2008-2015（金融危机后漫长修复）或 2030-2040（预测）",
    },
}

# ── 阶段判别指标：期望"历史分位区间"而非绝对值 ──
#
# 为什么用分位而不是绝对区间：CPI/PPI 的绝对水平在不同时代差异巨大（1990年代
# 中国 CPI 常年 5~15%，2020 年代不足 1%），手写绝对区间既覆盖不到高通胀时代，
# 又会在低通胀环境下让春/秋/冬三个阶段的区间彼此重叠、丧失区分力。改用各指标
# 在自身历史中的分位后，判据自动适应所处时代的均值水平。
#
# 四个指标的经济含义与方向：
#   spread  股债利差分位：越高 = 股票相对债券越便宜（底部特征）
#   credit  信用脉冲分位：越高 = 信用扩张越快（宽松/加杠杆）
#   pmi     制造业PMI分位：越高 = 产能越紧张（替代产能利用率，后者无稳定数据源）
#   m1m2    M1-M2剪刀差分位：越高 = 资金活化程度越高（扩张）
#
# 区分力主要来自 spread：春(60~95) 与 夏(0~30) 完全分离，秋(20~60) 与 冬(65~100)
# 也基本分离。credit/pmi/m1m2 在春夏（同为扩张期）和秋冬（同为收缩期）内部存在
# 重叠，属经济现实，它们的作用是佐证而非主判别。
_PHASE_INDICATOR_PROFILE: dict[str, dict[str, tuple[float, float]]] = {
    "spring": {"spread": (60, 95),  "credit": (60, 100), "pmi": (50, 90),  "m1m2": (55, 95)},
    "summer": {"spread": (0, 30),   "credit": (45, 85),  "pmi": (65, 100), "m1m2": (50, 90)},
    "autumn": {"spread": (20, 60),  "credit": (10, 50),  "pmi": (20, 55),  "m1m2": (5, 50)},
    "winter": {"spread": (65, 100), "credit": (0, 35),   "pmi": (0, 35),   "m1m2": (0, 40)},
}

_INDICATOR_LABELS: dict[str, str] = {
    "spread": "股债利差",
    "credit": "信用脉冲",
    "pmi": "制造业PMI",
    "m1m2": "M1-M2剪刀差",
}

_INDICATOR_UNITS: dict[str, str] = {
    "spread": "%",
    "credit": "%",
    "pmi": "",
    "m1m2": "%",
}


def _locate_phase(year: int) -> dict[str, Any]:
    """根据年份定位康波轮次与阶段，返回完整定位信息。"""
    for cycle in _KONDRATIEV_CYCLES:
        if cycle["start"] <= year < cycle["end"]:
            for p_start, p_end, p_key, p_name in cycle["phases"]:
                if p_start <= year < p_end:
                    phase_info = _PHASE_INFO[p_key]
                    elapsed = year - p_start
                    total = p_end - p_start
                    progress = round(elapsed / total * 100, 1) if total > 0 else 0
                    return {
                        "cycle": cycle["cycle"],
                        "cycle_start": cycle["start"],
                        "cycle_end": cycle["end"],
                        "tech": cycle["tech"],
                        "phase_key": p_key,
                        "phase_name": p_name,
                        "phase_season": phase_info["season"],
                        "phase_color": phase_info["color"],
                        "phase_start": p_start,
                        "phase_end": p_end,
                        "phase_progress_pct": progress,
                        "phase_elapsed_years": elapsed,
                        "phase_total_years": total,
                        "phase_remaining_years": total - elapsed,
                    }
            # 年份在轮次内但超出已知阶段（第6轮远期预测）
            last = cycle["phases"][-1]
            return {
                "cycle": cycle["cycle"],
                "cycle_start": cycle["start"],
                "cycle_end": cycle["end"],
                "tech": cycle["tech"],
                "phase_key": last[2],
                "phase_name": last[3] + "（延续）",
                "phase_season": _PHASE_INFO[last[2]]["season"],
                "phase_color": _PHASE_INFO[last[2]]["color"],
                "phase_start": last[0],
                "phase_end": last[1],
                "phase_progress_pct": 100.0,
                "phase_elapsed_years": year - last[0],
                "phase_total_years": last[1] - last[0],
                "phase_remaining_years": 0,
            }
    return {}


def _with_retry(fn, attempts: int = 3, wait: float = 1.0):
    """取数重试。异常与"成功但结果为空"都要重试。

    限流时 akshare 常常不抛异常、而是返回空 DataFrame。若只对异常重试，空结果会
    静默降级且日志无线索。全部失败时返回空列表而非抛出：单个指标缺失只降低置信度，
    不应让整个分析失败。
    """
    last = None
    for i in range(attempts):
        try:
            out = fn()
            if out:
                return out
            last = "返回空结果（通常是上游限流）"
        except Exception as exc:
            last = exc
        if i < attempts - 1:
            time.sleep(wait * (i + 1))
    print(f"[kondratiev] {getattr(fn, '__name__', fn)} 取数失败（重试{attempts}次）: {last}")
    return []


def _norm_month(raw: Any) -> str:
    """把各数据源的月份写法统一成 'YYYY-MM'。

    已知输入形态：'201501'（社融）、'2026年08月份'（PMI）、'2026-08-31'（PE/国债）、
    '2026.8'（货币供应量，月份不补零）。归一化是分位数正确性的前提——未归一化的
    '2026.10' 会在字符串排序中落到 '2026.2' 之前，导致序列错序、分位数失真。
    """
    s = str(raw).strip()
    if "年" in s:
        try:
            y = s.split("年")[0]
            m = s.split("年")[1].split("月")[0]
            return f"{int(y):04d}-{int(m):02d}"
        except (ValueError, IndexError):
            return s
    if "-" in s:
        parts = s[:7].split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0]):04d}-{int(parts[1]):02d}"
        return s
    if "." in s:
        parts = s.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0]):04d}-{int(parts[1]):02d}"
        return s
    if s.isdigit() and len(s) == 6:
        return f"{s[:4]}-{s[4:]}"
    return s


def _fetch_spread_series() -> list[tuple[str, float]]:
    """沪深300股债利差月度序列：盈利收益率(1/PE-TTM) − 10年期国债收益率。

    PE 用乐咕乐股月频序列（2005 起），国债收益率取该月最后一个交易日的值。
    返回 [(YYYY-MM, spread_pct), ...] 按月份升序。
    """
    import akshare as ak
    import pandas as pd

    pe_df = ak.stock_index_pe_lg()
    if pe_df is None or pe_df.empty:
        return []
    bond_df = ak.bond_zh_us_rate()
    if bond_df is None or bond_df.empty:
        return []

    bond_df = bond_df.dropna(subset=["中国国债收益率10年"]).copy()
    bond_df["_month"] = bond_df["日期"].astype(str).str[:7]
    # 每月最后一条观测代表该月利率水平
    bond_by_month = (bond_df.groupby("_month")["中国国债收益率10年"]
                     .last().to_dict())

    out: list[tuple[str, float]] = []
    for _, row in pe_df.iterrows():
        month = str(row["日期"])[:7]
        try:
            pe = float(row["滚动市盈率"])
        except (ValueError, TypeError):
            continue
        if pe <= 0 or pe != pe:
            continue
        bond = bond_by_month.get(month)
        if bond is None or bond != bond:
            continue
        out.append((month, round(100.0 / pe - float(bond), 3)))
    out.sort(key=lambda x: x[0])
    return out


def _fetch_credit_impulse_series() -> list[tuple[str, float]]:
    """信用脉冲月度序列。

    定义：社融增量 12 个月滚动和的同比变化率（%）。
    标准信用脉冲是"新增信用流量的变化 / GDP"，此处用滚动和的同比增速作为
    无量纲替代——它同样刻画信用的二阶变化（加速/减速），且避免了季度 GDP
    累计值对齐带来的口径风险。由于打分基于历史分位，绝对量纲不影响判别。

    社融数据自 2015-01 起，需 24 个月预热，故首个有效值约在 2017-01。
    返回 [(YYYY-MM, impulse_pct), ...] 按月份升序。
    """
    import akshare as ak

    df = ak.macro_china_shrzgm()
    if df is None or df.empty:
        return []

    rows: list[tuple[str, float]] = []
    for _, r in df.iterrows():
        try:
            rows.append((_norm_month(r["月份"]), float(r["社会融资规模增量"])))
        except (ValueError, TypeError):
            continue
    rows.sort(key=lambda x: x[0])
    if len(rows) < 24:
        return []

    vals = [v for _, v in rows]
    roll: list[float | None] = []
    for i in range(len(vals)):
        roll.append(sum(vals[i - 11:i + 1]) if i >= 11 else None)

    out: list[tuple[str, float]] = []
    for i in range(len(rows)):
        cur, prev = roll[i], roll[i - 12] if i >= 12 else None
        if cur is None or prev is None or prev == 0:
            continue
        out.append((rows[i][0], round((cur / prev - 1) * 100, 3)))
    return out


def _fetch_pmi_series() -> list[tuple[str, float]]:
    """制造业 PMI 月度序列（2008 起）。

    替代产能利用率：国家统计局的工业产能利用率为季度发布且无稳定 API，
    PMI 同样衡量产能松紧，月频且有天然的 50 荣枯线。
    """
    import akshare as ak

    df = ak.macro_china_pmi()
    if df is None or df.empty:
        return []
    out: list[tuple[str, float]] = []
    for _, r in df.iterrows():
        try:
            v = float(r["制造业-指数"])
        except (ValueError, TypeError):
            continue
        if v != v:
            continue
        out.append((_norm_month(r["月份"]), round(v, 2)))
    out.sort(key=lambda x: x[0])
    return out


def _fetch_m1m2_series() -> list[tuple[str, float]]:
    """M1-M2 剪刀差月度序列（同比增速之差，%）。

    akshare 的 macro_china_supply_of_money 上游会间歇性返回无法解码的响应
    （JSONDecodeError: No value to decode），故与 stock_server._fetch_m1m2_data
    采用同样的多源策略：akshare 优先，失败回退东财数据中心。
    """
    out: list[tuple[str, float]] = []

    # 方案1: akshare（历史最长，可回溯至 1978）
    try:
        import akshare as ak
        df = ak.macro_china_supply_of_money()
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                try:
                    m1 = float(r["货币(狭义货币M1)同比增长"])
                    m2 = float(r["货币和准货币（广义货币M2）同比增长"])
                except (ValueError, TypeError):
                    continue
                if m1 != m1 or m2 != m2:
                    continue
                out.append((_norm_month(r["统计时间"]), round(m1 - m2, 3)))
    except Exception:
        out = []

    # 方案2: 东财数据中心 RPT_ECONOMY_CURRENCY_SUPPLY
    if not out:
        try:
            import requests
            resp = requests.get(
                "https://datacenter-web.eastmoney.com/api/data/v1/get",
                params={
                    "sortColumns": "REPORT_DATE", "sortTypes": -1,
                    "pageSize": 600, "pageNumber": 1,
                    "reportName": "RPT_ECONOMY_CURRENCY_SUPPLY",
                    "columns": "REPORT_DATE,TIME,BASIC_CURRENCY_SAME,CURRENCY_SAME",
                },
                headers={"User-Agent": "Mozilla/5.0",
                         "Referer": "https://data.eastmoney.com/cjsj/hbgyl.html"},
                timeout=15,
            )
            data = resp.json()
            for item in (data.get("result") or {}).get("data") or []:
                m1, m2 = item.get("CURRENCY_SAME"), item.get("BASIC_CURRENCY_SAME")
                if m1 is None or m2 is None:
                    continue
                month = _norm_month(str(item.get("TIME", "")).replace("份", ""))
                out.append((month, round(float(m1) - float(m2), 3)))
        except Exception:
            pass

    out.sort(key=lambda x: x[0])
    return out


def _percentile(values: list[float], current: float) -> float:
    """current 在 values 中的百分位（0-100）。values 应只含当期及更早的数据。"""
    if not values:
        return 50.0
    below = sum(1 for v in values if v < current)
    equal = sum(1 for v in values if v == current)
    return round((below + equal / 2) / len(values) * 100, 1)


def _band_score(pct: float, band: tuple[float, float]) -> float:
    """分位 pct 相对期望区间 band 的吻合分（0-100）。

    区间内 80~100 分（越接近中心越高），区间外按超出距离线性衰减。
    """
    lo, hi = band
    center = (lo + hi) / 2
    half = (hi - lo) / 2 or 1.0
    d = abs(pct - center) / half
    if d <= 1:
        return round(100 - 20 * d, 1)
    return round(max(0.0, 80 - 40 * (d - 1)), 1)


def _score_all_phases(percentiles: dict[str, float | None]) -> list[dict[str, Any]]:
    """对四个阶段分别打分并按分数降序排名。

    与旧实现的关键差别：旧版只检验"数据是否符合日历给出的阶段"，无法暴露
    多个阶段同时满足的情形；此处对四个阶段并行打分，矛盾会直接体现在排名上。
    """
    ranking: list[dict[str, Any]] = []
    for key, profile in _PHASE_INDICATOR_PROFILE.items():
        per_ind: list[dict[str, Any]] = []
        scores: list[float] = []
        for ind, band in profile.items():
            pct = percentiles.get(ind)
            if pct is None:
                per_ind.append({"indicator": ind, "label": _INDICATOR_LABELS[ind],
                                "percentile": None, "band": list(band), "score": None})
                continue
            s = _band_score(pct, band)
            scores.append(s)
            per_ind.append({"indicator": ind, "label": _INDICATOR_LABELS[ind],
                            "percentile": pct, "band": list(band), "score": s})
        overall = round(sum(scores) / len(scores), 1) if scores else None
        ranking.append({
            "phase_key": key,
            "phase_name": _PHASE_INFO[key]["name"],
            "season": _PHASE_INFO[key]["season"],
            "color": _PHASE_INFO[key]["color"],
            "score": overall,
            "indicators": per_ind,
        })
    ranking.sort(key=lambda x: (x["score"] is not None, x["score"] or 0), reverse=True)
    return ranking


_SERIES_FETCHERS = {
    "spread": _fetch_spread_series,
    "credit": _fetch_credit_impulse_series,
    "pmi": _fetch_pmi_series,
    "m1m2": _fetch_m1m2_series,
}


def _fetch_all_series() -> dict[str, list[tuple[str, float]]]:
    """一次取齐四条指标序列，供快照与回测共用（避免重复网络开销）。"""
    return {key: _with_retry(fn) for key, fn in _SERIES_FETCHERS.items()}


def _snapshot_from_series(all_series: dict[str, list[tuple[str, float]]]) -> dict[str, Any]:
    """由已取得的序列计算最新值 + 历史分位（单项缺失不影响其他）。"""
    snapshot: dict[str, Any] = {}
    percentiles: dict[str, float | None] = {}
    for key in _SERIES_FETCHERS:
        series = all_series.get(key) or []
        if not series:
            snapshot[key] = {"label": _INDICATOR_LABELS[key], "value": None,
                             "month": None, "percentile": None, "history_n": 0,
                             "direction": None}
            percentiles[key] = None
            continue
        month, value = series[-1]
        vals = [v for _, v in series]
        pct = _percentile(vals, value)
        # 方向：与 6 个月前比较（展示用，不参与打分）
        direction = None
        if len(series) >= 7:
            prev = series[-7][1]
            direction = "上行" if value > prev else ("下行" if value < prev else "走平")
        snapshot[key] = {
            "label": _INDICATOR_LABELS[key],
            "unit": _INDICATOR_UNITS[key],
            "value": value,
            "month": month,
            "percentile": pct,
            "history_n": len(series),
            "history_from": series[0][0],
            "direction": direction,
        }
        percentiles[key] = pct
    return {"snapshot": snapshot, "percentiles": percentiles}


def _fetch_context_macro() -> dict[str, float | None]:
    """CPI/PPI/GDP 同比：仅作背景信息展示，不参与阶段打分。

    这三项的绝对水平在不同时代不可比，且在低通胀环境下无法区分春/秋/冬，
    故从判别体系中移除，只保留展示。
    """
    result: dict[str, float | None] = {"cpi_yoy": None, "ppi_yoy": None, "gdp_yoy": None}
    try:
        import akshare as ak
        cpi = ak.macro_china_cpi()
        if cpi is not None and not cpi.empty:
            v = float(cpi.iloc[0]["全国-同比增长"])
            result["cpi_yoy"] = v if v == v else None
    except Exception:
        pass
    try:
        import akshare as ak
        ppi = ak.macro_china_ppi()
        if ppi is not None and not ppi.empty:
            v = float(ppi.iloc[0]["当月同比增长"])
            result["ppi_yoy"] = v if v == v else None
    except Exception:
        pass
    try:
        import akshare as ak
        gdp = ak.macro_china_gdp()
        if gdp is not None and not gdp.empty:
            v = float(gdp.iloc[0]["国内生产总值-同比增长"])
            result["gdp_yoy"] = v if v == v else None
    except Exception:
        pass
    return result


def analyze_kondratiev(as_of: dt.date | None = None) -> dict[str, Any]:
    """康波周期完整分析，返回当前定位 + 阶段解读 + 宏观验证 + 历史时间轴。"""
    today = as_of or dt.date.today()
    year = today.year
    position = _locate_phase(year)
    if not position:
        return {"error": f"年份 {year} 超出已知康波范围", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    phase_key = position["phase_key"]
    phase_info = _PHASE_INFO[phase_key]

    all_series = _fetch_all_series()
    indicators = _snapshot_from_series(all_series)
    ranking = _score_all_phases(indicators["percentiles"])
    context = _fetch_context_macro()

    # 区分力实测：复用同一批序列，避免重复取数
    try:
        validation = _backtest_from_series(all_series, "2010-01")
        validation.pop("rows", None)  # 逐月明细不下发前端
    except Exception:
        validation = {}

    # 日历判定 vs 数据判定：不一致时如实暴露，而非强行给出"吻合"
    scored = [r for r in ranking if r["score"] is not None]
    data_phase = scored[0] if scored else None
    calendar_rank = next(
        (i + 1 for i, r in enumerate(ranking) if r["phase_key"] == phase_key), None
    )
    calendar_score = next(
        (r["score"] for r in ranking if r["phase_key"] == phase_key), None
    )
    margin = None
    if data_phase and len(scored) >= 2:
        margin = round(scored[0]["score"] - scored[1]["score"], 1)

    # 可用指标数量决定结论的可信程度：判别力来自四个指标的交叉印证，
    # 单一指标就能把某个阶段推到第一名（区间必然命中其中之一），此时的排名没有意义。
    available = sum(1 for v in indicators["percentiles"].values() if v is not None)
    total_inds = len(_PHASE_INDICATOR_PROFILE["spring"])

    if not scored:
        agreement, confidence = "无数据", "无数据"
    elif available < 2:
        agreement = "指标不足"
        confidence = "低"
    elif calendar_rank == 1:
        agreement = "一致"
        confidence = "高" if (margin or 0) >= 10 and available == total_inds else "中"
    elif calendar_rank == 2:
        agreement = "接近"
        confidence = "中" if available >= 3 else "低"
    else:
        agreement = "矛盾"
        confidence = "低"

    alignment = {
        "agreement": agreement,
        "confidence": confidence,
        "calendar_phase": phase_info["name"],
        "calendar_rank": calendar_rank,
        "calendar_score": calendar_score,
        "data_phase": data_phase["phase_name"] if data_phase else None,
        "data_phase_key": data_phase["phase_key"] if data_phase else None,
        "data_phase_score": data_phase["score"] if data_phase else None,
        "top_margin": margin,
        "indicators_available": available,
        "indicators_total": total_inds,
        "method": "各指标取自身历史分位，对四阶段并行打分后排名；日历定位为主判据，本表为独立验证",
    }
    if available < total_inds:
        missing = [_INDICATOR_LABELS[k] for k, v in indicators["percentiles"].items()
                   if v is None]
        alignment["data_warning"] = (
            f"仅 {available}/{total_inds} 个指标取到数据（缺失：{'、'.join(missing)}），"
            f"排名与命中率仅供参考，请点刷新重试"
        )

    # 历史时间轴（供前端渲染）
    timeline = []
    for cycle in _KONDRATIEV_CYCLES:
        phases_out = []
        for p_start, p_end, p_key, p_name in cycle["phases"]:
            phases_out.append({
                "start": p_start, "end": p_end,
                "key": p_key, "name": p_name,
                "season": _PHASE_INFO[p_key]["season"],
                "color": _PHASE_INFO[p_key]["color"],
                "is_current": (cycle["cycle"] == position["cycle"]
                               and p_key == phase_key
                               and p_start <= year < p_end),
            })
        timeline.append({
            "cycle": cycle["cycle"],
            "start": cycle["start"], "end": cycle["end"],
            "tech": cycle["tech"],
            "phases": phases_out,
            "is_current": cycle["cycle"] == position["cycle"],
        })

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": today.isoformat(),
        "position": position,
        "phase": {
            "key": phase_key,
            "name": phase_info["name"],
            "season": phase_info["season"],
            "color": phase_info["color"],
            "description": phase_info["description"],
            "features": phase_info["features"],
            "assets": phase_info["assets"],
            "risk": phase_info["risk"],
            "analogy": phase_info["analogy"],
        },
        "indicators": indicators["snapshot"],
        "phase_ranking": ranking,
        "macro_alignment": alignment,
        "context_macro": context,
        "validation": validation,
        "timeline": timeline,
        "summary": (
            f"当前处于第{position['cycle']}轮康波（{position['tech']}）的"
            f"「{phase_info['name']}」阶段（{phase_info['season']}），"
            f"本轮阶段已进行 {position['phase_elapsed_years']} 年"
            f"（进度 {position['phase_progress_pct']:.0f}%），"
            f"预计 {position['phase_end']} 年前后进入下一阶段。"
            + (
                f"判别指标最像「{alignment['data_phase']}」，与日历定位{alignment['agreement']}"
                f"（置信度{alignment['confidence']}）。"
                if alignment["data_phase"] else "判别指标暂无数据。"
            )
        ),
    }


def backtest_discrimination(start_month: str = "2010-01") -> dict[str, Any]:
    """回测入口：自行取数后委托 _backtest_from_series。"""
    return _backtest_from_series(_fetch_all_series(), start_month)


def _backtest_from_series(series: dict[str, list[tuple[str, float]]],
                          start_month: str = "2010-01") -> dict[str, Any]:
    """回测判别体系的区分力：逐月用当期及更早数据打分，与日历标签比对。

    分位数按扩张窗口计算（只用当期及之前的数据），避免未来信息泄漏。
    返回整体命中率、按阶段拆分的命中率，以及逐月明细。
    """
    failed = [k for k in _SERIES_FETCHERS if not series.get(k)]
    as_dict = {k: dict(v or []) for k, v in series.items()}
    months = sorted({m for v in series.values() for m, _ in (v or []) if m >= start_month})

    rows: list[dict[str, Any]] = []
    for month in months:
        year = int(month[:4])
        pos = _locate_phase(year)
        if not pos:
            continue
        calendar_key = pos["phase_key"]

        percentiles: dict[str, float | None] = {}
        for key, sr in series.items():
            cur = as_dict[key].get(month)
            if cur is None:
                percentiles[key] = None
                continue
            past = [v for m, v in (sr or []) if m <= month]
            # 分位数至少需要 24 个观测才有意义
            percentiles[key] = _percentile(past, cur) if len(past) >= 24 else None

        ranking = _score_all_phases(percentiles)
        scored = [r for r in ranking if r["score"] is not None]
        if not scored:
            continue
        top = scored[0]
        cal_rank = next(
            (i + 1 for i, r in enumerate(ranking) if r["phase_key"] == calendar_key), None
        )
        rows.append({
            "month": month,
            "calendar_phase": _PHASE_INFO[calendar_key]["name"],
            "calendar_key": calendar_key,
            "data_phase": top["phase_name"],
            "data_key": top["phase_key"],
            "hit": top["phase_key"] == calendar_key,
            "top2": cal_rank is not None and cal_rank <= 2,
            "calendar_rank": cal_rank,
        })

    n = len(rows)
    hit = sum(1 for r in rows if r["hit"])
    top2 = sum(1 for r in rows if r["top2"])
    by_phase: dict[str, dict[str, Any]] = {}
    for r in rows:
        b = by_phase.setdefault(r["calendar_phase"], {"n": 0, "hit": 0, "top2": 0})
        b["n"] += 1
        b["hit"] += 1 if r["hit"] else 0
        b["top2"] += 1 if r["top2"] else 0
    for b in by_phase.values():
        b["hit_rate"] = round(b["hit"] / b["n"] * 100, 1) if b["n"] else 0.0
        b["top2_rate"] = round(b["top2"] / b["n"] * 100, 1) if b["n"] else 0.0

    return {
        "months_tested": n,
        "range": f"{rows[0]['month']} ~ {rows[-1]['month']}" if rows else "",
        "hit_rate": round(hit / n * 100, 1) if n else 0.0,
        "top2_rate": round(top2 / n * 100, 1) if n else 0.0,
        "by_phase": by_phase,
        "series_lengths": {k: len(v) for k, v in series.items()},
        "failed_sources": failed,
        "rows": rows,
    }
