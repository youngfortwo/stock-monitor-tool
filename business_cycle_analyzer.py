#!/usr/bin/env python3
"""中短周期分析器：基钦（库存）/ 朱格拉（设备投资）/ 库兹涅茨（建筑地产）。

与康波模块的根本区别：这三个周期足够短，可用数据里存在多个完整循环，因此
阶段判定**直接从指标自身的波动位置推导**，不依赖手写日历表。周期长度也由
波谷检测实测得出，再与理论值对比——如果实测长度与理论严重偏离，说明该周期
在数据上并不成立，这一点会如实输出而非掩盖。

四阶段统一按（水平分位 × 变化方向）划分：
    分位低 + 上行 → 复苏
    分位高 + 上行 → 扩张
    分位高 + 下行 → 见顶回落
    分位低 + 下行 → 收缩

数据充分性：一个周期至少要有 3 个完整循环才谈得上统计可信。朱格拉与库兹涅茨
的可用数据分别只覆盖约 1.4 / 1.5 个循环，结论会标注为数据不足。
"""
from __future__ import annotations

import time
from typing import Any

# ── 周期定义 ──
# theory_months: 理论周期长度（月）
# smooth: 平滑窗口（月），用于压掉月度噪声后再做波谷检测
# min_cycles_for_confidence: 达到统计可信所需的最少完整循环数
_CYCLES: dict[str, dict[str, Any]] = {
    "kitchin": {
        "name": "基钦周期",
        "subtitle": "库存周期",
        "theory_months": 40,
        "theory_label": "约 3~4 年（40 个月）",
        "smooth": 7,
        "color": "#0ea5e9",
        "indicator": "PPI 同比",
        "indicator_note": (
            "工业企业产成品存货同比无稳定数据源（统计局仅提供通用树查询接口），"
            "此处以 PPI 同比作核心指标——它是中国库存周期公认的同步指标，"
            "但它是价格序列而非存货序列，属代理指标。"
        ),
        "corroborator": "工业增加值同比",
        "phase_names": {
            "recovery": "被动去库存（复苏）",
            "expansion": "主动补库存（过热）",
            "peak": "被动补库存（滞胀）",
            "contraction": "主动去库存（衰退）",
        },
        "phase_detail": {
            "recovery": {
                "desc": "需求回暖但企业尚未加产，库存被动消耗，价格开始企稳回升。",
                "implication": "周期资产最佳布局窗口：上游资源、工业品率先反应。",
                "risk": "低",
            },
            "expansion": {
                "desc": "需求确认，企业主动加产补库，价格与利润同步上行。",
                "implication": "盈利最强阶段，周期股/资源股表现最好，但已进入中后段。",
                "risk": "中",
            },
            "peak": {
                "desc": "需求转弱而库存仍在累积，价格见顶回落，利润受挤压。",
                "implication": "减配周期股，警惕业绩下修；防御性行业相对占优。",
                "risk": "高",
            },
            "contraction": {
                "desc": "需求低迷，企业主动砍产降库，价格与利润同步下行。",
                "implication": "周期底部酝酿期，等待价格企稳与需求回暖的确认信号。",
                "risk": "中高（但底部机会在酝酿）",
            },
        },
    },
    "juglar": {
        "name": "朱格拉周期",
        "subtitle": "设备投资周期",
        "theory_months": 108,
        "theory_label": "约 8~10 年",
        "smooth": 13,
        "color": "#8b5cf6",
        "indicator": "固定资产投资同比",
        "indicator_note": (
            "固定资产投资同比自 2012 年起可得，仅覆盖约 1.4 个理论循环，"
            "不足以统计验证周期长度，阶段判定仅反映当前指标位置。"
        ),
        "corroborator": None,
        "phase_names": {
            "recovery": "投资复苏",
            "expansion": "投资扩张",
            "peak": "投资见顶",
            "contraction": "投资收缩",
        },
        "phase_detail": {
            "recovery": {
                "desc": "产能利用率回升，企业开始重启资本开支，设备订单见到改善。",
                "implication": "设备制造、工业自动化、工程机械进入景气上行早期。",
                "risk": "低",
            },
            "expansion": {
                "desc": "资本开支全面铺开，中游设备订单饱满，产能建设加速。",
                "implication": "设备与资本品景气高点，但新增产能已在路上。",
                "risk": "中",
            },
            "peak": {
                "desc": "前期新增产能陆续投产，订单增速回落，产能利用率见顶。",
                "implication": "警惕产能过剩与价格战，减配重资产、高资本开支行业。",
                "risk": "高",
            },
            "contraction": {
                "desc": "产能过剩，资本开支收缩，设备更新推迟，投资增速低迷。",
                "implication": "等待产能出清；出清越彻底，下一轮设备周期弹性越大。",
                "risk": "中高",
            },
        },
    },
    "kuznets": {
        "name": "库兹涅茨周期",
        "subtitle": "建筑 / 房地产周期",
        "theory_months": 216,
        "theory_label": "约 18~20 年",
        "smooth": 25,
        "color": "#f97316",
        "indicator": "国房景气指数",
        "indicator_note": (
            "国房景气指数自 1998 年起可得，仅覆盖约 1.5 个理论循环，"
            "不足以统计验证周期长度。该序列更新偏滞后，注意数据月份。"
        ),
        "corroborator": None,
        "phase_names": {
            "recovery": "地产回升",
            "expansion": "地产扩张",
            "peak": "地产见顶",
            "contraction": "地产下行",
        },
        "phase_detail": {
            "recovery": {
                "desc": "销售与新开工回暖，库存去化加快，景气指数自低位回升。",
                "implication": "地产链（建材、家居、工程机械）需求改善。",
                "risk": "低",
            },
            "expansion": {
                "desc": "投资与新开工高增，土地与信用扩张，景气处于高位。",
                "implication": "地产链景气高位，但对利率与政策收紧最敏感。",
                "risk": "中",
            },
            "peak": {
                "desc": "销售转弱而投资惯性仍在，库存开始累积，景气自高位回落。",
                "implication": "减配地产链，关注房企现金流与信用风险。",
                "risk": "高",
            },
            "contraction": {
                "desc": "销售与投资双降，去库存漫长，景气持续低迷。",
                "implication": "地产链承压；该阶段跨度可达数年，需等政策与人口两端的转向信号。",
                "risk": "高",
            },
        },
    },
}

# 四阶段的循环顺序，用于展示"下一阶段"
_PHASE_ORDER = ["recovery", "expansion", "peak", "contraction"]

_MIN_CYCLES_FOR_CONFIDENCE = 3


def _norm_month(raw: Any) -> str:
    """统一月份写法为 'YYYY-MM'（见 kondratiev_analyzer 同名函数的踩坑说明）。"""
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


def _with_retry(fn, attempts: int = 3, wait: float = 1.0):
    """取数重试。异常与"成功但结果为空"都要重试。

    限流时 akshare 常常不抛异常、而是返回空 DataFrame。若只对异常重试，空结果会
    静默降级成"数据源不可用"且日志里没有任何线索，排查时无从下手。
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
    print(f"[cycles] {getattr(fn, '__name__', fn)} 取数失败（重试{attempts}次）: {last}")
    return []


def _series_from_df(df, month_col: str, value_col: str) -> list[tuple[str, float]]:
    """DataFrame → 按月份升序的 [(YYYY-MM, value)]，跳过空值。"""
    out: list[tuple[str, float]] = []
    if df is None or getattr(df, "empty", True):
        return out
    for _, r in df.iterrows():
        try:
            v = float(r[value_col])
        except (ValueError, TypeError, KeyError):
            continue
        if v != v:  # NaN
            continue
        out.append((_norm_month(r[month_col]), round(v, 3)))
    out.sort(key=lambda x: x[0])
    return out


def _fetch_ppi_yoy() -> list[tuple[str, float]]:
    """PPI 同比（2006 起）。当月同比缺失时用"当月"指数减 100 兜底。"""
    import akshare as ak

    df = ak.macro_china_ppi()
    out = _series_from_df(df, "月份", "当月同比增长")
    if out:
        return out
    rows = _series_from_df(df, "月份", "当月")
    return [(m, round(v - 100.0, 3)) for m, v in rows]


def _fetch_industrial_yoy() -> list[tuple[str, float]]:
    """工业增加值同比（2008 起），作基钦周期的需求侧佐证。"""
    import akshare as ak
    return _series_from_df(ak.macro_china_gyzjz(), "月份", "同比增长")


def _fetch_fai_yoy() -> list[tuple[str, float]]:
    """固定资产投资同比（2012 起）。"""
    import akshare as ak
    return _series_from_df(ak.macro_china_gdzctz(), "月份", "同比增长")


def _fetch_real_estate_index() -> list[tuple[str, float]]:
    """国房景气指数（1998 起）。该接口用"日期"列且为日期格式。"""
    import akshare as ak
    return _series_from_df(ak.macro_china_real_estate(), "日期", "最新值")


_FETCHERS = {
    "kitchin": _fetch_ppi_yoy,
    "juglar": _fetch_fai_yoy,
    "kuznets": _fetch_real_estate_index,
}


def _smooth(values: list[float], window: int) -> list[float | None]:
    """居中移动平均。两端不足半窗的位置返回 None，避免用残缺窗口造出假拐点。"""
    if window <= 1:
        return list(values)
    half = window // 2
    out: list[float | None] = []
    for i in range(len(values)):
        if i < half or i >= len(values) - half:
            out.append(None)
            continue
        seg = values[i - half:i + half + 1]
        out.append(sum(seg) / len(seg))
    return out


def _find_troughs(smoothed: list[float | None], min_gap: int) -> list[int]:
    """在平滑序列上找局部极小点，相邻波谷至少间隔 min_gap 个月。

    不用 scipy.argrelextrema：它对 None 缺口和平台期处理麻烦，这里的朴素实现
    更可控——先取严格局部极小，再按 min_gap 贪心去重（保留更低的那个）。
    """
    idx = [i for i, v in enumerate(smoothed) if v is not None]
    if len(idx) < 3:
        return []
    raw: list[int] = []
    for k in range(1, len(idx) - 1):
        i_prev, i, i_next = idx[k - 1], idx[k], idx[k + 1]
        v_prev, v, v_next = smoothed[i_prev], smoothed[i], smoothed[i_next]
        if v <= v_prev and v <= v_next and (v < v_prev or v < v_next):
            raw.append(i)
    if not raw:
        return []
    kept: list[int] = [raw[0]]
    for i in raw[1:]:
        if i - kept[-1] < min_gap:
            if smoothed[i] < smoothed[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    return kept


def _percentile(values: list[float], current: float) -> float:
    if not values:
        return 50.0
    below = sum(1 for v in values if v < current)
    equal = sum(1 for v in values if v == current)
    return round((below + equal / 2) / len(values) * 100, 1)


def _classify_phase(pct: float, direction: str, phase_names: dict[str, str]) -> tuple[str, str]:
    """(水平分位, 方向) → (阶段 key, 阶段名)。分位 50 为高低分界。"""
    high = pct >= 50
    if direction == "上行":
        key = "expansion" if high else "recovery"
    elif direction == "下行":
        key = "peak" if high else "contraction"
    else:
        key = "expansion" if high else "contraction"
    return key, phase_names[key]


def analyze_cycle(cycle_key: str, series: list[tuple[str, float]],
                  corroborator: list[tuple[str, float]] | None = None) -> dict[str, Any]:
    """单个周期的完整分析：实测周期长度 + 当前阶段 + 数据充分性。"""
    cfg = _CYCLES[cycle_key]
    theory = cfg["theory_months"]
    result: dict[str, Any] = {
        "key": cycle_key,
        "name": cfg["name"],
        "subtitle": cfg["subtitle"],
        "color": cfg["color"],
        "theory_months": theory,
        "theory_label": cfg["theory_label"],
        "indicator": cfg["indicator"],
        "indicator_note": cfg["indicator_note"],
    }
    if not series:
        result.update(error="数据源不可用", months=0)
        return result

    months = [m for m, _ in series]
    values = [v for _, v in series]
    result.update(months=len(series), range=f"{months[0]} ~ {months[-1]}",
                  latest_month=months[-1], latest_value=values[-1])

    # ── 实测周期长度：平滑后找波谷，取相邻波谷间隔 ──
    smoothed = _smooth(values, cfg["smooth"])
    troughs = _find_troughs(smoothed, min_gap=max(6, int(theory * 0.5)))
    intervals = [troughs[i] - troughs[i - 1] for i in range(1, len(troughs))]
    measured = round(sum(intervals) / len(intervals), 1) if intervals else None
    result["troughs"] = [months[i] for i in troughs]
    result["measured_intervals_months"] = intervals
    result["measured_cycle_months"] = measured
    if measured:
        result["theory_deviation_pct"] = round((measured / theory - 1) * 100, 1)
    else:
        # 相邻波谷间隔至少需要 2 个波谷。检出不足时如实说明原因，
        # 放宽 min_gap 虽能凑出"周期"，但那是噪声而非周期。
        result["measured_note"] = (
            f"仅检出 {len(troughs)} 个波谷（算间隔需 ≥2 个），"
            f"可用数据不足以实测周期长度"
        )

    # ── 数据充分性：完整循环数 ──
    complete_cycles = round(len(series) / theory, 1)
    result["complete_cycles_in_data"] = complete_cycles
    result["data_sufficient"] = complete_cycles >= _MIN_CYCLES_FOR_CONFIDENCE
    if not result["data_sufficient"]:
        result["data_warning"] = (
            f"可用数据仅覆盖约 {complete_cycles} 个理论循环"
            f"（统计可信需 {_MIN_CYCLES_FOR_CONFIDENCE} 个以上），"
            f"周期长度无法验证，阶段判定仅反映当前指标位置"
        )

    # ── 当前阶段：水平分位 × 方向 ──
    cur = values[-1]
    pct = _percentile(values, cur)
    look = max(3, cfg["smooth"] // 2)
    direction = "走平"
    if len(values) > look:
        prev = values[-1 - look]
        if cur > prev:
            direction = "上行"
        elif cur < prev:
            direction = "下行"
    phase_key, phase_name = _classify_phase(pct, direction, cfg["phase_names"])
    detail = cfg["phase_detail"][phase_key]
    nxt = _PHASE_ORDER[(_PHASE_ORDER.index(phase_key) + 1) % len(_PHASE_ORDER)]
    result.update(percentile=pct, direction=direction,
                  direction_lookback_months=look,
                  phase_key=phase_key, phase_name=phase_name,
                  phase_desc=detail["desc"],
                  phase_implication=detail["implication"],
                  phase_risk=detail["risk"],
                  next_phase_name=cfg["phase_names"][nxt])
    # 四阶段序列（含当前标记），供前端画阶段轮盘
    result["phase_sequence"] = [
        {
            "key": k,
            "name": cfg["phase_names"][k],
            "desc": cfg["phase_detail"][k]["desc"],
            "is_current": k == phase_key,
        }
        for k in _PHASE_ORDER
    ]

    # ── 距上一个波谷的月数 / 在实测周期中的位置 ──
    if troughs:
        since = len(values) - 1 - troughs[-1]
        result["months_since_trough"] = since
        result["last_trough_month"] = months[troughs[-1]]
        base = measured or theory
        result["cycle_progress_pct"] = round(min(since / base * 100, 999), 1)

    # ── 佐证指标（仅基钦有）──
    if corroborator:
        c_vals = [v for _, v in corroborator]
        c_look = max(3, cfg["smooth"] // 2)
        c_dir = "走平"
        if len(c_vals) > c_look:
            if c_vals[-1] > c_vals[-1 - c_look]:
                c_dir = "上行"
            elif c_vals[-1] < c_vals[-1 - c_look]:
                c_dir = "下行"
        result["corroborator"] = {
            "label": cfg["corroborator"],
            "month": corroborator[-1][0],
            "value": c_vals[-1],
            "percentile": _percentile(c_vals, c_vals[-1]),
            "direction": c_dir,
            "agrees": c_dir == direction,
        }

    # ── 完整历史序列（前端画走势用，含平滑值）──
    # 不做截断：截断会把早期波谷挡在窗口外，前端按月份定位波谷竖线时找不到对应
    # 位置，检出的周期在图上看不见——这正是"周期不够长"的观感来源。
    result["history"] = {
        "months": months,
        "values": values,
        "smoothed": [None if v is None else round(v, 3) for v in smoothed],
    }
    return result


def analyze_business_cycles() -> dict[str, Any]:
    """三个中短周期的合并分析结果。"""
    out: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycles": [],
        "method": (
            "阶段由指标自身波动位置推导（水平分位 × 变化方向），周期长度由平滑序列"
            "的波谷间隔实测，再与理论值对比；不使用手写日历表"
        ),
    }
    corr = _with_retry(_fetch_industrial_yoy)
    for key in ["kitchin", "juglar", "kuznets"]:
        series = _with_retry(_FETCHERS[key])
        out["cycles"].append(
            analyze_cycle(key, series, corr if key == "kitchin" else None)
        )

    ok = [c for c in out["cycles"] if not c.get("error")]
    if ok:
        parts = [f"{c['name']}处于「{c['phase_name']}」" for c in ok]
        out["summary"] = "；".join(parts) + "。"
        weak = [c["name"] for c in ok if not c.get("data_sufficient")]
        if weak:
            out["summary"] += f"其中 {'、'.join(weak)} 的可用数据不足 3 个完整循环，结论仅供参考。"
    else:
        out["summary"] = "三个周期的数据源均不可用。"
    return out
