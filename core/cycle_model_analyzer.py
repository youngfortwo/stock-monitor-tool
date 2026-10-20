#!/usr/bin/env python3
"""四周期数据驱动模型分析器（方案 B）。

不再使用理论正弦波，而是从真实宏观指标中提取周期成分：
- 基钦：PPI 同比（2006 年起）
- 朱格拉：固定资产投资累计同比（官方可比口径，2021-11 月起）
- 库兹涅茨：国房景气指数（1998 年起）
- 康波：沿用康波阶段日历定位（中国缺乏百年以上月度数据，无法做数据分解）

提取方法：对月度序列做 HP 滤波得到周期项，再归一化到 [-1,1]。
历史区间使用真实数据；未来区间用拟合的阻尼周期外推，并标注为预测。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

# 复用 business_cycle_analyzer 里已经验证过的取数逻辑
from business_cycle_analyzer import (
    _fetch_fai_yoy,
    _fetch_industrial_yoy,
    _fetch_ppi_yoy,
    _fetch_real_estate_index,
)
from kondratiev_analyzer import _KONDRATIEV_CYCLES, _PHASE_INFO, _locate_phase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_FILE = PROJECT_ROOT / "cycle_model_cache.json"

# 月度 HP 滤波 lambda：周期越长，lambda 越大
_LAMBDA_KITCHIN = 10000      # ~3.5 年
_LAMBDA_JUGLAR = 129600      # ~9 年（标准月度商务周期）
_LAMBDA_KUZNETS = 1000000    # ~20 年

def _fetch_juglar_proxy() -> list[tuple[str, float]]:
    """朱格拉周期代理指标。

    官方固定资产投资可比口径（_fetch_fai_yoy）仅回溯到 2021-11，不足以覆盖 9 年周期，
    因此优先使用工业增加值同比作为代理；若其不足再回退到 FAI。
    """
    industrial = _fetch_industrial_yoy()
    if len(industrial) >= 96:
        return industrial
    fai = _fetch_fai_yoy()
    return fai if fai else industrial


_CYCLES: dict[str, dict[str, Any]] = {
    "kitchin": {
        "name": "基钦",
        "subtitle": "库存周期",
        "period_months": 42,
        "theory_label": "约 3.5 年（42 个月）",
        "color": "#0ea5e9",
        "fetcher": _fetch_ppi_yoy,
        "indicator": "PPI 同比",
        "lambda": _LAMBDA_KITCHIN,
    },
    "juglar": {
        "name": "朱格拉",
        "subtitle": "资本支出周期",
        "period_months": 108,
        "theory_label": "约 9 年",
        "color": "#10b981",
        "fetcher": _fetch_juglar_proxy,
        "indicator": "工业增加值同比（FAI 过短时的代理）",
        "lambda": _LAMBDA_JUGLAR,
    },
    "kuznets": {
        "name": "库兹涅茨",
        "subtitle": "地产/建筑周期",
        "period_months": 240,
        "theory_label": "约 20 年",
        "color": "#f59e0b",
        "fetcher": _fetch_real_estate_index,
        "indicator": "国房景气指数",
        "lambda": _LAMBDA_KUZNETS,
    },
}


def _month_to_offset(month: str, base_month: str) -> int:
    """'YYYY-MM' 相对 base_month 的月数偏移（可为负）。"""
    y, m = int(month[:4]), int(month[5:7])
    by, bm = int(base_month[:4]), int(base_month[5:7])
    return (y - by) * 12 + (m - bm)


def _offset_to_month(offset: int, base_month: str) -> str:
    """月数偏移转 'YYYY-MM'。"""
    by, bm = int(base_month[:4]), int(base_month[5:7])
    total = by * 12 + bm - 1 + offset
    y, mo = divmod(total, 12)
    return f"{y:04d}-{mo + 1:02d}"


def _month_range(start_month: str, end_month: str) -> list[str]:
    """生成从 start_month 到 end_month（含）的月份列表。"""
    months = []
    y, m = int(start_month[:4]), int(start_month[5:7])
    ey, em = int(end_month[:4]), int(end_month[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def _hp_filter(x: np.ndarray, lamb: float) -> np.ndarray:
    """Hodrick-Prescott 滤波，返回周期项（原序列 - 趋势项）。

    解 (I + lambda * K'K) tau = y，其中 K 为二阶差分矩阵。
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    n = len(x)
    if n < 5:
        return np.zeros(n)

    main = np.ones(n)
    main[0] = 1 + lamb
    main[-1] = 1 + lamb
    main[1] = 1 + 5 * lamb
    main[-2] = 1 + 5 * lamb
    main[2:-2] = 1 + 6 * lamb

    upper = np.zeros(n - 1)
    upper[0] = -2 * lamb
    upper[1:-1] = -4 * lamb
    upper[-1] = -2 * lamb

    lower = np.zeros(n - 1)
    lower[0] = -2 * lamb
    lower[1:-1] = -4 * lamb
    lower[-1] = -2 * lamb

    A = diags([lower, main, upper], [-1, 0, 1], format="csc")
    trend = spsolve(A, x)
    return x - trend


def _damped_sine(t: np.ndarray, A: float, damping: float, T: float,
                 phi: float, C: float) -> np.ndarray:
    """阻尼正弦：A * exp(-damping*t) * sin(2*pi*t/T + phi) + C。"""
    return A * np.exp(-np.maximum(damping * t, -50)) * np.sin(2 * math.pi * t / T + phi) + C


def _fit_damped_sine(months: list[int], values: list[float],
                     period_months: int) -> tuple[float, ...]:
    """对提取出的周期项拟合阻尼正弦，用于未来外推。"""
    from scipy.optimize import curve_fit

    t = np.array(months, dtype=float)
    y = np.array(values, dtype=float)
    if len(t) < 12:
        A0 = (np.max(y) - np.min(y)) / 2 if len(y) else 0.5
        C0 = float(np.mean(y)) if len(y) else 0.0
        return (A0, 0.0, float(period_months), 0.0, C0)

    A0 = (np.percentile(y, 95) - np.percentile(y, 5)) / 2
    C0 = float(np.median(y))
    phi0 = 0.0
    damping0 = 0.0005
    p0 = [A0, damping0, float(period_months), phi0, C0]

    bounds = (
        [-abs(A0) * 3, 0.0, period_months * 0.4, -2 * math.pi, C0 - abs(A0) * 3],
        [abs(A0) * 3, 0.05, period_months * 2.5, 2 * math.pi, C0 + abs(A0) * 3],
    )
    try:
        popt, _ = curve_fit(_damped_sine, t, y, p0=p0, bounds=bounds, maxfev=8000)
        return tuple(popt)
    except Exception:
        return (A0, 0.0, float(period_months), phi0, C0)


def _extract_and_project(
    series: list[tuple[str, float]],
    all_months: list[str],
    period_months: int,
    lamb: float,
) -> tuple[list[float | None], dict[str, Any]]:
    """对单条序列提取 HP 周期项，历史用真实值，未来用阻尼正弦外推。"""
    if not series:
        return [None] * len(all_months), {"error": "无数据"}

    data_map = dict(series)
    values = [data_map.get(m) for m in all_months]

    # 缺失值用线性插值填充（短期缺口），然后取最长连续非空段做 HP 滤波
    filled = _linear_fill(values)
    first = next((i for i, v in enumerate(filled) if v is not None), None)
    last = next((i for i in range(len(filled) - 1, -1, -1) if filled[i] is not None), None)
    if first is None or last is None or last - first + 1 < 24:
        return [None] * len(all_months), {"error": "数据不足 24 个月，无法提取周期"}

    segment = filled[first:last + 1]
    y = np.array(segment, dtype=float)
    cycle = _hp_filter(y, lamb)

    # 历史真实观测的偏移（未填充的位置）
    hist_offsets = [i for i in range(first, last + 1) if values[i] is not None]
    hist_cycle = [cycle[i - first] for i in hist_offsets]
    params = _fit_damped_sine(hist_offsets, hist_cycle, period_months)

    # 组装结果：历史区间用 HP 周期项，未来用拟合模型
    out: list[float | None] = [None] * len(all_months)
    for i in range(first, last + 1):
        out[i] = float(cycle[i - first])
    last_real_idx = max(hist_offsets) if hist_offsets else last
    t_all = np.arange(last_real_idx + 1, len(all_months))
    if len(t_all):
        projected = _damped_sine(t_all, *params)
        for j, idx in enumerate(t_all):
            out[idx] = float(projected[j])

    meta = {
        "indicator": series[0][0] if series else None,
        "history_from": all_months[min(hist_offsets)],
        "history_to": all_months[max(hist_offsets)],
        "history_n": len(hist_offsets),
        "fit_params": {
            "A": round(params[0], 4),
            "damping": round(params[1], 6),
            "T": round(params[2], 2),
            "phi": round(params[3], 4),
            "C": round(params[4], 4),
        },
    }
    return out, meta


def _linear_fill(values: list[float | None]) -> list[float | None]:
    """对内部缺失值做线性插值；首尾缺失保留 None。"""
    out = list(values)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        if i > 0 and j < n:
            a, b = out[i - 1], out[j]
            steps = j - i + 1
            for k in range(i, j):
                out[k] = a + (b - a) * (k - i + 1) / steps
        i = j if j > i else i + 1
    return out


def _normalize(vals: list[float | None]) -> list[float | None]:
    """把序列稳健归一化到 [-1, 1]，使用 2.5 倍标准差截断。"""
    numeric = [v for v in vals if v is not None]
    if not numeric:
        return vals
    arr = np.array(numeric)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    scale = 2.5 * std if std > 0 else 1.0
    out: list[float | None] = []
    for v in vals:
        if v is None:
            out.append(None)
        else:
            nv = (v - mean) / scale
            out.append(round(max(-1.0, min(1.0, nv)), 4))
    return out


def _kondratiev_monthly(all_months: list[str]) -> list[float | None]:
    """根据康波阶段日历生成月度连续值。

    春季（回升）：-cos(pi * progress) 由 -1 _smooth_ 升至 +1
    夏季（繁荣）：+1
    秋季（滞胀）：+cos(pi * progress) 由 +1 _smooth_ 降至 -1
    冬季（萧条）：-1
    """
    PHASE_VALUE: dict[str, float] = {
        "spring": 0.0,
        "summer": 1.0,
        "autumn": 0.0,
        "winter": -1.0,
    }
    out: list[float | None] = []
    for m in all_months:
        year = int(m[:4])
        pos = _locate_phase(year)
        if not pos:
            out.append(None)
            continue
        phase_key = pos["phase_key"]
        p_start = pos["phase_start"]
        p_end = pos["phase_end"]
        total = p_end - p_start
        if total <= 0:
            out.append(PHASE_VALUE.get(phase_key, 0.0))
            continue
        # 年内进度：用月份估算
        mo = int(m[5:7])
        progress = (year - p_start + (mo - 1) / 12) / total
        progress = max(0.0, min(1.0, progress))
        if phase_key == "spring":
            v = -math.cos(math.pi * progress)
        elif phase_key == "summer":
            v = 1.0
        elif phase_key == "autumn":
            v = math.cos(math.pi * progress)
        elif phase_key == "winter":
            v = -1.0
        else:
            v = 0.0
        out.append(round(v, 4))
    return out


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def analyze_cycle_model(
    start_year: int = 1980,
    end_year: int = 2060,
    force: bool = False,
) -> dict[str, Any]:
    """主入口：返回数据驱动的四周期月度序列与综合景气线。"""
    today = dt.date.today()
    cache_key = f"{today.isoformat()}:{start_year}:{end_year}"
    cache = _load_cache()
    if not force and cache.get("cache_key") == cache_key:
        return cache

    start_month = f"{start_year}-01"
    end_month = f"{end_year}-12"
    all_months = _month_range(start_month, end_month)
    current_month = today.strftime("%Y-%m")

    # 取数
    series_data: dict[str, list[tuple[str, float]]] = {}
    metas: dict[str, Any] = {}
    for key, cfg in _CYCLES.items():
        raw = cfg["fetcher"]()
        series_data[key] = raw
        metas[key] = {"indicator": cfg["indicator"], "theory_label": cfg["theory_label"]}

    # 提取各周期成分
    result_series: list[dict[str, Any]] = []
    composite_sum: list[float] = [0.0] * len(all_months)
    composite_count: list[int] = [0] * len(all_months)

    for key, cfg in _CYCLES.items():
        raw = series_data[key]
        extracted, meta = _extract_and_project(
            raw, all_months, cfg["period_months"], cfg["lambda"]
        )
        norm = _normalize(extracted)
        result_series.append({
            "key": key,
            "name": cfg["name"],
            "color": cfg["color"],
            "desc": f"{cfg['theory_label']} · {cfg['subtitle']}",
            "indicator": cfg["indicator"],
            "theory_label": cfg["theory_label"],
            "values": norm,
            "meta": {**metas[key], **meta},
        })
        for i, v in enumerate(norm):
            if v is not None:
                composite_sum[i] += v
                composite_count[i] += 1

    # 康波：数据不足以做 HP 分解，沿用阶段日历
    kd_values = _kondratiev_monthly(all_months)
    result_series.append({
        "key": "kondratievi",
        "name": "康波",
        "color": "#6366f1",
        "desc": "约 55 年技术革命周期 · 阶段日历定位（中国缺乏百年以上月度数据）",
        "indicator": "康波阶段日历",
        "theory_label": "约 55 年",
        "values": kd_values,
        "meta": {"note": "中国宏观数据无法覆盖完整康波，使用康波阶段日历映射为连续值"},
    })
    for i, v in enumerate(kd_values):
        if v is not None:
            composite_sum[i] += v
            composite_count[i] += 1

    composite = [
        round(composite_sum[i] / composite_count[i], 4) if composite_count[i] else None
        for i in range(len(all_months))
    ]

    # 数据说明
    sources = []
    for key, cfg in _CYCLES.items():
        meta = next(s for s in result_series if s["key"] == key)["meta"]
        if meta.get("error"):
            sources.append(f"{cfg['name']}：{meta['error']}")
        else:
            sources.append(
                f"{cfg['name']}（{cfg['indicator']}）{meta.get('history_from', '')}~"
                f"{meta.get('history_to', '')} 共 {meta.get('history_n', 0)} 个月"
            )
    sources.append("康波：阶段日历定位")

    result = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache_key": cache_key,
        "start_year": start_year,
        "end_year": end_year,
        "current_month": current_month,
        "months": all_months,
        "series": result_series,
        "composite": {
            "name": "四周期综合景气",
            "color": "#dc2626",
            "values": composite,
        },
        "method": (
            "基钦/朱格拉/库兹涅茨：对真实宏观指标做 HP 滤波提取周期项，历史区间使用真实数据，"
            "未来区间用拟合阻尼正弦外推；康波因数据长度不足沿用阶段日历定位。"
            "各序列独立归一化到 [-1,1]，0 为景气中性线。"
        ),
        "sources": sources,
        "note": (
            "本图由真实宏观数据驱动：历史部分为 HP 滤波提取的周期成分，"
            "未来部分为模型外推（灰色/半透明区域为预测）；综合景气为四周期等权平均。"
        ),
    }
    _save_cache(result)
    return result


if __name__ == "__main__":
    # 本地快速验证
    r = analyze_cycle_model(start_year=1980, end_year=2060, force=True)
    print(json.dumps({
        "generated_at": r["generated_at"],
        "months_n": len(r["months"]),
        "series_keys": [s["key"] for s in r["series"]],
        "sources": r["sources"],
    }, ensure_ascii=False, indent=2))
