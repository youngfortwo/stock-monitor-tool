#!/usr/bin/env python3
"""中短周期分析器单元测试。

核心是用合成序列验证周期检测的准确度：如果在已知周期长度的正弦波上都测不准，
那么在真实数据上给出的"实测周期长度"就没有意义。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from business_cycle_analyzer import (
    _CYCLES, _classify_phase, _find_troughs, _norm_month, _percentile,
    _smooth, analyze_cycle,
)


def sine_series(period: int, n: int, amp: float = 5.0, phase: float = 0.0,
                start_year: int = 2000) -> list[tuple[str, float]]:
    """构造已知周期的正弦序列，月份从 start_year-01 起递增。"""
    out = []
    for i in range(n):
        y, m = divmod(i, 12)
        val = amp * math.sin(2 * math.pi * (i + phase) / period)
        out.append((f"{start_year + y:04d}-{m + 1:02d}", round(val, 4)))
    return out


# ── 用例1：月份归一化 ──
assert _norm_month("2026年08月份") == "2026-08"
assert _norm_month("2026-08-31") == "2026-08"
assert _norm_month("2026.8") == "2026-08"
assert _norm_month("201501") == "2015-01"
assert _norm_month("2026.10") > _norm_month("2026.2")
print("用例1 月份归一化: PASS")

# ── 用例2：居中平滑两端置 None，不用残缺窗口造假拐点 ──
sm = _smooth([1.0] * 20, 7)
assert sm[:3] == [None, None, None] and sm[-3:] == [None, None, None]
assert all(abs(v - 1.0) < 1e-9 for v in sm[3:-3])
assert _smooth([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]
print("用例2 居中平滑边界: PASS")

# ── 用例3：正弦波上实测周期长度应接近真值 ──
for period in (40, 108):
    n = period * 5
    series = sine_series(period, n)
    vals = [v for _, v in series]
    sm = _smooth(vals, 7 if period == 40 else 13)
    troughs = _find_troughs(sm, min_gap=max(6, int(period * 0.5)))
    intervals = [troughs[i] - troughs[i - 1] for i in range(1, len(troughs))]
    measured = sum(intervals) / len(intervals) if intervals else 0
    err = abs(measured - period) / period * 100
    print(f"用例3 周期={period:>3} 实测={measured:>6.1f} 误差={err:>4.1f}% "
          f"波谷数={len(troughs)}")
    assert intervals, f"period={period} 未检出波谷间隔"
    assert err < 5.0, f"period={period} 实测误差 {err:.1f}% 超过 5%"
print("用例3 正弦周期检测精度: PASS")

# ── 用例4：min_gap 抑制高频噪声造成的伪波谷 ──
noisy = []
for i, (m, v) in enumerate(sine_series(40, 200)):
    noisy.append((m, v + (1.5 if i % 2 else -1.5)))  # 叠加逐月抖动
vals = [v for _, v in noisy]
raw_troughs = _find_troughs(_smooth(vals, 1), min_gap=1)   # 不平滑不限距
good_troughs = _find_troughs(_smooth(vals, 7), min_gap=20)  # 平滑 + 限距
print(f"用例4 噪声序列: 未处理检出 {len(raw_troughs)} 个波谷，"
      f"平滑+限距后 {len(good_troughs)} 个（真值约 {200 // 40}）")
assert len(raw_troughs) > 20, "构造的噪声应产生大量伪波谷"
assert len(good_troughs) <= 6, f"平滑+限距后仍有 {len(good_troughs)} 个伪波谷"
print("用例4 伪波谷抑制: PASS")

# ── 用例5：四阶段分类的四种组合 ──
names = _CYCLES["kitchin"]["phase_names"]
assert _classify_phase(20, "上行", names)[0] == "recovery"
assert _classify_phase(80, "上行", names)[0] == "expansion"
assert _classify_phase(80, "下行", names)[0] == "peak"
assert _classify_phase(20, "下行", names)[0] == "contraction"
print("用例5 四阶段分类: PASS")

# ── 用例6：分位数 ──
assert _percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 0.0
assert _percentile([1.0, 2.0, 3.0, 4.0], 5.0) == 100.0
assert _percentile([], 1.0) == 50.0
print("用例6 分位数: PASS")

# ── 用例7：analyze_cycle 在已知相位上给出预期阶段 ──
# sin(2πi/period) 的波谷在 i/period = 0.75 处。方向由 look 个月的回看窗口决定，
# 因此刚过波谷时窗口仍跨着谷底、净变化为负，会读出"下行"——这是回看窗口固有的
# 滞后，不是缺陷，下面把两种相位都显式断言出来。
period = 40
full = sine_series(period, period * 5)          # 索引 0..199

# 谷后第 1 个月（i=191，相位 0.775）：仍读下行
r_lag = analyze_cycle("kitchin", full[:192])
print(f"用例7a 谷后1个月: 方向={r_lag['direction']} 分位={r_lag['percentile']}% "
      f"阶段={r_lag['phase_name']}")
assert r_lag["direction"] == "下行", r_lag["direction"]
assert r_lag["phase_key"] == "contraction"

# 谷后第 8 个月（i=198，相位 0.95）：确认转为低位上行 → 复苏
r = analyze_cycle("kitchin", full[:199])
print(f"用例7b 谷后8个月: 方向={r['direction']} 分位={r['percentile']}% "
      f"阶段={r['phase_name']} 实测周期={r['measured_cycle_months']} "
      f"偏离理论={r.get('theory_deviation_pct')}%")
assert r["direction"] == "上行", r["direction"]
assert r["phase_key"] == "recovery", r["phase_key"]
assert r["measured_cycle_months"] is not None
assert abs(r["measured_cycle_months"] - period) / period < 0.1
assert r["months_since_trough"] >= 1
print("用例7 analyze_cycle 端到端（含回看滞后）: PASS")

# ── 用例8：数据不足必须被标记 ──
short = sine_series(216, 60)  # 库兹涅茨理论 216 个月，只给 60 个月
r8 = analyze_cycle("kuznets", short)
print(f"用例8 短序列: 完整循环={r8['complete_cycles_in_data']} "
      f"充分={r8['data_sufficient']}")
assert r8["data_sufficient"] is False
assert r8.get("data_warning"), "数据不足时必须给出警告"
print("用例8 数据充分性标记: PASS")

# ── 用例9：空序列不崩溃 ──
r9 = analyze_cycle("juglar", [])
assert r9.get("error") and r9["months"] == 0
print("用例9 空数据降级: PASS")

# ── 用例10：佐证指标一致性判定 ──
up = sine_series(40, 120)
r10 = analyze_cycle("kitchin", up, corroborator=up)
assert r10["corroborator"]["agrees"] is True, r10["corroborator"]
print(f"用例10 佐证指标: 方向={r10['corroborator']['direction']} "
      f"一致={r10['corroborator']['agrees']}: PASS")

# ── 用例11：_with_retry 对"成功但返回空"也要重试 ──
# 限流时 akshare 返回空 DataFrame 而不抛异常，只对异常重试会导致静默降级
import business_cycle_analyzer as _bc

calls = {"n": 0}


def _empty_then_ok():
    calls["n"] += 1
    return [] if calls["n"] < 3 else [("2026-01", 1.0)]


assert _bc._with_retry(_empty_then_ok, attempts=3, wait=0) == [("2026-01", 1.0)]
assert calls["n"] == 3, f"应重试到第3次才成功，实际调用 {calls['n']} 次"

always_empty = {"n": 0}


def _always_empty():
    always_empty["n"] += 1
    return []


assert _bc._with_retry(_always_empty, attempts=3, wait=0) == []
assert always_empty["n"] == 3, "空结果必须用满重试次数"

raises = {"n": 0}


def _raises_then_ok():
    raises["n"] += 1
    if raises["n"] < 2:
        raise RuntimeError("boom")
    return [("2026-01", 2.0)]


assert _bc._with_retry(_raises_then_ok, attempts=3, wait=0) == [("2026-01", 2.0)]
print(f"用例11 空结果与异常均重试: 空→成功用了{calls['n']}次，"
      f"全空用满{always_empty['n']}次: PASS")

print("\n全部单元用例通过")
