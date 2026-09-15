#!/usr/bin/env python3
"""超跌反弹策略单元测试：构造K线验证左侧埋伏 / 右侧启动两个分支的条件边界。

共享门槛：超跌(回撤≥30%) + 未突破(C<HH90*0.75) + 排雷。
左侧埋伏：不创新低 + 距低点反弹<30% + 均线抬头(MA5>MA13>MA21) + 量能初现（未爆拉）。
右侧启动：刚站上MA30(1~5天) + 近3日放量≥2倍VOL30 + MA5>MA13 + 距低点反弹<50%。

否定用例断言 left_match / right_match 而非 is_match：两分支为 OR 关系，
断言总结果会让"某分支该拒却被另一分支放行"的缺陷被掩盖。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from oversold_rebound_scanner import evaluate_oversold_rebound, is_risk_name


def make_history(prices, volumes, start="2026-01-01"):
    """按收盘价序列构造K线，high/low 在收盘价 ±0.5% 内。"""
    dates = pd.date_range(start, periods=len(prices), freq="B")
    closes = np.array(prices, dtype=float)
    return pd.DataFrame({
        "date": dates, "open": closes * 0.998, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes, "volume": volumes,
        "amount": closes * np.array(volumes, dtype=float),
    })


def build_case(crash_to=60, recover_to=65, spike_vol=900_000,
               shrink_to=450_000, crash_vol=2_500_000):
    """标准抬头形态：涨到100 → 崩盘到 crash_to（放量）→ 缩量筑底 → 温和放量缓升。
    末段缓升，MA5>MA13>MA21 多头抬头；recover_to 较低，未突破 HH*0.75。"""
    p1 = list(np.linspace(70, 100, 60))
    p2 = list(np.linspace(100, crash_to, 30))
    p3 = list(np.linspace(crash_to, crash_to * 1.02, 15))
    p4 = list(np.linspace(crash_to * 1.02, recover_to, 15))
    prices = p1 + p2 + p3 + p4
    v1 = [1_000_000] * 60
    v2 = [crash_vol] * 30
    v3 = list(np.linspace(900_000, shrink_to, 15))
    v4 = list(np.linspace(shrink_to + 30_000, 870_000, 14)) + [spike_vol]
    vols = v1 + v2 + v3 + v4
    return make_history(prices, vols)


def build_pullback(crash_to=60, peak=72, pullback_to=68):
    """非多头（回调）形态：涨到100 → 崩盘到60 → 反弹到72 → 近15日回落到68。
    末段回落使 MA5<MA13（短期死叉），但仍在底部区间（未突破 HH*0.75）。"""
    p1 = list(np.linspace(70, 100, 60))
    p2 = list(np.linspace(100, crash_to, 30))
    p3 = list(np.linspace(crash_to, peak, 15))
    p4 = list(np.linspace(peak, pullback_to, 15))
    prices = p1 + p2 + p3 + p4
    v1 = [1_000_000] * 60
    v2 = [2_500_000] * 30
    v3 = list(np.linspace(700_000, 500_000, 15))                  # 地量
    v4 = list(np.linspace(520_000, 750_000, 14)) + [760_000]      # 温和放量
    vols = v1 + v2 + v3 + v4
    return make_history(prices, vols)


def failed_flags(r):
    """返回未通过的条件名列表（用于失败诊断）。"""
    return [k for k, v in (r.get("flags") or {}).items() if not v]


# ── 用例1：标准抬头形态（应入选左侧）──
hist = build_case()
r = evaluate_oversold_rebound("600000", "测试股份", "银行", hist)
print("用例1 标准抬头:", "入选" if r["is_match"] else "未入选",
      f"signal={r['signal_type']} dd={r['max_drawdown_pct']}% "
      f"below_high={r['pct_below_high']}% "
      f"MA=({r['ma5']}>{r['ma13']}>{r['ma21']}) score={r['score']}")
assert r["is_match"], f"应入选: 未通过 {failed_flags(r)}"
assert r["signal_type"] == "左侧埋伏", f"应标记为左侧埋伏，实为 {r['signal_type']}"

# ── 用例2：回撤不足 35%（共享门槛，两分支都应拒）──
hist2 = build_case(crash_to=75, recover_to=82)  # 100→75 仅约26%回撤
r2 = evaluate_oversold_rebound("600001", "测试二", "银行", hist2)
print("用例2 回撤不足:", "入选" if r2["is_match"] else "未入选", f"dd={r2['max_drawdown_pct']}%")
assert not r2["is_match"] and "oversold" in failed_flags(r2)

# ── 用例3：末日创新低（左侧应拒）──
hist3 = build_case()
hist3.loc[hist3.index[-1], "low"] = 55.0  # 最后一根K线砸出60日新低
r3 = evaluate_oversold_rebound("600002", "测试三", "银行", hist3)
print("用例3 创新低:", "左侧入选" if r3["left_match"] else "左侧未入选")
assert not r3["left_match"] and "no_new_low" in failed_flags(r3)

# ── 用例4：非多头（回调，MA5<MA13，左侧应拒）──
hist4 = build_pullback()
r4 = evaluate_oversold_rebound("600003", "测试四", "银行", hist4)
print("用例4 非多头回调:", "左侧入选" if r4["left_match"] else "左侧未入选",
      f"MA5={r4['ma5']} MA13={r4['ma13']} MA21={r4['ma21']}")
assert not r4["left_match"] and "ma_up" in failed_flags(r4)

# ── 用例5：末日爆量（左侧应拒，未爆拉条件）──
hist5 = build_case(spike_vol=2_000_000)  # 超过60日峰量70%
r5 = evaluate_oversold_rebound("600004", "测试五", "银行", hist5)
print("用例5 爆量:", "左侧入选" if r5["left_match"] else "左侧未入选")
assert not r5["left_match"] and "vol_launch" in failed_flags(r5)

# ── 用例6：末日缩量（左侧应拒，量能初现要求 V>REF(V,1)）──
hist6 = build_case(spike_vol=300_000)  # 低于前一日量
r6 = evaluate_oversold_rebound("600005", "测试六", "银行", hist6)
print("用例6 缩量:", "左侧入选" if r6["left_match"] else "左侧未入选")
assert not r6["left_match"] and "vol_launch" in failed_flags(r6)

# ── 用例7：已突破底部区间（共享门槛，两分支都应拒）──
hist7 = build_case(recover_to=80)  # 80 > 100.5*0.75=75.4
r7 = evaluate_oversold_rebound("600006", "测试七", "银行", hist7)
print("用例7 已突破:", "入选" if r7["is_match"] else "未入选",
      f"below_high={r7['pct_below_high']}%")
assert not r7["is_match"] and "not_broken" in failed_flags(r7)

# ── 用例8：ST 排雷 ──
assert is_risk_name("ST摩登") and is_risk_name("*ST海投") and is_risk_name("S*ST前锋") and is_risk_name("退市博元")
assert not is_risk_name("贵州茅台") and not is_risk_name("TCL科技")
print("用例8 ST排雷: PASS")

# ── 用例9：距低点反弹超过30%（左侧应拒）──
hist9 = build_case(crash_to=50, recover_to=70)  # 反弹约40%，超过30%上限
r9 = evaluate_oversold_rebound("600007", "测试九", "银行", hist9)
print("用例9 距低点反弹超30%:", "左侧入选" if r9["left_match"] else "左侧未入选",
      f"rebound={r9['rebound_from_low_pct']}%")
assert not r9["left_match"] and "near_low" in failed_flags(r9)


def build_right_case(crash_to=60, base_days=25, launch_days=4, launch_to=70,
                     base_vol=500_000, launch_vols=None):
    """右侧启动形态：涨到100 → 崩盘到 crash_to → 缩量横盘 base_days 天 →
    最后 launch_days 天放量拉升到 launch_to（刚刚站上 MA30）。"""
    p1 = list(np.linspace(70, 100, 60))
    p2 = list(np.linspace(100, crash_to, 30))
    p3 = [float(crash_to)] * base_days
    p4 = list(np.linspace(crash_to, launch_to, launch_days + 1))[1:]
    if launch_vols is None:
        launch_vols = list(np.linspace(1_500_000, 2_500_000, launch_days))
    vols = ([1_000_000] * 60 + [2_500_000] * 30 + [base_vol] * base_days
            + list(launch_vols))
    return make_history(p1 + p2 + p3 + p4, vols)


# ── 用例10：右侧刚启动（应入选，且标记为右侧启动）──
hist10 = build_right_case()
r10 = evaluate_oversold_rebound("600008", "测试十", "银行", hist10)
print("用例10 右侧刚启动:", "入选" if r10["is_match"] else "未入选",
      f"signal={r10['signal_type']} ma30_days={r10['days_above_ma30']} "
      f"surge={r10['vol_surge']}x rebound={r10['rebound_from_low_pct']}% "
      f"score={r10['score']}")
assert r10["is_match"], f"应入选: 未通过 {failed_flags(r10)}"
assert r10["signal_type"] == "右侧启动", f"应标记为右侧启动，实为 {r10['signal_type']}"
assert r10["matched_reason"].startswith("右侧启动: ")

# ── 用例11：已站上 MA30 太久（右侧应拒，不算"刚启动"）──
hist11 = build_right_case(launch_days=10, launch_vols=np.linspace(1_500_000, 2_500_000, 10))
r11 = evaluate_oversold_rebound("600009", "测试十一", "银行", hist11)
print("用例11 启动过久:", "右侧入选" if r11["right_match"] else "右侧未入选",
      f"ma30_days={r11['days_above_ma30']}")
assert not r11["right_match"] and "fresh_start" in failed_flags(r11)

# ── 用例12：拉升但放量不足（右侧应拒，缩量反抽不算启动）──
hist12 = build_right_case(launch_vols=[600_000, 620_000, 640_000, 660_000])
r12 = evaluate_oversold_rebound("600010", "测试十二", "银行", hist12)
print("用例12 放量不足:", "右侧入选" if r12["right_match"] else "右侧未入选",
      f"surge={r12['vol_surge']}x")
assert not r12["right_match"] and "vol_surge_ok" in failed_flags(r12)

# ── 用例13：距低点反弹超过右侧上限50%（右侧应拒，已涨透）──
hist13 = build_right_case(crash_to=40, launch_to=65)  # 40→65 反弹约63%
r13 = evaluate_oversold_rebound("600011", "测试十三", "银行", hist13)
print("用例13 反弹超50%:", "右侧入选" if r13["right_match"] else "右侧未入选",
      f"rebound={r13['rebound_from_low_pct']}%")
assert not r13["right_match"] and "not_overextended" in failed_flags(r13)

# ── 用例14：signal_mode 单边开关 ──
r14l = evaluate_oversold_rebound("600008", "测试十", "银行", hist10, signal_mode="left")
r14r = evaluate_oversold_rebound("600000", "测试股份", "银行", hist, signal_mode="right")
print(f"用例14 单边开关: 右侧样本在 left 模式下 is_match={r14l['is_match']}，"
      f"左侧样本在 right 模式下 is_match={r14r['is_match']}")
assert not r14l["is_match"], "signal_mode=left 时不应放行右侧信号"
assert not r14r["is_match"], "signal_mode=right 时不应放行左侧信号"

print("\n全部用例通过")
