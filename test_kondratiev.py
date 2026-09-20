#!/usr/bin/env python3
"""康波周期判别体系单元测试 + 区分力回测。

重点验证两件事：
1. 打分引擎的边界行为（区间内/外、缺数据、四阶段并行排名）
2. 判别指标在历史数据上的实际区分力（Top1/Top2 命中率）

设计背景：上一版用手写的 CPI/PPI 绝对区间做验证，回测发现低通胀环境下
春/秋/冬三个阶段区间重叠、同一组数据可同时命中三个阶段，且覆盖不到
1990 年代高通胀时代。本版改为各指标取自身历史分位 + 四阶段并行打分。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kondratiev_analyzer import (
    _PHASE_INDICATOR_PROFILE, _band_score, _locate_phase, _norm_month,
    _percentile, _score_all_phases, backtest_discrimination,
)

# ── 用例1：月份归一化覆盖各数据源写法 ──
assert _norm_month("201501") == "2015-01"
assert _norm_month("2026年08月份") == "2026-08"
assert _norm_month("2026年8月") == "2026-08"
assert _norm_month("2026-08-31") == "2026-08"
assert _norm_month("2026.8") == "2026-08"      # 货币供应量口径，月份不补零
assert _norm_month("2026.10") == "2026-10"
# 归一化后必须可按字符串正确排序（未补零时 '2026.10' < '2026.2'，会错序）
assert _norm_month("2026.10") > _norm_month("2026.2")
print("用例1 月份归一化: PASS")

# ── 用例2：分位数计算 ──
vals = [1.0, 2.0, 3.0, 4.0, 5.0]
assert _percentile(vals, 0.5) == 0.0, _percentile(vals, 0.5)
assert _percentile(vals, 5.5) == 100.0
assert _percentile(vals, 3.0) == 50.0, _percentile(vals, 3.0)
assert _percentile([], 1.0) == 50.0  # 无历史时取中性值
print("用例2 分位数计算: PASS")

# ── 用例3：区间打分（中心最高、边界 80、区间外衰减）──
assert _band_score(75, (60, 90)) == 100.0        # 正中心
assert _band_score(60, (60, 90)) == 80.0         # 下边界
assert _band_score(90, (60, 90)) == 80.0         # 上边界
assert _band_score(105, (60, 90)) < 80.0         # 区间外
assert _band_score(0, (60, 90)) == 0.0           # 远离区间归零
print("用例3 区间打分边界: PASS")

# ── 用例4：四阶段并行排名（构造典型"冬"特征）──
# 冬：股债利差极高（股票最便宜）+ 信用收缩 + PMI 低迷 + 资金不活化
winter_like = {"spread": 90.0, "credit": 10.0, "pmi": 15.0, "m1m2": 12.0}
ranking = _score_all_phases(winter_like)
assert ranking[0]["phase_key"] == "winter", [r["phase_key"] for r in ranking]
print(f"用例4 典型冬特征 -> Top1={ranking[0]['phase_name']}"
      f"({ranking[0]['score']}分), Top2={ranking[1]['phase_name']}"
      f"({ranking[1]['score']}分): PASS")

# ── 用例5：典型"夏"特征（泡沫期：股债利差极低 + 信用扩张 + PMI 高）──
summer_like = {"spread": 8.0, "credit": 70.0, "pmi": 85.0, "m1m2": 72.0}
ranking5 = _score_all_phases(summer_like)
assert ranking5[0]["phase_key"] == "summer", [r["phase_key"] for r in ranking5]
print(f"用例5 典型夏特征 -> Top1={ranking5[0]['phase_name']}({ranking5[0]['score']}分): PASS")

# ── 用例6：缺数据时不崩溃，且仅用可得指标打分 ──
partial = {"spread": 90.0, "credit": None, "pmi": None, "m1m2": None}
ranking6 = _score_all_phases(partial)
assert ranking6[0]["score"] is not None
assert any(i["score"] is None for i in ranking6[0]["indicators"])
all_none = _score_all_phases({"spread": None, "credit": None, "pmi": None, "m1m2": None})
assert all(r["score"] is None for r in all_none)
print("用例6 缺数据降级: PASS")

# ── 用例6b：单指标可用时不得给出高置信度 ──
# 单一指标必然命中某个阶段的区间，排名无意义，须标记为"指标不足"
import kondratiev_analyzer as _ka
_orig = _ka._fetch_all_series
try:
    _ka._fetch_all_series = lambda: {
        "spread": [], "pmi": [], "m1m2": [],
        "credit": [(f"20{y:02d}-{m:02d}", float((y * 12 + m) % 17 - 8))
                   for y in range(17, 27) for m in range(1, 13)],
    }
    r6b = _ka.analyze_kondratiev()
    a6b = r6b["macro_alignment"]
    print(f"用例6b 单指标: 可用={a6b['indicators_available']}/{a6b['indicators_total']} "
          f"一致性={a6b['agreement']} 置信度={a6b['confidence']}")
    assert a6b["indicators_available"] == 1
    assert a6b["agreement"] == "指标不足"
    assert a6b["confidence"] == "低"
    assert a6b.get("data_warning"), "指标缺失时必须给出警告文案"
    print("用例6b 单指标不得高置信: PASS")
finally:
    _ka._fetch_all_series = _orig

# ── 用例7：区间不再像旧版那样三阶段重叠 ──
# 对 spread 这一主判别指标，春/夏 与 秋/冬 的期望区间应基本分离
spring_b = _PHASE_INDICATOR_PROFILE["spring"]["spread"]
summer_b = _PHASE_INDICATOR_PROFILE["summer"]["spread"]
assert spring_b[0] > summer_b[1], "春与夏的股债利差区间必须分离"
print(f"用例7 主判别指标区间分离 春{spring_b} vs 夏{summer_b}: PASS")

# ── 用例8：阶段定位 ──
assert _locate_phase(2026)["phase_key"] == "autumn"
assert _locate_phase(2026)["cycle"] == 6
assert _locate_phase(2010)["phase_key"] == "winter"
assert _locate_phase(2010)["cycle"] == 5
assert _locate_phase(1995)["phase_key"] == "summer"
print("用例8 阶段日历定位: PASS")

print("\n全部单元用例通过\n")

# ── 区分力回测（需要网络，失败不算用例失败）──
if "--no-backtest" in sys.argv:
    sys.exit(0)

print("=== 区分力回测（拉取历史数据，约需 30 秒）===")
try:
    r = backtest_discrimination("2010-01")
except Exception as exc:
    print(f"回测跳过（数据源不可用）: {exc}")
    sys.exit(0)

if not r["months_tested"]:
    print("回测无有效样本（数据源返回为空）")
    sys.exit(0)

print(f"测试区间: {r['range']}   月数: {r['months_tested']}")
print(f"各序列长度: {r['series_lengths']}")
if r["failed_sources"]:
    print(f"失败数据源: {r['failed_sources']}")
print(f"Top1 命中率: {r['hit_rate']}%    Top2 命中率: {r['top2_rate']}%")
print("（随机基准：Top1 25%，Top2 50%）")
print("\n按日历阶段拆分:")
print(f"{'阶段':<8}{'月数':>6}{'Top1':>9}{'Top2':>9}")
for k, v in r["by_phase"].items():
    print(f"{k:<8}{v['n']:>6}{v['hit_rate']:>8.1f}%{v['top2_rate']:>8.1f}%")

from collections import Counter
print("\n日历阶段 -> 数据最像阶段 的分布:")
cm = Counter((x["calendar_phase"], x["data_phase"]) for x in r["rows"])
for (cal, dat), c in sorted(cm.items(), key=lambda x: -x[1]):
    print(f"  日历={cal}  数据={dat}: {c} 个月{'  [一致]' if cal == dat else ''}")

print("\n近 12 个月明细:")
for x in r["rows"][-12:]:
    print(f"  {x['month']}  日历={x['calendar_phase']}  数据={x['data_phase']}"
          f"  日历排名={x['calendar_rank']}{'  HIT' if x['hit'] else ''}")
