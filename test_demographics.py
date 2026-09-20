#!/usr/bin/env python3
"""人口结构分析器单元测试：新生儿推算、老龄化阶段判定、缺数据降级。

用构造数据验证逻辑，不依赖网络；末尾可选跑一次真实取数做口径校验。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demographics_analyzer import (
    _AGING_STAGES, _DIVERGENCE_PCT, _NBS_BIRTHS_WAN, _NBS_BIRTH_RATE,
    _NBS_LAST_YEAR, _REPLACEMENT_FERTILITY, _tail_years, analyze_aging,
    analyze_births, population_net_increase, wb_derived_births,
)


def make_series(**kw):
    """构造 {字段: {年份: 值}} 形态的输入。"""
    return {k: dict(v) for k, v in kw.items()}


# ── 用例1：取尾部年份且按年份升序 ──
d = {2003: 1.0, 2001: 2.0, 2002: 3.0}
assert _tail_years(d, 2) == [(2002, 3.0), (2003, 1.0)]
assert _tail_years({}, 5) == []
print("用例1 取尾部年份并升序: PASS")

# ── 用例2：世行口径推算公式 ──
# 出生率 10‰、总人口 14 亿 → 1400 万
wb = wb_derived_births(make_series(
    birth_rate={2023: 10.0, 2024: 5.0},
    population={2023: 1_400_000_000, 2024: 1_400_000_000},
))
assert wb == {2023: 1400.0, 2024: 700.0}, wb
print(f"用例2 世行口径推算: 10‰×14亿={wb[2023]}万: PASS")

# ── 用例3：主序列取统计局值，峰值不受世行口径影响 ──
# 世行口径下 2012 年推算约 1973 万，会把峰值错判到 2012；主序列必须给出 2016 年
s = make_series(
    birth_rate={2012: 14.57, 2016: 13.57, 2024: 6.77},
    population={2012: 1_354_200_000, 2016: 1_387_800_000, 2024: 1_409_000_000},
    fertility={2024: 1.05},
)
r = analyze_births(s, years=20)
assert r["latest_year"] == _NBS_LAST_YEAR
assert r["latest_births_wan"] == _NBS_BIRTHS_WAN[_NBS_LAST_YEAR]
assert r["peak_year"] == 2016, f"峰值应为 2016，实得 {r['peak_year']}"
assert r["peak_births_wan"] == 1786
print(f"用例3 峰值取统计局口径: {r['peak_year']} 年 {r['peak_births_wan']} 万 "
      f"（世行口径会误判为 2012）: PASS")

# ── 用例4：两口径分歧年份被标出 ──
div = {d["year"]: d for d in r["divergence"]}
assert 2012 in div, f"2012 年分歧未被标出: {r['divergence']}"
assert div[2012]["nbs"] == 1635 and div[2012]["diff_pct"] > 20, div[2012]
assert 2024 not in div, "2024 年两口径一致，不应列为分歧"
print(f"用例4 分歧标注: 2012 统计局{div[2012]['nbs']} vs 世行{div[2012]['wb']} "
      f"({div[2012]['diff_pct']:+}%): PASS")

# ── 用例5：净增人口可比出生人口晚一年，且会合并到同一历史年份轴 ──
net = population_net_increase(make_series(
    population={2023: 1_410_700_000, 2024: 1_409_000_000, 2025: 1_406_585_000},
))
assert net == {2024: -170.0, 2025: -241.5}, net
r_net = analyze_births(make_series(
    birth_rate={2024: 6.77},
    population={2023: 1_410_700_000, 2024: 1_409_000_000, 2025: 1_406_585_000},
), years=20)
assert r_net["latest_year"] == 2024
assert r_net["latest_net_population_year"] == 2025
assert r_net["latest_net_population_wan"] == -241.5
assert r_net["history"]["years"][-1] == 2025, "历史轴应延伸到净增人口最新年份"
assert r_net["history"]["births_wan"][-1] is None, "2025 出生人口缺失时应留空"
assert r_net["history"]["net_population_wan"][-1] == -241.5
assert r_net.get("births_missing_note"), "出生人口缺失但净增人口已更新时必须提示"
print(f"用例5 净增人口: 2025={r_net['latest_net_population_wan']}万，"
      f"出生人口仍截至{r_net['latest_year']}年: PASS")

# ── 用例6：总和生育率与世代更替水平的对比 ──
assert r["replacement_level"] == _REPLACEMENT_FERTILITY
assert r["latest_fertility"] == 1.05
assert r["fertility_gap_pct"] == -50.0, r["fertility_gap_pct"]
print(f"用例6 生育率缺口: 1.05 对 2.1 = {r['fertility_gap_pct']}%: PASS")

# ── 用例7：内嵌表结构自检（两表年份一致、数值区间合理）──
assert set(_NBS_BIRTHS_WAN) == set(_NBS_BIRTH_RATE), "出生人口与出生率的年份必须一一对应"
assert max(_NBS_BIRTHS_WAN) == _NBS_LAST_YEAR, "_NBS_LAST_YEAR 与表内最大年份不一致"
assert len(_NBS_BIRTHS_WAN) >= 20, "至少需覆盖 20 年"
for _y, _v in _NBS_BIRTHS_WAN.items():
    assert 500 <= _v <= 2500, f"{_y} 年出生人口 {_v} 万超出合理区间"
for _y, _v in _NBS_BIRTH_RATE.items():
    assert 3 <= _v <= 25, f"{_y} 年出生率 {_v}‰ 超出合理区间"
# 内嵌的出生人口与出生率应自洽：出生人口/出生率 ≈ 总人口/1000，逐年比值应平滑
ratios = [_NBS_BIRTHS_WAN[y] / _NBS_BIRTH_RATE[y] for y in sorted(_NBS_BIRTHS_WAN)]
assert max(ratios) / min(ratios) < 1.25, (
    f"出生人口与出生率不自洽，比值区间 {min(ratios):.1f}~{max(ratios):.1f}，"
    f"可能有录入错误")
print(f"用例7 内嵌表自检: {len(_NBS_BIRTHS_WAN)} 年、比值 "
      f"{min(ratios):.1f}~{max(ratios):.1f}（应接近总人口/1000）: PASS")

# ── 用例8：老龄化四档阶段判定（含边界值）──
cases = [
    (6.9, "尚未进入老龄化社会"),
    (7.0, "老龄化社会"),
    (13.9, "老龄化社会"),
    (14.0, "深度老龄化社会"),
    (19.9, "深度老龄化社会"),
    (20.0, "超老龄化社会"),
    (25.0, "超老龄化社会"),
]
for val, expect in cases:
    a = analyze_aging(make_series(elderly_share={2024: val}), 20)
    assert a["stage"]["name"] == expect, f"{val}% -> {a['stage']['name']}，期望 {expect}"
print(f"用例8 老龄化阶段判定（{len(cases)} 个边界）: PASS")

# ── 用例9：下一门槛与距离 ──
a = analyze_aging(make_series(elderly_share={2024: 14.91}), 20)
assert a["next_stage"]["threshold"] == 20.0
assert a["next_stage"]["gap_pct_points"] == 5.09, a["next_stage"]["gap_pct_points"]
# 已达最高档时不应再给下一门槛
a_top = analyze_aging(make_series(elderly_share={2024: 21.0}), 20)
assert a_top["next_stage"] is None, a_top["next_stage"]
print(f"用例9 下一门槛: 14.91% 距超老龄化 {a['next_stage']['gap_pct_points']} 个百分点: PASS")

# ── 用例10：门槛跨越年份取首次达到的年份 ──
hist = {2000: 6.8, 2001: 7.1, 2002: 7.5, 2020: 13.5, 2021: 14.2, 2022: 14.9}
a7 = analyze_aging(make_series(elderly_share=hist), 20)
cross = {c["threshold"]: c["year"] for c in a7["crossings"]}
assert cross[7.0] == 2001, cross
assert cross[14.0] == 2021, cross
assert cross[20.0] is None, cross
print(f"用例10 门槛跨越年份: 7%→{cross[7.0]}, 14%→{cross[14.0]}, 20%→{cross[20.0]}: PASS")

# ── 用例11：20 年变化幅度 ──
assert a7["change_pct_points"] == round(14.9 - 6.8, 2), a7["change_pct_points"]
assert a7["start_year"] == 2000
print(f"用例11 区间变化: +{a7['change_pct_points']} 个百分点: PASS")

# ── 用例12：老龄化数据缺失时降级 ──
assert analyze_aging(make_series(elderly_share={}), 20).get("error")
print("用例12 老龄化数据缺失降级: PASS")

# ── 用例13：阶段门槛表单调递增，且每档都有文案 ──
thr = [t for t, _n, _c, _d in _AGING_STAGES]
assert thr == sorted(thr), thr
for t, n, c, dsc in _AGING_STAGES:
    assert n and c.startswith("#") and dsc, (t, n, c, dsc)
print("用例13 门槛表单调且文案齐备: PASS")

print("\n全部单元用例通过")

if "--with-live" not in sys.argv:
    sys.exit(0)

# ── 真实取数的口径校验（需要网络）──
print("\n=== 真实数据口径校验 ===")
from demographics_analyzer import analyze_demographics

live = analyze_demographics(years=20)
b, ag = live["births"], live["aging"]
if b.get("error") or ag.get("error"):
    print("  取数失败:", b.get("error"), ag.get("error"))
    sys.exit(0)
print(f"  出生: {b['range']} 最新 {b['latest_year']}={b['latest_births_wan']}万 "
      f"峰值 {b['peak_year']}={b['peak_births_wan']}万 较峰值{b['from_peak_pct']}%")

# 内嵌表的录入校验：在两口径本应一致的年份上必须吻合。
# 这能抓出内嵌数字的打字错误——分歧年份（2011-2014、2016-2017）不参与校验。
div_years = {d["year"] for d in b["divergence"]}
checked = 0
for y, nbs, wbv in zip(b["history"]["years"], b["history"]["births_wan"],
                       b["history"]["births_wan_wb"]):
    if y in div_years or wbv is None:
        continue
    dev = abs(wbv / nbs - 1) * 100
    assert dev < _DIVERGENCE_PCT, (
        f"{y} 年未列入分歧却相差 {dev:.1f}%（内嵌 {nbs} vs 世行 {wbv}），"
        f"内嵌值可能录错")
    checked += 1
print(f"  内嵌表校验: {checked} 个非分歧年份与世行口径吻合（<{_DIVERGENCE_PCT}%），"
      f"{len(div_years)} 个年份为已知分歧 {sorted(div_years)}")
if b.get("nbs_stale"):
    print("  ⚠", b["nbs_stale"])
print(f"  老龄: {ag['latest_year']} 年 65+ {ag['elderly_share']}% "
      f"阶段={ag['stage']['name']}")
print("  口径校验通过")
