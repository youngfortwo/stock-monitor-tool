#!/usr/bin/env python3
"""超跌反弹选股扫描器：底部区域的"左侧埋伏"与"右侧刚启动"双信号实现。

三条共享硬门槛（两个分支都必须满足）：
    1. 阶段超跌：N日内最大回撤 ≥ M%（(HH-LL)/HH*100 >= M）
    2. 未突破：C < HHV(H,90)*0.75（相对 90 日高点仍有空间，避免"发现太晚"）
    3. 排雷：剔除 ST/*ST/退市股

左侧埋伏分支（启动前介入，通达信"超跌筑底 抬头版"原策略）：
    4. 底部探明：近20天不创新低（LLV(L,20) > LLV(L,N)）
    5. 距低点反弹 < 30%（排除暴涨后回调股）
    6. 均线抬头：MA5 > MA13 > MA21（短期多头排列）
    7. 量能初现：VOL10 > VOL30 且 V > REF(V,1) 且 V < HHV(V,60)*0.7（地量后温和放量，未爆拉）

右侧启动分支（启动后数日内介入，捕捉左侧分支按设计会排除的放量突破）：
    8. 启动新鲜：收盘连续站上 MA30 的天数在 1~right_max_days 之间（刚刚翻上来）
    9. 放量确认：近3日最大量 / VOL30 ≥ right_min_surge（真实放量，非缩量反抽）
   10. 短期转向：MA5 > MA13（刚启动来不及形成 MA13>MA21，故不作要求）
   11. 涨幅未透支：距 N 日低点反弹 < right_max_rebound%

两个分支为 OR 关系，同时成立时标记为左侧（左侧判据更严格）。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from sepa_stage2_scanner import (
    fetch_history, get_board, get_industry_map, get_stock_pool, get_sub_industry_map,
)

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


def is_risk_name(name: str) -> bool:
    """排雷：ST / *ST / S*ST / 退市股（对应 NAMELIKE('ST') 等通达信函数）。"""
    n = str(name).upper()
    return ("ST" in n) or ("退" in str(name))


def evaluate_oversold_rebound(code: str, name: str, industry: str, history: pd.DataFrame,
                              n_days: int = 60, m_pct: float = 30.0,
                              right_max_days: int = 5, right_min_surge: float = 2.0,
                              right_max_rebound: float = 50.0,
                              signal_mode: str = "both") -> dict:
    """评估单只股票（左侧埋伏 + 右侧启动双分支），返回完整指标 dict。

    signal_mode: both / left / right，控制启用哪些分支（诊断与回测用）。
    不匹配时 is_match=False，signal_type 为空串。
    """
    # 数据要求：N 日窗口 + 均线/量能中枢计算余量 + 90 日高点观察窗口
    min_rows = max(n_days, 90)
    if history.empty or len(history) < min_rows:
        return {"code": code, "name": name, "industry": industry,
                "is_match": False, "score": 0, "error": "历史数据不足"}

    data = history.copy().reset_index(drop=True)
    close_s = data["close"].astype(float)
    high_s = data["high"].astype(float)
    low_s = data["low"].astype(float)
    vol_s = data["volume"].astype(float)

    latest = data.iloc[-1]
    close = float(latest["close"])
    prev_close = float(data.iloc[-2]["close"]) if len(data) >= 2 else close
    pct_chg = (close / prev_close - 1) * 100 if prev_close > 0 else 0.0

    # ── 1. 阶段超跌：N 日最大回撤 ──
    hh = float(high_s.tail(n_days).max())   # HHV(H,N) 含当日
    ll = float(low_s.tail(n_days).min())    # LLV(L,N) 含当日
    drawdown_pct = (hh - ll) / hh * 100 if hh > 0 else 0.0
    oversold = drawdown_pct >= m_pct
    rebound_pct = (close / ll - 1) * 100 if ll > 0 else 0.0   # 现价距 N 日低点反弹幅度
    near_low = rebound_pct < 30.0                             # 距低点反弹 < 30%（排除暴涨后回调股）

    # 未突破的观察窗口：90 日高点（比超跌窗口更长，判断离历史前高还有多少空间）
    hh_90 = float(high_s.tail(90).max())    # HHV(H,90)
    pct_below_high = (close / hh_90 - 1) * 100 if hh_90 > 0 else 0.0  # 现价距 90 日高点（负值）

    # ── 2. 底部探明：近 20 天不创新低（N 日最低点不在近 20 天内）──
    low_20 = float(low_s.tail(20).min())    # LLV(L,20)
    no_new_low = low_20 > ll

    # ── 3. 均线抬头：短期多头排列（MA5>MA13>MA21，有抬头迹象）──
    ma5 = close_s.rolling(5).mean()
    ma13 = close_s.rolling(13).mean()
    ma21 = close_s.rolling(21).mean()
    ma30 = close_s.rolling(30).mean()
    ma5_v, ma13_v, ma21_v, ma30_v = (float(x.iloc[-1]) for x in (ma5, ma13, ma21, ma30))

    conv_pct = (ma30_v - ma5_v) / ma5_v * 100 if ma5_v > 0 else 999.0  # 均线乖离（展示用，负值=MA5在MA30上方）
    ma_up = ma5_v > ma13_v > ma21_v  # 短期多头排列
    ma_turn = ma5_v > ma13_v         # 短期转向（右侧分支：刚启动来不及形成完整多头排列）

    # 启动新鲜度：从最新一根K线往回数，收盘连续站上 MA30 的天数（0 = 当前在 MA30 下方）
    days_above_ma30 = 0
    for i in range(len(close_s) - 1, -1, -1):
        ma30_i = ma30.iloc[i]
        if pd.isna(ma30_i) or close_s.iloc[i] <= ma30_i:
            break
        days_above_ma30 += 1

    # ── 4. 量能初现：地量后温和放量，未爆拉 ──
    vol_today = float(vol_s.iloc[-1])
    vol_prev = float(vol_s.iloc[-2]) if len(vol_s) >= 2 else 0.0
    vol_60_max = float(vol_s.tail(60).max())                # HHV(V,60)
    vol10 = float(vol_s.rolling(10).mean().iloc[-1])
    vol30 = float(vol_s.rolling(30).mean().iloc[-1])
    vol_ratio = vol_today / vol10 if vol10 > 0 else 0.0      # 今日量 / 10日均量
    vol_center = vol10 / vol30 if vol30 > 0 else 0.0        # 量能中枢：10日均量 / 30日均量
    vol_launch = (vol10 > vol30
                  and vol_today > vol_prev
                  and vol_today < vol_60_max * 0.7)

    # 放量确认（右侧分支）：近3日最大量相对 30 日均量的倍数
    vol_3d_max = float(vol_s.tail(3).max())
    vol_surge = vol_3d_max / vol30 if vol30 > 0 else 0.0

    # ── 5. 未突破：仍在底部区间，没大涨（C < HHV(H,90)*0.75）──
    not_broken = close < hh_90 * 0.75

    # ── 6. 排雷 ──
    no_risk = not is_risk_name(name)

    # ── 右侧启动分支专属条件 ──
    fresh_start = 1 <= days_above_ma30 <= right_max_days
    vol_surge_ok = vol_surge >= right_min_surge
    not_overextended = rebound_pct < right_max_rebound

    # ── 综合判定：共享硬门槛 + 左右分支 OR ──
    shared_gate = oversold and not_broken and no_risk
    left_match = shared_gate and no_new_low and near_low and ma_up and vol_launch
    right_match = (shared_gate and fresh_start and vol_surge_ok
                   and ma_turn and not_overextended)
    if signal_mode == "left":
        right_match = False
    elif signal_mode == "right":
        left_match = False

    is_match = left_match or right_match
    signal_type = "左侧埋伏" if left_match else ("右侧启动" if right_match else "")

    # ── 评分（0-100，仅对入选股排序用）──
    score = 0.0
    score += min(30.0, max(0.0, (drawdown_pct - m_pct) * 0.6))    # 超跌深度
    score += min(25.0, max(0.0, -pct_below_high * 0.5))           # 距高点空间（越深越左侧）
    score += min(20.0, max(0.0, (vol_center - 1.0) * 40.0))       # 量能中枢温和抬升
    score += min(25.0, max(0.0, (ma5_v / ma13_v - 1) * 300))      # 抬头强度（MA5相对MA13）
    score = max(0.0, round(score, 2))

    # 各条件通过情况（测试/诊断用，不写入CSV）
    flags = {
        "oversold": oversold,
        "no_new_low": no_new_low,
        "near_low": near_low,
        "ma_up": ma_up,
        "vol_launch": vol_launch,
        "not_broken": not_broken,
        "no_risk": no_risk,
        "fresh_start": fresh_start,
        "vol_surge_ok": vol_surge_ok,
        "ma_turn": ma_turn,
        "not_overextended": not_overextended,
    }

    # 入选理由（按命中分支分别生成）
    if right_match and not left_match:
        reasons = [
            f"{n_days}日回撤{drawdown_pct:.0f}%",
            f"站上MA30第{days_above_ma30}天",
            f"近3日放量{vol_surge:.1f}倍",
            f"距低点反弹{rebound_pct:.0f}%",
            f"距高点{pct_below_high:.0f}%",
        ]
        matched_reason = "右侧启动: " + "; ".join(reasons)
    else:
        reasons = []
        if oversold:
            reasons.append(f"{n_days}日回撤{drawdown_pct:.0f}%")
        if no_new_low:
            reasons.append("近20日未创新低")
        if near_low:
            reasons.append(f"距低点反弹{rebound_pct:.0f}%")
        if ma_up:
            reasons.append("短期多头抬头")
        if vol_launch:
            reasons.append(f"量能中枢{vol_center:.2f}·温和放量")
        if not_broken:
            reasons.append(f"距高点{pct_below_high:.0f}%")
        matched_reason = "左侧埋伏: " + "; ".join(reasons)

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "board": get_board(code),
        "is_match": is_match,
        "signal_type": signal_type,
        "date": str(latest["date"])[:10],
        "close": round(close, 2),
        "pct_chg": round(pct_chg, 2),
        "high_n": round(hh, 2),
        "low_n": round(ll, 2),
        "high_90d": round(hh_90, 2),
        "max_drawdown_pct": round(drawdown_pct, 2),
        "rebound_from_low_pct": round(rebound_pct, 2),
        "pct_below_high": round(pct_below_high, 2),
        "ma_convergence_pct": round(conv_pct, 2),
        "days_above_ma30": days_above_ma30,
        "ma5": round(ma5_v, 2),
        "ma13": round(ma13_v, 2),
        "ma21": round(ma21_v, 2),
        "ma30": round(ma30_v, 2),
        "vol_today": round(vol_today, 0),
        "vol_ratio": round(vol_ratio, 2),
        "vol_center": round(vol_center, 2),
        "vol_surge": round(vol_surge, 2),
        "amount_cny": round(float(latest["amount"]), 2),
        "score": score,
        "left_match": left_match,
        "right_match": right_match,
        "flags": flags,
        "matched_reason": matched_reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="超跌反弹选股扫描器（左侧埋伏 + 右侧刚启动）")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", type=str, default="oversold_rebound_candidates_test.csv")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--include-bj", action="store_true", default=False)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--n-days", type=int, default=60, help="下跌观察周期 N（可改120看更长周期）")
    parser.add_argument("--m-pct", type=float, default=30.0, help="最低回撤幅度 M（%%，腰斩可改50）")
    parser.add_argument("--right-max-days", type=int, default=5,
                        help="右侧分支：连续站上 MA30 的最大天数（越小越接近刚启动）")
    parser.add_argument("--right-min-surge", type=float, default=2.0,
                        help="右侧分支：近3日最大量 / 30日均量 的最小倍数")
    parser.add_argument("--right-max-rebound", type=float, default=50.0,
                        help="右侧分支：距 N 日低点反弹幅度上限（%%，防追高）")
    parser.add_argument("--signal-mode", choices=["both", "left", "right"], default="both",
                        help="启用的信号分支：both=左侧+右侧，left/right=只跑单边")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = get_stock_pool(args.include_bj, args.limit, args.offset)
    industry_map = get_industry_map()

    # 排雷在数据层面直接剔除，节省行情拉取时间
    pool = pool[~pool["名称"].apply(is_risk_name)]
    print(
        f"Scanning {len(pool)} stocks for 超跌反弹 (mode={args.signal_mode}, "
        f"N={args.n_days}, M={args.m_pct}%, right: days<={args.right_max_days} "
        f"surge>={args.right_min_surge}x rebound<{args.right_max_rebound}%), "
        f"offset={args.offset}, limit={args.limit}..."
    )

    matches: list[dict] = []
    for index, row in pool.iterrows():
        code = str(row["代码"])
        name = str(row["名称"])
        try:
            history = fetch_history(code, args.min_history_days, args.sleep_seconds)
            result = evaluate_oversold_rebound(
                code, name, industry_map.get(code, "Unknown"), history,
                n_days=args.n_days, m_pct=args.m_pct,
                right_max_days=args.right_max_days,
                right_min_surge=args.right_min_surge,
                right_max_rebound=args.right_max_rebound,
                signal_mode=args.signal_mode,
            )
            if result.get("is_match"):
                matches.append(result)
                print(f"MATCH [{result['signal_type']}] {result['code']} {result['name']} "
                      f"{result['industry']} score={result['score']} "
                      f"dd={result['max_drawdown_pct']}% below_high={result['pct_below_high']}% "
                      f"ma30_days={result['days_above_ma30']} surge={result['vol_surge']}x")
        except Exception as exc:
            print(f"WARN failed {code} {name}: {exc}")

        if (index + 1) % 100 == 0:
            print(f"Progress: {index + 1}/{len(pool)}, matches={len(matches)}")

    # 对匹配的候选股批量获取细分行业（三级分类），写入 display_industry 字段（参考 SEPA Stage2）
    if matches:
        print(f"Fetching sub-industry (3rd-level) for {len(matches)} candidates...")
        match_codes = [m["code"] for m in matches]
        sub_industry_map = get_sub_industry_map(match_codes)
        for m in matches:
            sub_ind = sub_industry_map.get(m["code"])
            if sub_ind:
                m["display_industry"] = sub_ind
        print(f"Sub-industry mapped for {len(sub_industry_map)}/{len(matches)} candidates")

    result_df = pd.DataFrame(matches)
    if not result_df.empty:
        # left_match/right_match 已由 signal_type 概括，不落 CSV
        result_df = result_df.drop(columns=["flags", "left_match", "right_match"], errors="ignore")
        result_df = result_df.sort_values(["score", "amount_cny"], ascending=[False, False])

    output = Path(args.output)
    result_df.to_csv(output, index=False, encoding="utf-8-sig")
    counts = result_df["signal_type"].value_counts().to_dict() if not result_df.empty else {}
    print(f"Saved {len(result_df)} 超跌反弹 candidates to {output} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
