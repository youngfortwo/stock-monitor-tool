#!/usr/bin/env python3
"""Scan A-share stocks that match SEPA Stage 1 (基底/筑底) criteria."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

from financial_filter import check_financial_acceleration, load_cache, save_cache
from sepa_stage2_scanner import (
    build_market_cap_cache,
    calc_stage2_core_conditions,
    calc_all_rps,
    calculate_pe_ttm,
    detect_cup_handle,
    detect_vcp,
    detect_transition,
    fetch_history,
    get_board,
    get_industry_map,
    get_market_cap,
    get_stock_pool,
    next_report_date,
)

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


def evaluate_stage1(code: str, name: str, industry: str, history: pd.DataFrame,
                    financial_cache: dict | None = None,
                    market_caps: dict[str, float] | None = None,
                    rps_120: float | None = None) -> dict:
    if history.empty or len(history) < 220:
        return {
            "code": code, "name": name, "industry": industry,
            "is_stage1": False, "score": 0,
            "error": "Not enough history data.",
            "conditions": {},
        }

    data = history.copy()
    data["ma50"] = data["close"].rolling(50).mean()
    data["ma150"] = data["close"].rolling(150).mean()
    data["ma200"] = data["close"].rolling(200).mean()
    data["vol_avg"] = data["volume"].rolling(20).mean()

    latest = data.iloc[-1]
    high_52w = data.tail(252)["high"].max()
    low_52w = data.tail(252)["low"].min()
    close = float(latest["close"])
    ma50 = float(latest["ma50"])
    ma150 = float(latest["ma150"])
    ma200 = float(latest["ma200"])
    ma200_20d_ago = float(data.iloc[-21]["ma200"])

    pct_off_high = (close / float(high_52w) - 1) * 100
    pct_above_low = (close / float(low_52w) - 1) * 100

    return_20d = (close / float(data.iloc[-21]["close"]) - 1) * 100 if len(data) >= 21 else 0
    return_120d = (close / float(data.iloc[-121]["close"]) - 1) * 100 if len(data) >= 121 else 0

    # MA200 flattening: change over last 20 trading days
    ma200_change_pct = (ma200 / ma200_20d_ago - 1) * 100 if ma200_20d_ago > 0 else -99

    # Volume contraction: compare recent 20d vs 60-120d ago
    recent_vol = data.tail(20)["volume"].mean()
    prior_vol = data.iloc[-121:-21]["volume"].mean() if len(data) >= 121 else recent_vol
    vol_contraction_ratio = recent_vol / prior_vol if prior_vol > 0 else 1

    # Price consolidation: range over last 20d as % of close
    recent_high = data.tail(20)["high"].max()
    recent_low = data.tail(20)["low"].min()
    range_pct = (recent_high / recent_low - 1) * 100 if recent_low > 0 else 100

    # Price near MA200: how close is current price to MA200
    pct_off_ma200 = (close / ma200 - 1) * 100 if ma200 > 0 else 999

    # Stage2 核心条件（复用公共函数，规则与Stage2完全一致）
    stage2_core_count, core_stage2 = calc_stage2_core_conditions(data)
    in_stage2 = stage2_core_count == 9

    # Stage 1 conditions
    conditions = {
        "corrected_25pct": pct_off_high <= -20,
        "ma200_flattening": abs(ma200_change_pct) <= 5,
        "near_ma200": abs(pct_off_ma200) <= 8,
        "volume_contracting": vol_contraction_ratio < 0.85,
        "tight_range": range_pct < 25,
        "near_low": pct_above_low <= 60,
        "not_in_stage2": not in_stage2,
    }

    cup_handle_pattern, cup_handle_details = detect_cup_handle(data)
    vcp_pattern, vcp_details = detect_vcp(data)

    # 财务加速检查（放在过渡态检测之前，以便传入 override）
    fin_pass = False
    fin_score = 0
    fin_rev = []
    fin_profit = []
    fin_margin = []
    if financial_cache is not None:
        try:
            fin = check_financial_acceleration(code, financial_cache)
            fin_pass = fin["financial_pass"]
            fin_score = fin["financial_score"]
            fin_rev = fin["rev_growth"]
            fin_profit = fin["profit_growth"]
            fin_margin = fin["profit_margin"]
        except Exception:
            pass

    # 检测过渡态（传入已计算的值，避免内部重复计算）
    is_transition, transition_info = detect_transition(
        code, data,
        financial_cache=financial_cache,
        rps_120=rps_120,
        core_count_override=stage2_core_count,
        has_vcp_override=vcp_pattern,
        has_cup_override=cup_handle_pattern,
        financial_ready_override=fin_pass,
    )

    # Scoring
    score = 0
    score += 20 if conditions["corrected_25pct"] else max(0, 10 + pct_off_high / -2)
    score += 20 if conditions["ma200_flattening"] else max(0, 20 - abs(ma200_change_pct) * 2)
    score += 20 if conditions["near_ma200"] else max(0, 20 - abs(pct_off_ma200) * 2)
    score += 25 if conditions["volume_contracting"] else max(0, 25 * (1 - min(vol_contraction_ratio, 1.5)))
    score += 20 if conditions["tight_range"] else max(0, 20 * (1 - range_pct / 25))
    score += 10 if cup_handle_pattern else 0
    score += 10 if vcp_pattern else 0
    score += fin_score * 0.5

    # 过渡态额外加分：越接近Stage2，优先级越高
    score += stage2_core_count * 3  # 每多满足1个核心条件，加3分
    if transition_info.get("is_base_transition", False):
        score += 15  # 基础过渡态加15分
    if transition_info.get("is_high_quality_transition", False):
        score += 20  # 高质量过渡态再加20分（优先级最高）

    # Gate: must be corrected, not in stage2, and meet at least one quality signal
    quality_count = sum([conditions["volume_contracting"], conditions["tight_range"],
                         conditions["near_ma200"], cup_handle_pattern, vcp_pattern])

    is_stage1 = (
        conditions["not_in_stage2"]
        and pct_off_high <= -15
        and stage2_core_count <= 8  # 从6放宽到8，纳入过渡态标的
        and quality_count >= 1
    )

    reasons = []
    if conditions["corrected_25pct"]:
        reasons.append(f"从高点回调 ≥20%")
    if conditions["ma200_flattening"]:
        reasons.append("MA200 走平")
    if conditions["near_ma200"]:
        reasons.append(f"贴近MA200（{pct_off_ma200:+.1f}%）")
    if conditions["volume_contracting"]:
        reasons.append("成交量萎缩")
    if conditions["tight_range"]:
        reasons.append("价格波动收窄")
    if cup_handle_pattern:
        reasons.append("杯柄形态")
    if vcp_pattern:
        reasons.append("VCP 波动收缩")
    if fin_pass:
        reasons.append("业绩加速")
    if transition_info.get("is_high_quality_transition", False):
        reasons.append("高质量过渡态：技术面+业绩+RPS三重共振")
    elif transition_info.get("is_base_transition", False):
        reasons.append("Stage1末期→即将进入Stage2")

    # 标签分类
    if transition_info.get("is_high_quality_transition", False):
        label = "高质量过渡态"
    elif transition_info.get("is_base_transition", False):
        label = "基础过渡态"
    else:
        label = "纯筑底股"

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "board": get_board(code),
        "is_stage1": is_stage1,
        "label": label,
        "date": str(latest["date"]),
        "close": round(close, 2),
        "ma50": round(ma50, 2),
        "ma150": round(ma150, 2),
        "ma200": round(ma200, 2),
        "52w_high": round(float(high_52w), 2),
        "52w_low": round(float(low_52w), 2),
        "pct_above_52w_low": round(float(pct_above_low), 2),
        "pct_below_52w_high": round(float(pct_off_high), 2),
        "return_20d_pct": round(float(return_20d), 2),
        "return_120d_pct": round(float(return_120d), 2),
        "amount_cny": round(float(latest["amount"]), 2),
        "volume": round(float(latest["volume"]), 2),
        "score": round(float(score), 2),
        "conditions": {k: bool(v) for k, v in conditions.items()},
        "cup_handle_details": cup_handle_details,
        "vcp_details": vcp_details,
        "financial_pass": fin_pass,
        "financial_score": fin_score,
        "is_transition": is_transition,
        "is_base_transition": transition_info.get("is_base_transition", False),
        "is_high_quality_transition": transition_info.get("is_high_quality_transition", False),
        "stage2_core_count": stage2_core_count,
        "transition_details": transition_info,
        "market_cap_cny": get_market_cap(code, close, market_caps) if market_caps else 0,
        "pe_ttm": calculate_pe_ttm(code, close, financial_cache) if financial_cache else None,
        "next_report_period": next_report_date()["period"],
        "next_report_deadline": next_report_date()["deadline"],
        "rev_growth": json.dumps(fin_rev[-3:] if len(fin_rev) >= 3 else fin_rev, ensure_ascii=False) if fin_rev else "",
        "profit_growth": json.dumps(fin_profit[-3:] if len(fin_profit) >= 3 else fin_profit, ensure_ascii=False) if fin_profit else "",
        "profit_margin": json.dumps(fin_margin[-3:] if len(fin_margin) >= 3 else fin_margin, ensure_ascii=False) if fin_margin else "",
        "matched_reason": "SEPA Stage 1: " + "; ".join(reasons) if reasons else "SEPA Stage 1",
        "ma200_change_pct": round(float(ma200_change_pct), 2),
        "vol_contraction_ratio": round(float(vol_contraction_ratio), 2),
        "range_pct": round(float(range_pct), 2),
        "pct_off_ma200": round(float(pct_off_ma200), 2),
    }


def analyze_stage1(
    code: str, name: str, industry: str, history: pd.DataFrame,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
    rps_120: float | None = None,
) -> dict | None:
    result = evaluate_stage1(code, name, industry, history,
                             financial_cache=financial_cache, market_caps=market_caps,
                             rps_120=rps_120)
    if not result.get("is_stage1"):
        return None
    return result


def scan_one(
    row: pd.Series, industry_map: dict[str, str], args: argparse.Namespace,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
    rps_map: dict[str, float] | None = None,
) -> dict | None:
    code = row["代码"]
    name = row["名称"]
    history = fetch_history(code, args.min_history_days, args.sleep_seconds)
    rps_120 = rps_map.get(code) if rps_map else None
    return analyze_stage1(code, name, industry_map.get(code, "Unknown"), history,
                          financial_cache=financial_cache, market_caps=market_caps,
                          rps_120=rps_120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEPA Stage 1 (基底) scanner")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", type=str, default="sepa_stage1_candidates.csv")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-bj", action="store_true", default=False)
    parser.add_argument("--min-history-days", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        print("WARN --workers is currently forced to 1 because AkShare daily data is not thread-safe.")

    pool = get_stock_pool(args.include_bj, args.limit, args.offset)
    industry_map = get_industry_map()
    financial_cache = load_cache()
    # market_caps 仅用于展示市值，批量扫描时不构建以节省时间
    market_caps = None
    print(
        f"Scanning {len(pool)} stocks for SEPA Stage 1, "
        f"offset={args.offset}, limit={args.limit}, industry mappings={len(industry_map)}..."
    )

    # 单遍：拉取历史数据时收集120日涨跌幅，存内存避免 Pass 2 读缓存 I/O
    print("Pass 1/2: fetching history and collecting 120d returns...")
    stock_returns: dict[str, float] = {}
    history_cache: dict[str, pd.DataFrame] = {}  # 存内存，避免 Pass 2 重复读文件
    for index, row in pool.iterrows():
        code = row["代码"]
        history = fetch_history(code, args.min_history_days, args.sleep_seconds)
        history_cache[code] = history
        if len(history) >= 121:
            close_now = float(history.iloc[-1]["close"])
            close_120d = float(history.iloc[-121]["close"])
            stock_returns[code] = round((close_now / close_120d - 1) * 100, 2)
        if (index + 1) % 100 == 0:
            print(f"  Pass 1 progress: {index + 1}/{len(pool)}")

    # 统一计算 RPS
    print(f"Computing RPS for {len(stock_returns)} stocks...")
    rps_map = calc_all_rps(stock_returns)

    # Pass 2：直接用内存中的 history，不再调 fetch_history（避免 API 请求）
    print("Pass 2/2: evaluating Stage 1 candidates with RPS...")
    matches: list[dict] = []
    for index, row in pool.iterrows():
        code = row["代码"]
        name = row["名称"]
        history = history_cache.get(code, pd.DataFrame())
        try:
            rps_120 = rps_map.get(code)
            result = analyze_stage1(code, name, industry_map.get(code, "Unknown"), history,
                                    financial_cache=financial_cache, market_caps=market_caps,
                                    rps_120=rps_120)
            if result:
                matches.append(result)
                label = result.get("label", "")
                print(f"MATCH {result['code']} {result['name']} {result['industry']} score={result['score']} RPS={rps_120 or 0} {label}")
        except Exception as exc:
            print(f"WARN failed {row['代码']} {row['名称']}: {exc}")

        if (index + 1) % 100 == 0:
            print(f"  Pass 2 progress: {index + 1}/{len(pool)}, matches={len(matches)}")

    save_cache(financial_cache)
    result_df = pd.DataFrame(matches)
    if not result_df.empty:
        result_df = result_df.sort_values(["score", "amount_cny"], ascending=[False, False])

    output = Path(args.output)
    result_df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result_df)} SEPA Stage 1 candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
