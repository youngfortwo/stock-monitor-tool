#!/usr/bin/env python3
"""Scan A-share stocks near their Sep 2024 bottom with good moat & fundamentals."""
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

from financial_filter import check_financial_acceleration, fetch_financial_indicators, load_cache, save_cache
from sepa_stage2_scanner import (
    calculate_pe_ttm,
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


def fetch_history_long(symbol: str, min_history_days: int, sleep_seconds: float) -> pd.DataFrame:
    from daily_cache import load_cached, save_cached

    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"

    # Step 1: Get standard daily data via shared cache (Stage1/Stage2 also fill this)
    base = load_cached(symbol, min_days=min_history_days)
    if base is None or len(base) < min_history_days:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        start = (dt.date.today() - dt.timedelta(days=520)).strftime("%Y%m%d")
        end = dt.date.today().strftime("%Y%m%d")
        base = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", start_date=start, end_date=end, adjust="qfq")
        if base is not None and not base.empty:
            for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                if col in base.columns:
                    base[col] = pd.to_numeric(base[col], errors="coerce")
            base = base.dropna(subset=["close", "volume", "amount"]).reset_index(drop=True)
            if not base.empty and len(base) >= 250:
                save_cached(symbol, base)
        else:
            base = pd.DataFrame()

    # Step 2: Fetch Q3 2024 data (2024-07-01 to 2024-10-31) for 924 bottom
    q3_path = Path("daily_cache") / f"{symbol}_q3.csv"
    if q3_path.exists():
        try:
            q3 = pd.read_csv(q3_path)
        except Exception:
            q3 = pd.DataFrame()
    else:
        q3 = pd.DataFrame()
    if q3.empty:
        q3 = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", start_date="20240701", end_date="20241031", adjust="qfq")
        if q3 is not None and not q3.empty:
            for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                if col in q3.columns:
                    q3[col] = pd.to_numeric(q3[col], errors="coerce")
            q3 = q3.dropna(subset=["close", "volume", "amount"]).reset_index(drop=True)
            if not q3.empty:
                try:
                    Path("daily_cache").mkdir(exist_ok=True)
                    q3.to_csv(q3_path, index=False, encoding="utf-8")
                except Exception:
                    pass
        else:
            q3 = pd.DataFrame()

    # Combine: deduplicate by date, sort chronologically
    frames = []
    if isinstance(base, pd.DataFrame) and not base.empty:
        base["date"] = pd.to_datetime(base["date"])
        frames.append(base)
    if not q3.empty:
        q3["date"] = pd.to_datetime(q3["date"])
        frames.append(q3)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.dropna(subset=["close", "volume", "amount"]).reset_index(drop=True)


def check_moat(code: str, cache: dict | None = None) -> dict:
    """Check moat quality: ROE, gross margin, net margin, FCF/NI ratio.

    Uses cached financial data (same cache as financial_filter).
    Returns scores per dimension + overall moat_pass.
    """
    if cache is None:
        cache = load_cache()

    df = fetch_financial_indicators(code, cache)
    if df is None or df.empty or len(df) < 3:
        return {"moat_pass": False, "moat_score": 0, "moat_roe": [], "moat_gross_margin": [], "moat_net_margin": [], "moat_fcf_ratio": []}

    col_map = {
        "roe": "净资产收益率(%)",
        "gross_margin": "销售毛利率(%)",
        "net_margin": "销售净利率(%)",
        "fcf_ratio": "经营现金净流量与净利润的比率(%)",
    }

    vals = {}
    for key, col in col_map.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna().tail(8)
            vals[key] = [round(float(v), 2) for v in s.tolist()]
        else:
            vals[key] = []

    moat_score = 0
    details = {}

    # ROE: average >= 12%, no consecutive decline
    roe_vals = vals["roe"]
    if len(roe_vals) >= 4:
        avg_roe = sum(roe_vals) / len(roe_vals)
        recent_rising = len(roe_vals) >= 3 and roe_vals[-1] > roe_vals[-2]
        if avg_roe >= 15:
            moat_score += 15
        elif avg_roe >= 10:
            moat_score += 10
        elif avg_roe >= 5:
            moat_score += 5
        if recent_rising:
            moat_score += 3
        details["roe_avg"] = round(avg_roe, 2)
    else:
        details["roe_avg"] = 0

    # Gross margin: average >= 20%, stable
    gm_vals = vals["gross_margin"]
    if len(gm_vals) >= 4:
        avg_gm = sum(gm_vals) / len(gm_vals)
        if avg_gm >= 40:
            moat_score += 12
        elif avg_gm >= 20:
            moat_score += 8
        elif avg_gm >= 10:
            moat_score += 4
        details["gm_avg"] = round(avg_gm, 2)
    else:
        details["gm_avg"] = 0

    # Net margin: average >= 10%
    nm_vals = vals["net_margin"]
    if len(nm_vals) >= 4:
        avg_nm = sum(nm_vals) / len(nm_vals)
        if avg_nm >= 20:
            moat_score += 12
        elif avg_nm >= 10:
            moat_score += 8
        elif avg_nm >= 5:
            moat_score += 4
        details["nm_avg"] = round(avg_nm, 2)
    else:
        details["nm_avg"] = 0

    # FCF/NI: average >= 0.6 (earnings are real cash, ratio form)
    fcf_vals = vals["fcf_ratio"]
    if len(fcf_vals) >= 4:
        avg_fcf = sum(fcf_vals) / len(fcf_vals)
        if avg_fcf >= 1.0:
            moat_score += 10
        elif avg_fcf >= 0.6:
            moat_score += 7
        elif avg_fcf >= 0.3:
            moat_score += 4
        details["fcf_avg"] = round(avg_fcf, 2)
    else:
        details["fcf_avg"] = 0

    moat_pass = moat_score >= 12

    return {
        "moat_pass": moat_pass,
        "moat_score": moat_score,
        "moat_roe": vals["roe"][-4:] if len(vals["roe"]) >= 4 else vals["roe"],
        "moat_gross_margin": vals["gross_margin"][-4:] if len(vals["gross_margin"]) >= 4 else vals["gross_margin"],
        "moat_net_margin": vals["net_margin"][-4:] if len(vals["net_margin"]) >= 4 else vals["net_margin"],
        "moat_fcf_ratio": vals["fcf_ratio"][-4:] if len(vals["fcf_ratio"]) >= 4 else vals["fcf_ratio"],
        "moat_details": details,
    }


def evaluate_value_bottom(code: str, name: str, industry: str, history: pd.DataFrame,
                          financial_cache: dict | None = None,
                          market_caps: dict[str, float] | None = None) -> dict:
    if history.empty or len(history) < 250:
        return {
            "code": code, "name": name, "industry": industry,
            "is_match": False, "score": 0,
            "error": "Not enough history data.",
            "924_bottom": 0, "close": 0, "pct_above_924": 0, "market_cap_b": 0,
            "financial_pass": False, "moat_pass": False, "moat_score": 0,
            "moat_roe": "", "moat_gross_margin": "", "moat_net_margin": "", "moat_fcf_ratio": "", "moat_details": "{}",
        }

    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["ma200"] = data["close"].rolling(200).mean()

    latest = data.iloc[-1]
    close = float(latest["close"])
    ma200 = float(latest["ma200"]) if not pd.isna(latest["ma200"]) else close

    # 1. Find Sep 2024 bottom
    q3_start = pd.Timestamp("2024-07-01")
    q3_end = pd.Timestamp("2024-10-31")
    q3_data = data[(data["date"] >= q3_start) & (data["date"] <= q3_end)]
    if q3_data.empty:
        return {
            "code": code, "name": name, "industry": industry,
            "is_match": False, "score": 0,
            "error": "No Q3 2024 data.",
            "924_bottom": 0, "close": close, "pct_above_924": 0, "market_cap_b": 0,
            "financial_pass": False, "moat_pass": False, "moat_score": 0,
            "moat_roe": "", "moat_gross_margin": "", "moat_net_margin": "", "moat_fcf_ratio": "", "moat_details": "{}",
        }
    bottom_924 = float(q3_data["low"].min())
    bottom_date = str(q3_data.loc[q3_data["low"].idxmin(), "date"])[:10] if len(q3_data) else ""
    pct_above_bottom = (close / bottom_924 - 1) * 100

    # 2. 52w metrics
    high_52w = data.tail(252)["high"].max()
    low_52w = data.tail(252)["low"].min()
    pct_off_high = (close / float(high_52w) - 1) * 100
    pct_above_low = (close / float(low_52w) - 1) * 100

    # 3. Volume contraction to confirm low activity
    recent_vol = data.tail(20)["volume"].mean()
    prior_vol = data.iloc[-121:-21]["volume"].mean() if len(data) >= 121 else recent_vol
    vol_ratio = recent_vol / prior_vol if prior_vol > 0 else 1

    # 4. Price near MA200
    pct_off_ma200 = (close / ma200 - 1) * 100 if ma200 > 0 else 999

    # 5. Market cap
    mcap = get_market_cap(code, close, market_caps) if market_caps else 0
    mcap_b = mcap / 1e8 if mcap else 0

    # 6. Moat check (ROE + gross margin + net margin + FCF/NI)
    moat_pass = False
    moat_score = 0
    moat_roe = []
    moat_gm = []
    moat_nm = []
    moat_fcf = []
    moat_details = {}
    if financial_cache is not None:
        try:
            moat = check_moat(code, financial_cache)
            moat_pass = moat["moat_pass"]
            moat_score = moat["moat_score"]
            moat_roe = moat["moat_roe"]
            moat_gm = moat["moat_gross_margin"]
            moat_nm = moat["moat_net_margin"]
            moat_fcf = moat["moat_fcf_ratio"]
            moat_details = moat.get("moat_details", {})
        except Exception:
            pass

    # 7. Financial acceleration
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

    # Score
    score = 0

    # Near Sep 2024 bottom: < 25% above
    near_bottom = pct_above_bottom <= 25
    score += 30 if near_bottom else max(0, int(30 - (pct_above_bottom - 15) * 1.5))

    # Moat score
    score += moat_score

    # Financial growth
    score += fin_score * 0.5 if fin_score else 0

    # Near MA200
    near_ma200 = abs(pct_off_ma200) <= 15
    score += 10 if near_ma200 else max(0, int(10 - abs(pct_off_ma200)))

    # Volume low (below average)
    low_vol = vol_ratio < 1.0
    score += 10 if low_vol else max(0, int(10 * (1 - min(vol_ratio, 1.5))))

    # Market cap bonus (not a gate, just extra points for scale)
    if mcap >= 1e11:
        score += 10
    elif mcap >= 5e10:
        score += 5

    # Gate conditions
    is_match = (
        near_bottom
        and moat_pass
        and fin_pass
        and pct_off_high <= -10
    )

    reasons = []
    if near_bottom:
        reasons.append(f"距924底+{pct_above_bottom:.0f}%")
    if moat_pass:
        d = moat_details
        parts = []
        if d.get("roe_avg"):
            parts.append(f"ROE{d['roe_avg']:.1f}%")
        if d.get("gm_avg"):
            parts.append(f"毛利率{d['gm_avg']:.1f}%")
        if d.get("nm_avg"):
            parts.append(f"净利率{d['nm_avg']:.1f}%")
        reasons.append("护城河(" + ",".join(parts) + ")")
    if fin_pass:
        reasons.append("业绩增长")
    if near_ma200:
        reasons.append(f"贴近MA200")
    if low_vol:
        reasons.append("低成交量")
    if mcap_b >= 500:
        reasons.append(f"市值{mcap_b:.0f}亿")

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "board": get_board(code),
        "is_match": is_match,
        "date": str(latest["date"]),
        "close": round(close, 2),
        "ma200": round(ma200, 2),
        "924_bottom": round(bottom_924, 2),
        "924_bottom_date": bottom_date,
        "pct_above_924": round(float(pct_above_bottom), 2),
        "52w_high": round(float(high_52w), 2),
        "52w_low": round(float(low_52w), 2),
        "pct_above_52w_low": round(float(pct_above_low), 2),
        "pct_below_52w_high": round(float(pct_off_high), 2),
        "vol_ratio": round(float(vol_ratio), 2),
        "pct_off_ma200": round(float(pct_off_ma200), 2),
        "market_cap_cny": round(mcap, 2) if mcap else 0,
        "pe_ttm": calculate_pe_ttm(code, close, financial_cache) if financial_cache else None,
        "market_cap_b": round(mcap_b, 2) if mcap else 0,
        "score": round(float(score), 2),
        "financial_pass": fin_pass,
        "financial_score": fin_score,
        "moat_pass": moat_pass,
        "moat_score": moat_score,
        "moat_roe": json.dumps(moat_roe, ensure_ascii=False) if moat_roe else "",
        "moat_gross_margin": json.dumps(moat_gm, ensure_ascii=False) if moat_gm else "",
        "moat_net_margin": json.dumps(moat_nm, ensure_ascii=False) if moat_nm else "",
        "moat_fcf_ratio": json.dumps(moat_fcf, ensure_ascii=False) if moat_fcf else "",
        "moat_details": json.dumps(moat_details, ensure_ascii=False) if moat_details else "{}",
        "rev_growth": json.dumps(fin_rev[-3:] if len(fin_rev) >= 3 else fin_rev, ensure_ascii=False) if fin_rev else "",
        "profit_growth": json.dumps(fin_profit[-3:] if len(fin_profit) >= 3 else fin_profit, ensure_ascii=False) if fin_profit else "",
        "profit_margin": json.dumps(fin_margin[-3:] if len(fin_margin) >= 3 else fin_margin, ensure_ascii=False) if fin_margin else "",
        "matched_reason": "价值底部: " + "; ".join(reasons) if reasons else "价值底部",
        "amount_cny": round(float(latest["amount"]), 2),
        "return_20d_pct": round((close / float(data.iloc[-21]["close"]) - 1) * 100, 2) if len(data) >= 21 else 0,
        "return_120d_pct": round((close / float(data.iloc[-121]["close"]) - 1) * 100, 2) if len(data) >= 121 else 0,
        "next_report_period": next_report_date()["period"],
        "next_report_deadline": next_report_date()["deadline"],
    }


def analyze_value_bottom(
    code: str, name: str, industry: str, history: pd.DataFrame,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
) -> dict | None:
    result = evaluate_value_bottom(code, name, industry, history,
                                   financial_cache=financial_cache, market_caps=market_caps)
    if not result.get("is_match"):
        return None
    return result


def scan_one(
    row: pd.Series, industry_map: dict[str, str], args: argparse.Namespace,
    financial_cache: dict | None = None,
    market_caps: dict[str, float] | None = None,
) -> dict | None:
    code = row["代码"]
    name = row["名称"]
    history = fetch_history_long(code, args.min_history_days, args.sleep_seconds)
    return analyze_value_bottom(code, name, industry_map.get(code, "Unknown"), history,
                                financial_cache=financial_cache, market_caps=market_caps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Value bottom (924底部 + 护城河) scanner")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", type=str, default="value_bottom_candidates.csv")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-bj", action="store_true", default=False)
    parser.add_argument("--min-history-days", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        print("WARN --workers is currently forced to 1 because AkShare daily data is not thread-safe.")

    pool = get_stock_pool(args.include_bj, args.limit, args.offset)
    industry_map = get_industry_map()
    financial_cache = load_cache()
    # Skip market_caps in batch mode to avoid O(N) API calls per batch
    print(
        f"Scanning {len(pool)} stocks for value bottom, "
        f"offset={args.offset}, limit={args.limit}, industry mappings={len(industry_map)}..."
    )

    matches: list[dict] = []
    for index, row in pool.iterrows():
        try:
            result = scan_one(row, industry_map, args, financial_cache=financial_cache, market_caps=None)
            if result:
                matches.append(result)
                print(f"MATCH {result['code']} {result['name']} {result['industry']} score={result['score']}")
        except Exception as exc:
            print(f"WARN failed {row['代码']} {row['名称']}: {exc}")

        if (index + 1) % 100 == 0:
            print(f"Progress: {index + 1}/{len(pool)}, matches={len(matches)}")

    save_cache(financial_cache)
    result_df = pd.DataFrame(matches)
    if not result_df.empty:
        result_df = result_df.sort_values(["score", "market_cap_b"], ascending=[False, False])

    output = Path(args.output)
    result_df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result_df)} value bottom candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
