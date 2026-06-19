#!/usr/bin/env python3
"""
A-share stock scanner for rule-based watchlist generation.
This tool fetches public market data with AkShare, applies a small set of
trend/volume/risk filters, and writes the matched stocks to a CSV file.
It is for research and monitoring only, not investment advice.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import akshare as ak
import pandas as pd

from financial_filter import check_financial_acceleration, load_cache, save_cache
from sepa_stage2_scanner import get_industry_map, load_industry_overrides

# Disable proxies to avoid issues with AkShare
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)


@dataclass(frozen=True)
class ScanConfig:
    history_days: int
    min_history_days: int
    min_amount: float
    volume_multiplier: float
    min_return_20d: float
    max_return_20d: float
    max_drawdown_10d: float
    exclude_bj: bool
    sleep_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan A-share stocks by trend, volume, liquidity, and drawdown rules."
    )
    parser.add_argument(
        "--output", default="stock_candidates.csv", help="Output CSV path."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Only scan first N stocks for testing."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N stocks before scanning."
    )
    parser.add_argument(
        "--history-days", type=int, default=180, help="Trading days to fetch per stock."
    )
    parser.add_argument(
        "--min-history-days", type=int, default=120, help="Minimum usable history days."
    )
    parser.add_argument(
        "--min-amount", type=float, default=100_000_000,
        help="Minimum latest daily amount in CNY."
    )
    parser.add_argument(
        "--volume-multiplier", type=float, default=1.5,
        help="Latest volume vs 5-day average."
    )
    parser.add_argument(
        "--min-return-20d", type=float, default=5.0,
        help="Minimum 20-day return percentage."
    )
    parser.add_argument(
        "--max-return-20d", type=float, default=30.0,
        help="Maximum 20-day return percentage."
    )
    parser.add_argument(
        "--max-drawdown-10d", type=float, default=15.0,
        help="Maximum 10-day drawdown percentage."
    )
    parser.add_argument(
        "--include-bj", action="store_true", help="Include Beijing Stock Exchange stocks."
    )
    parser.add_argument(
        "--sleep-seconds", type=float, default=0.15,
        help="Sleep between history requests."
    )
    return parser.parse_args()


def today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


def start_yyyymmdd(history_days: int) -> str:
    # Calendar days are intentionally wider than trading days to cover weekends and holidays.
    return (dt.date.today() - dt.timedelta(days=history_days * 2)).strftime("%Y%m%d")


def get_stock_pool(exclude_bj: bool, limit: int, offset: int) -> pd.DataFrame:
    errors: list[str] = []
    for source_name, loader in (
        ("stock_zh_a_spot_em", ak.stock_zh_a_spot_em),
        ("stock_info_a_code_name", ak.stock_info_a_code_name),
        ("stock_zh_a_spot", ak.stock_zh_a_spot),
    ):
        try:
            spot = loader()
            spot = spot.rename(columns={"code": "代码", "name": "名称"})
            required_columns = {"代码", "名称"}
            missing = required_columns.difference(spot.columns)
            if missing:
                raise RuntimeError(f"missing columns: {sorted(missing)}")
            print(f"Using stock pool source: {source_name}")
            break
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
    else:
        raise RuntimeError("All stock pool sources failed: " + " | ".join(errors))

    pool = spot[["代码", "名称"]].copy()
    pool["代码"] = pool["代码"].astype(str).str.zfill(6)
    pool = pool[~pool["名称"].str.contains("ST|退", regex=True, na=False)]

    if exclude_bj:
        pool = pool[~pool["代码"].str.startswith(("8", "4"))]
    if offset > 0:
        pool = pool.iloc[offset:]
    if limit > 0:
        pool = pool.head(limit)

    return pool.reset_index(drop=True)


def fetch_history(symbol: str, config: ScanConfig) -> pd.DataFrame:
    errors: list[str] = []
    try:
        history = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_yyyymmdd(config.history_days),
            end_date=today_yyyymmdd(),
            adjust="qfq",
        )
    except Exception as exc:
        errors.append(f"stock_zh_a_hist: {exc}")
        market_symbol = f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"
        try:
            history = ak.stock_zh_a_daily(
                symbol=market_symbol,
                start_date=start_yyyymmdd(config.history_days),
                end_date=today_yyyymmdd(),
                adjust="qfq",
            )
        except Exception as fallback_exc:
            errors.append(f"stock_zh_a_daily: {fallback_exc}")
            raise RuntimeError("All history sources failed: " + " | ".join(errors)) from fallback_exc

    if history.empty:
        return history

    history = history.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_change",
            "换手率": "turnover",
        }
    )

    numeric_columns = ["open", "close", "high", "low", "volume", "amount", "pct_change", "turnover"]
    for column in numeric_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    if "pct_change" not in history.columns:
        history["pct_change"] = history["close"].pct_change() * 100
    if "turnover" not in history.columns:
        history["turnover"] = 0

    return history.dropna(subset=["close", "volume", "amount"]).reset_index(drop=True)


def max_drawdown_percent(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdown = (series / running_max - 1) * 100
    return abs(float(drawdown.min()))


def analyze_stock(
    code: str,
    name: str,
    industry: str,
    history: pd.DataFrame,
    config: ScanConfig,
    financial_cache: dict | None = None,
    skip_financial: bool = False,
) -> dict | None:
    if len(history) < config.min_history_days:
        return None

    history = history.copy()
    history["ma20"] = history["close"].rolling(20).mean()
    history["ma60"] = history["close"].rolling(60).mean()
    history["vol5"] = history["volume"].rolling(5).mean()
    history["vol20"] = history["volume"].rolling(20).mean()

    latest = history.iloc[-1]
    close_20d_ago = history.iloc[-21]["close"]
    return_20d = (latest["close"] / close_20d_ago - 1) * 100
    drawdown_10d = max_drawdown_percent(history.tail(10)["close"])
    volume_ratio = latest["volume"] / latest["vol5"] if latest["vol5"] else 0

    matched = (
        latest["close"] > latest["ma20"] > latest["ma60"]
        and volume_ratio >= config.volume_multiplier
        and latest["amount"] >= config.min_amount
        and config.min_return_20d <= return_20d <= config.max_return_20d
        and drawdown_10d <= config.max_drawdown_10d
    )
    if not matched:
        return None

    score = 0
    score += 30 if latest["close"] > latest["ma20"] > latest["ma60"] else 0
    score += min(25, volume_ratio * 8)
    score += 20 if config.min_return_20d <= return_20d <= config.max_return_20d else 0
    score += max(0, 15 - drawdown_10d)
    score += 10 if latest["amount"] >= config.min_amount * 2 else 5

    fin_pass = False
    fin_score = 0
    fin_rev = []
    fin_profit = []
    fin_margin = []
    if not skip_financial and financial_cache is not None:
        try:
            fin = check_financial_acceleration(code, financial_cache)
            fin_pass = fin["financial_pass"]
            fin_score = fin["financial_score"]
            fin_rev = fin["rev_growth"]
            fin_profit = fin["profit_growth"]
            fin_margin = fin["profit_margin"]
            score += fin_score
        except Exception:
            pass

    reasons = ["close>MA20>MA60", "volume breakout", "liquid", "20d return in range", "drawdown controlled"]
    if fin_pass:
        reasons.append("acceleration: rev+profit up, margin improving")

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "date": latest["date"],
        "close": round(float(latest["close"]), 2),
        "pct_change": round(float(latest.get("pct_change", 0)), 2),
        "amount_cny": round(float(latest["amount"]), 2),
        "volume_ratio_vs_5d": round(float(volume_ratio), 2),
        "return_20d_pct": round(float(return_20d), 2),
        "max_drawdown_10d_pct": round(float(drawdown_10d), 2),
        "ma20": round(float(latest["ma20"]), 2),
        "ma60": round(float(latest["ma60"]), 2),
        "score": round(float(score), 2),
        "financial_pass": fin_pass,
        "financial_score": fin_score,
        "rev_growth": json.dumps(fin_rev[-3:] if len(fin_rev) >= 3 else fin_rev, ensure_ascii=False) if fin_rev else "",
        "profit_growth": json.dumps(fin_profit[-3:] if len(fin_profit) >= 3 else fin_profit, ensure_ascii=False) if fin_profit else "",
        "profit_margin": json.dumps(fin_margin[-3:] if len(fin_margin) >= 3 else fin_margin, ensure_ascii=False) if fin_margin else "",
        "matched_reason": "; ".join(reasons),
    }


def run_scan(args: argparse.Namespace) -> pd.DataFrame:
    config = ScanConfig(
        history_days=args.history_days,
        min_history_days=args.min_history_days,
        min_amount=args.min_amount,
        volume_multiplier=args.volume_multiplier,
        min_return_20d=args.min_return_20d,
        max_return_20d=args.max_return_20d,
        max_drawdown_10d=args.max_drawdown_10d,
        exclude_bj=not args.include_bj,
        sleep_seconds=args.sleep_seconds,
    )
    pool = get_stock_pool(config.exclude_bj, args.limit, args.offset)
    industry_map = get_industry_map()
    financial_cache = load_cache()
    skip_financial = getattr(args, "skip_financial", False)
    print(f"Scanning {len(pool)} stocks, offset={args.offset}, limit={args.limit}...")

    matches: list[dict] = []
    for index, row in pool.iterrows():
        code = row["代码"]
        name = row["名称"]
        try:
            history = fetch_history(code, config)
            result = analyze_stock(code, name, industry_map.get(code, "Unknown"), history, config,
                                   financial_cache=financial_cache, skip_financial=skip_financial)
            if result:
                matches.append(result)
                print(f"MATCH {code} {name} score={result['score']}")
        except Exception as exc:
            # Keep a full-market scan running when one symbol fails.
            print(f"WARN failed {code} {name}: {exc}", file=sys.stderr)
        finally:
            if config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)

        if (index + 1) % 100 == 0:
            print(f"Progress: {index + 1}/{len(pool)}, matches={len(matches)}")

    save_cache(financial_cache)
    if not matches:
        return pd.DataFrame()
    return pd.DataFrame(matches).sort_values(["score", "amount_cny"], ascending=[False, False])


def main() -> int:
    args = parse_args()
    result = run_scan(args)
    output = Path(args.output)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
