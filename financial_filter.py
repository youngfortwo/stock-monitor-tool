#!/usr/bin/env python3
"""Financial data filter: check for accelerating net profit, revenue, and profit margin."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import akshare as ak
import pandas as pd

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

CACHE_FILE = Path("financial_cache.json")


def load_cache() -> dict[str, dict]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_financial_indicators(code: str, cache: dict) -> pd.DataFrame | None:
    """Fetch financial analysis indicators for a single stock, with caching."""
    if code in cache and cache[code]:
        rows = cache[code]
        if isinstance(rows, list) and rows:
            return pd.DataFrame(rows)

    try:
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2024")
        if df is None or df.empty:
            cache[code] = []
            return None
        result = df.tail(6).copy()
        for col in result.select_dtypes(include=["datetime64", "object"]).columns:
            try:
                result[col] = result[col].astype(str)
            except Exception:
                pass
        cache[code] = result.to_dict(orient="records")
        return result
    except Exception:
        cache[code] = []
        return None


def check_financial_acceleration(code: str, cache: dict | None = None) -> dict:
    """Check if a stock has accelerating net profit, revenue, and improving profit margin.

    Returns a dict with:
        - financial_pass: bool — all three metrics show acceleration/improvement
        - rev_growth: list — last 3 quarters revenue growth rates
        - profit_growth: list — last 3 quarters net profit growth rates
        - profit_margin: list — last 3 quarters net profit margins
        - financial_score: int — 0-15 bonus score
    """
    if cache is None:
        cache = load_cache()

    df = fetch_financial_indicators(code, cache)
    if df is None or df.empty or len(df) < 3:
        return {
            "financial_pass": False,
            "rev_growth": [],
            "profit_growth": [],
            "profit_margin": [],
            "financial_score": 0,
        }

    rev_col = "主营业务收入增长率(%)"
    profit_growth_col = "净利润增长率(%)"
    margin_col = "销售净利率(%)"

    available_cols = {c for c in [rev_col, profit_growth_col, margin_col] if c in df.columns}
    if len(available_cols) < 2:
        return {
            "financial_pass": False,
            "rev_growth": [],
            "profit_growth": [],
            "profit_margin": [],
            "financial_score": 0,
        }

    recent = df.tail(4)
    recent = recent.apply(pd.to_numeric, errors="coerce")
    score = 0

    rev_growth_vals = []
    if rev_col in df.columns:
        series = pd.to_numeric(df[rev_col], errors="coerce").dropna().tail(4)
        rev_growth_vals = [round(float(v), 2) for v in series.tolist()]
        if len(rev_growth_vals) >= 3:
            latest = rev_growth_vals[-1]
            prev = rev_growth_vals[-2]
            prev2 = rev_growth_vals[-3]
            if latest > prev and prev > prev2 and latest > 0:
                score += 5

    profit_growth_vals = []
    if profit_growth_col in df.columns:
        series = pd.to_numeric(df[profit_growth_col], errors="coerce").dropna().tail(4)
        profit_growth_vals = [round(float(v), 2) for v in series.tolist()]
        if len(profit_growth_vals) >= 3:
            latest = profit_growth_vals[-1]
            prev = profit_growth_vals[-2]
            if latest > prev and latest > 0:
                score += 5
                if prev > 0:
                    score += 2

    margin_vals = []
    if margin_col in df.columns:
        series = pd.to_numeric(df[margin_col], errors="coerce").dropna().tail(4)
        margin_vals = [round(float(v), 2) for v in series.tolist()]
        if len(margin_vals) >= 3:
            latest = margin_vals[-1]
            prev = margin_vals[-2]
            if latest > prev:
                score += 3

    financial_pass = score >= 8
    return {
        "financial_pass": financial_pass,
        "rev_growth": rev_growth_vals,
        "profit_growth": profit_growth_vals,
        "profit_margin": margin_vals,
        "financial_score": score,
    }


if __name__ == "__main__":
    cache = load_cache()
    result = check_financial_acceleration("000001", cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    save_cache(cache)
    print(f"Cache has {len(cache)} entries")
