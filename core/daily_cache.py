#!/usr/bin/env python3
"""Daily stock data cache: stores ak.stock_zh_a_daily results as local CSV files.

Allows Stage1 and Stage2 scanners to share fetched data, eliminating duplicate
API calls when both scans are run in the same session.

TTL: 4 hours (stale cache won't break scoring, just returns slightly outdated data)
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("daily_cache")
CACHE_TTL_SECONDS = 4 * 3600  # 4 hours


def cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code}.csv"


def load_cached(code: str, min_days: int = 0) -> pd.DataFrame | None:
    """Load daily data from local cache if fresh enough and has enough rows."""
    p = cache_path(code)
    if not p.exists():
        return None
    try:
        mtime = os.path.getmtime(str(p))
        if time.time() - mtime > CACHE_TTL_SECONDS:
            return None
        df = pd.read_csv(p, dtype={"code": str})
        if df.empty or (min_days > 0 and len(df) < min_days):
            return None
        for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume", "amount"]).reset_index(drop=True)
        if min_days > 0 and len(df) < min_days:
            return None
        return df
    except Exception:
        return None


def save_cached(code: str, df: pd.DataFrame) -> None:
    """Save daily data to local cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        df.to_csv(cache_path(code), index=False, encoding="utf-8")
    except Exception:
        pass


def clear_expired() -> int:
    """Remove cache files older than TTL. Returns count removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    cutoff = time.time() - CACHE_TTL_SECONDS
    for f in CACHE_DIR.glob("*.csv"):
        try:
            if os.path.getmtime(str(f)) < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass
    return removed


def stats() -> dict:
    """Return cache statistics."""
    if not CACHE_DIR.exists():
        return {"count": 0, "size_mb": 0}
    files = list(CACHE_DIR.glob("*.csv"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"count": len(files), "size_mb": round(total_bytes / 1024 / 1024, 2)}
