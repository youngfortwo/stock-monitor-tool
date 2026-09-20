#!/usr/bin/env python3
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "core", PROJECT_ROOT / "db", PROJECT_ROOT / "api"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Disable all proxies
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

print("Testing AkShare API (no proxy)...")
print(f"  Python: {sys.executable}")

import akshare as ak

try:
    print("\n1. Fetching stock list...")
    df = ak.stock_info_a_code_name()
    print(f"   ✓ Success: Got {len(df)} stocks")
    print(f"   First 3: {list(df[['code', 'name']].head(3).itertuples(index=False, name=None))}")

    print("\n2. Fetching history for 000001...")
    df_hist = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", end_date="20240601", adjust="qfq")
    print(f"   ✓ Success: Got {len(df_hist)} trading days")
    print(f"   Columns: {list(df_hist.columns)}")

    print("\nAll tests passed! API is working correctly.")

except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
