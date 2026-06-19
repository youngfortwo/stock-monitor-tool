#!/usr/bin/env python3
"""Generate A-share market breadth and a simple panic index."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import akshare as ak
import pandas as pd

EPSILON = 0.005


def fetch_spot_data() -> pd.DataFrame:
    errors: list[str] = []
    for source_name, loader in (
        ("stock_zh_a_spot_em", ak.stock_zh_a_spot_em),
        ("stock_zh_a_spot", ak.stock_zh_a_spot),
    ):
        try:
            spot = loader()
            required_columns = {"代码", "名称", "涨跌幅"}
            missing = required_columns.difference(spot.columns)
            if missing:
                raise RuntimeError(f"missing columns: {sorted(missing)}")
            print(f"Using market breadth source: {source_name}")
            return spot
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
    raise RuntimeError("All market breadth sources failed: " + " | ".join(errors))


def build_market_breadth(spot: pd.DataFrame) -> dict:
    data = spot[["代码", "名称", "涨跌幅"]].copy()
    data["代码"] = data["代码"].astype(str).str.zfill(6)
    data["涨跌幅"] = pd.to_numeric(data["涨跌幅"], errors="coerce")
    data = data.dropna(subset=["涨跌幅"])

    up_count = int((data["涨跌幅"] > EPSILON).sum())
    down_count = int((data["涨跌幅"] < -EPSILON).sum())
    flat_count = int((data["涨跌幅"].abs() <= EPSILON).sum())
    moving_count = up_count + down_count
    total_count = int(len(data))

    panic_index = round((down_count / moving_count * 100) if moving_count else 0, 2)
    up_ratio = round((up_count / total_count * 100) if total_count else 0, 2)
    down_ratio = round((down_count / total_count * 100) if total_count else 0, 2)

    return {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": total_count,
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "up_ratio": up_ratio,
        "down_ratio": down_ratio,
        "panic_index": panic_index,
        "panic_index_note": "0-100; calculated as down_count / (up_count + down_count) * 100",
        "avg_pct_change": round(float(data["涨跌幅"].mean()), 2),
        "median_pct_change": round(float(data["涨跌幅"].median()), 2),
    }


def main() -> int:
    spot = fetch_spot_data()
    breadth = build_market_breadth(spot)
    output = Path("market_breadth.json")
    output.write_text(json.dumps(breadth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved market breadth to {output}")
    print(json.dumps(breadth, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())