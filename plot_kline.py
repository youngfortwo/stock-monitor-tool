#!/usr/bin/env python3
"""Generate a candlestick chart for an A-share symbol."""
from __future__ import annotations
import argparse
from pathlib import Path

import mplfinance as mpf
import pandas as pd

from stock_scanner import ScanConfig, fetch_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an A-share candlestick chart.")
    parser.add_argument(
        "--symbol", default="000027", help="A-share stock code, e.g. 000027."
    )
    parser.add_argument(
        "--name", default="", help="Optional stock name for the chart title."
    )
    parser.add_argument(
        "--days", type=int, default=120, help="Number of recent trading days to plot."
    )
    parser.add_argument("--output", default="", help="Output PNG path.")
    return parser.parse_args()


def build_config(days: int) -> ScanConfig:
    return ScanConfig(
        history_days=max(days, 180),
        min_history_days=60,
        min_amount=0,
        volume_multiplier=0,
        min_return_20d=-100,
        max_return_20d=1000,
        max_drawdown_10d=100,
        exclude_bj=True,
        sleep_seconds=0,
    )


def prepare_plot_data(history: pd.DataFrame, days: int) -> pd.DataFrame:
    plot_data = history.tail(days).copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"])
    plot_data = plot_data.set_index("date")
    
    plot_data = plot_data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return plot_data[["Open", "High", "Low", "Close", "Volume"]]


def main() -> int:
    args = parse_args()
    config = build_config(args.days)
    history = fetch_history(args.symbol, config)

    if history.empty:
        raise RuntimeError(f"No history data returned for {args.symbol}")

    plot_data = prepare_plot_data(history, args.days)
    title = f"{args.symbol} {args.name}".strip()
    output = Path(args.output or f"kline_{args.symbol}.png")

    mpf.plot(
        plot_data,
        type="candle",
        mav=(5, 20, 60),
        volume=True,
        style="yahoo",
        title=title,
        ylabel="Price",
        ylabel_lower="Volume",
        figratio=(16, 9),
        figscale=1.2,
        savefig=str(output),
    )
    
    print(f"Saved chart to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())