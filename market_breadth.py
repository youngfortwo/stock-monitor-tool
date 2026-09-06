#!/usr/bin/env python3
"""Generate A-share market breadth and a simple panic index."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

EPSILON = 0.005


def _fetch_eastmoney_direct() -> pd.DataFrame:
    """直接调用东方财富 clist 接口，作为 akshare 失败时的备选。

    分页获取全市场 A 股（沪深京），返回与 akshare stock_zh_a_spot_em 相同结构的 DataFrame。
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })

    # 尝试多个 push2 域名
    hosts = [
        "https://82.push2.eastmoney.com",
        "https://19.push2.eastmoney.com",
        "https://push2.eastmoney.com",
        "http://82.push2.eastmoney.com",
    ]
    # 沪深 A 股 + 北交所
    fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    all_rows = []
    page = 1
    page_size = 2000
    total = None
    base_url = None
    last_error = None
    fields = "f12,f14,f3,f6,f62"

    for host in hosts:
        try:
            test_url = f"{host}/api/qt/clist/get"
            r = session.get(test_url, params={
                "pn": 1, "pz": 10, "po": 1, "np": 1,
                "fltt": 2, "invt": 2, "fid": "f3",
                "fs": fs, "fields": fields,
            }, timeout=10)
            if r.status_code == 200 and r.json().get("data"):
                base_url = test_url
                break
        except Exception as e:
            last_error = e
            continue

    if not base_url:
        raise RuntimeError(f"all eastmoney push2 hosts unavailable: {last_error}")

    while True:
        params = {
            "pn": page, "pz": page_size, "po": 1, "np": 1,
            "fltt": 2, "invt": 2, "fid": "f12",
            "fs": fs,
            "fields": fields,
        }
        r = session.get(base_url, params=params, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"eastmoney direct HTTP {r.status_code}")
        data = r.json()
        if not data.get("data"):
            break
        if total is None:
            total = data["data"].get("total", 0)
        diff = data["data"].get("diff") or []
        if not diff:
            break
        for item in diff:
            code = str(item.get("f12", "")).zfill(6)
            name = item.get("f14", "")
            pct = item.get("f3")
            if pct is None or pct == "-":
                continue
            row = {"代码": code, "名称": name, "涨跌幅": float(pct)}
            # f6=成交额(元), f62=主力净流入-净额(元)
            amt = item.get("f6")
            if amt is not None and amt != "-":
                row["成交额"] = float(amt)
            mf = item.get("f62")
            if mf is not None and mf != "-":
                row["主力净流入"] = float(mf)
            all_rows.append(row)
        if len(all_rows) >= (total or 0):
            break
        page += 1
        time.sleep(0.1)
    if not all_rows:
        raise RuntimeError("eastmoney direct returned no rows")
    print(f"Using market breadth source: eastmoney_direct ({len(all_rows)} rows)")
    return pd.DataFrame(all_rows)


def fetch_spot_data() -> pd.DataFrame:
    errors: list[str] = []
    for source_name, loader in (
        ("stock_zh_a_spot_em", ak.stock_zh_a_spot_em),
        ("stock_zh_a_spot", ak.stock_zh_a_spot),
        ("eastmoney_direct", _fetch_eastmoney_direct),
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

    # 总成交额（元）— akshare 源有"成交额"列，eastmoney_direct 源也有
    total_amount = None
    if "成交额" in spot.columns:
        amt = pd.to_numeric(spot["成交额"], errors="coerce").dropna()
        if not amt.empty:
            total_amount = round(float(amt.sum()), 2)

    # 主力净流入（元）— 仅 eastmoney_direct 源有"主力净流入"列
    net_inflow = None
    if "主力净流入" in spot.columns:
        mf = pd.to_numeric(spot["主力净流入"], errors="coerce").dropna()
        if not mf.empty:
            net_inflow = round(float(mf.sum()), 2)

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
        "total_amount": total_amount,
        "net_inflow": net_inflow,
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