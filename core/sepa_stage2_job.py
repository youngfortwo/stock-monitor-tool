#!/usr/bin/env python3
"""SEPA Stage2 独立定时扫描任务（部署在局域网另一台扫描机上）。

完整流程：
1. 交易日判断（周一~周五；节假日 best-effort 用新浪交易日历，缓存 90 天）
2. 分批调用 sepa_stage2_scanner.py（默认 200 只/批，单批 900s 超时，与 _scan_worker 约定一致）
3. 合并去重结果写入本地 SQLite（sepa_stage2.db）—— 断网兜底 + 历史归档
4. HTTP POST 上报主机 stock_server（/api/sepa/stage2/upload），失败重试 3 次
   主机收到后写入自己的 SQLite 并原子覆写 sepa_stage2_candidates_test.csv，
   dashboard 页面 / 下载 Excel 链路零改动。

用法（cron 或 launchd 每交易日 18:00 触发，见 com.stock.sepa-stage2.plist）：
    python3 sepa_stage2_job.py --server http://192.168.1.100:8001 --token XXX

参数速查：
    --force            非交易日强制运行
    --total/--batch    扫描总数 / 每批数量（默认 5000 / 200）
    --db               本地 SQLite 路径（默认 sepa_stage2.db）
    --no-upload        只落本地 SQLite，不上报（调试用）
    --reupload DATE    跳过扫描，从本地 SQLite 补传指定日期数据（网络恢复后用）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "db"
if str(DB_DIR) not in sys.path:
    sys.path.insert(0, str(DB_DIR))

from sepa_db import save_candidates, load_candidates

BATCH_TIMEOUT = 900          # 单批超时（秒），与 _scan_worker.py 约定一致
SCANNER_SCRIPT = Path(__file__).with_name("sepa_stage2_scanner.py")
UPLOAD_RETRIES = 3           # 上报失败重试次数
UPLOAD_RETRY_WAIT = 10       # 重试间隔（秒）
CALENDAR_CACHE = PROJECT_ROOT / "trade_calendar_cache.json"
CALENDAR_TTL_DAYS = 90        # 交易日历缓存有效期


def is_trading_day(day: dt.date) -> bool:
    """周一~周五 + 新浪交易日历（best-effort，日历失败时仅按周末判断）。"""
    if day.weekday() >= 5:
        return False
    key = day.strftime("%Y-%m-%d")
    try:
        cache = json.loads(CALENDAR_CACHE.read_text(encoding="utf-8"))
        cached_at = dt.date.fromisoformat(cache["fetched_at"])
        if (dt.date.today() - cached_at).days <= CALENDAR_TTL_DAYS:
            return key in cache["trade_dates"]
    except Exception:
        pass
    # 缓存缺失/过期：拉取新浪交易日历
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        dates = {str(d) for d in cal["trade_date"]}
        CALENDAR_CACHE.write_text(
            json.dumps({"fetched_at": str(dt.date.today()), "trade_dates": sorted(dates)}),
            encoding="utf-8",
        )
        return key in dates
    except Exception as exc:
        print(f"WARN 交易日历获取失败（按周末规则执行）: {exc}")
        return True


def run_batches(total: int, batch: int) -> pd.DataFrame:
    """分批调用 scanner，增量合并 batch_results/sepa_*.csv。"""
    batch_dir = Path("batch_results")
    batch_dir.mkdir(exist_ok=True)
    for f in batch_dir.glob("job_sepa_*.csv"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass

    total_batches = (total + batch - 1) // batch
    merged = pd.DataFrame()
    for batch_no in range(total_batches):
        offset = batch_no * batch
        limit = min(batch, total - offset)
        print(f"[job] 第 {batch_no + 1}/{total_batches} 批（{offset}-{offset + limit}）扫描中…", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(SCANNER_SCRIPT),
                 "--offset", str(offset), "--limit", str(limit),
                 "--output", f"batch_results/job_sepa_{offset}.csv",
                 "--sleep-seconds", "0.15"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=BATCH_TIMEOUT,
            )
            if proc.returncode != 0:
                print(f"WARN 第 {batch_no + 1} 批失败: {proc.stderr[:200]}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"WARN 第 {batch_no + 1} 批超时（>{BATCH_TIMEOUT}s），跳过", file=sys.stderr)
            continue

        part = f"batch_results/job_sepa_{offset}.csv"
        try:
            frame = pd.read_csv(part, dtype={"code": str})
        except Exception:
            continue
        if frame.empty:
            continue
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        merged = frame if merged.empty else pd.concat([merged, frame], ignore_index=True)

    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=["code"], keep="first")
    sort_cols = [c for c in ("score", "amount_cny") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    # 清理 np.True_/np.False_ 等 repr 残留（与 _scan_worker.merge_and_write 保持一致）
    for col in ["conditions", "cup_handle_details", "vcp_details", "pullback_details"]:
        if col in merged.columns:
            merged[col] = (merged[col].astype(str)
                           .str.replace(r"np\.True_", "true", regex=True)
                           .str.replace(r"np\.False_", "false", regex=True)
                           .str.replace(r"np\.float64\(([\d.]+)\)", r"\1", regex=True)
                           .str.replace(r"np\.int64\((\d+)\)", r"\1", regex=True)
                           .str.replace("'", '"', regex=False)
                           .str.replace(r"\bTrue\b", "true", regex=True)
                           .str.replace(r"\bFalse\b", "false", regex=True))
    return merged


def read_rps_csv() -> str:
    """读取 scanner 输出的 rps_all.csv（分批增量合并后的全量 RPS），随 payload 一并上报。"""
    path = Path("rps_all.csv")
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def upload(server: str, token: str, payload: dict, retries: int = UPLOAD_RETRIES) -> bool:
    """POST 上报主机，带重试。"""
    import requests

    url = server.rstrip("/") + "/api/sepa/stage2/upload"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Upload-Token"] = token
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200 and resp.json().get("ok"):
                print(f"[job] 上报成功: {resp.json()}")
                return True
            print(f"WARN 上报被拒（HTTP {resp.status_code}）: {resp.text[:200]}", file=sys.stderr)
        except Exception as exc:
            print(f"WARN 上报失败（第 {attempt}/{retries} 次）: {exc}", file=sys.stderr)
        if attempt < retries:
            time.sleep(UPLOAD_RETRY_WAIT)
    return False


def _json_safe(v):
    """NaN/±inf → None（json.dumps 不接受非有限浮点数）。"""
    if v is None:
        return None
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def build_payload(df: pd.DataFrame, scan_date: str, generated_at: str) -> dict:
    """DataFrame → 紧凑 JSON payload（列数组 + 行数组的数组）。"""
    data = df.astype(object).where(pd.notna(df), None)
    rows = [[_json_safe(v) for v in row] for row in data.values.tolist()]
    return {
        "scan_date": scan_date,
        "generated_at": generated_at,
        "count": len(rows),
        "columns": list(data.columns),
        "rows": rows,
        "rps_csv": read_rps_csv(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEPA Stage2 standalone daily job.")
    parser.add_argument("--server", default="", help="主机 stock_server 地址，如 http://192.168.1.100:8001")
    parser.add_argument("--token", default="", help="主机设置的 SEPA_UPLOAD_TOKEN（可选）")
    parser.add_argument("--total", type=int, default=5000, help="扫描股票总数")
    parser.add_argument("--batch", type=int, default=200, help="每批数量")
    parser.add_argument("--db", default="sepa_stage2.db", help="本地 SQLite 路径")
    parser.add_argument("--force", action="store_true", help="非交易日强制运行")
    parser.add_argument("--no-upload", action="store_true", help="只写本地 SQLite，不上报")
    parser.add_argument("--reupload", default="", metavar="YYYY-MM-DD", help="跳过扫描，补传指定日期本地数据")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 补传模式：从本地 SQLite 读指定日期数据重新上报
    if args.reupload:
        df = load_candidates(args.db, scan_date=args.reupload)
        if df.empty:
            print(f"[job] 本地无 {args.reupload} 的数据，退出")
            return 1
        df = df.drop(columns=["scan_date"], errors="ignore")
        payload = build_payload(df, args.reupload, time.strftime("%Y-%m-%d %H:%M:%S"))
        payload["rps_csv"] = ""  # 补传不带 RPS，避免旧数据覆盖
        ok = upload(args.server, args.token, payload)
        return 0 if ok else 1

    today = dt.date.today()
    if not args.force and not is_trading_day(today):
        print(f"[job] {today} 非交易日，跳过（--force 可强制运行）")
        return 0

    scan_date = today.isoformat()
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[job] 开始 SEPA Stage2 扫描：{scan_date}，共 {args.total} 只，{args.batch} 只/批", flush=True)

    df = run_batches(args.total, args.batch)
    print(f"[job] 扫描完成：{len(df)} 只候选股", flush=True)

    # 空结果也照常落库/上报（表示"今日无候选"），但保留上次 CSV 的行为由主机端决定
    if df.empty:
        print("[job] 今日无候选股")

    # 1) 本地 SQLite 兜底存储
    try:
        saved = save_candidates(df, args.db, scan_date)
        print(f"[job] 本地 SQLite 写入 {saved} 行 → {args.db}")
    except Exception:
        traceback.print_exc()

    # 2) 上报主机
    if args.no_upload or not args.server:
        print("[job] 未指定 --server 或 --no-upload，跳过上报")
        return 0
    df_out = df.copy()
    df_out["scanned_at"] = generated_at
    payload = build_payload(df_out, scan_date, generated_at)
    ok = upload(args.server, args.token, payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
