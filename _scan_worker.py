#!/usr/bin/env python3
"""Background scan worker, invoked by the dashboard server."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
import pandas as pd

HISTORY_MAX = 10


def save_to_history(output: str, scan_type: str) -> None:
    """保存扫描结果到历史记录，保留最近10次"""
    src = Path(output)
    if not src.exists() or src.stat().st_size == 0:
        return
    history_dir = Path("scan_history")
    history_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = history_dir / f"{scan_type}_{ts}.csv"
    shutil.copy2(src, dst)
    # 清理旧记录，只保留最近10次
    prefix = f"{scan_type}_"
    files = sorted(history_dir.glob(f"{prefix}*.csv"), reverse=True)
    for old in files[HISTORY_MAX:]:
        try:
            old.unlink()
        except FileNotFoundError:
            pass
    print(f"History saved: {dst} ({len(files[:HISTORY_MAX])} kept)")


def merge_and_write(pattern: str, output: str, sort_cols: list[str]) -> int:
    frames = []
    for p in sorted(Path("batch_results").glob(pattern)):
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            f = pd.read_csv(p, dtype={"code": str})
        except Exception:
            continue
        if not f.empty:
            f["code"] = f["code"].astype(str).str.zfill(6)
            frames.append(f)
    if not frames:
        # 不覆写空文件，保留上次结果
        return 0
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["code"], keep="first")
    result["scanned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    existing = [c for c in sort_cols if c in result.columns]
    if existing:
        result = result.sort_values(existing, ascending=[False] * len(existing))
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return len(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["stage1", "stage2", "value_bottom"])
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args()

    if args.type == "stage1":
        script = "sepa_stage1_scanner.py"
        prefix = "stage1_"
        output = "sepa_stage1_candidates_test.csv"
        progress_file = "scan_progress_s1.json"
    elif args.type == "value_bottom":
        script = "value_bottom_scanner.py"
        prefix = "vb_"
        output = "value_bottom_candidates_test.csv"
        progress_file = "scan_progress_vb.json"
    else:
        script = "sepa_stage2_scanner.py"
        prefix = "sepa_"
        output = "sepa_stage2_candidates_test.csv"
        progress_file = "scan_progress_s2.json"
    sort_cols = ["score", "amount_cny"]

    import subprocess
    total_batches = (args.total + args.batch - 1) // args.batch

    # 扫描开始前：备份上次结果 + 清理旧分批文件
    save_to_history(output, args.type)
    import glob as _glob
    for f in _glob.glob(f"batch_results/{prefix}*.csv"):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass

    def write_progress(batch_no, status="running", message=""):
        pct = round(batch_no / total_batches * 100, 1) if total_batches else 0
        scanned = min(batch_no * args.batch, args.total)
        payload = {
            "status": status,
            "current_batch": batch_no,
            "total_batches": total_batches,
            "total": args.total,
            "processed_stocks": scanned,
            "percent": pct,
            "message": message or f"第 {batch_no}/{total_batches} 批",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        Path(progress_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    write_progress(0, message=f"正在拉取第 1/{total_batches} 批（{0}-{min(args.batch, args.total)}），请耐心等待（约3分钟/批）...")

    for batch_no in range(total_batches):
        offset = batch_no * args.batch
        limit = min(args.batch, args.total - offset)
        write_progress(batch_no, message=f"第 {batch_no + 1}/{total_batches} 批（{offset}-{offset + limit}），处理中…")

        cmd = [
            sys.executable, script,
            "--offset", str(offset),
            "--limit", str(limit),
            "--output", f"batch_results/{prefix}{offset}.csv",
            "--sleep-seconds", "0.15",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(f"Batch {batch_no + 1} failed: {proc.stderr[:200]}", file=sys.stderr)

        # Merge incrementally
        count = merge_and_write(f"{prefix}*.csv", output, sort_cols)
        print(f"Batch {batch_no + 1}/{total_batches}: merged {count} candidates")

    write_progress(total_batches, "done")
    save_to_history(output, args.type)
    print("Scan complete.")


if __name__ == "__main__":
    main()
