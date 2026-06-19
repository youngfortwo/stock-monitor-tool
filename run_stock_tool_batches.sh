#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="/usr/bin/python3"
TOTAL="${1:-500}"
BATCH_SIZE="${2:-100}"
PORT="${PORT:-8000}"
TMP_DIR="batch_results"
PROGRESS_FILE="scan_progress.json"
TOTAL_BATCHES=$(((TOTAL + BATCH_SIZE - 1) / BATCH_SIZE))

merge_batch_results() {
    echo "Merging current batch results..."
    "${PYTHON}" - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def merge_trend(pattern: str, output: str, sort_columns: list[str]) -> int:
    frames = []
    for path in sorted(Path("batch_results").glob(pattern)):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except EmptyDataError:
            continue
        if not frame.empty:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            frames.append(frame)
    if not frames:
        Path(output).write_text("", encoding="utf-8-sig")
        return 0

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["code"], keep="first")
    existing_sort_columns = [column for column in sort_columns if column in result.columns]
    if existing_sort_columns:
        result = result.sort_values(existing_sort_columns, ascending=[False] * len(existing_sort_columns))
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return len(result)


def merge_sepa_with_rps(pattern: str, output: str, sort_columns: list[str]) -> int:
    frames = []
    for path in sorted(Path("batch_results").glob(pattern)):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except EmptyDataError:
            continue
        if not frame.empty:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            frames.append(frame)
    if not frames:
        Path(output).write_text("", encoding="utf-8-sig")
        return 0

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["code"], keep="first")

    if "return_120d_pct" in result.columns:
        valid = result["return_120d_pct"].notna()
        if valid.sum() >= 5:
            rps_values = result.loc[valid, "return_120d_pct"]
            result.loc[valid, "rps"] = rps_values.rank(pct=True) * 100
            result["rps"] = result["rps"].round(2)
            result = result[result["rps"] >= 70]
        else:
            result["rps"] = 0
    else:
        result["rps"] = 0

    existing_sort_columns = [column for column in sort_columns if column in result.columns]
    if existing_sort_columns:
        result = result.sort_values(existing_sort_columns, ascending=[False] * len(existing_sort_columns))
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return len(result)


trend_count = merge_trend("trend_*.csv", "test_candidates.csv", ["score", "amount_cny"])
sepa_count = merge_sepa_with_rps("sepa_*.csv", "sepa_stage2_candidates_test.csv", ["score", "amount_cny"])
stage1_count = merge_trend("stage1_*.csv", "sepa_stage1_candidates_test.csv", ["score", "amount_cny"])
print(f"Merged: {trend_count} trend candidates, {sepa_count} SEPA Stage 2 candidates, {stage1_count} SEPA Stage 1 candidates")
PY
}

write_progress() {
    local status="$1"
    local stage="$2"
    local current_batch="$3"
    local completed_batches="$4"
    local current_offset="$5"
    local current_limit="$6"
    local message="$7"

    "${PYTHON}" - "$PROGRESS_FILE" "$status" "$stage" "$TOTAL" "$BATCH_SIZE" "$TOTAL_BATCHES" \
        "$current_batch" "$completed_batches" "$current_offset" "$current_limit" "$message" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

(
    output,
    status,
    stage,
    total,
    batch_size,
    total_batches,
    current_batch,
    completed_batches,
    current_offset,
    current_limit,
    message,
) = sys.argv[1:]

total = int(total)
batch_size = int(batch_size)
total_batches = int(total_batches)
current_batch = int(current_batch)
completed_batches = int(completed_batches)
current_offset = int(current_offset)
current_limit = int(current_limit)

processed_stocks = min(total, completed_batches * batch_size)
percent = round((processed_stocks / total * 100) if total else 0, 2)

payload = {
    "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "status": status,
    "stage": stage,
    "message": message,
    "total": total,
    "batch_size": batch_size,
    "total_batches": total_batches,
    "current_batch": current_batch,
    "completed_batches": completed_batches,
    "current_offset": current_offset,
    "current_limit": current_limit,
    "processed_stocks": processed_stocks,
    "percent": percent,
}

Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

write_progress "running" "init" 0 0 0 0 "初始化分批扫描"

rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

echo "Installing dependencies..."
write_progress "running" "install" 0 0 0 0 "安装或确认依赖"
"${PYTHON}" -m pip install -r requirements-stock-scanner.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Generating market breadth and panic index..."
write_progress "running" "market_breadth" 0 0 0 0 "更新市场情绪和恐慌指数"
if "${PYTHON}" market_breadth.py; then
    write_progress "running" "market_breadth_done" 0 0 0 0 "市场情绪更新完成"
else
    echo "WARN market breadth update failed; keeping previous market_breadth.json and continuing."
    write_progress "running" "market_breadth_warn" 0 0 0 0 "市场情绪接口失败，保留旧数据并继续扫描"
fi

for ((OFFSET=0; OFFSET<TOTAL; OFFSET+=BATCH_SIZE)); do
    CURRENT_LIMIT="${BATCH_SIZE}"
    REMAINING=$((TOTAL - OFFSET))
    if (( REMAINING < BATCH_SIZE )); then
        CURRENT_LIMIT="${REMAINING}"
    fi
    BATCH_NO=$((OFFSET / BATCH_SIZE + 1))

    echo
    echo "Batch ${BATCH_NO}: offset=${OFFSET}, limit=${CURRENT_LIMIT}"

    write_progress "running" "trend_scan" "${BATCH_NO}" $((BATCH_NO - 1)) "${OFFSET}" "${CURRENT_LIMIT}" \
        "第 ${BATCH_NO}/${TOTAL_BATCHES} 批：扫描趋势候选股"
    "${PYTHON}" stock_scanner.py \
        --offset "${OFFSET}" \
        --limit "${CURRENT_LIMIT}" \
        --output "${TMP_DIR}/trend_${OFFSET}.csv"

    write_progress "running" "sepa_scan" "${BATCH_NO}" $((BATCH_NO - 1)) "${OFFSET}" "${CURRENT_LIMIT}" \
        "第 ${BATCH_NO}/${TOTAL_BATCHES} 批：扫描 SEPA 第二阶段"
    "${PYTHON}" sepa_stage2_scanner.py \
        --offset "${OFFSET}" \
        --limit "${CURRENT_LIMIT}" \
        --output "${TMP_DIR}/sepa_${OFFSET}.csv"

    write_progress "running" "stage1_scan" "${BATCH_NO}" $((BATCH_NO - 1)) "${OFFSET}" "${CURRENT_LIMIT}" \
        "第 ${BATCH_NO}/${TOTAL_BATCHES} 批：扫描 SEPA 第一阶段"
    "${PYTHON}" sepa_stage1_scanner.py \
        --offset "${OFFSET}" \
        --limit "${CURRENT_LIMIT}" \
        --output "${TMP_DIR}/stage1_${OFFSET}.csv"

    # 每批完成后立即合并结果，更新 Dashboard
    write_progress "running" "batch_merge" "${BATCH_NO}" "${BATCH_NO}" "${OFFSET}" "${CURRENT_LIMIT}" \
        "第 ${BATCH_NO}/${TOTAL_BATCHES} 批完成，正在合并结果"
    merge_batch_results

    # 每 5 批刷新一次市场情绪数据
    REFRESH_INTERVAL=5
    if (( BATCH_NO % REFRESH_INTERVAL == 0 )); then
        echo "Refreshing market breadth data..."
        write_progress "running" "market_breadth" "${BATCH_NO}" "${BATCH_NO}" "${OFFSET}" "${CURRENT_LIMIT}" \
            "第 ${BATCH_NO}/${TOTAL_BATCHES} 批：刷新市场情绪"
        "${PYTHON}" market_breadth.py 2>/dev/null || true
    fi

    write_progress "running" "batch_done" "${BATCH_NO}" "${BATCH_NO}" "${OFFSET}" "${CURRENT_LIMIT}" \
        "第 ${BATCH_NO}/${TOTAL_BATCHES} 批完成"
done

echo
echo "All batches complete. Final merge..."
write_progress "running" "merge" "${TOTAL_BATCHES}" "${TOTAL_BATCHES}" "${TOTAL}" 0 "合并所有批次结果"
merge_batch_results

echo "Generating K-line chart for the top merged trend candidate..."
write_progress "running" "chart" "${TOTAL_BATCHES}" "${TOTAL_BATCHES}" "${TOTAL}" 0 "生成最高分候选股 K 线图"

TOP_CODE="$("${PYTHON}" - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("test_candidates.csv")
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit("")

df = pd.read_csv(path, dtype={"code": str})
if df.empty:
    raise SystemExit("")

print(str(df.iloc[0]["code"]).zfill(6))
PY
)"

if [[ -n "${TOP_CODE}" ]]; then
    "${PYTHON}" plot_kline.py --symbol "${TOP_CODE}" --days 120 --output "kline_${TOP_CODE}.png"
else
    echo "No trend candidate found; skipped K-line chart."
fi

echo
echo "Done."
echo "Scanned total=${TOTAL}, batch_size=${BATCH_SIZE}"
write_progress "done" "done" "${TOTAL_BATCHES}" "${TOTAL_BATCHES}" "${TOTAL}" 0 "分批扫描完成"
echo "Dashboard:"
echo "  http://localhost:${PORT}/stock-monitor-tool/stock_dashboard.html"
