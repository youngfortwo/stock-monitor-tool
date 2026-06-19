#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LIMIT="${1:-100}"
PORT="${PORT:-8000}"

echo "Installing dependencies..."
# 强制系统Python安装依赖（核心修复）
/usr/bin/python3 -m pip install -r requirements-stock-scanner.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Generating market breadth and panic index..."
/usr/bin/python3 market_breadth.py

echo "Scanning trend candidates, limit=${LIMIT}..."
/usr/bin/python3 stock_scanner.py --limit "${LIMIT}" --output test_candidates.csv

echo "Generating K-line chart for the top candidate..."
TOP_CODE="$(/usr/bin/python3 - <<'PY'
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
    /usr/bin/python3 plot_kline.py --symbol "${TOP_CODE}" --days 120 --output "kline_${TOP_CODE}.png"
else
    echo "No trend candidate found; skipped K-line chart."
fi

echo "Scanning SEPA Stage 2 candidates, limit=${LIMIT}..."
/usr/bin/python3 sepa_stage2_scanner.py --limit "${LIMIT}" --output sepa_stage2_candidates_test.csv

echo
echo "Done."
echo "Dashboard:"
echo "  http://localhost:${PORT}/stock-monitor-tool/stock_dashboard.html"
echo
echo "If the local web server is not running, start it from the workspace root:"
echo "  cd /Users/XYa116/test5613 && /usr/bin/python3 -m http.server ${PORT}"