#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TOTAL="${1:-1000}"
BATCH_SIZE="${2:-200}"
PORT="${PORT:-8001}"

PORT_PID=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${PORT_PID}" ]]; then
    echo "Port ${PORT} is in use by PID ${PORT_PID}, killing..."
    kill -9 ${PORT_PID} 2>/dev/null || true
    sleep 1
    echo "Port ${PORT} released."
fi

echo "Step 1/2: Starting background data refresh, total=${TOTAL}, batch_size=${BATCH_SIZE}..."
./run_stock_tool_batches.sh "${TOTAL}" "${BATCH_SIZE}" > start_stock_tool_data_refresh.log 2>&1 &
REFRESH_PID="$!"
echo "Data refresh is running in the background. PID=${REFRESH_PID}"

echo
echo "Step 2/2: Starting dashboard server..."
echo "Open: http://localhost:${PORT}/stock_dashboard.html"
echo "Refresh progress is shown on the dashboard and logged to start_stock_tool_data_refresh.log"
echo

PORT="${PORT}" /usr/bin/python3 stock_server.py
