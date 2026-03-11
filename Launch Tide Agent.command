#!/bin/zsh

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8502
URL="http://localhost:${PORT}"
LOG_FILE="/tmp/tide_agent_streamlit.log"

cd "$APP_DIR"

if [[ ! -x "./tide_env/bin/streamlit" ]]; then
  osascript -e 'display alert "Tide Agent" message "Python environment not found at ./tide_env/bin/streamlit" as critical'
  exit 1
fi

# If app already runs, just open browser.
if lsof -iTCP:${PORT} -sTCP:LISTEN >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

nohup ./tide_env/bin/streamlit run app.py --server.headless true --server.port ${PORT} >"$LOG_FILE" 2>&1 &

# Give server time to start.
sleep 2
open "$URL"

exit 0
