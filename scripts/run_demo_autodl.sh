#!/usr/bin/env bash
# Start the demo on an AutoDL instance and print the SSH tunnel command.
#
#   bash scripts/run_demo_autodl.sh
#   PORT=7862 DEVICE=cpu bash scripts/run_demo_autodl.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found; run: bash scripts/setup_demo_autodl.sh" >&2
  exit 1
fi
source .venv/bin/activate

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7861}"
DEVICE="${DEVICE:-auto}"

echo "UI will listen on http://${HOST}:${PORT} (device=${DEVICE})"
echo "From your laptop, forward the port (keep this terminal open):"
echo "  ssh -L ${PORT}:127.0.0.1:${PORT} -p <instance-ssh-port> root@<instance-host>"
echo "Then open http://127.0.0.1:${PORT} in the browser."
echo

exec python -m table_vlm.demo --host "$HOST" --port "$PORT" --device "$DEVICE"
