#!/bin/sh

# Start bgutil PO-token server
node /opt/bgutil/build/main.js &

# Start AutoDub-Pro
exec python main.py --web --port "$PORT"