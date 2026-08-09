#!/bin/sh

echo "=== Starting bgutil PO-token server ==="

node /opt/bgutil/build/main.js &
BGUTIL_PID=$!

sleep 2

echo "=== Checking bgutil server ==="
curl -s http://127.0.0.1:4416/ping || true

echo ""
echo "=== Checking yt-dlp bgutil plugin ==="
yt-dlp --verbose --simulate "https://youtu.be/5hPtU8Jbpg0" 2>&1 | grep -E "PO Token Providers|bgutil" || true

echo "=== Starting AutoDub-Pro ==="

exec python main.py --web --port "$PORT"