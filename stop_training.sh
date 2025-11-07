#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Stop Gaussian Splatting Training"
echo "=========================================="

# Find ns-train processes
PIDS=$(pgrep -f "ns-train splatfacto" || true)

if [ -z "$PIDS" ]; then
    echo "❌ No training process found"
    echo ""
    echo "Checking for any nerfstudio processes..."
    ps aux | grep -E "ns-train|nerfstudio" | grep -v grep || echo "None found"
    exit 0
fi

echo "Found training process(es):"
ps -p $PIDS -o pid,etime,cmd | head -20

echo ""
read -p "Stop these processes? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping training..."
    kill $PIDS
    sleep 2
    
    # Check if still running
    STILL_RUNNING=$(pgrep -f "ns-train splatfacto" || true)
    if [ -n "$STILL_RUNNING" ]; then
        echo "Process still running, forcing kill..."
        kill -9 $STILL_RUNNING
    fi
    
    echo "✅ Training stopped"
else
    echo "Cancelled"
fi

echo "=========================================="
