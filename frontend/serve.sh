#!/bin/bash

echo "=================================================="
echo "  Frontend Preview Server"
echo "=================================================="
echo ""
echo "  Open in browser: http://localhost:8000"
echo "  Press Ctrl+C to stop"
echo ""
echo "  Note: Chat requires the Modal backend to be running."
echo "        This is for UI preview only."
echo "=================================================="
echo ""

cd "$(dirname "$0")"
python3 -m http.server 8000
