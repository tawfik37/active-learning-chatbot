#!/bin/bash

echo "=================================================="
echo "  QWEN2 FINE-TUNING & RAG PIPELINE"
echo "   Initializing Environment..."
echo "=================================================="

# Exit on any error
set -e

echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "=================================================="
echo "Next step: Run ./start_pipeline.sh"
echo "=================================================="