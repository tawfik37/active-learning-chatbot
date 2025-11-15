#!/bin/bash

echo "=================================================="
echo "🚀 QWEN2 FINE-TUNING & RAG PIPELINE"
echo "   Starting Pipeline..."
echo "=================================================="

# Exit on any error
set -e

# Run the pipeline
python3 pipeline.py

echo ""
echo "=================================================="
echo "✅ Pipeline execution complete!"
echo "=================================================="
