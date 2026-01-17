#!/bin/bash
#SBATCH --job-name=rerank_quick
#SBATCH --output=/local/scratch/tkim462/rerank/slurm-%j.out
#SBATCH --error=/local/scratch/tkim462/rerank/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=h200

# CRITICAL: Set cache directories BEFORE any other commands
# Flashinfer reads these at import time, so they must be set first
export XDG_CACHE_HOME=/local/scratch/tkim462/.cache
export XDG_CONFIG_HOME=/local/scratch/tkim462/.config
export HF_HOME=/local/scratch/tkim462/.cache/huggingface
export TRANSFORMERS_CACHE=/local/scratch/tkim462/.cache/huggingface
export HF_DATASETS_CACHE=/local/scratch/tkim462/.cache/huggingface/datasets
export FLASHINFER_WORKSPACE_DIR=/local/scratch/tkim462/.cache/flashinfer
export VLLM_CACHE_ROOT=/local/scratch/tkim462/.cache/vllm

# Quick Start Script for LLM Ranker Evaluation
# This script sets up and runs a quick test evaluation

set -e  # Exit on error
echo "available devices"
echo $CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "LLM Ranker Evaluation - Quick Start"
echo "=========================================="
echo ""

cd /local/scratch/tkim462/rerank

# Create cache directories
mkdir -p $XDG_CACHE_HOME
mkdir -p $HF_HOME
mkdir -p $FLASHINFER_WORKSPACE_DIR
mkdir -p $VLLM_CACHE_ROOT
echo "Cache directories created in /local/scratch"
echo ""

# 1. Activate virtual environment (if exists)
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
    
    # Verify cache variables are set
    echo "Verifying cache environment variables:"
    echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
    echo "  FLASHINFER_WORKSPACE_DIR=$FLASHINFER_WORKSPACE_DIR"
    echo ""
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
    
    echo "✓ Installing dependencies..."
    pip install -r requirements_new.txt
fi

echo ""

# 2. Check API keys for closed-source models
# if [ -z "$OPENAI_API_KEY" ]; then
#     echo "⚠️  OPENAI_API_KEY not set"
#     echo "   Set it with: export OPENAI_API_KEY='your-key'"
#     echo ""
# fi

# if [ -z "$ANTHROPIC_API_KEY" ]; then
#     echo "⚠️  ANTHROPIC_API_KEY not set"
#     echo "   Set it with: export ANTHROPIC_API_KEY='your-key'"
#     echo ""
# fi

# 3. Run quick test
echo "=========================================="
echo "Running Quick Test (10 queries)"
echo "=========================================="
echo ""

# Check which models are enabled
echo "Checking enabled models in configs/models.yaml..."
echo ""

# Run the pipeline
echo "Starting evaluation pipeline..."
echo ""

python3 main.py "$@"

echo ""
echo "=========================================="
echo "Quick start complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check results in: results/"
echo "  2. View comparison: cat results/comparison.txt"
echo "  3. Configure more models in: configs/models.yaml"
echo "  4. Modify evaluation settings in: configs/config.yaml"
echo ""
