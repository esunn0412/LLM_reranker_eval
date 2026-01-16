#!/bin/bash
#SBATCH --job-name=rerank_msmarco
#SBATCH --output=/local/scratch/tkim462/rerank_%j.out
#SBATCH --error=/local/scratch/tkim462/rerank_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=h200

# Set cache directories
export HF_HOME='/local/scratch/tkim462/.cache/huggingface'
export XDG_CACHE_HOME='/local/scratch/tkim462/.cache'

# Activate virtual environment
source /local/scratch/tkim462/rerank/rerank_eval/bin/activate

# Verify environment
echo "Python location: $(which python3)"
echo "Python version: $(python3 --version)"
echo "Torch installed: $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"

# Run the Python script
python3 run_reranking.py

echo "Job completed!"
