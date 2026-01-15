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

# Activate your virtual environment

# Run the Python script
python /local/scratch/tkim462/run_reranking.py

echo "Job completed!"
