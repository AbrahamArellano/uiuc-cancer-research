#!/bin/bash
#SBATCH --job-name=tabnet_hyperparameter_opt
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=36:00:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/scripts/optimization/hyperopt_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/scripts/optimization/hyperopt_%j.err

set -euo pipefail

echo "🧬 H100 TABNET HYPERPARAMETER OPTIMIZATION"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Started: $(date)"

cd /u/aa107/uiuc-cancer-research
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate tabnet-prostate

echo "🚀 Starting 36-hour optimization with heat map generation..."
python scripts/optimization/hyperparameter_optimization.py

echo "✅ Optimization completed: $(date)"