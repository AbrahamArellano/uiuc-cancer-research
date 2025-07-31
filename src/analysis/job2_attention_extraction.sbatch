#!/bin/bash
#SBATCH --job-name=attention_extraction
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/src/analysis/attention_extraction_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/src/analysis/attention_extraction_%j.err

# ================================================================
# TABNET ATTENTION EXTRACTION - EXCLUSIVE GPU JOB
# ================================================================

set -euo pipefail

echo "🧠 TABNET ATTENTION EXTRACTION - EXCLUSIVE GPU ACCESS"
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "GPU Mode: EXCLUSIVE ACCESS"
echo ""

# Configuration
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
SCRIPT="${PROJECT_DIR}/src/analysis/attention_extractor.py"
CONDA_ENV="tabnet-prostate"

# GPU verification
if command -v nvidia-smi &> /dev/null; then
    echo "🎯 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Environment setup
echo "🔧 ENVIRONMENT SETUP"
echo "-------------------"
cd "$PROJECT_DIR"

# Load anaconda and initialize conda
module load anaconda3
eval "$(conda shell.bash hook)"

# Activate environment
conda activate "$CONDA_ENV"
echo "✅ Environment activated: $CONDA_DEFAULT_ENV"
echo "✅ Python: $(which python)"
echo ""

# Prerequisite checks
echo "📋 PREREQUISITE CHECKS"
echo "--------------------"

# Check selected variants from Job 1
VARIANTS_FILE="${PROJECT_DIR}/results/attention_analysis/selected_variants.csv"
if [ ! -f "$VARIANTS_FILE" ]; then
    echo "❌ Selected variants not found: $VARIANTS_FILE"
    echo "💡 Job 1 (variant selection) must complete first"
    exit 1
else
    echo "✅ Selected variants found: $(wc -l < "$VARIANTS_FILE") lines"
fi

# Check TabNet model
MODEL_PATH="/u/aa107/uiuc-cancer-research/results/training/tabnet_training_20250728_075348/tabnet_model_20250728_072312.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ TabNet model not found: $MODEL_PATH"
    exit 1
else
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "✅ TabNet model found: $MODEL_SIZE"
fi

# Verify script exists
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Attention extractor script not found: $SCRIPT"
    exit 1
else
    echo "✅ Script found: attention_extractor.py"
fi

echo ""

python -c "
import torch
import pytorch_tabnet
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

# Execute attention extraction
echo "🚀 STARTING ATTENTION EXTRACTION"
echo "================================"
echo "Purpose: Extract TabNet attention weights with exclusive GPU access"
echo "Script: attention_extractor.py"
echo ""

# Run the script
python "$SCRIPT"
EXTRACTION_EXIT_CODE=$?

echo ""
if [ $EXTRACTION_EXIT_CODE -eq 0 ]; then
    echo "✅ ATTENTION EXTRACTION COMPLETED SUCCESSFULLY"
    echo "📁 Results saved to: ${PROJECT_DIR}/results/attention_analysis/attention_weights"
    echo "🎯 Ready for Job 3: Analysis Pipeline"
else
    echo "❌ ATTENTION EXTRACTION FAILED (Exit code: $EXTRACTION_EXIT_CODE)"
    exit $EXTRACTION_EXIT_CODE
fi

echo ""
echo "Job completed: $(date)"