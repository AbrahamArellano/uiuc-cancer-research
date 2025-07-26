#!/bin/bash
#SBATCH --job-name=tabnet_roc_auc_gpu
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/src/model/evaluation_auc_roc/roc_auc_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/src/model/evaluation_auc_roc/roc_auc_%j.err

# ================================================================
# TabNet ROC-AUC Calculation on GPU Node
# Loads GPU-saved model and calculates multi-class ROC curves
# ================================================================

set -euo pipefail

echo "🎯 TABNET ROC-AUC CALCULATION ON GPU NODE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
EVAL_DIR="${PROJECT_DIR}/src/model/evaluation_auc_roc"
MODEL_PATH="/u/aa107/scratch/tabnet_model_20250719_055806.pkl"
OUTPUT_DIR="${PROJECT_DIR}/results/roc_analysis"
FIGURES_DIR="${PROJECT_DIR}/results/figures"
CONDA_ENV="tabnet-prostate"

echo "📁 Configuration:"
echo "  Project: ${PROJECT_DIR}"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# === ENVIRONMENT SETUP ===
echo "🔧 ENVIRONMENT SETUP"
echo "===================="

# Navigate to project directory
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

# Load anaconda module (required on UIUC Campus Cluster)
echo "Loading anaconda module..."
if module load anaconda3 2>/dev/null; then
    echo "✅ Anaconda module loaded"
else
    echo "⚠️  Anaconda module not available - using system conda"
fi

# Initialize conda for shell script (critical for compute nodes)
echo "Initializing conda for bash..."
eval "$(conda shell.bash hook)"

# Check if environment exists
if conda env list | grep -q "${CONDA_ENV}"; then
    echo "✅ Found ${CONDA_ENV} environment"
else
    echo "❌ ${CONDA_ENV} environment not found"
    exit 1
fi

# Activate the conda environment
echo "Activating ${CONDA_ENV} environment..."
conda activate "${CONDA_ENV}"

# Verify environment activation
echo "✅ Environment activated:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"

# === GPU VERIFICATION ===
echo ""
echo "🚀 GPU VERIFICATION"
echo "==================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    
    # Check PyTorch CUDA
    python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️  No CUDA available - will still work since model will be on CPU')
"
else
    echo "⚠️  GPU info command not available"
fi

# === CREATE OUTPUT DIRECTORIES ===
echo ""
echo "📁 Creating output directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FIGURES_DIR"
mkdir -p "${FIGURES_DIR}/supplementary"

# === RUN ROC-AUC CALCULATION ===
echo ""
echo "📊 CALCULATING ROC-AUC METRICS"
echo "=============================="

# Execute the calculation script
python "${EVAL_DIR}/calculate_roc_auc.py"

CALC_EXIT_CODE=$?

if [ $CALC_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ ROC-AUC calculation completed successfully!"
    
    # Verify outputs
    if [ -f "${OUTPUT_DIR}/roc_auc_metrics.json" ]; then
        echo "📋 Results summary:"
        python -c "
import json
with open('${OUTPUT_DIR}/roc_auc_metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f'  Overall AUC: {metrics[\"summary\"][\"overall_auc\"]:.3f}')
    print(f'  Pathogenic AUC: {metrics[\"summary\"][\"pathogenic_auc\"]:.3f}')
    print(f'  Benign AUC: {metrics[\"summary\"][\"benign_auc\"]:.3f}')
    print(f'  VUS AUC: {metrics[\"summary\"][\"vus_auc\"]:.3f}')
"
    fi
    
    # === RUN VISUALIZATION ===
    echo ""
    echo "🎨 GENERATING PUBLICATION FIGURES"
    echo "================================"
    
    python "${EVAL_DIR}/compare_baseline_tools.py"
    
    VIS_EXIT_CODE=$?
    
    if [ $VIS_EXIT_CODE -eq 0 ]; then
        echo "✅ Visualization completed successfully!"
        
        echo ""
        echo "📁 Output files:"
        ls -la "${OUTPUT_DIR}/"*.json
        ls -la "${FIGURES_DIR}/"*.png 2>/dev/null || echo "No main figures"
        ls -la "${FIGURES_DIR}/supplementary/"*.png 2>/dev/null || echo "No supplementary figures"
    else
        echo "❌ Visualization failed with exit code: $VIS_EXIT_CODE"
    fi
    
else
    echo "❌ ROC-AUC calculation failed with exit code: $CALC_EXIT_CODE"
fi

# === FINAL SUMMARY ===
echo ""
echo "🎯 JOB SUMMARY"
echo "=============="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Node: ${SLURMD_NODENAME}"
echo "Job ID: ${SLURM_JOB_ID}"

if [ $CALC_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: ROC-AUC analysis completed on GPU node"
    echo "📊 Results: ${OUTPUT_DIR}/roc_auc_metrics.json"
    echo "📈 Figure: ${FIGURES_DIR}/figure_6_roc_curves.png"
else
    echo ""
    echo "❌ FAILED: Check error log at:"
    echo "   /u/aa107/uiuc-cancer-research/src/model/evaluation_auc_roc/roc_auc_${SLURM_JOB_ID}.err"
fi

exit $CALC_EXIT_CODE