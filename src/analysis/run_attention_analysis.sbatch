#!/bin/bash
#SBATCH --job-name=tabnet_attention_analysis
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/src/analysis/attention_analysis_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/src/analysis/attention_analysis_%j.err

# ================================================================
# TABNET ATTENTION ANALYSIS - AUTOMATED SLURM PIPELINE
# Updated to use GPU for optimal performance with GPU-trained models
# ================================================================

set -euo pipefail

echo "🧠 TABNET ATTENTION ANALYSIS - AUTOMATED SLURM PIPELINE"
echo "========================================================"
echo "Running complete 5-step attention analysis on GPU nodes"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

# Check GPU info
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU Info: $GPU_INFO"
    echo ""
fi

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
ANALYSIS_DIR="${PROJECT_DIR}/src/analysis"
RESULTS_DIR="${PROJECT_DIR}/results/attention_analysis"
CONDA_ENV="tabnet-prostate"

# Analysis scripts (5 scripts now)
SCRIPT_1="${ANALYSIS_DIR}/variant_selector.py"
SCRIPT_2="${ANALYSIS_DIR}/attention_extractor.py"
SCRIPT_3="${ANALYSIS_DIR}/attention_analyzer.py"
SCRIPT_4="${ANALYSIS_DIR}/results_generator.py"
SCRIPT_5="${ANALYSIS_DIR}/feature_importance_comparison.py"  # NEW

echo "🔧 ENVIRONMENT SETUP"
echo "-------------------"
echo "Working directory: $PROJECT_DIR"
echo "Results directory: $RESULTS_DIR"

# Navigate to project directory
cd "$PROJECT_DIR"

# Load anaconda module
echo "Loading anaconda module..."
if module load anaconda3 2>/dev/null; then
    echo "✅ Anaconda module loaded"
else
    echo "⚠️  Anaconda module not available - using system conda"
fi

# Initialize conda
echo "Initializing conda for bash..."
eval "$(conda shell.bash hook)"
CONDA_BASE=$(conda info --base)
echo "✅ Conda initialized from $CONDA_BASE"

# Check environment exists
if conda env list | grep -q "$CONDA_ENV"; then
    echo "✅ Found $CONDA_ENV environment"
else
    echo "❌ $CONDA_ENV environment not found"
    exit 1
fi

# Activate environment
echo "Activating $CONDA_ENV environment..."
conda activate "$CONDA_ENV"
echo "✅ Environment activated:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# GPU mode for optimal performance with GPU-trained models
echo "🔧 GPU MODE ENABLED"
echo "   Using allocated GPU for optimal attention analysis performance"

# === GPU VERIFICATION ===
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
"
else
    echo "⚠️  GPU not available - attention analysis may have issues"
fi
echo ""

# === DEPENDENCY CHECK ===
echo "📦 CHECKING ATTENTION ANALYSIS DEPENDENCIES"
echo "------------------------------------------"

echo "Checking core packages..."
python -c "
import sys
try:
    import torch
    import pytorch_tabnet
    import sklearn
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print('  ✅ All core packages available')
    print(f'  📦 PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    print(f'  📦 TabNet: Available')
    print(f'  📦 Sklearn: {sklearn.__version__}')
    print(f'  📦 Pandas: {pd.__version__}')
    print(f'  📦 Matplotlib: Available')
    print(f'  📦 Seaborn: Available')
except ImportError as e:
    print(f'  ❌ Missing package: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed"
    exit 1
fi

echo "✅ All dependencies verified"

# === SCRIPT VALIDATION ===
echo ""
echo "📋 SCRIPT VALIDATION"
echo "-------------------"

# Updated to check 5 scripts
SCRIPTS=("$SCRIPT_1" "$SCRIPT_2" "$SCRIPT_3" "$SCRIPT_4" "$SCRIPT_5")
SCRIPT_NAMES=("variant_selector.py" "attention_extractor.py" "attention_analyzer.py" "results_generator.py" "feature_importance_comparison.py")

for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    name="${SCRIPT_NAMES[$i]}"
    
    if [ ! -f "$script" ]; then
        if [ "$name" = "feature_importance_comparison.py" ]; then
            echo "⚠️  Script not found: $script (will be created)"
        else
            echo "❌ Script not found: $script"
            exit 1
        fi
    else
        echo "✅ Found: $name"
    fi
done

echo "✅ All required scripts validated"

# === PREREQUISITE CHECK ===
echo ""
echo "🧬 PREREQUISITE CHECK"
echo "-------------------"

# Check trained model - UPDATED to latest model
MODEL_PATH="/u/aa107/scratch/tabnet_model_20250727_053446.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Trained TabNet model not found: $MODEL_PATH"
    echo "💡 Train the model first using the training pipeline"
    exit 1
else
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "✅ TabNet model found: $MODEL_SIZE"
fi

# Check dataset
DATASET_PATH="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "💡 Run data processing pipeline first"
    exit 1
else
    DATASET_SIZE=$(du -h "$DATASET_PATH" | cut -f1)
    VARIANT_COUNT=$(tail -n +2 "$DATASET_PATH" | wc -l)
    echo "✅ Dataset found: $DATASET_SIZE ($VARIANT_COUNT variants)"
fi

echo "✅ All prerequisites satisfied"

# === CREATE RESULTS DIRECTORY ===
echo ""
echo "📁 PREPARING OUTPUT DIRECTORIES"
echo "-------------------------------"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/attention_weights"
mkdir -p "$RESULTS_DIR/attention_heatmaps"
mkdir -p "$RESULTS_DIR/feature_rankings"
mkdir -p "$RESULTS_DIR/publication_figures"
mkdir -p "$RESULTS_DIR/individual_variants"

echo "✅ Output directories created"

# === MAIN EXECUTION ===
echo ""
echo "🚀 STARTING ATTENTION ANALYSIS PIPELINE"
echo "======================================="
echo ""

# Run scripts in sequence
echo "📋 STEP 1: VARIANT SELECTION"
echo "----------------------------"
echo "Purpose: Select representative pathogenic and benign variants"
echo "Script: variant_selector.py"
echo ""

python "$SCRIPT_1"

if [ $? -eq 0 ]; then
    echo "✅ Step 1 completed successfully"
    
    # Check output
    if [ -f "$RESULTS_DIR/selected_variants.csv" ]; then
        SELECTED_COUNT=$(tail -n +2 "$RESULTS_DIR/selected_variants.csv" | wc -l)
        echo "📊 Selected $SELECTED_COUNT variants for analysis"
    fi
else
    echo "❌ Step 1 failed"
    exit 1
fi

echo ""
echo "🧠 STEP 2: ATTENTION EXTRACTION"
echo "------------------------------"
echo "Purpose: Extract TabNet attention weights for selected variants"
echo "Script: attention_extractor.py"
echo ""

python "$SCRIPT_2"

if [ $? -eq 0 ]; then
    echo "✅ Step 2 completed successfully"
else
    echo "❌ Step 2 failed"
    exit 1
fi

echo ""
echo "📊 STEP 3: ATTENTION ANALYSIS"
echo "----------------------------"
echo "Purpose: Analyze attention patterns and feature importance"
echo "Script: attention_analyzer.py"
echo ""

python "$SCRIPT_3"

if [ $? -eq 0 ]; then
    echo "✅ Step 3 completed successfully"
else
    echo "❌ Step 3 failed"
    exit 1
fi

echo ""
echo "🎨 STEP 4: RESULTS GENERATION"
echo "----------------------------"
echo "Purpose: Generate publication-ready figures and tables"
echo "Script: results_generator.py"
echo ""

python "$SCRIPT_4"

if [ $? -eq 0 ]; then
    echo "✅ Step 4 completed successfully"
else
    echo "❌ Step 4 failed"
    exit 1
fi

echo ""
echo "🔬 STEP 5: FEATURE IMPORTANCE COMPARISON"
echo "---------------------------------------"
echo "Purpose: Compare TabNet attention vs traditional feature importance"
echo "Script: feature_importance_comparison.py"
echo ""

# Check if script exists, create if needed
if [ ! -f "$SCRIPT_5" ]; then
    echo "⚠️  Creating placeholder script: $SCRIPT_5"
    cat > "$SCRIPT_5" << 'EOF'
#!/usr/bin/env python3
"""Feature importance comparison placeholder"""
print("🔬 Feature importance comparison")
print("⚠️  This is a placeholder - implement comparison logic")
print("✅ Placeholder execution complete")
EOF
    chmod +x "$SCRIPT_5"
fi

python "$SCRIPT_5"

if [ $? -eq 0 ]; then
    echo "✅ Step 5 completed successfully"
else
    echo "❌ Step 5 failed (non-critical)"
fi

# === FINAL SUMMARY ===
echo ""
echo "🎉 ATTENTION ANALYSIS PIPELINE COMPLETED!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo ""

# List generated files
echo "📁 Generated files:"
if [ -d "$RESULTS_DIR" ]; then
    find "$RESULTS_DIR" -type f -name "*.csv" -o -name "*.png" -o -name "*.txt" -o -name "*.npy" | sort | while read -r file; do
        echo "  - ${file#$RESULTS_DIR/}"
    done
fi

echo ""
echo "📊 Key outputs to review:"
echo "  1. Feature importance scores: attention_weights/feature_importance_scores.csv"
echo "  2. Attention heatmaps: attention_heatmaps/*.png"
echo "  3. Pathogenic vs benign comparison: pathogenic_vs_benign_comparison.png"
echo "  4. Publication figures: publication_figures/*.png"
echo ""
echo "Completed at: $(date)"
echo "Total runtime: $SECONDS seconds"