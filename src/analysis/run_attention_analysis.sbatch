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
# Updated to include 5 scripts and use latest model
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
    echo "⚠️  GPU not available - attention analysis may have device issues"
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
    echo "❌ Clean dataset not found: $DATASET_PATH"
    echo "💡 Run data processing pipeline first"
    exit 1
else
    DATASET_SIZE=$(du -h "$DATASET_PATH" | cut -f1)
    DATASET_LINES=$(wc -l < "$DATASET_PATH")
    echo "✅ Dataset found: $DATASET_SIZE ($DATASET_LINES variants)"
fi

echo "✅ All prerequisites satisfied"

# === ATTENTION ANALYSIS PIPELINE ===
echo ""
echo "🚀 STARTING ATTENTION ANALYSIS PIPELINE"
echo "======================================="

# Step 1: Variant Selection
echo ""
echo "📋 STEP 1: VARIANT SELECTION"
echo "----------------------------"
echo "Purpose: Select representative pathogenic and benign variants"
echo "Script: variant_selector.py"
echo ""

START_TIME=$(date +%s)

python "$SCRIPT_1"
STEP1_EXIT_CODE=$?

if [ $STEP1_EXIT_CODE -eq 0 ]; then
    echo "✅ Step 1 completed successfully"
    
    # Check output
    SELECTED_FILE="${RESULTS_DIR}/selected_variants.csv"
    if [ -f "$SELECTED_FILE" ]; then
        VARIANT_COUNT=$(tail -n +2 "$SELECTED_FILE" | wc -l)
        echo "📊 Selected $VARIANT_COUNT variants for analysis"
    fi
else
    echo "❌ Step 1 failed (exit code: $STEP1_EXIT_CODE)"
    exit 1
fi

# Step 2: Attention Extraction
echo ""
echo "🧠 STEP 2: ATTENTION EXTRACTION"
echo "------------------------------"
echo "Purpose: Extract TabNet attention weights for selected variants"
echo "Script: attention_extractor.py"
echo ""

python "$SCRIPT_2"
STEP2_EXIT_CODE=$?

if [ $STEP2_EXIT_CODE -eq 0 ]; then
    echo "✅ Step 2 completed successfully"
    
    # Check output
    ATTENTION_DIR="${RESULTS_DIR}/attention_weights"
    if [ -d "$ATTENTION_DIR" ]; then
        ATTENTION_FILES=$(find "$ATTENTION_DIR" -name "*_attention.csv" | wc -l)
        echo "📊 Generated $ATTENTION_FILES attention files"
    fi
else
    echo "❌ Step 2 failed (exit code: $STEP2_EXIT_CODE)"
    exit 1
fi

# Step 3: Pattern Analysis
echo ""
echo "📊 STEP 3: PATTERN ANALYSIS"
echo "--------------------------"
echo "Purpose: Analyze attention patterns without medical interpretation"
echo "Script: attention_analyzer.py"
echo ""

python "$SCRIPT_3"
STEP3_EXIT_CODE=$?

if [ $STEP3_EXIT_CODE -eq 0 ]; then
    echo "✅ Step 3 completed successfully"
    
    # Check output
    PATTERNS_DIR="${RESULTS_DIR}/pattern_analysis"
    if [ -d "$PATTERNS_DIR" ]; then
        PATTERN_FILES=$(find "$PATTERNS_DIR" -name "*.txt" -o -name "*.png" | wc -l)
        echo "📊 Generated $PATTERN_FILES analysis files"
    fi
else
    echo "❌ Step 3 failed (exit code: $STEP3_EXIT_CODE)"
    exit 1
fi

# Step 4: Results Generation
echo ""
echo "📋 STEP 4: RESULTS GENERATION"
echo "----------------------------"
echo "Purpose: Generate final publication-ready results"
echo "Script: results_generator.py"
echo ""

python "$SCRIPT_4"
STEP4_EXIT_CODE=$?

if [ $STEP4_EXIT_CODE -eq 0 ]; then
    echo "✅ Step 4 completed successfully"
    
    # Check output
    FINAL_DIR="${RESULTS_DIR}/final_results"
    if [ -d "$FINAL_DIR" ]; then
        FINAL_FILES=$(find "$FINAL_DIR" -type f | wc -l)
        echo "📊 Generated $FINAL_FILES final result files"
    fi
else
    echo "❌ Step 4 failed (exit code: $STEP4_EXIT_CODE)"
    exit 1
fi

# Step 5: Feature Importance Comparison (NEW)
echo ""
echo "🔍 STEP 5: FEATURE IMPORTANCE COMPARISON"
echo "---------------------------------------"
echo "Purpose: Compare training vs attention feature importance"
echo "Script: feature_importance_comparison.py"
echo ""

# Check if script exists, if not skip with warning
if [ -f "$SCRIPT_5" ]; then
    python "$SCRIPT_5"
    STEP5_EXIT_CODE=$?
    
    if [ $STEP5_EXIT_CODE -eq 0 ]; then
        echo "✅ Step 5 completed successfully"
    else
        echo "⚠️  Step 5 failed but continuing (exit code: $STEP5_EXIT_CODE)"
    fi
else
    echo "⚠️  Feature importance comparison script not found - skipping"
    echo "   Create $SCRIPT_5 to enable this analysis"
fi

# === COMPLETION SUMMARY ===
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "🎉 ATTENTION ANALYSIS PIPELINE COMPLETED!"
echo "========================================="
echo "Completed: $(date)"
echo "Duration: $DURATION seconds"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Count total outputs
echo "📁 Output Directory Structure:"
echo "   ${RESULTS_DIR}/"
echo "   ├── selected_variants.csv                    # Selected variants"
echo "   ├── attention_weights/                       # Individual attention files"
echo "   │   ├── variant_*_attention.csv"
echo "   │   ├── attention_summary.csv"
echo "   │   └── extraction_metadata.json"
echo "   ├── pattern_analysis/                        # Pattern analysis"
echo "   │   ├── pathogenic_vs_benign_patterns.txt"
echo "   │   ├── feature_group_analysis.txt"
echo "   │   ├── decision_step_patterns.txt"
echo "   │   └── *.png visualizations"
echo "   └── final_results/                           # Publication materials"
echo "       ├── tabnet_attention_analysis_report.md"
echo "       ├── validation_dashboard.png"
echo "       └── analysis_summary.json"
echo ""

# Calculate total files and size
if [ -d "$RESULTS_DIR" ]; then
    TOTAL_FILES=$(find "$RESULTS_DIR" -type f | wc -l)
    TOTAL_SIZE=$(du -sh "$RESULTS_DIR" | cut -f1)
    echo "📊 Analysis Summary:"
    echo "   Total files generated: $TOTAL_FILES"
    echo "   Total output size: $TOTAL_SIZE"
    echo "   Analysis duration: $DURATION seconds"
    echo "   GPU utilization: Available"
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. Review attention patterns in: ${RESULTS_DIR}/pattern_analysis/"
echo "   2. Examine final results in: ${RESULTS_DIR}/final_results/"
echo "   3. Check feature comparison in: ${RESULTS_DIR}/feature_importance_comparison/"
echo "   4. Share results with clinical experts for validation"
echo "   5. Prepare manuscript using generated materials"
echo ""
echo "✅ TabNet attention analysis pipeline completed successfully!"
echo "🎉 Ready for clinical expert review and publication!"
echo ""
echo "📋 SLURM Job Summary:"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $SLURMD_NODENAME"
echo "   Exit Status: SUCCESS"
echo "   Log files: attention_analysis_${SLURM_JOB_ID}.out/.err"