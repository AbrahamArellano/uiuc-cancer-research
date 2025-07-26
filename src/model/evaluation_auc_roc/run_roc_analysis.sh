#!/bin/bash
# File: /u/aa107/uiuc-cancer-research/src/model/evaluation_auc_roc/run_roc_analysis.sh
# Purpose: Run complete ROC-AUC analysis pipeline on UIUC Campus Cluster
# Version: Debug-enhanced

set -e  # Exit on error

echo "🎯 COMPLETE ROC-AUC ANALYSIS PIPELINE"
echo "===================================="
echo "Started at: $(date)"
echo ""

# Configuration
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
EVAL_DIR="${PROJECT_DIR}/src/model/evaluation_auc_roc"
MODEL_PATH="/u/aa107/scratch/tabnet_model_20250719_055806.pkl"
OUTPUT_DIR="${PROJECT_DIR}/results/roc_analysis"
FIGURES_DIR="${PROJECT_DIR}/results/figures"
CONDA_ENV="tabnet-prostate"

# Debug: Show all paths
echo "🔍 DEBUG: Path Configuration"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  EVAL_DIR: $EVAL_DIR"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Change to project directory
cd $PROJECT_DIR

# Set Python path
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"
echo "🔍 DEBUG: PYTHONPATH=$PYTHONPATH"

# === ENVIRONMENT SETUP (UIUC CAMPUS CLUSTER) ===
echo "🔧 Setting up environment..."

# Load anaconda module (required on UIUC Campus Cluster)
echo "Loading anaconda module..."
if module load anaconda3 2>/dev/null; then
    echo "✅ Anaconda module loaded"
else
    echo "⚠️  Anaconda module not available - using system conda"
fi

# Initialize conda for shell script (required for compute nodes)
echo "Initializing conda for bash..."
eval "$(conda shell.bash hook)"

# Check if environment exists
if conda env list | grep -q "${CONDA_ENV}"; then
    echo "✅ Found ${CONDA_ENV} environment"
else
    echo "❌ ${CONDA_ENV} environment not found"
    echo "💡 Create it first by running:"
    echo "   bash src/model/tests/run_tabnet_tests.sh"
    exit 1
fi

# Activate the conda environment
echo "Activating ${CONDA_ENV} environment..."
conda activate "${CONDA_ENV}"

# Verify environment
echo "📍 Current directory: $(pwd)"
echo "📍 Python: $(which python)"
echo "📍 Python version: $(python --version)"
echo "📍 Conda env: ${CONDA_DEFAULT_ENV}"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# 1. Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $MODEL_PATH"
    echo "   Please ensure you have the trained TabNet model"
    exit 1
else
    echo "✅ Model file found"
    echo "   Size: $(ls -lh "$MODEL_PATH" | awk '{print $5}')"
fi

# 2. Check if dataset exists
DATASET_PATH="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    exit 1
else
    echo "✅ Dataset found ($(wc -l < "$DATASET_PATH") lines)"
fi

# 3. Check Python imports
echo "🐍 Checking Python dependencies..."
python -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
try:
    import pandas, numpy, sklearn, matplotlib
    print('✅ Core packages available')
    from pytorch_tabnet.tab_model import TabNetClassifier
    print('✅ TabNet available')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Python dependency check failed"
    exit 1
fi

# Create output directories
echo ""
echo "📁 Creating output directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $FIGURES_DIR
mkdir -p "${FIGURES_DIR}/supplementary"

echo "🔍 DEBUG: Output directories created:"
ls -la "${OUTPUT_DIR}/" 2>/dev/null || echo "  OUTPUT_DIR not accessible"
ls -la "${FIGURES_DIR}/" 2>/dev/null || echo "  FIGURES_DIR not accessible"

# STEP 1: Calculate ROC-AUC metrics
echo ""
echo "📊 STEP 1: Calculating ROC-AUC metrics..."
echo "========================================="

# Check if calculate_roc_auc.py exists
CALC_SCRIPT="${EVAL_DIR}/calculate_roc_auc.py"
echo "🔍 DEBUG: Checking for script at: $CALC_SCRIPT"

if [ ! -f "$CALC_SCRIPT" ]; then
    echo "❌ Error: calculate_roc_auc.py not found"
    echo "   Expected at: $CALC_SCRIPT"
    echo "   Directory contents:"
    ls -la "$EVAL_DIR" 2>/dev/null || echo "   EVAL_DIR not accessible"
    exit 1
else
    echo "✅ Script found: $CALC_SCRIPT"
    echo "   Size: $(ls -lh "$CALC_SCRIPT" | awk '{print $5}')"
    echo "   Permissions: $(ls -l "$CALC_SCRIPT" | awk '{print $1}')"
fi

# Check if ROC data already exists
echo ""
echo "🔍 DEBUG: Checking for existing ROC data..."
if [ -f "${OUTPUT_DIR}/roc_auc_metrics.json" ]; then
    echo "  Found: roc_auc_metrics.json"
    echo "  Size: $(ls -lh "${OUTPUT_DIR}/roc_auc_metrics.json" | awk '{print $5}')"
fi
if [ -f "${OUTPUT_DIR}/roc_curve_data.json" ]; then
    echo "  Found: roc_curve_data.json"
    echo "  Size: $(ls -lh "${OUTPUT_DIR}/roc_curve_data.json" | awk '{print $5}')"
fi

# Run ROC calculation
echo ""
echo "🚀 EXECUTING: python ${CALC_SCRIPT}"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"

# Execute with full error capture
python "${CALC_SCRIPT}" 2>&1 | tee /tmp/roc_calc_output_$$.log

# Capture exit code
CALC_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "🔍 DEBUG: Python script exit code: $CALC_EXIT_CODE"

if [ $CALC_EXIT_CODE -ne 0 ]; then
    echo "❌ ROC calculation failed with exit code: $CALC_EXIT_CODE"
    echo "📋 Last 20 lines of output:"
    tail -20 /tmp/roc_calc_output_$$.log
    exit 1
else
    echo "✅ Python script completed with exit code 0"
fi

# Verify ROC outputs
echo ""
echo "🔍 Verifying ROC calculation outputs..."
echo "Checking in directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" 2>/dev/null || echo "❌ Cannot list OUTPUT_DIR"

if [ -f "${OUTPUT_DIR}/roc_auc_metrics.json" ] && [ -f "${OUTPUT_DIR}/roc_curve_data.json" ]; then
    echo "✅ ROC data files generated successfully"
    
    # Display AUC summary
    echo ""
    echo "📈 AUC Summary:"
    python -c "
import json
try:
    with open('${OUTPUT_DIR}/roc_auc_metrics.json', 'r') as f:
        metrics = json.load(f)
        print(f\"  Overall AUC: {metrics['summary']['overall_auc']:.3f}\")
        print(f\"  Pathogenic AUC: {metrics['summary']['pathogenic_auc']:.3f}\")
        print(f\"  Benign AUC: {metrics['summary']['benign_auc']:.3f}\")
        print(f\"  VUS AUC: {metrics['summary']['vus_auc']:.3f}\")
except Exception as e:
    print(f\"❌ Error reading metrics: {e}\")
"
else
    echo "❌ ROC data files not found!"
    echo "Expected files:"
    echo "  ${OUTPUT_DIR}/roc_auc_metrics.json"
    echo "  ${OUTPUT_DIR}/roc_curve_data.json"
    echo ""
    echo "🔍 DEBUG: Checking if script created any files..."
    find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.png" 2>/dev/null || echo "  No files found"
    exit 1
fi

# STEP 2: Generate visualization figures
echo ""
echo "🎨 STEP 2: Generating publication figures..."
echo "==========================================="

VIS_SCRIPT="${EVAL_DIR}/compare_baseline_tools.py"
echo "🔍 DEBUG: Checking visualization script: $VIS_SCRIPT"

if [ ! -f "$VIS_SCRIPT" ]; then
    echo "❌ Visualization script not found"
    exit 1
else
    echo "✅ Found visualization script"
fi

echo "🚀 EXECUTING: python ${VIS_SCRIPT}"
python "${VIS_SCRIPT}" 2>&1 | tee /tmp/vis_output_$$.log

VIS_EXIT_CODE=${PIPESTATUS[0]}
echo "🔍 DEBUG: Visualization script exit code: $VIS_EXIT_CODE"

if [ $VIS_EXIT_CODE -ne 0 ]; then
    echo "❌ Figure generation failed!"
    echo "📋 Last 20 lines of output:"
    tail -20 /tmp/vis_output_$$.log
    exit 1
fi

# Verify outputs
echo ""
echo "📁 Final output verification:"
echo "============================"

echo ""
echo "ROC Analysis Data:"
ls -la "${OUTPUT_DIR}/"

echo ""
echo "Publication Figures:"
ls -la "${FIGURES_DIR}/"*.png 2>/dev/null || echo "  No figures in main directory"

echo ""
echo "Supplementary Figures:"
ls -la "${FIGURES_DIR}/supplementary/"*.png 2>/dev/null || echo "  No supplementary figures"

# Cleanup temp files
rm -f /tmp/roc_calc_output_$$.log /tmp/vis_output_$$.log

# Summary
echo ""
echo "✅ ROC-AUC ANALYSIS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "📊 Key Results Location:"
echo "  - Metrics: ${OUTPUT_DIR}/roc_auc_metrics.json"
echo "  - Main Figure: ${FIGURES_DIR}/figure_6_roc_curves.png"
echo "  - Comparison: ${FIGURES_DIR}/figure_roc_comparison.png"
echo "  - Summary Table: ${FIGURES_DIR}/auc_summary_table.png"
echo ""
echo "Completed at: $(date)"