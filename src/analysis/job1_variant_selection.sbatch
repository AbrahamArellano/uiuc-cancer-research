#!/bin/bash
#SBATCH --job-name=variant_selection
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=15:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/src/analysis/variant_selection_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/src/analysis/variant_selection_%j.err

# ================================================================
# TABNET VARIANT SELECTION - CPU-ONLY JOB
# ================================================================

set -euo pipefail

echo "🧬 TABNET VARIANT SELECTION - CPU PROCESSING"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Mode: CPU-ONLY (Fast variant selection)"
echo ""

# Configuration
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
SCRIPT="${PROJECT_DIR}/src/analysis/variant_selector.py"
CONDA_ENV="tabnet-prostate"

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

# Check dataset
DATASET_PATH="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "💡 Run data processing pipeline first"
    exit 1
else
    DATASET_SIZE=$(du -h "$DATASET_PATH" | cut -f1)
    DATASET_LINES=$(wc -l < "$DATASET_PATH")
    echo "✅ Dataset found: $DATASET_SIZE ($DATASET_LINES variants)"
fi

# Verify script exists
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Variant selector script not found: $SCRIPT"
    exit 1
else
    echo "✅ Script found: variant_selector.py"
fi

# Create output directory
OUTPUT_DIR="${PROJECT_DIR}/results/attention_analysis"
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory: $OUTPUT_DIR"

echo ""

# Execute variant selection
echo "🚀 STARTING VARIANT SELECTION"
echo "============================="
echo "Purpose: Select representative pathogenic and benign variants"
echo "Script: variant_selector.py"
echo ""

# Run the script
python "$SCRIPT"
SELECTION_EXIT_CODE=$?

echo ""
if [ $SELECTION_EXIT_CODE -eq 0 ]; then
    echo "✅ VARIANT SELECTION COMPLETED SUCCESSFULLY"
    
    # Verify output files
    VARIANTS_FILE="${OUTPUT_DIR}/selected_variants.csv"
    if [ -f "$VARIANTS_FILE" ]; then
        SELECTED_COUNT=$(wc -l < "$VARIANTS_FILE")
        echo "📁 Selected variants: $VARIANTS_FILE ($SELECTED_COUNT variants)"
        echo "🎯 Ready for Job 2: Attention Extraction"
    else
        echo "⚠️  Output file not found: $VARIANTS_FILE"
        exit 1
    fi
else
    echo "❌ VARIANT SELECTION FAILED (Exit code: $SELECTION_EXIT_CODE)"
    exit $SELECTION_EXIT_CODE
fi

echo ""
echo "Job completed: $(date)"