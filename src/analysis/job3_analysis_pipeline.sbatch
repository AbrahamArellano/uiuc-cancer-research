#!/bin/bash
#SBATCH --job-name=analysis_pipeline
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/src/analysis/analysis_pipeline_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/src/analysis/analysis_pipeline_%j.err

# ================================================================
# TABNET ANALYSIS PIPELINE - CPU-ONLY FINAL PROCESSING
# ================================================================

set -euo pipefail

echo "📊 TABNET ANALYSIS PIPELINE - FINAL PROCESSING"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Mode: CPU-ONLY (Analysis and reporting)"
echo ""

# Configuration
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
ANALYSIS_DIR="${PROJECT_DIR}/src/analysis"
CONDA_ENV="tabnet-prostate"

# Analysis scripts
SCRIPT_1="${ANALYSIS_DIR}/attention_analyzer.py"
SCRIPT_2="${ANALYSIS_DIR}/results_generator.py"
SCRIPT_3="${ANALYSIS_DIR}/feature_importance_comparison.py"

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

# Check attention weights from Job 2
ATTENTION_DIR="${PROJECT_DIR}/results/attention_analysis/attention_weights"
if [ ! -d "$ATTENTION_DIR" ]; then
    echo "❌ Attention weights directory not found: $ATTENTION_DIR"
    echo "💡 Job 2 (attention extraction) must complete first"
    exit 1
else
    ATTENTION_FILES=$(find "$ATTENTION_DIR" -name "*.csv" | wc -l)
    echo "✅ Attention weights found: $ATTENTION_FILES files in $ATTENTION_DIR"
fi

# Verify Job 2 completed fully (should have 16+ individual files + 3 summaries)
EXPECTED_FILES=19
ACTUAL_FILES=$(find "$ATTENTION_DIR" -name "*.csv" -o -name "*.json" | wc -l)
if [ $ACTUAL_FILES -lt $EXPECTED_FILES ]; then
    echo "❌ Incomplete Job 2 results: $ACTUAL_FILES/$EXPECTED_FILES files"
    exit 1
fi

# Check selected variants from Job 1
VARIANTS_FILE="${PROJECT_DIR}/results/attention_analysis/selected_variants.csv"
if [ ! -f "$VARIANTS_FILE" ]; then
    echo "❌ Selected variants not found: $VARIANTS_FILE"
    echo "💡 Job 1 (variant selection) must complete first"
    exit 1
else
    echo "✅ Selected variants found: $(wc -l < "$VARIANTS_FILE") lines"
fi

# Verify analysis scripts exist
for script in "$SCRIPT_1" "$SCRIPT_2" "$SCRIPT_3"; do
    script_name=$(basename "$script")
    if [ ! -f "$script" ]; then
        if [ "$script_name" = "feature_importance_comparison.py" ]; then
            echo "⚠️  Script not found: $script_name (will be created if needed)"
        else
            echo "❌ Script not found: $script"
            exit 1
        fi
    else
        echo "✅ Script found: $script_name"
    fi
done

echo ""

# Execute analysis pipeline
echo "🚀 STARTING ANALYSIS PIPELINE"
echo "============================="
echo "Purpose: Analyze attention weights and generate final reports"
echo "Scripts: 3-step analysis pipeline"
echo ""

# Step 1: Attention Analysis
echo "📊 STEP 1: ATTENTION ANALYSIS"
echo "----------------------------"
echo "Script: attention_analyzer.py"
echo ""

python "$SCRIPT_1"
STEP1_EXIT_CODE=$?

if [ $STEP1_EXIT_CODE -ne 0 ]; then
    echo "❌ Step 1 failed (Exit code: $STEP1_EXIT_CODE)"
    exit $STEP1_EXIT_CODE
fi

echo "✅ Step 1 completed successfully"
echo ""

# Step 2: Results Generation
echo "📈 STEP 2: RESULTS GENERATION"
echo "----------------------------"
echo "Script: results_generator.py"
echo ""

python "$SCRIPT_2"
STEP2_EXIT_CODE=$?

if [ $STEP2_EXIT_CODE -ne 0 ]; then
    echo "❌ Step 2 failed (Exit code: $STEP2_EXIT_CODE)"
    exit $STEP2_EXIT_CODE
fi

echo "✅ Step 2 completed successfully"
echo ""

# Step 3: Feature Importance Comparison
echo "🔍 STEP 3: FEATURE IMPORTANCE COMPARISON"
echo "---------------------------------------"
echo "Script: feature_importance_comparison.py"
echo ""

if [ -f "$SCRIPT_3" ]; then
    python "$SCRIPT_3"
    STEP3_EXIT_CODE=$?
    
    if [ $STEP3_EXIT_CODE -ne 0 ]; then
        echo "❌ Step 3 failed (Exit code: $STEP3_EXIT_CODE)"
        exit $STEP3_EXIT_CODE
    fi
    
    echo "✅ Step 3 completed successfully"
else
    echo "⚠️  Step 3 script not found - skipping feature importance comparison"
fi

echo ""

# Final verification
echo "🎉 ANALYSIS PIPELINE COMPLETED"
echo "=============================="

# Check final outputs
RESULTS_DIR="${PROJECT_DIR}/results/attention_analysis"
if [ -d "$RESULTS_DIR" ]; then
    RESULT_FILES=$(find "$RESULTS_DIR" -name "*.png" -o -name "*.csv" -o -name "*.txt" | wc -l)
    echo "📁 Analysis results: $RESULT_FILES files in $RESULTS_DIR"
    echo "✅ Complete attention analysis pipeline finished successfully"
else
    echo "⚠️  Results directory not found: $RESULTS_DIR"
fi

echo ""
echo "🎯 All analysis steps completed!"
echo "📊 Results available for review and publication"
echo ""
echo "Job completed: $(date)"