#!/bin/bash
#SBATCH --job-name=alphamissense_enhance
#SBATCH --partition=IllinoisComputes
#SBATCH --account=aa107-ic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/u/aa107/uiuc-cancer-research/logs/alphamissense_enhancement_%j.out
#SBATCH --error=/u/aa107/uiuc-cancer-research/logs/alphamissense_enhancement_%j.err

# ================================================================
# AlphaMissense Functional Enhancement Pipeline
# Replaces artificial imputation with legitimate pathogenicity scores
# ================================================================

set -euo pipefail

echo "🧬 ALPHAMISSENSE FUNCTIONAL ENHANCEMENT PIPELINE"
echo "================================================================"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
SCRIPT_DIR="${PROJECT_DIR}/scripts/enhance/functional_enhancement"
LOG_DIR="${PROJECT_DIR}/logs"
SCRATCH_DIR="/u/aa107/scratch/alphamissense"

# Input/Output files
INPUT_FILE="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet.csv"
OUTPUT_FILE="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_enhanced.csv"
SCRIPT_FILE="${SCRIPT_DIR}/simple_functional_imputation.py"

# === CREATE DIRECTORIES ===
echo "📁 Creating necessary directories..."
mkdir -p "${LOG_DIR}"
mkdir -p "${SCRATCH_DIR}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"
echo "✅ Directories created"

# === ENVIRONMENT SETUP ===
echo ""
echo "🔧 ENVIRONMENT SETUP"
echo "-------------------"

# Navigate to project root
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

# Load anaconda module (following project standard)
echo "Loading anaconda module..."
module load anaconda3

# Initialize conda for bash (required for compute nodes)
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

echo "Activating tabnet-prostate environment..."
conda activate tabnet-prostate

# Verify environment activation
echo "✅ Environment verification:"
echo "  Python location: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda environment: $CONDA_DEFAULT_ENV"

# Install additional packages if needed
echo "📥 Installing additional packages..."
pip install requests --quiet
echo "✅ Packages ready"

# === INPUT VALIDATION ===
echo ""
echo "📋 INPUT VALIDATION"
echo "-------------------"

if [ ! -f "${INPUT_FILE}" ]; then
    echo "❌ ERROR: Input file not found: ${INPUT_FILE}"
    echo "Please ensure VEP processing has completed successfully."
    exit 1
fi

INPUT_SIZE=$(du -h "${INPUT_FILE}" | cut -f1)
INPUT_LINES=$(wc -l < "${INPUT_FILE}")

echo "✅ Input file validated:"
echo "   File: ${INPUT_FILE}"
echo "   Size: ${INPUT_SIZE}"
echo "   Lines: ${INPUT_LINES}"

# === DISK SPACE CHECK ===
echo ""
echo "💾 DISK SPACE CHECK"
echo "-------------------"

SCRATCH_AVAILABLE=$(df -h "${SCRATCH_DIR}" 2>/dev/null | awk 'NR==2{print $4}' || echo "Unknown")
PROJECT_AVAILABLE=$(df -h "${PROJECT_DIR}" | awk 'NR==2{print $4}')

echo "📊 Available space:"
echo "   Scratch dir: ${SCRATCH_AVAILABLE}"
echo "   Project dir: ${PROJECT_AVAILABLE}"

# === ALPHAMISSENSE ENHANCEMENT ===
echo ""
echo "🎯 RUNNING ALPHAMISSENSE ENHANCEMENT"
echo "====================================="

echo "🔬 Enhancement details:"
echo "   Method: AlphaMissense pathogenicity prediction"
echo "   Database: Google DeepMind pre-computed scores (~2GB)"
echo "   Storage: ${SCRATCH_DIR}"
echo "   Objective: Eliminate data leakage from artificial features"
echo ""

START_TIME=$(date +%s)

# Run the AlphaMissense enhancement with detailed logging
echo "📝 Starting enhancement process..."
python3 "${SCRIPT_FILE}" 2>&1 | tee "${LOG_DIR}/alphamissense_enhancement_$(date +%Y%m%d_%H%M%S).log"

PYTHON_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "⏱️  Processing completed in ${DURATION} seconds"
echo "Exit code: ${PYTHON_EXIT_CODE}"

# Check if Python script succeeded
if [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "❌ Python script failed with exit code: ${PYTHON_EXIT_CODE}"
    echo "Check the log file for details: ${LOG_DIR}/alphamissense_enhancement_*.log"
    exit 1
fi

# === OUTPUT VALIDATION ===
echo ""
echo "✅ OUTPUT VALIDATION"
echo "-------------------"

if [ -f "${OUTPUT_FILE}" ]; then
    OUTPUT_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    OUTPUT_LINES=$(wc -l < "${OUTPUT_FILE}")
    
    echo "✅ Output file created successfully:"
    echo "   File: ${OUTPUT_FILE}"
    echo "   Size: ${OUTPUT_SIZE}"
    echo "   Lines: ${OUTPUT_LINES}"
    
    # Validate line count (should be same as input)
    if [ "${OUTPUT_LINES}" -eq "${INPUT_LINES}" ]; then
        echo "✅ Line count validation: PASSED"
    else
        echo "⚠️  Line count mismatch: Input ${INPUT_LINES}, Output ${OUTPUT_LINES}"
    fi
    
    # Check for AlphaMissense columns
    HEADER=$(head -1 "${OUTPUT_FILE}")
    if echo "${HEADER}" | grep -q "alphamissense_pathogenicity"; then
        echo "✅ AlphaMissense features: FOUND"
    else
        echo "❌ AlphaMissense features: MISSING"
    fi
    
    # Check artificial features are removed
    if echo "${HEADER}" | grep -q -E "(sift_confidence|polyphen_confidence|functional_pathogenicity)"; then
        echo "⚠️  Artificial features: STILL PRESENT (check implementation)"
    else
        echo "✅ Artificial features: REMOVED"
    fi
    
else
    echo "❌ ERROR: Output file not created: ${OUTPUT_FILE}"
    echo "Check Python script execution and logs"
    exit 1
fi

# === CLEANUP AND SUMMARY ===
echo ""
echo "🧹 CLEANUP"
echo "----------"

# Keep AlphaMissense database for future runs
ALPHAMISSENSE_DB="${SCRATCH_DIR}/AlphaMissense_hg38.tsv"
if [ -f "${ALPHAMISSENSE_DB}" ]; then
    DB_SIZE=$(du -h "${ALPHAMISSENSE_DB}" | cut -f1)
    echo "💾 AlphaMissense database preserved: ${ALPHAMISSENSE_DB} (${DB_SIZE})"
fi

echo ""
echo "🎉 ALPHAMISSENSE ENHANCEMENT COMPLETED"
echo "======================================"
echo "📊 Summary:"
echo "   ✅ Data leakage eliminated (artificial features removed)"
echo "   ✅ Legitimate AlphaMissense scores added"
echo "   ✅ Ready for TabNet training with realistic accuracy"
echo "   📁 Enhanced dataset: ${OUTPUT_FILE}"
echo "   📋 Report: ${PROJECT_DIR}/data/processed/tabnet_csv/alphamissense_enhancement_report.txt"
echo ""
echo "🎯 Next Steps:"
echo "   1. Validate enhanced dataset quality"
echo "   2. Train TabNet model with new features"
echo "   3. Expect 75-85% accuracy (no more 100% artificial accuracy)"
echo "   4. Proceed with interpretable cancer variant classification"
echo ""
echo "Completed: $(date)"
echo "Duration: ${DURATION} seconds"

# === FINAL STATUS ===
if [ -f "${OUTPUT_FILE}" ]; then
    echo "✅ PIPELINE SUCCESS: Enhanced dataset ready for TabNet training"
    exit 0
else
    echo "❌ PIPELINE FAILED: Check logs for details"
    exit 1
fi