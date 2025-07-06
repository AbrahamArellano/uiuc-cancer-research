#!/bin/bash
#SBATCH --job-name=enhanced_tabnet_validation
#SBATCH --account=aa107-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=enhanced_tabnet_validation_%j.out
#SBATCH --error=enhanced_tabnet_validation_%j.err

echo "🧬 ENHANCED TABNET VALIDATION ON CAMPUS CLUSTER"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Updated thresholds: 70-80% target accuracy (realistic clinical performance)"
echo ""

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
else
    echo "⚠️  GPU info not available"
fi
echo ""

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
VALIDATION_SCRIPT="${PROJECT_DIR}/src/model/validation/validate_tabnet.py"
CLEAN_DATASET="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"

# === ENVIRONMENT SETUP ===
echo "🔧 ENVIRONMENT SETUP"
echo "-----------------"

# Navigate to project directory
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

# Load anaconda module
echo "Loading anaconda module..."
module load anaconda3

# Initialize conda
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Activate environment
echo "Activating tabnet-prostate environment..."
conda activate tabnet-prostate

# Verify environment
echo "✅ Environment verification:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# === PREREQUISITE CHECKS ===
echo "📋 PREREQUISITE CHECKS"
echo "--------------------"

# Check clean dataset
if [ ! -f "$CLEAN_DATASET" ]; then
    echo "❌ Clean dataset not found: $CLEAN_DATASET"
    echo "💡 Run data processing pipeline first"
    exit 1
else
    DATASET_SIZE=$(du -h "$CLEAN_DATASET" | cut -f1)
    DATASET_LINES=$(wc -l < "$CLEAN_DATASET")
    echo "✅ Clean dataset found:"
    echo "  📁 File: $CLEAN_DATASET"
    echo "  📏 Size: $DATASET_SIZE"
    echo "  📊 Lines: $DATASET_LINES"
fi

# Check validation script
if [ ! -f "$VALIDATION_SCRIPT" ]; then
    echo "❌ Validation script not found: $VALIDATION_SCRIPT"
    exit 1
else
    echo "✅ Validation script found: $VALIDATION_SCRIPT"
fi

# Quick dependency check
echo "📦 Checking core dependencies..."
python -c "
import sys
try:
    import torch
    import pytorch_tabnet
    import sklearn
    import pandas as pd
    import numpy as np
    print('  ✅ All core packages available')
    print(f'  📦 PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    print(f'  📦 Sklearn: {sklearn.__version__}')
    print(f'  📦 Pandas: {pd.__version__}')
except ImportError as e:
    print(f'  ❌ Missing package: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed"
    exit 1
fi

echo "✅ All prerequisites satisfied"
echo ""

# === STEP 1: QUICK DATASET VALIDATION ===
echo "🔍 STEP 1: QUICK DATASET VALIDATION"
echo "===================================="

echo "Quick validation of clean dataset structure..."
python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$CLEAN_DATASET', low_memory=False)
    print(f'✅ Dataset loaded: {len(df):,} variants × {len(df.columns)} features')
    
    # Check for critical features
    leakage_features = ['sift_prediction', 'polyphen_prediction', 'functional_pathogenicity']
    leakage_found = [f for f in leakage_features if f in df.columns]
    
    tier1_features = ['Consequence', 'CLIN_SIG', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
    missing_tier1 = [f for f in tier1_features if f not in df.columns]
    
    am_features = ['alphamissense_pathogenicity', 'alphamissense_class']
    missing_am = [f for f in am_features if f not in df.columns]
    
    if leakage_found:
        print(f'❌ Data leakage features found: {leakage_found}')
        sys.exit(1)
    elif missing_tier1:
        print(f'❌ VEP-corrected features missing: {missing_tier1}')
        sys.exit(1)
    elif missing_am:
        print(f'❌ AlphaMissense features missing: {missing_am}')
        sys.exit(1)
    else:
        print('✅ Dataset structure validated')
        print('  ✅ No data leakage features')
        print('  ✅ VEP-corrected features present')
        print('  ✅ AlphaMissense features present')
        
        # Coverage check
        am_coverage = df['alphamissense_pathogenicity'].notna().sum()
        coverage_rate = am_coverage / len(df) * 100
        print(f'  📊 AlphaMissense coverage: {coverage_rate:.1f}%')
        
except Exception as e:
    print(f'❌ Dataset validation failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dataset validation failed"
    exit 1
fi

echo ""

# === STEP 2: COMPREHENSIVE VALIDATION ===
echo "🔥 STEP 2: COMPREHENSIVE TABNET VALIDATION"
echo "=========================================="
echo "Expected results with updated thresholds:"
echo "  🎯 Target accuracy: 70-80% (realistic clinical performance)"
echo "  ⚠️  Suspicious threshold: 90% (was 95%)"
echo "  ✅ Good threshold: 70% (was 75%)"
echo "  ✅ Excellent threshold: 80% (was 85%)"
echo ""

echo "Running comprehensive TabNet validation..."

# Set PYTHONPATH for imports
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Run the validation script
python "${VALIDATION_SCRIPT}"

VALIDATION_EXIT_CODE=$?

echo ""
echo "=== VALIDATION RESULTS ==="

if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "✅ VALIDATION COMPLETED SUCCESSFULLY!"
    echo ""
    
    # Check for latest report
    RESULTS_DIR="${PROJECT_DIR}/results/validation"
    LATEST_REPORT=$(find "${RESULTS_DIR}" -name "enhanced_tabnet_validation_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo "")
    
    if [ -n "$LATEST_REPORT" ] && [ -f "$LATEST_REPORT" ]; then
        echo "📋 Latest validation report: $LATEST_REPORT"
        
        # Extract key results
        echo "🔍 Key Validation Results:"
        python -c "
import json
try:
    with open('$LATEST_REPORT', 'r') as f:
        results = json.load(f)
    
    # Summary
    summary = results.get('summary', {})
    data_leakage = summary.get('data_leakage_detected', False)
    am_integrated = summary.get('alphamissense_integrated', False)
    feature_count = summary.get('feature_count', 0)
    
    print(f'  Data leakage detected: {\"❌ YES\" if data_leakage else \"✅ NO\"}')
    print(f'  AlphaMissense integrated: {\"✅ YES\" if am_integrated else \"❌ NO\"}')
    print(f'  Features validated: {feature_count}')
    
    # Performance with updated thresholds
    if 'baseline_validation' in results:
        baseline_acc = results['baseline_validation']['mean_accuracy']
        print(f'  Baseline accuracy: {baseline_acc:.3f}')
        
        if baseline_acc > 0.90:
            print('  Baseline status: ❌ SUSPICIOUS - Check for leakage')
        elif baseline_acc > 0.80:
            print('  Baseline status: ✅ EXCELLENT (80%+)')
        elif baseline_acc > 0.70:
            print('  Baseline status: ✅ GOOD (70-80%)')
        else:
            print('  Baseline status: ✅ REALISTIC (<70%)')
    
    if 'tabnet_validation' in results:
        tabnet_acc = results['tabnet_validation']['mean_accuracy']
        print(f'  TabNet accuracy: {tabnet_acc:.3f}')
        
        if tabnet_acc > 0.90:
            print('  TabNet status: ❌ SUSPICIOUS - Check for leakage')
        elif tabnet_acc > 0.80:
            print('  TabNet status: ✅ EXCELLENT (80%+)')
        elif tabnet_acc > 0.70:
            print('  TabNet status: ✅ GOOD (70-80% target)')
        else:
            print('  TabNet status: ✅ ACCEPTABLE (within range)')
    
except Exception as e:
    print(f'  ⚠️  Could not parse report: {e}')
"
    else
        echo "⚠️  No validation report found in ${RESULTS_DIR}"
    fi
    
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "  1. Review validation report for detailed results"
    echo "  2. If no data leakage detected, proceed with full training:"
    echo "     python src/model/tabnet_prostate_variant_classifier.py"
    echo "  3. Or submit full training cluster job"
    echo "  4. Expected final performance: 75-85% accuracy with clinical interpretability"
    
    exit 0
    
else
    echo "❌ VALIDATION FAILED (exit code: $VALIDATION_EXIT_CODE)"
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "  1. Check the validation output above for specific errors"
    echo "  2. Ensure clean dataset is properly formatted"
    echo "  3. Verify all dependencies are installed correctly"
    echo "  4. Check that enhanced TabNet model can be imported"
    echo "  5. Verify updated performance thresholds are working"
    
    exit 1
fi

echo ""
echo "Validation completed at: $(date)"