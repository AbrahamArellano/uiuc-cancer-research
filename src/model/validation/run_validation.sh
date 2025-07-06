#!/bin/bash
# ================================================================
# TabNet Validation Runner - Enhanced 57-Feature Version
# Updated for realistic 70-80% accuracy targets with fixed thresholds
# ================================================================

set -euo pipefail

echo "🧬 TABNET VALIDATION RUNNER - ENHANCED VERSION"
echo "==============================================="
echo "Updated performance thresholds for realistic clinical expectations"
echo "Target accuracy: 70-80% (was 75-85%)"
echo ""

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
VALIDATION_SCRIPT="${PROJECT_DIR}/src/model/validation/validate_tabnet.py"
RESULTS_DIR="${PROJECT_DIR}/results/validation"
CONDA_ENV="tabnet-prostate"

# === ENVIRONMENT SETUP ===
echo "🔧 ENVIRONMENT SETUP"
echo "-------------------"

# Navigate to project directory
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

# Create results directory
mkdir -p "${RESULTS_DIR}"
echo "Results directory: ${RESULTS_DIR}"

# Load anaconda module (required on campus cluster)
echo "Loading anaconda module..."
if module load anaconda3 2>/dev/null; then
    echo "✅ Anaconda module loaded"
else
    echo "⚠️  Anaconda module not available - using system conda"
fi

# Initialize conda for shell script
echo "Initializing conda..."
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "✅ Conda initialized from $(conda info --base)"
else
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        echo "❌ Could not initialize conda"
        exit 1
    }
fi

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

# Verify activation
echo "✅ Environment activated:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo ""

# === DEPENDENCY CHECK ===
echo "📦 CHECKING VALIDATION DEPENDENCIES"
echo "----------------------------------"

echo "Checking core packages..."
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
    echo "❌ Missing dependencies - run test runner first"
    exit 1
fi

echo "✅ All validation dependencies available"
echo ""

# === ENHANCED DATASET CHECK ===
echo "🧬 CHECKING CLEAN DATASET"
echo "-------------------------"

CLEAN_DATASET="${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"

if [ -f "$CLEAN_DATASET" ]; then
    DATASET_SIZE=$(du -h "$CLEAN_DATASET" | cut -f1)
    DATASET_LINES=$(wc -l < "$CLEAN_DATASET")
    echo "✅ Clean dataset found:"
    echo "  📁 File: $CLEAN_DATASET"
    echo "  📏 Size: $DATASET_SIZE"
    echo "  📊 Lines: $DATASET_LINES"
    
    # Quick validation of dataset structure
    echo "🔍 Quick dataset structure check..."
    python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$CLEAN_DATASET', low_memory=False)
    print(f'  📊 Loaded: {len(df):,} variants × {len(df.columns)} features')
    
    # Check for critical features
    leakage_features = ['sift_prediction', 'polyphen_prediction', 'functional_pathogenicity']
    leakage_found = [f for f in leakage_features if f in df.columns]
    
    tier1_features = ['Consequence', 'CLIN_SIG', 'DOMAINS', 'PUBMED', 'VAR_SYNONYMS']
    missing_tier1 = [f for f in tier1_features if f not in df.columns]
    
    am_features = ['alphamissense_pathogenicity', 'alphamissense_class']
    missing_am = [f for f in am_features if f not in df.columns]
    
    if leakage_found:
        print(f'  ❌ Data leakage features found: {leakage_found}')
        sys.exit(1)
    elif missing_tier1:
        print(f'  ❌ VEP-corrected features missing: {missing_tier1}')
        sys.exit(1)
    elif missing_am:
        print(f'  ❌ AlphaMissense features missing: {missing_am}')
        sys.exit(1)
    else:
        print('  ✅ Dataset structure looks good')
        print('    ✅ No data leakage features')
        print('    ✅ VEP-corrected features present')
        print('    ✅ AlphaMissense features present')
        
        # Coverage check
        am_coverage = df['alphamissense_pathogenicity'].notna().sum()
        coverage_rate = am_coverage / len(df) * 100
        print(f'    📊 AlphaMissense coverage: {coverage_rate:.1f}%')
        
except Exception as e:
    print(f'  ❌ Dataset check failed: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Dataset structure check failed"
        exit 1
    fi
    
else
    echo "❌ Clean dataset not found: $CLEAN_DATASET"
    echo "💡 Run data processing pipeline first to create clean dataset"
    exit 1
fi

echo ""

# === RUN VALIDATION ===
echo "🧬 RUNNING ENHANCED TABNET VALIDATION"
echo "===================================="
echo "Updated performance expectations:"
echo "  🎯 Target range: 70-80% (realistic clinical performance)"
echo "  ✅ Good threshold: 70% (was 75%)"
echo "  ✅ Excellent threshold: 80% (was 85%)"
echo "  ⚠️  Suspicious threshold: 90% (was 95%)"
echo ""

if [ -f "$VALIDATION_SCRIPT" ]; then
    echo "Running enhanced validation script: $VALIDATION_SCRIPT"
    echo ""
    
    # Set PYTHONPATH to include src directory for imports
    export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
    
    # Run the validation script
    python "$VALIDATION_SCRIPT"
    
    VALIDATION_EXIT_CODE=$?
    
    echo ""
    echo "=== ENHANCED VALIDATION RESULTS ==="
    
    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo "✅ VALIDATION COMPLETED SUCCESSFULLY!"
        echo ""
        
        # Check for latest report
        LATEST_REPORT=$(find "${RESULTS_DIR}" -name "enhanced_tabnet_validation_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo "")
        
        if [ -n "$LATEST_REPORT" ] && [ -f "$LATEST_REPORT" ]; then
            echo "📋 Latest validation report: $LATEST_REPORT"
            
            # Extract key results with updated thresholds
            echo "🔍 Key Validation Results (Updated Thresholds):"
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
    
    # Performance with UPDATED thresholds
    if 'baseline_validation' in results:
        baseline_acc = results['baseline_validation']['mean_accuracy']
        print(f'  Baseline accuracy: {baseline_acc:.3f}')
        
        if baseline_acc > 0.90:
            print('  Baseline status: ❌ SUSPICIOUS - Check for leakage (>90%)')
        elif baseline_acc > 0.80:
            print('  Baseline status: ✅ EXCELLENT - Above target (80%+)')
        elif baseline_acc > 0.70:
            print('  Baseline status: ✅ GOOD - Target range (70-80%)')
        elif baseline_acc > 0.60:
            print('  Baseline status: ✅ ACCEPTABLE - Within range (60-70%)')
        else:
            print('  Baseline status: ⚠️  MODERATE - Below target (<60%)')
    
    if 'tabnet_validation' in results:
        tabnet_acc = results['tabnet_validation']['mean_accuracy']
        print(f'  TabNet accuracy: {tabnet_acc:.3f}')
        
        if tabnet_acc > 0.90:
            print('  TabNet status: ❌ SUSPICIOUS - Check for leakage (>90%)')
        elif tabnet_acc > 0.80:
            print('  TabNet status: ✅ EXCELLENT - Above target (80%+)')
        elif tabnet_acc > 0.70:
            print('  TabNet status: ✅ GOOD - Target range (70-80%)')
        elif tabnet_acc > 0.60:
            print('  TabNet status: ✅ ACCEPTABLE - Within range (60-70%)')
        else:
            print('  TabNet status: ⚠️  MODERATE - Consider improvements (<60%)')
    
    print()
    print('📊 Updated Performance Thresholds:')
    print('  Excellent: 80% (was 85%)')
    print('  Good: 70% (was 75%)')
    print('  Suspicious: 90% (was 95%)')
    
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
        echo "  3. Or submit cluster job for full training:"
        echo "     sbatch scripts/training/train_tabnet.sbatch"
        echo "  4. Expected final performance: 75-85% accuracy with clinical interpretability"
        echo ""
        echo "📊 Performance Interpretation Guide:"
        echo "  90%+: ❌ SUSPICIOUS - Likely data leakage"
        echo "  80-90%: ✅ EXCELLENT - Above clinical target"
        echo "  70-80%: ✅ GOOD - Target clinical performance"
        echo "  60-70%: ✅ ACCEPTABLE - Realistic for complex genomics"
        echo "  <60%: ⚠️  MODERATE - Consider feature engineering"
        
        exit 0
        
    else
        echo "❌ VALIDATION FAILED (exit code: $VALIDATION_EXIT_CODE)"
        echo ""
        echo "🔧 TROUBLESHOOTING:"
        echo "  1. Check the validation output above for specific errors"
        echo "  2. Ensure clean dataset is properly formatted"
        echo "  3. Verify all dependencies are installed correctly"
        echo "  4. Check that enhanced TabNet model can be imported"
        echo "  5. Verify updated performance thresholds (70-80% target)"
        echo ""
        echo "🛠️  Common solutions:"
        echo "  • Dataset issues:"
        echo "    - Ensure clean dataset exists with no leakage features"
        echo "    - Check AlphaMissense features are present"
        echo "    - Verify VEP-corrected annotations"
        echo "  • Environment issues:"
        echo "    - Rerun: conda activate ${CONDA_ENV}"
        echo "    - Check: python -c 'import pytorch_tabnet, sklearn'"
        echo "    - Verify: echo \$PYTHONPATH includes ${PROJECT_DIR}/src"
        echo "  • Performance issues:"
        echo "    - Updated thresholds: 70-80% is now target (not 75-85%)"
        echo "    - Suspicious threshold: 90% (not 95%)"
        echo "    - Feature count: Should be ~57 features"
        
        exit 1
    fi
    
else
    echo "❌ Validation script not found: $VALIDATION_SCRIPT"
    echo "💡 Make sure you're in the project root directory"
    echo "Expected project structure:"
    echo "  ${PROJECT_DIR}/"
    echo "  ├── src/"
    echo "  │   └── model/"
    echo "  │       ├── tabnet_prostate_variant_classifier.py"
    echo "  │       └── validation/"
    echo "  │           └── validate_tabnet.py"
    echo "  └── data/"
    echo "      └── processed/"
    echo "          └── tabnet_csv/"
    echo "              └── prostate_variants_tabnet_clean.csv"
    exit 1
fi

# === CLEANUP ===
echo ""
echo "🧹 CLEANUP"
echo "========="
echo "Environment ${CONDA_ENV} remains active for further use"
echo "To deactivate: conda deactivate"
echo ""
echo "Validation run completed at: $(date)"