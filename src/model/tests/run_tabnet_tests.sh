#!/bin/bash
# ================================================================
# TabNet Test Runner - Enhanced 57-Feature Version
# Properly sets up HPC conda environment and runs tests
# Updated for VEP-corrected annotations with 70-80% accuracy targets
# ================================================================

set -euo pipefail

echo "🧪 TABNET TEST RUNNER - ENHANCED 57-FEATURE VERSION"
echo "===================================================="
echo "Setting up HPC environment and running tests for VEP-corrected annotations..."
echo ""

# === CONFIGURATION ===
PROJECT_DIR="/u/aa107/uiuc-cancer-research"
TEST_SCRIPT="${PROJECT_DIR}/src/model/tests/test_environment.py"
CONDA_ENV="tabnet-prostate"

# === ENVIRONMENT SETUP ===
echo "🔧 ENVIRONMENT SETUP"
echo "-------------------"

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

# Initialize conda for shell script (required for compute nodes)
echo "Initializing conda for bash..."
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "✅ Conda initialized from $(conda info --base)"
else
    # Alternative initialization method
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        echo "❌ Could not initialize conda"
        echo "💡 Make sure conda is installed and in PATH"
        exit 1
    }
fi

# Check if environment exists, create if needed
echo "Checking conda environment: ${CONDA_ENV}..."
if conda env list | grep -q "${CONDA_ENV}"; then
    echo "✅ Found ${CONDA_ENV} environment"
else
    echo "🔨 Creating ${CONDA_ENV} environment..."
    conda create -n "${CONDA_ENV}" python=3.11 -y
    echo "✅ Environment created"
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

# === DEPENDENCY INSTALLATION ===
echo "📦 INSTALLING/CHECKING DEPENDENCIES"
echo "----------------------------------"

# Update pip first
echo "Updating pip..."
python -m pip install --upgrade pip --quiet

# Check and install PyTorch with CUDA support
echo "Checking PyTorch..."
if python -c "import torch; print(f'  ✅ PyTorch: {torch.__version__}')" 2>/dev/null; then
    echo "  ✅ PyTorch already installed"
else
    echo "  🔨 Installing PyTorch with CUDA support..."
    # Use conda for PyTorch to ensure CUDA compatibility
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y 2>/dev/null || {
        echo "  ⚠️  Conda install failed, trying pip..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
fi

# Check and install TabNet
echo "Checking TabNet..."
if python -c "import pytorch_tabnet; print('  ✅ TabNet: OK')" 2>/dev/null; then
    echo "  ✅ TabNet already installed"
else
    echo "  🔨 Installing TabNet..."
    pip install pytorch-tabnet
fi

# Check and install scikit-learn
echo "Checking scikit-learn..."
if python -c "import sklearn; print(f'  ✅ Sklearn: {sklearn.__version__}')" 2>/dev/null; then
    echo "  ✅ Sklearn already installed"
else
    echo "  🔨 Installing scikit-learn..."
    pip install scikit-learn
fi

# Check and install pandas/numpy
echo "Checking pandas/numpy..."
if python -c "import pandas as pd, numpy as np; print(f'  ✅ Pandas: {pd.__version__}, NumPy: {np.__version__}')" 2>/dev/null; then
    echo "  ✅ Pandas/NumPy already installed"
else
    echo "  🔨 Installing pandas/numpy..."
    pip install pandas numpy
fi

# Additional dependencies for enhanced TabNet
echo "Installing additional dependencies..."
pip install matplotlib seaborn pathlib --quiet

echo "✅ All dependencies installed/verified"
echo ""

# === VERIFY ENVIRONMENT ===
echo "🔍 FINAL ENVIRONMENT VERIFICATION"
echo "================================"

python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print(f'Python path:')
for path in sys.path[:3]:
    print(f'  {path}')

try:
    import torch
    import pytorch_tabnet
    import sklearn
    import pandas as pd
    import numpy as np
    print()
    print('✅ Core packages verification:')
    print(f'  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    print(f'  TabNet: OK')
    print(f'  Sklearn: {sklearn.__version__}')
    print(f'  Pandas: {pd.__version__}')
    print(f'  NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Environment verification failed"
    exit 1
fi

echo ""

# === RUN ENHANCED TESTS ===
echo "🧪 RUNNING ENHANCED TABNET TESTS (57 FEATURES)"
echo "=============================================="
echo "Expected results:"
echo "  • Feature count: ~57 (not 24)"
echo "  • Feature groups: 8 tiers"
echo "  • Target accuracy: 70-80% (not 35%)"
echo "  • VEP-corrected annotations validated"
echo ""

if [ -f "$TEST_SCRIPT" ]; then
    echo "Running: $TEST_SCRIPT"
    echo ""
    
    # Set PYTHONPATH to include src directory for imports
    export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
    
    # Run the enhanced test script
    python "$TEST_SCRIPT"
    
    TEST_EXIT_CODE=$?
    
    echo ""
    echo "=== ENHANCED TEST RESULTS ==="
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "✅ ALL ENHANCED TESTS PASSED!"
        echo "🎯 57-feature TabNet environment ready for training"
        echo ""
        echo "📊 Verified capabilities:"
        echo "  ✅ 57 VEP-corrected features properly selected"
        echo "  ✅ 8-tier feature organization working"
        echo "  ✅ VEP severity encoding tables functional"
        echo "  ✅ AlphaMissense integration validated"
        echo "  ✅ No data leakage detected"
        echo "  ✅ Realistic 70-80% accuracy expectations"
        echo ""
        echo "🚀 Next steps:"
        echo "  1. Run comprehensive validation:"
        echo "     python src/model/validation/validate_tabnet.py"
        echo "  2. Or submit cluster validation job:"
        echo "     sbatch src/model/validation/validate_tabnet.sbatch"
        echo "  3. Start full TabNet training:"
        echo "     python src/model/tabnet_prostate_variant_classifier.py"
        echo ""
        echo "🎯 Expected final performance: 75-85% accuracy with clinical interpretability"
        exit 0
    else
        echo "❌ SOME ENHANCED TESTS FAILED (exit code: $TEST_EXIT_CODE)"
        echo "🔧 Check the test output above for specific issues"
        echo ""
        echo "🛠️  Common troubleshooting:"
        echo "  • Dataset issues:"
        echo "    - Ensure clean dataset exists: ${PROJECT_DIR}/data/processed/tabnet_csv/prostate_variants_tabnet_clean.csv"
        echo "    - Check AlphaMissense features are present"
        echo "    - Verify no data leakage features remain"
        echo ""
        echo "  • Environment issues:"
        echo "    - Rerun: conda activate ${CONDA_ENV}"
        echo "    - Check: python -c 'import pytorch_tabnet, sklearn'"
        echo "    - Verify: echo \$PYTHONPATH includes ${PROJECT_DIR}/src"
        echo ""
        echo "  • Import issues:"
        echo "    - Check: tabnet_prostate_variant_classifier.py exists in src/model/"
        echo "    - Verify: Python can import from src.model.tabnet_prostate_variant_classifier"
        echo ""
        echo "  • Feature count issues:"
        echo "    - Expected ~57 features (45-65 range acceptable)"
        echo "    - Must have 8 feature groups populated"
        echo "    - VEP-corrected features must be present"
        echo ""
        exit 1
    fi
    
else
    echo "❌ Enhanced test script not found: $TEST_SCRIPT"
    echo "💡 File structure issue - check project organization"
    echo ""
    echo "Expected project structure:"
    echo "  ${PROJECT_DIR}/"
    echo "  ├── src/"
    echo "  │   └── model/"
    echo "  │       ├── tabnet_prostate_variant_classifier.py"
    echo "  │       └── tests/"
    echo "  │           └── test_environment.py"
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
echo "Test run completed at: $(date)"