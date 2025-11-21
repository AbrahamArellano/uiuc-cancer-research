#!/bin/bash
################################################################################
# UIUC Cancer Research - Full Pipeline Execution Script
################################################################################
# Prostate-VarBench: Interpretable Deep Learning for Prostate Cancer
# Variant Classification
#
# This script executes the complete end-to-end pipeline from data download
# to model training and analysis.
#
# Usage:
#   bash scripts/run_pipeline.sh                    # Run full pipeline
#   bash scripts/run_pipeline.sh --skip-download    # Skip data download
#   bash scripts/run_pipeline.sh --help             # Show help
#
# Author: UIUC Cancer Research Team
# License: MIT
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "📁 Loading environment variables from .env"
    set -a
    source .env
    set +a
else
    echo "⚠️  Warning: .env file not found. Using defaults."
    echo "   Copy .env.example to .env and configure it."
fi

# Export PROJECT_ROOT
export PROJECT_ROOT

# Default settings
SKIP_DOWNLOAD=false
SKIP_VEP=false
SKIP_OPTIMIZATION=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "$(printf '=%.0s' {1..70})"
    echo "$1"
    echo "$(printf '=%.0s' {1..70})"
    echo ""
}

print_step() {
    echo -e "${BLUE}[$1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "Required command not found: $1"
        echo "Please install $1 and try again."
        exit 1
    fi
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "Required file not found: $1"
        return 1
    fi
    return 0
}

# ============================================================================
# USAGE & HELP
# ============================================================================

show_help() {
    cat << EOF
UIUC Cancer Research Pipeline - Full Execution Script

Usage: bash run_full_pipeline.sh [OPTIONS]

OPTIONS:
    --skip-download         Skip data download step
    --skip-vep              Skip VEP annotation step (slow)
    --skip-optimization     Skip hyperparameter optimization
    --verbose               Enable verbose output
    -h, --help              Show this help message

PIPELINE STEPS:
    1. Data Download (COSMIC, ClinVar, TCGA-PRAD)
    2. Variant Filtering (COSMIC, ClinVar)
    3. Dataset Merging
    4. VEP Annotation
    5. VCF to CSV Conversion
    6. AlphaMissense Enhancement
    7. TabNet Model Training
    8. Hyperparameter Optimization (optional)
    9. Attention Analysis

REQUIREMENTS:
    - Conda environment: uiuc-cancer-research
    - COSMIC credentials in .env file
    - VEP installed with cache files
    - R with TCGAmutations package (for TCGA download)

EXAMPLES:
    # Run full pipeline
    bash run_full_pipeline.sh

    # Skip data download (already downloaded)
    bash run_full_pipeline.sh --skip-download

    # Skip slow VEP annotation
    bash run_full_pipeline.sh --skip-vep

    # Quick run (skip download and optimization)
    bash run_full_pipeline.sh --skip-download --skip-optimization

For more information, see README.md
EOF
    exit 0
}

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-vep)
            SKIP_VEP=true
            shift
            ;;
        --skip-optimization)
            SKIP_OPTIMIZATION=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

print_header "PROSTATE-VARBENCH PIPELINE - PRE-FLIGHT CHECKS"

# Check Python
check_command python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "uiuc-cancer-research" ]]; then
    print_warning "Conda environment 'uiuc-cancer-research' not active"
    echo "Activate with: conda activate uiuc-cancer-research"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "Conda environment: $CONDA_DEFAULT_ENV"
fi

# Check R for TCGA download
if [ "$SKIP_DOWNLOAD" = false ]; then
    if command -v Rscript &> /dev/null; then
        R_VERSION=$(Rscript --version 2>&1 | head -n1)
        print_success "R available: $R_VERSION"
    else
        print_warning "R not found - TCGA download will fail"
        print_warning "Install R or use --skip-download if data already exists"
    fi
fi

# Check VEP
if [ "$SKIP_VEP" = false ]; then
    VEP_PATH="${VEP_CACHE_DIR:-$HOME/.vep}"
    if [ -d "$VEP_PATH" ]; then
        print_success "VEP cache found: $VEP_PATH"
    else
        print_warning "VEP cache not found at $VEP_PATH"
        print_warning "VEP annotation may fail. See README.md for VEP setup."
    fi
fi

# Check credentials
if [ "$SKIP_DOWNLOAD" = false ]; then
    if [ -z "${COSMIC_EMAIL:-}" ] || [ -z "${COSMIC_PASSWORD:-}" ]; then
        print_error "COSMIC credentials not set in .env file"
        echo "Please configure COSMIC_EMAIL and COSMIC_PASSWORD"
        exit 1
    else
        print_success "COSMIC credentials configured"
    fi
fi

# Create required directories
mkdir -p data/raw/variants data/processed logs models results

print_success "Pre-flight checks complete!"
echo ""

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

START_TIME=$(date +%s)
print_header "PIPELINE EXECUTION START: $(date '+%Y-%m-%d %H:%M:%S')"

# ----------------------------------------------------------------------------
# STEP 1: DATA DOWNLOAD
# ----------------------------------------------------------------------------

if [ "$SKIP_DOWNLOAD" = false ]; then
    print_header "STEP 1/9: DATA DOWNLOAD"

    print_step "1.1" "Downloading COSMIC Mutant Census v102"
    python scripts/download/download_cosmic.py \
        --email "$COSMIC_EMAIL" \
        --password "$COSMIC_PASSWORD" \
        --version v102 \
        --assembly GRCh38 \
        || { print_error "COSMIC download failed"; exit 1; }

    print_step "1.2" "Downloading ClinVar (GRCh38)"
    python scripts/download/download_clinvar.py \
        --assembly GRCh38 \
        || { print_error "ClinVar download failed"; exit 1; }

    print_step "1.3" "Downloading TCGA-PRAD via R/TCGAmutations"
    if command -v Rscript &> /dev/null; then
        Rscript scripts/variants/download_tcga.R \
            || { print_error "TCGA-PRAD download failed"; exit 1; }
    else
        print_error "R not available for TCGA download"
        exit 1
    fi

    print_success "Data download complete!"
else
    print_warning "Skipping data download (--skip-download)"
fi

# ----------------------------------------------------------------------------
# STEP 2: VARIANT FILTERING
# ----------------------------------------------------------------------------

print_header "STEP 2/9: VARIANT FILTERING"

print_step "2.1" "Filtering COSMIC for prostate-specific variants"
if check_file "scripts/variants/filter_cosmic_prostate.py"; then
    python scripts/variants/filter_cosmic_prostate.py \
        || { print_error "COSMIC filtering failed"; exit 1; }
    print_success "COSMIC filtering complete"
else
    print_warning "COSMIC filter script not found, skipping..."
fi

print_step "2.2" "Filtering ClinVar for prostate cancer genes"
if check_file "scripts/variants/filter_clinvar_prostate.py"; then
    python scripts/variants/filter_clinvar_prostate.py \
        || { print_error "ClinVar filtering failed"; exit 1; }
    print_success "ClinVar filtering complete"
else
    print_warning "ClinVar filter script not found, skipping..."
fi

# ----------------------------------------------------------------------------
# STEP 3: DATASET MERGING
# ----------------------------------------------------------------------------

print_header "STEP 3/9: DATASET MERGING"

print_step "3" "Merging COSMIC, ClinVar, and TCGA-PRAD datasets"
if check_file "scripts/merge/merge_datasets.py"; then
    python scripts/merge/merge_datasets.py \
        || { print_error "Dataset merging failed"; exit 1; }
    print_success "Dataset merging complete"
else
    print_error "Merge script not found: scripts/merge/merge_datasets.py"
    exit 1
fi

# ----------------------------------------------------------------------------
# STEP 4: VEP ANNOTATION
# ----------------------------------------------------------------------------

if [ "$SKIP_VEP" = false ]; then
    print_header "STEP 4/9: VEP ANNOTATION"

    print_step "4" "Running Variant Effect Predictor annotation"
    print_warning "This step may take 30-60 minutes..."

    if check_file "scripts/vep/run_vep_annotation.sh"; then
        bash scripts/vep/run_vep_annotation.sh \
            || { print_error "VEP annotation failed"; exit 1; }
        print_success "VEP annotation complete"
    else
        print_error "VEP script not found: scripts/vep/run_vep_annotation.sh"
        exit 1
    fi
else
    print_warning "Skipping VEP annotation (--skip-vep)"
fi

# ----------------------------------------------------------------------------
# STEP 5: VCF TO CSV CONVERSION
# ----------------------------------------------------------------------------

print_header "STEP 5/9: VCF TO CSV CONVERSION"

print_step "5" "Converting VEP-annotated VCF to TabNet CSV format"
if check_file "scripts/enhance/tabnet_vcf_to_csv/vcf_to_tabnet_csv.py"; then
    python scripts/enhance/tabnet_vcf_to_csv/vcf_to_tabnet_csv.py \
        || { print_error "VCF to CSV conversion failed"; exit 1; }
    print_success "VCF to CSV conversion complete"
else
    print_error "VCF conversion script not found"
    exit 1
fi

# ----------------------------------------------------------------------------
# STEP 6: ALPHAMISSENSE ENHANCEMENT
# ----------------------------------------------------------------------------

print_header "STEP 6/9: ALPHAMISSENSE ENHANCEMENT"

print_step "6" "Adding AlphaMissense pathogenicity scores"
print_warning "This will download ~2GB of AlphaMissense data if not cached"

if check_file "scripts/enhance/functional_enhancement/simple_functional_imputation.py"; then
    python scripts/enhance/functional_enhancement/simple_functional_imputation.py \
        || { print_error "AlphaMissense enhancement failed"; exit 1; }
    print_success "AlphaMissense enhancement complete"
else
    print_error "AlphaMissense script not found"
    exit 1
fi

# ----------------------------------------------------------------------------
# STEP 7: TABNET MODEL TRAINING
# ----------------------------------------------------------------------------

print_header "STEP 7/9: TABNET MODEL TRAINING"

print_step "7" "Training TabNet classifier (56 features, 8-tier system)"
if check_file "src/model/tabnet_prostate_variant_classifier.py"; then
    python src/model/tabnet_prostate_variant_classifier.py \
        || { print_error "TabNet training failed"; exit 1; }
    print_success "TabNet training complete"
else
    print_error "TabNet classifier not found"
    exit 1
fi

# ----------------------------------------------------------------------------
# STEP 8: HYPERPARAMETER OPTIMIZATION (OPTIONAL)
# ----------------------------------------------------------------------------

if [ "$SKIP_OPTIMIZATION" = false ]; then
    print_header "STEP 8/9: HYPERPARAMETER OPTIMIZATION"

    print_step "8" "Running hyperparameter grid search"
    print_warning "This may take up to 36 GPU hours (configurable)"

    if check_file "scripts/optimization/hyperparameter_optimization.py"; then
        python scripts/optimization/hyperparameter_optimization.py \
            || { print_warning "Hyperparameter optimization failed (non-critical)"; }
        print_success "Hyperparameter optimization complete"
    else
        print_warning "Optimization script not found, skipping..."
    fi
else
    print_warning "Skipping hyperparameter optimization (--skip-optimization)"
fi

# ----------------------------------------------------------------------------
# STEP 9: ATTENTION ANALYSIS
# ----------------------------------------------------------------------------

print_header "STEP 9/9: ATTENTION ANALYSIS"

print_step "9.1" "Extracting attention weights from TabNet model"
if check_file "src/analysis/attention_extractor.py"; then
    python src/analysis/attention_extractor.py \
        || { print_warning "Attention extraction failed (non-critical)"; }
else
    print_warning "Attention extractor not found, skipping..."
fi

print_step "9.2" "Analyzing attention patterns"
if check_file "src/analysis/attention_analyzer.py"; then
    python src/analysis/attention_analyzer.py \
        || { print_warning "Attention analysis failed (non-critical)"; }
else
    print_warning "Attention analyzer not found, skipping..."
fi

print_step "9.3" "Generating results reports"
if check_file "src/analysis/results_generator.py"; then
    python src/analysis/results_generator.py \
        || { print_warning "Results generation failed (non-critical)"; }
else
    print_warning "Results generator not found, skipping..."
fi

# ============================================================================
# PIPELINE COMPLETION
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

print_header "PIPELINE EXECUTION COMPLETE!"

echo "Execution time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results locations:"
echo "  - Trained models:    models/"
echo "  - Training results:  results/training/"
echo "  - Attention analysis: results/attention_analysis/"
echo "  - Performance metrics: results/metrics/"
echo ""
print_success "✓ All pipeline steps completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review model performance in results/training/"
echo "  2. Examine attention patterns in results/attention_analysis/"
echo "  3. Check feature importance rankings"
echo "  4. Validate on external test sets (if available)"
echo ""
echo "For detailed documentation, see README.md"
echo ""

exit 0
