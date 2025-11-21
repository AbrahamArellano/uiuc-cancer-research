#!/bin/bash
################################################################################
# Data Download Only Script - Prostate-VarBench
################################################################################
# Downloads all required datasets:
#   1. COSMIC Mutant Census v102 (SFTP)
#   2. ClinVar variants (FTP)
#   3. TCGA-PRAD mutations (R/TCGAmutations)
#
# Usage:
#   bash scripts/download/download_all.sh
#
# Prerequisites:
#   - .env file with COSMIC credentials
#   - R installed with TCGAmutations package
#   - Internet connection
#
# Author: UIUC Cancer Research Team
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
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

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

print_header "PROSTATE-VARBENCH - DATA DOWNLOAD PRE-FLIGHT CHECKS"

# Check for .env file
if [ -f ".env" ]; then
    print_success ".env file found"
    set -a
    source .env
    set +a
else
    print_error ".env file not found"
    echo ""
    echo "Please create .env file with COSMIC credentials:"
    echo "  cp .env.example .env"
    echo "  nano .env  # Edit with your credentials"
    echo ""
    exit 1
fi

# Check COSMIC credentials
if [ -z "${COSMIC_EMAIL:-}" ] || [ -z "${COSMIC_PASSWORD:-}" ]; then
    print_error "COSMIC credentials not set in .env"
    echo ""
    echo "Required variables in .env:"
    echo "  COSMIC_EMAIL=your@email.com"
    echo "  COSMIC_PASSWORD=your_token"
    echo ""
    echo "Register at: https://cancer.sanger.ac.uk/cosmic/register"
    exit 1
else
    print_success "COSMIC credentials configured"
    print_info "Email: $COSMIC_EMAIL"
fi

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    print_success "Python available: $PYTHON_VERSION"
else
    print_error "Python not found"
    echo "Please install Python 3.10+ or activate conda environment"
    exit 1
fi

# Check R
if command -v Rscript &> /dev/null; then
    R_VERSION=$(Rscript --version 2>&1 | head -n1)
    print_success "R available: $R_VERSION"
else
    print_warning "R not found - TCGA download will fail"
    echo ""
    read -p "Continue without R? (TCGA download will be skipped) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check internet connectivity
print_step "NET" "Checking internet connectivity..."
if ping -c 1 google.com &> /dev/null || ping -c 1 8.8.8.8 &> /dev/null; then
    print_success "Internet connection available"
else
    print_error "No internet connection detected"
    exit 1
fi

# Create output directories
mkdir -p data/raw/variants
print_success "Output directories created"

# Check disk space
AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available"
    print_warning "Recommended: 10GB+ free space"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "Sufficient disk space: ${AVAILABLE_SPACE}GB available"
fi

echo ""
print_success "Pre-flight checks complete!"

# ============================================================================
# DOWNLOAD SUMMARY
# ============================================================================

print_header "DOWNLOAD PLAN"

echo "The following datasets will be downloaded:"
echo ""
echo "┌────────────────────────────────────────────────────────────────┐"
echo "│ Dataset         │ Size      │ Time      │ Method             │"
echo "├────────────────────────────────────────────────────────────────┤"
echo "│ COSMIC v102     │ ~88 MB    │ 2-5 min   │ HTTP API (Python)  │"
echo "│ ClinVar GRCh38  │ ~500 MB   │ 5-15 min  │ FTP (Python)       │"
echo "│ TCGA-PRAD MC3   │ ~50 MB    │ 5-10 min  │ R/TCGAmutations    │"
echo "├────────────────────────────────────────────────────────────────┤"
echo "│ TOTAL           │ ~640 MB   │ 12-30 min │                    │"
echo "└────────────────────────────────────────────────────────────────┘"
echo ""
echo "Output directory: data/raw/variants/"
echo ""

read -p "Proceed with download? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_warning "Download cancelled by user"
    exit 0
fi

# ============================================================================
# START DOWNLOADS
# ============================================================================

START_TIME=$(date +%s)
print_header "STARTING DOWNLOADS - $(date '+%Y-%m-%d %H:%M:%S')"

DOWNLOAD_SUCCESS=0
DOWNLOAD_FAILED=0

# ============================================================================
# DOWNLOAD 1: COSMIC
# ============================================================================

print_header "DOWNLOAD 1/3: COSMIC MUTANT CENSUS v102"

print_info "Source: Catalogue of Somatic Mutations in Cancer"
print_info "Version: v102 (GRCh38)"
print_info "Method: HTTP API via cancer.sanger.ac.uk/api/mono/products"
print_info "Size: ~88 MB tar archive, ~512 MB uncompressed TSV"
echo ""

COSMIC_START=$(date +%s)

if python scripts/download/download_cosmic_http.py \
    --email "$COSMIC_EMAIL" \
    --password "$COSMIC_PASSWORD" \
    --version v102 \
    --assembly GRCh38 \
    --output data/raw/variants; then

    COSMIC_END=$(date +%s)
    COSMIC_TIME=$((COSMIC_END - COSMIC_START))
    COSMIC_MIN=$((COSMIC_TIME / 60))
    COSMIC_SEC=$((COSMIC_TIME % 60))

    print_success "COSMIC download complete! (${COSMIC_MIN}m ${COSMIC_SEC}s)"
    ((DOWNLOAD_SUCCESS++))

    # Check output file: Cosmic_MutantCensus_v102_GRCh38.tsv.gz
    if ls data/raw/variants/Cosmic_MutantCensus*.tsv* >/dev/null 2>&1; then
        FILE_SIZE=$(du -h data/raw/variants/Cosmic_MutantCensus*.tsv* | awk '{print $1}' | head -1)
        print_info "File size: $FILE_SIZE"
    fi
else
    print_error "COSMIC download failed!"
    print_warning "Common issues:"
    echo "  - Invalid credentials (check COSMIC_EMAIL and COSMIC_PASSWORD)"
    echo "  - Network connectivity problems"
    echo "  - COSMIC server maintenance"
    echo ""
    echo "Verify credentials at: https://cancer.sanger.ac.uk/cosmic/register"
    ((DOWNLOAD_FAILED++))
fi

# ============================================================================
# DOWNLOAD 2: CLINVAR
# ============================================================================

print_header "DOWNLOAD 2/3: CLINVAR VARIANTS (GRCh38)"

print_info "Source: NCBI ClinVar Database"
print_info "Assembly: GRCh38"
print_info "Method: FTP via ftp.ncbi.nlm.nih.gov"
print_info "Size: ~500 MB compressed, ~5 GB uncompressed"
echo ""

CLINVAR_START=$(date +%s)

if python scripts/download/download_clinvar.py \
    --assembly GRCh38 \
    --output data/raw/variants; then

    CLINVAR_END=$(date +%s)
    CLINVAR_TIME=$((CLINVAR_END - CLINVAR_START))
    CLINVAR_MIN=$((CLINVAR_TIME / 60))
    CLINVAR_SEC=$((CLINVAR_TIME % 60))

    print_success "ClinVar download complete! (${CLINVAR_MIN}m ${CLINVAR_SEC}s)"
    ((DOWNLOAD_SUCCESS++))

    # Check output files
    if [ -f "data/raw/variants/clinvar_GRCh38.vcf" ] || \
       [ -f "data/raw/variants/clinvar_GRCh38.vcf.gz" ]; then
        FILE_SIZE=$(du -h data/raw/variants/clinvar_GRCh38.vcf* | awk '{print $1}')
        print_info "File size: $FILE_SIZE"
    fi
else
    print_error "ClinVar download failed!"
    print_warning "Common issues:"
    echo "  - Network connectivity problems"
    echo "  - NCBI FTP server issues"
    echo "  - Disk space insufficient"
    ((DOWNLOAD_FAILED++))
fi

# ============================================================================
# DOWNLOAD 3: TCGA-PRAD
# ============================================================================

print_header "DOWNLOAD 3/3: TCGA-PRAD MUTATIONS (MC3)"

print_info "Source: The Cancer Genome Atlas - Prostate Adenocarcinoma"
print_info "Method: R package (TCGAmutations)"
print_info "Dataset: MC3 consensus mutations"
print_info "Size: ~50 MB"
echo ""

if ! command -v Rscript &> /dev/null; then
    print_warning "R not available - skipping TCGA download"
    print_info "Install R to download TCGA-PRAD data"
else
    TCGA_START=$(date +%s)

    if Rscript scripts/variants/download_tcga.R; then
        TCGA_END=$(date +%s)
        TCGA_TIME=$((TCGA_END - TCGA_START))
        TCGA_MIN=$((TCGA_TIME / 60))
        TCGA_SEC=$((TCGA_TIME % 60))

        print_success "TCGA-PRAD download complete! (${TCGA_MIN}m ${TCGA_SEC}s)"
        ((DOWNLOAD_SUCCESS++))

        # Check output file
        if [ -f "data/raw/variants/tcga_prad_mutations.csv" ]; then
            FILE_SIZE=$(du -h data/raw/variants/tcga_prad_mutations.csv | awk '{print $1}')
            VARIANT_COUNT=$(tail -n +2 data/raw/variants/tcga_prad_mutations.csv | wc -l)
            print_info "File size: $FILE_SIZE"
            print_info "Variants: $VARIANT_COUNT"
        fi
    else
        print_error "TCGA-PRAD download failed!"
        print_warning "Common issues:"
        echo "  - TCGAmutations package not installed"
        echo "  - Network connectivity problems"
        echo "  - R library path issues"
        echo ""
        echo "Install TCGAmutations:"
        echo "  Rscript -e 'devtools::install_github(\"PoisonAlien/TCGAmutations\")'"
        ((DOWNLOAD_FAILED++))
    fi
fi

# ============================================================================
# DOWNLOAD SUMMARY
# ============================================================================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_SEC=$((TOTAL_TIME % 60))

print_header "DOWNLOAD SUMMARY"

echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total duration: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""

# Calculate total downloads attempted
TOTAL_ATTEMPTED=$((DOWNLOAD_SUCCESS + DOWNLOAD_FAILED))

echo "Download Results:"
echo "┌──────────────────────────────────────┐"
printf "│ %-20s │ %15s │\n" "Successful" "$DOWNLOAD_SUCCESS / $TOTAL_ATTEMPTED"
printf "│ %-20s │ %15s │\n" "Failed" "$DOWNLOAD_FAILED / $TOTAL_ATTEMPTED"
echo "└──────────────────────────────────────┘"
echo ""

# List downloaded files
echo "Downloaded files in data/raw/variants/:"
if [ -d "data/raw/variants" ]; then
    ls -lh data/raw/variants/ | grep -E "\.(tsv|vcf|csv)" || echo "  (no files found)"
fi
echo ""

# Calculate total size
if [ -d "data/raw/variants" ]; then
    TOTAL_SIZE=$(du -sh data/raw/variants/ | awk '{print $1}')
    print_info "Total data downloaded: $TOTAL_SIZE"
fi

echo ""

# Final status
if [ $DOWNLOAD_FAILED -eq 0 ]; then
    print_success "✓ ALL DOWNLOADS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify downloaded files in data/raw/variants/"
    echo "  2. Run full pipeline: bash scripts/run_pipeline.sh"
    echo "  3. Or continue with manual processing steps"
    echo ""
    exit 0
else
    print_warning "⚠ SOME DOWNLOADS FAILED"
    echo ""
    echo "Failed downloads: $DOWNLOAD_FAILED"
    echo "Please review errors above and retry failed downloads."
    echo ""
    echo "To retry individual downloads:"
    echo "  COSMIC:    python scripts/download/download_cosmic.py --email ... --password ..."
    echo "  ClinVar:   python scripts/download/download_clinvar.py"
    echo "  TCGA-PRAD: Rscript scripts/variants/download_tcga.R"
    echo ""
    exit 1
fi
