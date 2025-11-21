# Prostate-VarBench Setup Guide

Comprehensive setup instructions for the UIUC Cancer Research project.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Environment Setup](#environment-setup)
- [VEP Installation](#vep-installation)
- [R Setup for TCGA Download](#r-setup-for-tcga-download)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Disk: 100 GB free space
- GPU: Optional (CPU training supported)

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- Disk: 200 GB free space (SSD recommended)
- GPU: NVIDIA with 8GB+ VRAM (for faster training)
  - CUDA 11.7+ or 12.1+
  - cuDNN compatible version

**For Hyperparameter Optimization:**
- GPU: NVIDIA A100 or H100
- GPU Memory: 40GB+ recommended
- Time Budget: 36 GPU hours

### Software Requirements

**Required:**
- Linux/Unix-based OS (tested on CentOS 7, Ubuntu 20.04+)
- Python 3.10 or higher
- Conda or Miniconda
- Perl 5.10+ (for VEP)
- R 4.0+ (for TCGA-PRAD download via TCGAmutations)
- Git
- wget or curl

**Optional:**
- SLURM (for cluster computing)
- Docker (for containerized deployment)

---

## Installation Steps

### Step 1: Clone Repository

```bash
# Navigate to your projects directory
cd /path/to/your/projects

# Clone the repository
git clone https://github.com/AbrahamArellano/uiuc-cancer-research.git
cd uiuc-cancer-research

# Verify directory structure
ls -la
```

**Expected output:**
```
config/
data/
docs/
scripts/
src/
tools/
environment.yml
README.md
run_full_pipeline.sh
```

### Step 2: Create Conda Environment

```bash
# Create environment from YAML file
conda env create -f environment.yml

# This will:
# - Create environment named 'uiuc-cancer-research'
# - Install Python 3.10
# - Install all required packages (numpy, pandas, scikit-learn, pytorch, etc.)
# - Install bioinformatics tools (pysam, cyvcf2)
# - Install TabNet and other ML libraries

# Expected time: 10-20 minutes
```

**Activate the environment:**
```bash
conda activate uiuc-cancer-research

# Verify Python version
python --version  # Should be Python 3.10.x

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### Step 3: Install Additional Dependencies

**Install development tools (optional but recommended):**
```bash
conda activate uiuc-cancer-research
pip install pytest black flake8 jupyter
```

---

## Environment Setup

### Step 1: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your settings
nano .env  # or use vim, emacs, etc.
```

### Step 2: Required Configuration

Edit `.env` and set these **required** variables:

```bash
# Project root (use absolute path)
PROJECT_ROOT=/path/to/uiuc-cancer-research

# COSMIC credentials (required for data download)
# Register at: https://cancer.sanger.ac.uk/cosmic/register
COSMIC_EMAIL=your.email@example.com
COSMIC_PASSWORD=your_cosmic_token_here

# VEP cache directory
VEP_CACHE_DIR=/home/yourusername/.vep
```

### Step 3: Optional Configuration

**For GPU acceleration:**
```bash
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (change for multi-GPU systems)
```

**For controlled-access TCGA data:**
```bash
# Download token from: https://portal.gdc.cancer.gov/
GDC_TOKEN_PATH=/path/to/gdc-user-token.txt
```

**For experiment tracking:**
```bash
# Weights & Biases (optional)
WANDB_PROJECT=uiuc-cancer-research
WANDB_ENTITY=your_username
WANDB_API_KEY=your_wandb_api_key
```

---

## VEP Installation

VEP (Variant Effect Predictor) is required for variant annotation.

### Step 1: Install VEP

```bash
# Navigate to tools directory
cd tools
mkdir -p vep
cd vep

# Clone VEP repository
git clone https://github.com/Ensembl/ensembl-vep.git .

# Run installer
perl INSTALL.pl --AUTO a --SPECIES homo_sapiens --ASSEMBLY GRCh38

# This will:
# - Download and install VEP
# - Install required Perl modules
# - Download some cache files
# Expected time: 20-30 minutes
```

**Troubleshooting VEP Installation:**

If you encounter Perl module errors:
```bash
# Install missing Perl modules
cpan App::cpanminus
cpanm DBI
cpanm DBD::mysql
cpanm Archive::Zip
cpanm LWP::Simple
```

If you don't have sudo access:
```bash
# Install to local Perl library
perl INSTALL.pl --NO_UPDATE --AUTO p --SPECIES homo_sapiens --ASSEMBLY GRCh38
```

### Step 2: Download VEP Cache Files

VEP cache files (~15 GB) are required for offline annotation:

```bash
# Create VEP cache directory
mkdir -p ~/.vep
cd ~/.vep

# Download GRCh38 cache (VEP v110)
wget ftp://ftp.ensembl.org/pub/release-110/variation/vep/homo_sapiens_vep_110_GRCh38.tar.gz

# Extract cache
tar xzf homo_sapiens_vep_110_GRCh38.tar.gz

# Verify extraction
ls -lh homo_sapiens/110_GRCh38/

# Expected time: 30-60 minutes (depending on connection speed)
```

**Alternative: Download via VEP installer:**
```bash
cd /path/to/uiuc-cancer-research/tools/vep
perl INSTALL.pl --AUTO c --SPECIES homo_sapiens --ASSEMBLY GRCh38
```

### Step 3: Verify VEP Installation

```bash
# Test VEP
cd /path/to/uiuc-cancer-research/tools/vep
./vep --help

# Test with cache
./vep --cache --offline --assembly GRCh38 --help

# Should output VEP help message without errors
```

---

## R Setup for TCGA Download

The project uses R with the TCGAmutations package to download TCGA-PRAD data.

### Step 1: Install R

**On CentOS/RHEL:**
```bash
sudo yum install R
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install r-base r-base-dev
```

**Verify installation:**
```bash
R --version
# Should be R version 4.0.0 or higher
```

### Step 2: Install Required R Packages

**Launch R:**
```bash
R
```

**Install packages in R console:**
```r
# Set user library path
user_lib <- "~/Rlibs"
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

# Install devtools
install.packages("devtools", lib = user_lib, repos = "https://cran.r-project.org")

# Install TCGAmutations from GitHub
devtools::install_github("PoisonAlien/TCGAmutations", lib = user_lib)

# Verify installation
library(TCGAmutations)
?tcga_load

# Exit R
q()
```

**Alternative: Use the R script:**
```bash
# The download_tcga.R script will auto-install packages
Rscript scripts/variants/download_tcga.R
```

### Step 3: Verify R Setup

```bash
# Test TCGAmutations
Rscript -e "library(TCGAmutations); print('TCGAmutations loaded successfully')"
```

---

## Configuration

### Step 1: Copy Configuration Template

```bash
cd /path/to/uiuc-cancer-research

# Copy pipeline configuration template
cp config/pipeline_config_template.py config/pipeline_config.py
```

### Step 2: Customize Configuration (Optional)

Edit `config/pipeline_config.py` to customize:

```python
# Example customizations:

# Change project root
PROJECT_ROOT = Path('/custom/path/to/project')

# Modify TabNet hyperparameters
TABNET_N_D = 128  # Increase model capacity
TABNET_N_A = 128
TABNET_LEARNING_RATE = 1e-2

# Add custom prostate cancer genes
IMPORTANT_PROSTATE_GENES.extend(['GENE1', 'GENE2'])

# Adjust GPU settings
GPU_ENABLED = True
CUDA_VISIBLE_DEVICES = '0,1'  # Use multiple GPUs
```

### Step 3: Create Required Directories

The pipeline will create directories automatically, but you can pre-create them:

```bash
mkdir -p data/raw/variants
mkdir -p data/processed/{cosmic_prostate,clinvar_prostate,tcga_prad_prostate,merged,vep,tabnet_csv}
mkdir -p models
mkdir -p results/{training,optimization,attention_analysis,metrics}
mkdir -p logs
```

---

## Verification

### Pre-Flight Checklist

Run these checks before executing the pipeline:

**1. Conda Environment:**
```bash
conda activate uiuc-cancer-research
python -c "import torch, pandas, sklearn, pysam; print('✓ All core packages available')"
```

**2. Environment Variables:**
```bash
source .env
echo "Project root: $PROJECT_ROOT"
echo "COSMIC email: $COSMIC_EMAIL"
echo "VEP cache: $VEP_CACHE_DIR"
```

**3. VEP Installation:**
```bash
cd tools/vep
./vep --help | head -5
ls -d ~/.vep/homo_sapiens/110_GRCh38 && echo "✓ VEP cache found"
```

**4. R and TCGAmutations:**
```bash
Rscript -e "library(TCGAmutations)" && echo "✓ TCGAmutations available"
```

**5. Disk Space:**
```bash
df -h $PROJECT_ROOT
# Should have at least 100GB free
```

**6. GPU (Optional):**
```bash
nvidia-smi  # Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test Run

Test the pipeline setup with a minimal example:

```bash
# Activate environment
conda activate uiuc-cancer-research

# Test data download (skip if you don't have credentials yet)
# python scripts/download/download_clinvar.py --assembly GRCh38

# Test configuration
python -c "from config.pipeline_config import PipelineConfig; PipelineConfig.print_config()"

# Run pipeline help
bash run_full_pipeline.sh --help
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Conda Environment Creation Fails

**Error:** `CondaValueError: prefix already exists`

**Solution:**
```bash
# Remove existing environment
conda env remove -n uiuc-cancer-research

# Recreate
conda env create -f environment.yml
```

#### Issue 2: VEP Installation Fails

**Error:** `Can't locate Module.pm in @INC`

**Solution:**
```bash
# Install missing Perl modules
cpan install Module::Build
cpan install DBI
cpan install Archive::Zip

# Retry VEP installation
cd tools/vep
perl INSTALL.pl --AUTO a
```

#### Issue 3: R Package Installation Fails

**Error:** `package 'TCGAmutations' is not available`

**Solution:**
```r
# Update devtools first
install.packages("devtools", repos = "https://cran.r-project.org")

# Install from specific commit
devtools::install_github("PoisonAlien/TCGAmutations@master")
```

#### Issue 4: GPU Not Detected

**Error:** `CUDA not available`

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### Issue 5: Disk Space Issues

**Error:** `No space left on device`

**Solution:**
```bash
# Check disk usage
df -h

# Clean conda cache
conda clean --all

# Clean pip cache
pip cache purge

# Remove old log files
find logs/ -name "*.log" -mtime +30 -delete
```

#### Issue 6: COSMIC Download Fails

**Error:** `Authentication failed`

**Solution:**
1. Verify credentials at https://cancer.sanger.ac.uk/cosmic/register
2. Check `.env` file has correct `COSMIC_EMAIL` and `COSMIC_PASSWORD`
3. Try manual download first to test credentials

#### Issue 7: Memory Errors During Training

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size in .env
echo "TABNET_BATCH_SIZE=512" >> .env  # Instead of 1024

# Or disable GPU
echo "GPU_ENABLED=false" >> .env
```

### Getting Help

If you encounter issues not covered here:

1. **Check logs:** `ls -lt logs/` and review most recent log files
2. **GitHub Issues:** https://github.com/AbrahamArellano/uiuc-cancer-research/issues
3. **Contact:**
   - Abraham Arellano: aa107@illinois.edu
   - Umesh Kumar: umesh2@illinois.edu

### Additional Resources

- **Ensembl VEP Documentation:** https://www.ensembl.org/info/docs/tools/vep/
- **COSMIC Documentation:** https://cancer.sanger.ac.uk/cosmic/help
- **TCGA Documentation:** https://www.cancer.gov/tcga
- **PyTorch Documentation:** https://pytorch.org/docs/
- **TabNet Paper:** https://arxiv.org/abs/1908.07442

---

## Next Steps

After completing setup:

1. **Download Data:** Follow [DATA_SOURCES.md](DATA_SOURCES.md) for data acquisition
2. **Run Pipeline:** Execute `bash run_full_pipeline.sh`
3. **Review Results:** Check `results/` directory for model outputs
4. **Read README:** See [README.md](../README.md) for usage examples

---

**Setup Complete!** You're ready to run the Prostate-VarBench pipeline.
