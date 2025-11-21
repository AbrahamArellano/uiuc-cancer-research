# Troubleshooting Guide

This guide covers common issues encountered when running the Prostate-VarBench pipeline.

## Table of Contents

- [Data Download Issues](#data-download-issues)
- [VEP Annotation Issues](#vep-annotation-issues)
- [Model Training Issues](#model-training-issues)
- [Environment Issues](#environment-issues)
- [HPC/Cluster Issues](#hpccluster-issues)

---

## Data Download Issues

### COSMIC Authentication Failed

**Error**: `401 Unauthorized` or `Authentication failed`

**Solution**:
1. Verify your COSMIC credentials at https://cancer.sanger.ac.uk/cosmic/register
2. Check `.env` file has correct values:
   ```bash
   COSMIC_EMAIL=your@email.com
   COSMIC_PASSWORD=your_token
   ```
3. Ensure you're using the API token, not your login password
4. Test credentials:
   ```bash
   python scripts/download/diagnose_cosmic.py
   ```

### COSMIC Download Returns HTML

**Error**: Response contains HTML instead of JSON

**Solution**:
The COSMIC API endpoint may have changed. The correct endpoint is:
```
https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted
```

Use the HTTP-based downloader:
```bash
python scripts/download/download_cosmic_http.py \
    --email $COSMIC_EMAIL \
    --password $COSMIC_PASSWORD
```

### ClinVar Download Timeout

**Error**: Connection timeout or slow download

**Solution**:
1. Check network connectivity
2. Try during off-peak hours (NCBI servers can be busy)
3. Use a different mirror if available
4. Increase timeout in script

### TCGA-PRAD R Package Issues

**Error**: `TCGAmutations package not found`

**Solution**:
```R
# Install from GitHub
install.packages("devtools")
devtools::install_github("PoisonAlien/TCGAmutations")

# Verify installation
library(TCGAmutations)
```

---

## VEP Annotation Issues

### VEP Cache Not Found

**Error**: `Could not find cache directory`

**Solution**:
1. Download VEP cache:
   ```bash
   cd ~/.vep
   wget ftp://ftp.ensembl.org/pub/release-110/variation/vep/homo_sapiens_vep_110_GRCh38.tar.gz
   tar xzf homo_sapiens_vep_110_GRCh38.tar.gz
   ```
2. Set correct path in `.env`:
   ```bash
   VEP_CACHE_DIR=/home/username/.vep
   ```

### VEP Perl Dependencies Missing

**Error**: `Can't locate DBI.pm` or similar Perl errors

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install perl libdbi-perl libdbd-mysql-perl

# CentOS/RHEL
sudo yum install perl-DBI perl-DBD-MySQL

# Conda
conda install -c bioconda ensembl-vep
```

### VEP Out of Memory

**Error**: `Out of memory` during VEP annotation

**Solution**:
1. Use scratch storage on HPC:
   ```bash
   export VEP_SCRATCH=/scratch/$USER/vep_cache
   ```
2. Process in smaller batches
3. Increase SLURM memory allocation:
   ```bash
   #SBATCH --mem=32G
   ```

### VEP Concatenation Issues

**Error**: Multiple values concatenated in VEP output fields

**Solution**:
Run the VEP correction script:
```bash
python scripts/enhance/correction_vcf/post_process_vep_concatenation.py
```

This fixes ~15.8% of variants affected by concatenation issues.

---

## Model Training Issues

### GPU Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
1. Reduce batch size in `.env`:
   ```bash
   TABNET_BATCH_SIZE=512  # Default is 1024
   ```
2. Use gradient accumulation
3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### TabNet Training Not Converging

**Error**: Loss not decreasing or accuracy stuck

**Solution**:
1. Check learning rate (try 1e-2 to 2e-2)
2. Verify data preprocessing completed correctly
3. Check for NaN values in features:
   ```python
   import pandas as pd
   df = pd.read_csv("data/processed/tabnet_csv/prostate_variants_tabnet_enhanced.csv")
   print(df.isnull().sum())
   ```
4. Ensure class balance is reasonable

### Missing AlphaMissense Scores

**Error**: `alphamissense_pathogenicity` column all NaN

**Solution**:
1. Re-run functional enhancement:
   ```bash
   python scripts/enhance/functional_enhancement/simple_functional_imputation.py
   ```
2. Check AlphaMissense database downloaded correctly (~2GB)
3. Verify variants are missense type

---

## Environment Issues

### Conda Environment Creation Failed

**Error**: Package conflicts during `conda env create`

**Solution**:
1. Use mamba for faster resolution:
   ```bash
   conda install mamba -n base -c conda-forge
   mamba env create -f environment.yml
   ```
2. Try minimal environment:
   ```bash
   conda env create -f environment_minimal.yml
   ```
3. Install packages incrementally

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
1. Activate correct environment:
   ```bash
   conda activate uiuc-cancer-research
   ```
2. Verify installation:
   ```bash
   pip list | grep package_name
   ```
3. Install missing package:
   ```bash
   pip install package_name
   ```

### Path Issues

**Error**: `FileNotFoundError` for scripts or data

**Solution**:
1. Run from project root directory
2. Set PROJECT_ROOT in `.env`:
   ```bash
   PROJECT_ROOT=/path/to/uiuc-cancer-research
   ```
3. Use absolute paths in scripts

---

## HPC/Cluster Issues

### SLURM Job Fails Immediately

**Error**: Job exits with no output

**Solution**:
1. Check SLURM logs:
   ```bash
   cat slurm-JOBID.out
   cat slurm-JOBID.err
   ```
2. Verify resource requests are valid for your cluster
3. Load required modules:
   ```bash
   module load python/3.10
   module load R/4.5
   ```

### Scratch Storage Full

**Error**: `No space left on device` on scratch

**Solution**:
1. Clean old files:
   ```bash
   rm -rf /scratch/$USER/old_files
   ```
2. Check quota:
   ```bash
   quota -s
   ```
3. Use project storage instead

### Singularity Container Issues

**Error**: VEP container fails to run

**Solution**:
1. Pull container fresh:
   ```bash
   singularity pull docker://ensemblorg/ensembl-vep
   ```
2. Bind paths correctly:
   ```bash
   singularity exec --bind /path/to/data vep.sif vep ...
   ```

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/AbrahamArellano/uiuc-cancer-research/issues)
2. Search existing discussions
3. Open a new issue with:
   - Error message (full traceback)
   - Steps to reproduce
   - Environment details (`conda list`)
   - Log files if available

## Contact

- Abraham Arellano: aa107@illinois.edu
- Project: https://github.com/AbrahamArellano/uiuc-cancer-research/
