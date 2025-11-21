# Prostate-VarBench: Interpretable Deep Learning for Prostate Cancer Variant Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ML4H%202025-green.svg)](https://proceedings.mlr.press/v297)

> **Official implementation of "Prostate-VarBench: A Benchmark with Interpretable TabNet Framework for Prostate Cancer Variant Classification"**
> *Published in Proceedings of Machine Learning Research (ML4H 2025)*

## Overview

Prostate cancer genomic testing frequently yields **Variants of Uncertain Significance (VUS)** in 30-50% of cases, delaying diagnosis and treatment decisions. This project provides:

1. **Prostate-VarBench**: A curated, prostate-specific benchmark integrating COSMIC, ClinVar, and TCGA-PRAD datasets (193,278 variants)
2. **Interpretable TabNet Model**: Deep learning classifier with native attention mechanisms for transparent clinical decision support
3. **VEP Concatenation Correction**: Systematic data quality fix affecting 15.8% of annotations
4. **8-Tier Feature System**: 56 clinically relevant features across interpretable hierarchical tiers
5. **VUS Reduction**: 6.5% absolute reduction in uncertain classifications (62.8% → 56.3%)

### Key Results

- **Test Accuracy**: 89.9% with balanced class metrics
- **Performance**: Pathogenic (P/R/F1: 0.89/0.86/0.88), VUS (0.92/0.91/0.91), Benign (0.87/0.90/0.88)
- **Interpretability**: Step-wise attention masks highlight clinically meaningful features (VAR_SYNONYMS, AlphaMissense, clinical context)
- **Clinical Impact**: ~12,600 variants reclassified from uncertain to actionable categories

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Acquisition](#data-acquisition)
- [Pipeline Execution](#pipeline-execution)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Results & Analysis](#results--analysis)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Data Pipeline
- **Multi-Source Integration**: COSMIC (somatic mutations), ClinVar (clinical variants), TCGA-PRAD (prostate tumors)
- **Automated Downloads**: Scripts for COSMIC (SFTP), ClinVar (FTP), TCGA-PRAD (GDC API)
- **VEP Annotation**: Variant Effect Predictor integration with concatenation correction
- **AlphaMissense Enhancement**: AI-powered pathogenicity scores for missense variants

### Machine Learning
- **TabNet Architecture**: Interpretable deep learning with sequential attention mechanisms
- **8-Tier Feature System**: VEP-corrected, Core VEP, AlphaMissense, Population Genetics, Functional Predictions, Clinical Context, Variant Properties, Prostate Biology
- **Hyperparameter Optimization**: Systematic grid search with H100 GPU support
- **Attention Analysis**: Per-variant explanations and feature importance ranking

### Clinical Utility
- **VUS Reduction**: 6.5% absolute decrease in uncertain classifications
- **Transparent Decisions**: Step-wise feature selection masks for molecular tumor board review
- **Therapy Guidance**: Support for PARP inhibitor, hormone therapy, and immunotherapy selection

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Conda**: Recommended for environment management
- **Disk Space**: ~100 GB (data + models + results)
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: Optional but recommended (NVIDIA with 8GB+ VRAM)

### Step 1: Clone Repository

```bash
git clone https://github.com/AbrahamArellano/uiuc-cancer-research.git
cd uiuc-cancer-research
```

### Step 2: Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate uiuc-cancer-research
```

**Alternative: Manual Installation**
```bash
conda create -n uiuc-cancer-research python=3.10
conda activate uiuc-cancer-research
pip install -r requirements.txt
```

### Step 3: Install Ensembl VEP

VEP (Variant Effect Predictor) is required for variant annotation:

```bash
# Clone VEP repository
cd tools
git clone https://github.com/Ensembl/ensembl-vep.git vep
cd vep

# Install VEP (requires Perl)
perl INSTALL.pl --AUTO a --SPECIES homo_sapiens --ASSEMBLY GRCh38

# Return to project root
cd ../..
```

### Step 4: Download VEP Cache

```bash
# Download VEP cache files (~15 GB, takes 30-60 minutes)
cd ~/.vep
wget ftp://ftp.ensembl.org/pub/release-110/variation/vep/homo_sapiens_vep_110_GRCh38.tar.gz
tar xzf homo_sapiens_vep_110_GRCh38.tar.gz
cd -
```

### Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env  # or use your preferred editor
```

**Required settings in `.env`**:
- `COSMIC_EMAIL`: Your COSMIC account email
- `COSMIC_PASSWORD`: Your COSMIC token
- `PROJECT_ROOT`: Absolute path to project directory
- `VEP_CACHE_DIR`: Path to VEP cache (default: `~/.vep`)

---

## Data Acquisition

### Required Accounts

1. **COSMIC**: Register at https://cancer.sanger.ac.uk/cosmic/register (free for academic use)
2. **GDC** (optional): For controlled-access TCGA data at https://gdc.cancer.gov/

### Automated Download (Recommended)

Download all datasets with a single command:

```bash
python scripts/download/download_all_data.py \
    --cosmic-email your@email.com \
    --cosmic-password your_cosmic_token
```

**Expected Download:**
- **Size**: ~4-5 GB compressed
- **Time**: 25-65 minutes (depending on connection speed)
- **Datasets**: COSMIC v102, ClinVar (GRCh38), TCGA-PRAD

**With configuration file:**
```bash
# Create config file
cat > download_config.json << EOF
{
  "cosmic_email": "your@email.com",
  "cosmic_password": "your_token",
  "gdc_token": "/path/to/gdc-token.txt"
}
EOF

# Run download
python scripts/download/download_all_data.py --config download_config.json
```

### Manual Downloads (Alternative)

#### COSMIC
```bash
python scripts/download/download_cosmic.py \
    --email your@email.com \
    --password your_token \
    --version v102 \
    --assembly GRCh38
```

#### ClinVar
```bash
python scripts/download/download_clinvar.py \
    --assembly GRCh38
```

#### TCGA-PRAD
```bash
python scripts/download/download_tcga_prad.py \
    --data-types "Masked Somatic Mutation" \
    --token /path/to/gdc-token.txt  # Optional
```

---

## Pipeline Execution

### Quick Start: Full Pipeline

Run the complete pipeline with a single command:

```bash
bash scripts/run_pipeline.sh
```

This executes all 9 steps:
1. Data filtering (COSMIC, ClinVar)
2. Dataset merging
3. VEP annotation
4. VCF to CSV conversion
5. AlphaMissense enhancement
6. TabNet model training
7. Hyperparameter optimization (optional)
8. Attention analysis
9. Results generation

### Step-by-Step Execution

For more control, run individual pipeline steps:

#### Step 1-2: Filter Variants

```bash
# Filter COSMIC for prostate-specific variants
python scripts/variants/filter_cosmic_prostate.py

# Filter ClinVar for prostate cancer genes
python scripts/variants/filter_clinvar_prostate.py
```

**Expected Output**:
- `data/processed/cosmic_prostate/cosmic_prostate.csv`
- `data/processed/clinvar_prostate/clinvar_prostate.csv`

**Runtime**: 5-10 minutes each

#### Step 3: Merge Datasets

```bash
python scripts/merge/merge_datasets.py
```

**Expected Output**:
- `data/processed/merged/merged_prostate_variants.csv`
- `data/processed/merged_vcf/merged_prostate_variants.vcf`

**Runtime**: 10-15 minutes

#### Step 4: VEP Annotation

```bash
bash scripts/vep/run_vep_annotation.sh
```

**Expected Output**:
- `data/processed/vep/vep_annotated.vcf`

**Runtime**: 30-60 minutes (varies by dataset size)

#### Step 5: VCF to CSV Conversion

```bash
python scripts/enhance/tabnet_vcf_to_csv/vcf_to_tabnet_csv.py
```

**Expected Output**:
- `data/processed/tabnet_csv/prostate_variants_tabnet.csv`

**Runtime**: 5-10 minutes

#### Step 6: AlphaMissense Enhancement

```bash
python scripts/enhance/functional_enhancement/simple_functional_imputation.py
```

**Expected Output**:
- `data/processed/tabnet_csv/prostate_variants_tabnet_enhanced.csv`
- Downloads AlphaMissense scores (~2 GB) automatically

**Runtime**: 20-30 minutes (includes download)

#### Step 7: Train TabNet Model

```bash
python src/model/tabnet_prostate_variant_classifier.py
```

**Expected Output**:
- `models/tabnet_model_TIMESTAMP.pkl`
- `results/training/tabnet_training_TIMESTAMP/`
- Performance metrics (accuracy, F1, ROC-AUC)

**Runtime**: 15-25 minutes (with GPU), 60-90 minutes (CPU only)

**Performance Metrics**:
- Cross-validation accuracy: 88.0% ± 2.0%
- Test accuracy: 89.9%
- Balanced accuracy: 89.05%
- Weighted F1: 0.8991
- Macro ROC-AUC: 0.9701

#### Step 8: Hyperparameter Optimization (Optional)

```bash
# Requires H100 or A100 GPU
python scripts/optimization/hyperparameter_optimization.py
```

**Expected Output**:
- `results/optimization/TIMESTAMP/optimization_report.txt`
- Heatmaps for architecture, learning dynamics, regularization

**Runtime**: Up to 36 GPU hours (configurable budget)

#### Step 9: Attention Analysis

```bash
# Extract attention weights
python src/analysis/attention_extractor.py

# Analyze attention patterns
python src/analysis/attention_analyzer.py

# Generate result reports
python src/analysis/results_generator.py
```

**Expected Output**:
- `results/attention_analysis/attention_weights/`
- `results/attention_analysis/pattern_analysis/`
- Feature importance rankings by tier

**Runtime**: 10-15 minutes

---

## Project Structure

```
uiuc-cancer-research/
├── config/                          # Configuration files
│   ├── pipeline_config_template.py  # Configuration template
│   └── pipeline_config.py          # Your local config (git-ignored)
│
├── data/                            # Data directory (git-ignored)
│   ├── raw/                        # Raw downloaded data
│   │   ├── variants/               # COSMIC, ClinVar files
│   │   └── TCGA-PRAD/             # TCGA prostate cancer data
│   ├── processed/                  # Processed datasets
│   │   ├── cosmic_prostate/       # Filtered COSMIC variants
│   │   ├── clinvar_prostate/      # Filtered ClinVar variants
│   │   ├── merged/                # Merged CSV datasets
│   │   ├── merged_vcf/            # Merged VCF files
│   │   ├── vep/                   # VEP-annotated variants
│   │   └── tabnet_csv/            # Final TabNet training data
│   └── data_versions.json         # Data provenance tracking
│
├── src/                            # Source code
│   ├── data/preprocessing/        # Data loading & preprocessing
│   ├── model/                     # TabNet classifier
│   ├── analysis/                  # Attention & interpretability analysis
│   └── utils/                     # Logging & utilities
│
├── scripts/                        # Pipeline scripts
│   ├── download/                  # Data download scripts
│   │   ├── download_all_data.py  # Master download orchestrator
│   │   ├── download_cosmic.py    # COSMIC SFTP downloader
│   │   ├── download_clinvar.py   # ClinVar FTP downloader
│   │   └── download_tcga_prad.py # TCGA GDC API downloader
│   ├── variants/                  # Variant filtering
│   ├── merge/                     # Dataset merging
│   ├── vep/                       # VEP annotation
│   ├── enhance/                   # Functional enhancement
│   ├── optimization/              # Hyperparameter optimization
│   └── validation/                # Data validation
│
├── models/                         # Saved trained models (git-ignored)
├── results/                        # Training & analysis results
│   ├── training/                  # Model training results
│   ├── optimization/              # Hyperparameter search results
│   ├── attention_analysis/        # TabNet attention patterns
│   └── metrics/                   # Performance metrics
│
├── tools/                          # External tools
│   └── vep/                       # Ensembl VEP installation
│
├── docs/                           # Documentation
│   ├── SETUP.md                   # Detailed setup guide
│   └── DATA_SOURCES.md           # Data provenance documentation
│
├── logs/                           # Execution logs (git-ignored)
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore patterns
├── scripts/
│   ├── run_pipeline.sh            # Master pipeline execution script
│   └── download/download_all.sh   # Data download script
└── README.md                       # This file
```

---

## Configuration

### Environment Variables

All configuration is managed through environment variables in `.env`:

```bash
# Project paths
PROJECT_ROOT=/path/to/uiuc-cancer-research

# COSMIC credentials
COSMIC_EMAIL=your@email.com
COSMIC_PASSWORD=your_token

# VEP configuration
VEP_CACHE_DIR=/home/username/.vep

# Model parameters
TABNET_N_D=64
TABNET_N_A=64
TABNET_N_STEPS=6
TABNET_LEARNING_RATE=0.02

# GPU configuration
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0
```

See [.env.example](.env.example) for all available options.

### Pipeline Configuration

Edit `config/pipeline_config.py` to customize:
- Data directories
- Model hyperparameters
- Feature engineering settings
- Prostate-specific gene lists

---

## Model Training

### TabNet Architecture

**Configuration** (8-tier, 56-feature system):
- **n_d=64**: Decision prediction layer width
- **n_a=64**: Attention embedding dimension
- **n_steps=6**: Sequential decision steps
- **gamma=1.3**: Feature reuse parameter
- **lambda_sparse=1e-3**: Sparsity regularization

**Feature Tiers**:
1. **VEP-Corrected** (4 features, 29.8% attention): Post-correction IDs/evidence
2. **Core VEP** (10 features, 11.0% attention): Gene/transcript context
3. **AlphaMissense** (2 features, 14.0% attention): AI pathogenicity scores
4. **Population Genetics** (17 features, 9.0% attention): Ancestry frequencies
5. **Functional Predictions** (6 features, 7.8% attention): In-silico effects
6. **Clinical Context** (5 features, 14.8% attention): Somatic/phenotype context
7. **Variant Properties** (8 features, 11.1% attention): Structural attributes
8. **Prostate Biology** (4 features, 2.5% attention): Pathway disruptions

### Training Parameters

```python
MAX_EPOCHS = 200
PATIENCE = 15  # Early stopping
BATCH_SIZE = 1024
VIRTUAL_BATCH_SIZE = 128
LEARNING_RATE = 2e-2
```

### Hyperparameter Optimization

Grid search over 108 configurations:
- **n_d**: [32, 64, 128]
- **n_a**: [32, 64, 128]
- **n_steps**: [6, 7, 8]
- **gamma**: [1.0, 1.3, 1.5]
- **lambda_sparse**: [1e-4, 1e-3, 1e-2]
- **learning_rate**: [1e-3, 2e-3, 1e-2, 2e-2]

**Best Configuration**: n_d=64, n_a=64, n_steps=6, lr=1e-2 (89.1% accuracy)

---

## Results & Analysis

### Performance Metrics

**Test Set (38,656 variants)**:
- **Overall Accuracy**: 89.9%
- **Balanced Accuracy**: 89.05%
- **Cohen's Kappa**: 0.8263
- **Weighted F1**: 0.8991
- **Macro ROC-AUC**: 0.9701

**Class-Specific Performance**:

| Class       | Precision | Recall | F1 Score | Support | Clinical Impact                              |
|-------------|-----------|--------|----------|---------|----------------------------------------------|
| Benign      | 0.87      | 0.90   | 0.88     | 10,870  | High sensitivity for excluding actionable variants |
| Pathogenic  | 0.89      | 0.86   | 0.88     | 6,037   | Balanced detection of therapeutic targets    |
| VUS         | 0.92      | 0.91   | 0.91     | 21,749  | Superior handling of uncertain classifications |

### Attention Analysis

**Top 5 Decision-Driving Features**:
1. **VAR_SYNONYMS** (20.3%): Variant database identifiers
2. **AlphaMissense Class** (12.8%): AI pathogenicity predictions
3. **Existing_variation** (11.2%): Known variant IDs
4. **is_lof** (10.1%): Loss-of-function indicator
5. **is_snv** (7.8%): Single nucleotide variant flag

### VUS Reduction Impact

- **Before**: 62.8% VUS rate
- **After**: 56.3% VUS rate
- **Reduction**: 6.5% absolute (12,600 variants)
- **Clinical Impact**: Enhanced therapeutic decision-making for PARP inhibitors, hormone therapy, precision medicine

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{tavara2025prostatevarbench,
  title={Prostate-VarBench: A Benchmark with Interpretable TabNet Framework for Prostate Cancer Variant Classification},
  author={Tavara, Abraham Francisco Arellano and Kumar, Umesh and Pradeepkumar, Jathurshan and Sun, Jimeng},
  booktitle={Proceedings of Machine Learning Research},
  volume={297},
  year={2025},
  organization={ML4H}
}
```

**Paper**: [ML4H 2025 Proceedings](https://proceedings.mlr.press/v297)
**Code**: [GitHub Repository](https://github.com/AbrahamArellano/uiuc-cancer-research/)

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Setup**:
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest

# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

---

## Troubleshooting

### Common Issues

**1. COSMIC Download Fails**
- **Error**: Authentication failed
- **Solution**: Verify COSMIC credentials at https://cancer.sanger.ac.uk/cosmic/register
- Check `.env` file for correct `COSMIC_EMAIL` and `COSMIC_PASSWORD`

**2. VEP Installation Issues**
- **Error**: Perl dependencies missing
- **Solution**: Install Perl and required modules
  ```bash
  sudo apt-get install perl libdbi-perl libdbd-mysql-perl
  ```

**3. GPU Out of Memory**
- **Error**: CUDA out of memory
- **Solution**: Reduce batch size in `.env`
  ```bash
  TABNET_BATCH_SIZE=512  # Instead of 1024
  ```

**4. Slow Download Speeds**
- **Solution**: Download during off-peak hours
- Use `--max-files` flag for testing:
  ```bash
  python scripts/download/download_tcga_prad.py --max-files 5
  ```

For more troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **TCGA-PRAD**: The Cancer Genome Atlas Research Network
- **COSMIC**: Catalogue of Somatic Mutations in Cancer (Wellcome Sanger Institute)
- **ClinVar**: NCBI ClinVar Database
- **Ensembl VEP**: Variant Effect Predictor
- **AlphaMissense**: DeepMind AlphaMissense pathogenicity predictions
- **University of Illinois Urbana-Champaign**: Department of Computer Science

---

## Contact

- **Abraham Arellano**: aa107@illinois.edu
- **Umesh Kumar**: umesh2@illinois.edu
- **Jathurshan Pradeepkumar**: jp65@illinois.edu
- **Jimeng Sun**: jimeng@illinois.edu

**Project Homepage**: https://github.com/AbrahamArellano/uiuc-cancer-research/

---

<p align="center">
  <b>Interpretable Deep Learning for Actionable Prostate Cancer Variant Classification</b><br>
  University of Illinois Urbana-Champaign | ML4H 2025
</p>
