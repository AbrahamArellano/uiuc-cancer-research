# Data Sources & Provenance

Documentation of all data sources used in the Prostate-VarBench project.

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
  - [COSMIC Mutant Census](#1-cosmic-mutant-census)
  - [ClinVar](#2-clinvar)
  - [TCGA-PRAD](#3-tcga-prad)
  - [AlphaMissense](#4-alphamissense)
- [Data Integration Pipeline](#data-integration-pipeline)
- [Data Quality & Validation](#data-quality--validation)
- [Data Versioning](#data-versioning)
- [Access & Licensing](#access--licensing)
- [Citation Requirements](#citation-requirements)

---

## Overview

Prostate-VarBench integrates four major genomic databases to create a comprehensive, prostate-specific variant classification benchmark:

| Source | Type | Variants | Assembly | Purpose |
|--------|------|----------|----------|---------|
| COSMIC | Somatic mutations | ~212,067 | GRCh38 | Cancer-associated variants |
| ClinVar | Clinical annotations | ~191,891 | GRCh38 | Expert-curated pathogenicity |
| TCGA-PRAD | Tumor genomics | ~20,054 | GRCh38 | Real-world clinical context |
| AlphaMissense | AI predictions | Genome-wide | hg38 | Missense pathogenicity scores |

**Final Benchmark:** 193,278 harmonized variants with 56 features across 8 clinical tiers.

---

## Data Sources

### 1. COSMIC Mutant Census

**Source:** Catalogue of Somatic Mutations in Cancer (Wellcome Sanger Institute)

#### Description
COSMIC is the world's largest database of somatic mutations in human cancer, containing expert-curated data from scientific literature and large-scale experimental screens.

#### Version Used
- **Version:** v102
- **Assembly:** GRCh38 (hg38)
- **Release Date:** 2023
- **File:** `Cosmic_MutantCensus_v102_GRCh38.tsv`

#### Data Content
- **Total Mutations:** 212,067 prostate-relevant variants
- **Genes Covered:** 10,408 genes
- **Data Types:**
  - Somatic mutations from literature
  - Tumor cell line mutations
  - Genomic coordinates (chr, pos, ref, alt)
  - Gene annotations
  - Mutation consequences
  - Sample annotations

#### Prostate-Specific Filtering
Variants selected based on:
1. **Direct tissue annotation:** Primary site = "prostate"
2. **Prostate cancer cell lines:**
   - PC3, LNCaP, DU145, 22Rv1, VCaP
   - LAPC4, C4-2, MDA-PCa series
3. **TCGA-PRAD samples:** Matched TCGA barcodes

#### Download Method
- **Protocol:** SFTP via sftp-cancer.sanger.ac.uk
- **Authentication:** Required (free academic registration)
- **Script:** `scripts/download/download_cosmic.py`
- **Size:** ~2 GB compressed

#### Access
1. Register at: https://cancer.sanger.ac.uk/cosmic/register
2. Obtain SFTP credentials
3. Download via automated script or manual SFTP

**Manual Download:**
```bash
sftp your_email@sftp-cancer.sanger.ac.uk
cd cosmic/v102/MutantCensus/
get Cosmic_MutantCensus_v102_GRCh38.tsv.gz
```

---

### 2. ClinVar

**Source:** NCBI ClinVar - Public archive of variant-disease relationships

#### Description
ClinVar aggregates information about genomic variation and its relationship to human health, with expert-reviewed clinical significance classifications.

#### Version Used
- **Assembly:** GRCh38
- **Download Date:** 2025-01 (monthly updates available)
- **File Format:** VCF (Variant Call Format)
- **File:** `clinvar_GRCh38.vcf.gz`

#### Data Content
- **Total Variants:** 191,891 clinically annotated variants
- **Clinical Significance Labels:**
  - Pathogenic
  - Likely Pathogenic
  - Benign
  - Likely Benign
  - Uncertain Significance (VUS)
  - Conflicting Interpretations
- **Additional Annotations:**
  - Review status (stars: 0-4)
  - Submitter organizations
  - Phenotype/disease associations
  - ACMG/AMP guidelines compliance

#### Prostate-Specific Filtering
Variants filtered for prostate cancer-relevant genes:
- DNA repair: BRCA1, BRCA2, ATM, CHEK2, PALB2
- Mismatch repair: MLH1, MSH2, MSH6, PMS2
- Tumor suppressors: TP53, PTEN, RB1
- Prostate-specific: AR, SPOP, FOXA1, NKX3-1, ERG

#### Download Method
- **Protocol:** HTTPS/FTP (public, no authentication)
- **URL:** https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/
- **Script:** `scripts/download/download_clinvar.py`
- **Size:** ~500 MB compressed, ~5 GB uncompressed
- **Update Frequency:** Monthly

#### Access
```bash
# Direct download
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz

# Or use automated script
python scripts/download/download_clinvar.py --assembly GRCh38
```

---

### 3. TCGA-PRAD

**Source:** The Cancer Genome Atlas - Prostate Adenocarcinoma

#### Description
TCGA-PRAD contains comprehensive genomic characterization of 500+ prostate adenocarcinoma samples with linked clinical outcomes and treatment data.

#### Version Used
- **Project:** TCGA-PRAD
- **Assembly:** GRCh38
- **Data Source:** MC3 (Multi-Center Mutation Calling in Multiple Cancers)
- **Library:** TCGAmutations (R package)
- **File:** `tcga_prad_mutations.csv`

#### Data Content
- **Total Mutations:** 20,054 variants
- **Sample Size:** 500+ patients
- **Data Types:**
  - Somatic mutations (SNVs, indels)
  - Mutation consequences
  - Protein changes
  - Clinical annotations
  - Treatment outcomes
  - Survival data

#### Clinical Annotations
- **Patient Demographics:**
  - Age at diagnosis
  - Race/ethnicity
  - Tumor stage (T, N, M)
  - Gleason score
  - PSA levels
- **Treatment Information:**
  - Hormone therapy
  - Radiation therapy
  - Surgical interventions
- **Outcomes:**
  - Overall survival
  - Disease-free survival
  - Biochemical recurrence

#### Download Method
- **Protocol:** R package (TCGAmutations) via GitHub
- **Repository:** PoisonAlien/TCGAmutations
- **Script:** `scripts/variants/download_tcga.R`
- **Size:** ~50-100 MB
- **Authentication:** Not required for MC3 public data

#### R Implementation

**Package Installation:**
```r
# Install TCGAmutations
install.packages("devtools")
devtools::install_github("PoisonAlien/TCGAmutations")
```

**Data Download:**
```r
library(TCGAmutations)

# Download TCGA-PRAD mutations from MC3 dataset
tcga_load(study = "PRAD", source = "MC3")

# Access mutation data
prad_mutations <- prad_mc3@data
```

**Automated Script:**
```bash
# Run automated R script
Rscript scripts/variants/download_tcga.R

# Output: data/raw/variants/tcga_prad_mutations.csv
```

#### Pathway Annotations
The download script adds prostate cancer pathway annotations:
- **Core prostate genes:** AR, PTEN, TP53, MYC, ERG, SPOP, FOXA1
- **DNA repair pathway:** BRCA1, BRCA2, ATM, CHEK2, PALB2
- **Hormone pathway:** CYP17A1, SRD5A2, AR, ESR1
- **PI3K pathway:** PIK3CA, AKT1, MTOR, TSC1

#### Alternative Access Methods
1. **GDC Data Portal:** https://portal.gdc.cancer.gov/projects/TCGA-PRAD
2. **GDC API:** Programmatic access via Python (see `scripts/download/download_tcga_prad.py`)
3. **TCGAbiolinks (R):** Alternative R package for TCGA data

---

### 4. AlphaMissense

**Source:** DeepMind AlphaMissense - AI-powered pathogenicity predictions

#### Description
AlphaMissense uses deep learning to predict the pathogenicity of all possible missense variants in the human proteome with high accuracy.

#### Version Used
- **Assembly:** hg38 (GRCh38)
- **Model Version:** 2023
- **Coverage:** ~71 million possible missense variants
- **File:** `AlphaMissense_hg38.tsv.gz`

#### Data Content
- **Pathogenicity Score:** Continuous score (0-1)
  - 0 = Likely benign
  - 1 = Likely pathogenic
- **Classification:**
  - Likely pathogenic: score ≥ 0.564
  - Likely benign: score ≤ 0.34
  - Ambiguous: 0.34 < score < 0.564
- **Coverage:** All canonical transcripts

#### Integration in Pipeline
AlphaMissense scores are used to:
1. **Enhance variant classification** (Tier 3 features)
2. **Reduce VUS uncertainty** (primary contribution)
3. **Provide AI-powered second opinion** for clinical annotations

#### Download Method
- **Protocol:** HTTPS (Google Cloud Storage)
- **URL:** https://storage.googleapis.com/dm_alphamissense/
- **Script:** Automatically downloaded by `simple_functional_imputation.py`
- **Size:** ~2 GB compressed
- **Authentication:** Not required (public dataset)

#### Download
```bash
# Manual download
wget https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz

# Or automatic via pipeline script
python scripts/enhance/functional_enhancement/simple_functional_imputation.py
```

#### Citation
```
Cheng et al. (2023). Accurate proteome-wide missense variant effect
prediction with AlphaMissense. Science, 381(6664), eadg7492.
```

---

## Data Integration Pipeline

### Pipeline Overview

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
│   COSMIC    │────▶│          │     │           │     │            │
│ (212K vars) │     │ Prostate │     │           │     │   Final    │
└─────────────┘     │ Filtering│────▶│  Merging  │────▶│ Benchmark  │
┌─────────────┐     │          │     │           │     │ (193K vars)│
│   ClinVar   │────▶│          │     │           │     │            │
│ (192K vars) │     └──────────┘     └───────────┘     └────────────┘
└─────────────┘                             │
┌─────────────┐                             │
│ TCGA-PRAD   │─────────────────────────────┘
│  (20K vars) │
└─────────────┘
       │
       │ Annotation
       ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│     VEP     │────▶│ AlphaMissense│────▶│ TabNet CSV   │
│ Annotation  │     │  Enhancement │     │ (56 features)│
└─────────────┘     └──────────────┘     └──────────────┘
```

### Integration Steps

**Step 1: Prostate-Specific Filtering**
- Filter COSMIC for prostate tissue/cell lines
- Filter ClinVar for prostate cancer genes
- TCGA-PRAD already prostate-specific

**Step 2: Coordinate Standardization**
- Convert all to GRCh38 coordinates
- Standardize chromosome naming (1-22, X, Y)
- Normalize ref/alt alleles

**Step 3: Variant Merging**
- Create unique variant keys: (chr, pos, ref, alt)
- Merge on genomic coordinates
- Resolve conflicts (prefer ClinVar > COSMIC > TCGA)

**Step 4: VEP Annotation**
- Run Ensembl VEP v110
- Add consequence predictions
- Add population frequencies
- Add functional predictions (SIFT, PolyPhen)

**Step 5: VEP Concatenation Correction**
- Fix systematic VEP multi-transcript merging issue
- Affects 15.8% of annotations
- Select canonical transcript per variant

**Step 6: AlphaMissense Enhancement**
- Match variants to AlphaMissense predictions
- Add pathogenicity scores (0-1)
- Classify as likely pathogenic/benign/ambiguous

**Step 7: Feature Engineering**
- Extract 56 features across 8 tiers
- Add prostate pathway annotations
- Create target labels (Pathogenic/VUS/Benign)

---

## Data Quality & Validation

### Quality Control Measures

**1. Coordinate Validation**
- All variants mapped to GRCh38
- Invalid coordinates removed
- Chromosome standardization (no "chr" prefix)

**2. Duplicate Removal**
- Remove exact duplicate variants
- Resolve multi-allelic sites
- Keep canonical transcripts only

**3. Missing Data Handling**
- Variants with missing essential fields removed
- Population frequencies: missing = 0 (rare)
- AlphaMissense: missing = use VEP IMPACT fallback

**4. Conflict Resolution**
When same variant has conflicting annotations:
```
Priority: ClinVar > AlphaMissense > VEP IMPACT
```

**5. Data Leakage Prevention**
- **Target leakage:** CLIN_SIG removed from features
- **Patient leakage:** Patient-aware splits in TCGA data
- **Gene leakage:** Gene-aware cross-validation

### Validation Metrics

**Dataset Statistics:**
| Metric | Value |
|--------|-------|
| Total variants | 193,278 |
| Unique genes | 10,408+ |
| Pathogenic | 6,037 (15.6%) |
| Benign | 10,870 (28.1%) |
| VUS | 21,749 (56.3%) |
| With AlphaMissense | ~140,000 (72%) |
| With population freq | ~180,000 (93%) |

**Quality Checks:**
```bash
# Run validation scripts
python scripts/variants/validate_cosmic.py
Rscript scripts/variants/validate_tcga.R
python scripts/validation/validate_merged_data.py
```

---

## Data Versioning

### Version Tracking

All downloaded data is tracked with metadata:

**File:** `data/data_versions.json`
```json
{
  "cosmic": {
    "version": "v102",
    "assembly": "GRCh38",
    "download_date": "2025-01-15",
    "file_size_mb": 2048,
    "sha256": "abc123...",
    "source_url": "sftp://sftp-cancer.sanger.ac.uk/..."
  },
  "clinvar": {
    "assembly": "GRCh38",
    "download_date": "2025-01-15",
    "file_size_mb": 512,
    "md5": "def456...",
    "source_url": "https://ftp.ncbi.nlm.nih.gov/..."
  },
  "tcga_prad": {
    "source": "MC3",
    "download_date": "2025-01-15",
    "sample_count": 500,
    "mutation_count": 20054
  }
}
```

### Reproducibility

To ensure reproducibility:

1. **Pin versions:** Use specific COSMIC/ClinVar releases
2. **Record checksums:** SHA256/MD5 for all downloads
3. **Log parameters:** Save all processing parameters
4. **Random seeds:** Fixed seeds for splitting (seed=42)

---

## Access & Licensing

### COSMIC

**License:** Free for academic use, commercial license available

**Registration Required:** Yes
- Academic users: Free registration
- Commercial users: Paid license

**Terms of Use:**
- Academic research only (free tier)
- Must cite COSMIC in publications
- No redistribution of raw data

**Access:** https://cancer.sanger.ac.uk/cosmic/license

### ClinVar

**License:** Public domain (US government work)

**Registration Required:** No

**Terms of Use:**
- Freely available for any use
- No restrictions on redistribution
- Attribution appreciated but not required

**Access:** https://www.ncbi.nlm.nih.gov/clinvar/

### TCGA-PRAD

**License:** Open access (dbGaP)

**Registration Required:**
- No (for published MC3 data)
- Yes (for controlled-access data)

**Terms of Use:**
- Open access for published data
- Must agree to dbGaP terms for controlled data
- Must cite TCGA consortium

**Access:** https://www.cancer.gov/tcga

### AlphaMissense

**License:** CC BY 4.0 (Creative Commons Attribution)

**Registration Required:** No

**Terms of Use:**
- Free for commercial and non-commercial use
- Must provide attribution
- Can redistribute with proper citation

**Access:** https://github.com/google-deepmind/alphamissense

---

## Citation Requirements

### Required Citations

**If you use this benchmark, please cite:**

**1. Prostate-VarBench (this work):**
```bibtex
@inproceedings{tavara2025prostatevarbench,
  title={Prostate-VarBench: A Benchmark with Interpretable TabNet Framework
         for Prostate Cancer Variant Classification},
  author={Tavara, Abraham Francisco Arellano and Kumar, Umesh and
          Pradeepkumar, Jathurshan and Sun, Jimeng},
  booktitle={Proceedings of Machine Learning Research},
  volume={297},
  year={2025},
  organization={ML4H}
}
```

**2. COSMIC:**
```bibtex
@article{tate2018cosmic,
  title={COSMIC: the catalogue of somatic mutations in cancer},
  author={Tate, John G and Bamford, Sally and Jubb, Harry C and others},
  journal={Nucleic Acids Research},
  volume={47},
  pages={D941--D947},
  year={2018}
}
```

**3. ClinVar:**
```bibtex
@article{landrum2016clinvar,
  title={ClinVar: public archive of interpretations of clinically relevant variants},
  author={Landrum, Melissa J and Lee, Jennifer M and others},
  journal={Nucleic Acids Research},
  volume={44},
  pages={D862--D868},
  year={2016}
}
```

**4. TCGA-PRAD:**
```bibtex
@article{tcga2015molecular,
  title={The molecular taxonomy of primary prostate cancer},
  author={{Cancer Genome Atlas Research Network}},
  journal={Cell},
  volume={163},
  pages={1011--1025},
  year={2015}
}
```

**5. AlphaMissense:**
```bibtex
@article{cheng2023alphamissense,
  title={Accurate proteome-wide missense variant effect prediction with AlphaMissense},
  author={Cheng, Jun and Novati, Guido and Pan, Joshua and others},
  journal={Science},
  volume={381},
  pages={eadg7492},
  year={2023}
}
```

**6. Ensembl VEP:**
```bibtex
@article{mclaren2016vep,
  title={The Ensembl Variant Effect Predictor},
  author={McLaren, William and Gil, Laurent and Hunt, Sarah E and others},
  journal={Genome Biology},
  volume={17},
  pages={122},
  year={2016}
}
```

---

## Data Update Policy

### When to Update Data

**COSMIC:** Yearly (major releases)
- New versions: 2-3 times per year
- Recommended: Update annually for latest cancer mutations

**ClinVar:** Monthly
- Updates: First week of each month
- Recommended: Update quarterly for clinical applications

**TCGA-PRAD:** Static (published dataset)
- No updates expected
- MC3 dataset is final consensus calls

**AlphaMissense:** Static (2023 release)
- One-time prediction set
- No updates unless new model version released

### Update Instructions

```bash
# Update COSMIC
python scripts/download/download_cosmic.py --version v103

# Update ClinVar
python scripts/download/download_clinvar.py

# Re-run pipeline
bash run_full_pipeline.sh
```

---

## Data Privacy & Ethics

### Patient Data Protection

- **No patient identifiers:** All data de-identified
- **Institutional Review Board:** Not required (public datasets only)
- **HIPAA Compliance:** Not applicable (no PHI)
- **Consent:** Not required (public research datasets)

### Ethical Use

This benchmark should be used for:
- ✓ Academic research
- ✓ Clinical decision support development
- ✓ Machine learning model training
- ✓ Variant interpretation improvement

This benchmark should NOT be used for:
- ✗ Individual patient diagnosis without clinical oversight
- ✗ Direct-to-consumer genetic testing
- ✗ Discriminatory practices
- ✗ Commercial use without appropriate licensing

---

**For questions about data sources, contact:**
- Abraham Arellano: aa107@illinois.edu
- Project Repository: https://github.com/AbrahamArellano/uiuc-cancer-research/
