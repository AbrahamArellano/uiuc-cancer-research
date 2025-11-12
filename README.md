# TCGA-PRAD Data Processing Pipeline

This project implements a data processing pipeline for analyzing TCGA-PRAD (The Cancer Genome Atlas Prostate Adenocarcinoma) dataset, integrating clinical, biospecimen, and medical imaging data.

## Project Structure

```
cancer-research/
├── data/
│   ├── raw/                    # Raw data directories
│   │   ├── TCGA-PRAD/         # TCGA Prostate Cancer data
│   │   ├── COSMIC/            # COSMIC database data
│   │   └── clinvar/           # ClinVar database data
│   ├── processed/             # Processed data files
│   └── interim/               # Intermediate data files
├── src/
│   ├── data/                  # Data processing scripts
│   │   ├── preprocessing/     # Data preprocessing modules
│   │   ├── validation/        # Data validation modules
│   │   └── transformation/    # Data transformation modules
│   ├── features/              # Feature engineering code
│   ├── visualization/         # Data visualization code
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                    # Test files
├── config/                   # Configuration files
├── docs/                     # Documentation
└── requirements.txt          # Project dependencies
```

## Features

- Processes TCGA-PRAD clinical and biospecimen data
- Extracts metadata from DICOM imaging files
- Merges clinical, biospecimen, and imaging data
- Performs data validation and cleaning
- Handles missing values and data transformations
- Provides comprehensive logging

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd cancer-research
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

1. Place your raw data files in the appropriate directories:
   - TCGA-PRAD data in `data/raw/TCGA-PRAD/`
   - Clinical JSON files in `data/raw/TCGA-PRAD/clinical/`
   - DICOM files in `data/raw/TCGA-PRAD/images/`

2. Update configuration in `config/pipeline_config.py` if needed

## Usage

Run the main pipeline:
```bash
python src/main.py
```

The pipeline will:
1. Load clinical, biospecimen, and DICOM data
2. Clean and preprocess the data
3. Merge all data sources
4. Validate the results
5. Save processed data to `data/processed/tcga_prad/`

## Data Processing Details

### DICOM Processing
- Extracts metadata from DICOM files
- Supports multiple imaging modalities (CT, PT, MR)
- Links imaging data with clinical records via patient ID

### Clinical Data Processing
- Processes clinical and biospecimen JSON files
- Handles missing values and data type conversions
- Merges multiple data sources

### Validation
- Checks for data completeness
- Validates relationships between datasets
- Ensures data quality standards

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TCGA-PRAD dataset providers
- The Cancer Imaging Archive (TCIA)
- National Cancer Institute (NCI) 