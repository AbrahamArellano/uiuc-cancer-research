"""
Main script to run the data processing pipeline.

This module provides a Python entry point for the Prostate-VarBench pipeline.
For full pipeline execution, use: bash scripts/run_pipeline.sh

Usage:
    python src/main.py [--skip-download] [--skip-vep]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing.tcga_prad_preprocessor import TCGAPRADPreprocessor
from src.utils.logger import setup_logger
from config.pipeline_config import (
    RAW_DATA,
    PROCESSED_DATA,
    PREPROCESSING_CONFIG,
    VALIDATION_CONFIG
)


def process_cosmic(config: Dict[str, Any], logger) -> Path:
    """
    Filter COSMIC data for prostate-specific variants.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Path to filtered COSMIC CSV
    """
    logger.info("Processing COSMIC data...")

    output_dir = PROCESSED_DATA / "cosmic_prostate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cosmic_prostate.csv"

    # Run the filter script
    script_path = Path(__file__).parent.parent / "scripts" / "variants" / "filter_cosmic_prostate.py"

    if script_path.exists():
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"COSMIC filtering failed: {result.stderr}")
            raise RuntimeError("COSMIC filtering failed")
        logger.info(f"COSMIC data filtered successfully: {output_path}")
    else:
        logger.warning(f"COSMIC filter script not found: {script_path}")

    return output_path


def process_clinvar(config: Dict[str, Any], logger) -> Path:
    """
    Filter ClinVar data for prostate cancer genes.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Path to filtered ClinVar CSV
    """
    logger.info("Processing ClinVar data...")

    output_dir = PROCESSED_DATA / "clinvar_prostate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clinvar_prostate.csv"

    # Run the filter script
    script_path = Path(__file__).parent.parent / "scripts" / "variants" / "filter_clinvar_prostate.py"

    if script_path.exists():
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"ClinVar filtering failed: {result.stderr}")
            raise RuntimeError("ClinVar filtering failed")
        logger.info(f"ClinVar data filtered successfully: {output_path}")
    else:
        logger.warning(f"ClinVar filter script not found: {script_path}")

    return output_path


def process_tcga_prad(config: Dict[str, Any], logger) -> Path:
    """
    Process TCGA-PRAD data.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Path to processed TCGA-PRAD CSV
    """
    logger.info("Processing TCGA-PRAD data...")

    try:
        # Initialize preprocessor
        preprocessor = TCGAPRADPreprocessor(config)

        # Create output directory
        output_dir = PROCESSED_DATA / "tcga_prad"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process data
        output_path = output_dir / "merged_clinical_dicom_dataset.csv"
        preprocessor.process(None, output_path)

        logger.info(f"TCGA-PRAD data processed successfully: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error processing TCGA-PRAD data: {str(e)}")
        raise


def merge_datasets(config: Dict[str, Any], logger) -> Path:
    """
    Merge COSMIC, ClinVar, and TCGA-PRAD datasets.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Path to merged dataset
    """
    logger.info("Merging datasets...")

    script_path = Path(__file__).parent.parent / "scripts" / "merge" / "merge_datasets.py"

    if script_path.exists():
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Dataset merging failed: {result.stderr}")
            raise RuntimeError("Dataset merging failed")

        output_path = PROCESSED_DATA / "merged" / "merged_prostate_variants.csv"
        logger.info(f"Datasets merged successfully: {output_path}")
        return output_path
    else:
        logger.warning(f"Merge script not found: {script_path}")
        return None


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description="Prostate-VarBench Data Processing Pipeline"
    )
    parser.add_argument(
        "--skip-cosmic", action="store_true",
        help="Skip COSMIC filtering"
    )
    parser.add_argument(
        "--skip-clinvar", action="store_true",
        help="Skip ClinVar filtering"
    )
    parser.add_argument(
        "--skip-tcga", action="store_true",
        help="Skip TCGA-PRAD processing"
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip dataset merging"
    )
    args = parser.parse_args()

    logger = setup_logger("main_pipeline")
    logger.info("Starting Prostate-VarBench data processing pipeline")

    # Create necessary directories
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        "RAW_DATA": RAW_DATA,
        "PREPROCESSING_CONFIG": PREPROCESSING_CONFIG,
        "VALIDATION_CONFIG": VALIDATION_CONFIG
    }

    try:
        # Step 1: Process COSMIC
        if not args.skip_cosmic:
            process_cosmic(config, logger)
        else:
            logger.info("Skipping COSMIC processing")

        # Step 2: Process ClinVar
        if not args.skip_clinvar:
            process_clinvar(config, logger)
        else:
            logger.info("Skipping ClinVar processing")

        # Step 3: Process TCGA-PRAD
        if not args.skip_tcga:
            process_tcga_prad(config, logger)
        else:
            logger.info("Skipping TCGA-PRAD processing")

        # Step 4: Merge datasets
        if not args.skip_merge:
            merge_datasets(config, logger)
        else:
            logger.info("Skipping dataset merging")

        logger.info("Pipeline completed successfully")
        logger.info("Next steps:")
        logger.info("  1. Run VEP annotation: bash scripts/vep/run_vep_annotation.sh")
        logger.info("  2. Run full pipeline: bash scripts/run_pipeline.sh")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 