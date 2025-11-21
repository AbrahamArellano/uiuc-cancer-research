#!/usr/bin/env python3
"""
Master Data Download Orchestrator

Downloads all required datasets for the UIUC Cancer Research project:
- COSMIC Mutant Census v102 (SFTP)
- ClinVar variants (FTP)
- TCGA-PRAD data (GDC API)

Usage:
    python download_all_data.py --cosmic-email user@example.com --cosmic-password token
    python download_all_data.py --config download_config.json
    python download_all_data.py --skip cosmic  # Skip COSMIC if already downloaded

Author: UIUC Cancer Research Team
License: MIT
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess


def run_download_script(script_name, args_list):
    """
    Run a download script with arguments

    Args:
        script_name: Name of the Python script to run
        args_list: List of command-line arguments

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args_list

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Script {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to run {script_name}: {e}")
        return False


def download_all_data(cosmic_email=None, cosmic_password=None, gdc_token=None,
                      output_dir=None, skip_datasets=None, config_file=None):
    """
    Download all required datasets

    Args:
        cosmic_email: COSMIC account email
        cosmic_password: COSMIC account password/token
        gdc_token: Path to GDC authentication token
        output_dir: Base output directory
        skip_datasets: List of datasets to skip
        config_file: Path to configuration JSON file

    Returns:
        Dictionary with download results
    """
    if skip_datasets is None:
        skip_datasets = []

    # Load config file if provided
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            cosmic_email = cosmic_email or config.get('cosmic_email')
            cosmic_password = cosmic_password or config.get('cosmic_password')
            gdc_token = gdc_token or config.get('gdc_token')
            output_dir = output_dir or config.get('output_dir')
            skip_datasets = skip_datasets or config.get('skip_datasets', [])
        else:
            print(f"WARNING: Config file not found: {config_file}")

    # Set default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"UIUC Cancer Research - Data Download Pipeline")
    print(f"{'='*70}")
    print(f"Start time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output dir:     {output_dir}")
    print(f"Skip datasets:  {', '.join(skip_datasets) if skip_datasets else 'None'}")
    print(f"{'='*70}\n")

    results = {
        'cosmic': None,
        'clinvar': None,
        'tcga_prad': None,
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'success': False
    }

    # =========================================================================
    # 1. Download COSMIC
    # =========================================================================
    if 'cosmic' not in skip_datasets:
        print(f"\n{'#'*70}")
        print(f"# Step 1/3: Downloading COSMIC Mutant Census")
        print(f"{'#'*70}\n")

        if not cosmic_email or not cosmic_password:
            print("ERROR: COSMIC credentials not provided")
            print("Please provide --cosmic-email and --cosmic-password")
            print("Or add them to a config file and use --config")
            print("\nSkipping COSMIC download...")
            results['cosmic'] = 'skipped'
        else:
            cosmic_args = [
                '--email', cosmic_email,
                '--password', cosmic_password,
                '--version', 'v102',
                '--assembly', 'GRCh38',
                '--output', str(output_dir / 'variants')
            ]

            results['cosmic'] = 'success' if run_download_script('download_cosmic.py', cosmic_args) else 'failed'
    else:
        print("\n[SKIPPED] COSMIC download")
        results['cosmic'] = 'skipped'

    # =========================================================================
    # 2. Download ClinVar
    # =========================================================================
    if 'clinvar' not in skip_datasets:
        print(f"\n{'#'*70}")
        print(f"# Step 2/3: Downloading ClinVar")
        print(f"{'#'*70}\n")

        clinvar_args = [
            '--assembly', 'GRCh38',
            '--output', str(output_dir / 'variants')
        ]

        results['clinvar'] = 'success' if run_download_script('download_clinvar.py', clinvar_args) else 'failed'
    else:
        print("\n[SKIPPED] ClinVar download")
        results['clinvar'] = 'skipped'

    # =========================================================================
    # 3. Download TCGA-PRAD
    # =========================================================================
    if 'tcga_prad' not in skip_datasets:
        print(f"\n{'#'*70}")
        print(f"# Step 3/3: Downloading TCGA-PRAD")
        print(f"{'#'*70}\n")

        tcga_args = [
            '--data-types', 'Masked Somatic Mutation',
            '--output', str(output_dir / 'TCGA-PRAD')
        ]

        if gdc_token:
            tcga_args.extend(['--token', gdc_token])

        results['tcga_prad'] = 'success' if run_download_script('download_tcga_prad.py', tcga_args) else 'failed'
    else:
        print("\n[SKIPPED] TCGA-PRAD download")
        results['tcga_prad'] = 'skipped'

    # =========================================================================
    # Summary
    # =========================================================================
    results['end_time'] = datetime.now().isoformat()
    results['success'] = all(r in ['success', 'skipped'] for r in [
        results['cosmic'], results['clinvar'], results['tcga_prad']
    ])

    print(f"\n{'='*70}")
    print(f"DOWNLOAD PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"COSMIC:         {results['cosmic'].upper()}")
    print(f"ClinVar:        {results['clinvar'].upper()}")
    print(f"TCGA-PRAD:      {results['tcga_prad'].upper()}")
    print(f"Overall:        {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")
    print(f"End time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Save results to file
    results_path = output_dir / f'download_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}\n")

    if not results['success']:
        print("WARNING: Some downloads failed. Please review errors above.")
        return results

    print("✓ All downloads completed successfully!")
    print("\nNext steps:")
    print("  1. Run data filtering scripts")
    print("  2. Merge datasets")
    print("  3. Run VEP annotation")
    print("  4. Train TabNet model")
    print("\nSee README.md for detailed pipeline instructions.\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Download all required datasets for UIUC Cancer Research project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all datasets with COSMIC credentials
    python download_all_data.py \\
        --cosmic-email user@example.com \\
        --cosmic-password your_token

    # Use configuration file
    python download_all_data.py --config download_config.json

    # Skip COSMIC (if already downloaded)
    python download_all_data.py --skip cosmic

    # Download to custom directory
    python download_all_data.py \\
        --cosmic-email user@example.com \\
        --cosmic-password your_token \\
        --output /path/to/output

Configuration File Format (JSON):
    {
        "cosmic_email": "user@example.com",
        "cosmic_password": "your_token",
        "gdc_token": "/path/to/gdc-token.txt",
        "output_dir": "/path/to/output",
        "skip_datasets": ["cosmic"]
    }

Required Accounts:
    - COSMIC: Register at https://cancer.sanger.ac.uk/cosmic/register
    - GDC: Optional, for controlled-access data (https://gdc.cancer.gov/)

Expected Download Sizes:
    - COSMIC:    ~2 GB (compressed)
    - ClinVar:   ~500 MB (compressed)
    - TCGA-PRAD: ~1-2 GB
    - Total:     ~4-5 GB

Expected Download Time:
    - COSMIC:    10-30 minutes
    - ClinVar:   5-15 minutes
    - TCGA-PRAD: 10-20 minutes
    - Total:     25-65 minutes (depends on connection speed)
        """
    )

    parser.add_argument('--cosmic-email', '-e',
                       help='COSMIC account email')

    parser.add_argument('--cosmic-password', '-p',
                       help='COSMIC account password/token')

    parser.add_argument('--gdc-token', '-t',
                       help='Path to GDC authentication token file')

    parser.add_argument('--output', '-o',
                       help='Base output directory (default: data/raw)')

    parser.add_argument('--skip',
                       nargs='+',
                       choices=['cosmic', 'clinvar', 'tcga_prad'],
                       help='Datasets to skip')

    parser.add_argument('--config', '-c',
                       help='Path to configuration JSON file')

    args = parser.parse_args()

    results = download_all_data(
        cosmic_email=args.cosmic_email,
        cosmic_password=args.cosmic_password,
        gdc_token=args.gdc_token,
        output_dir=args.output,
        skip_datasets=args.skip,
        config_file=args.config
    )

    # Exit with error code if downloads failed
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
