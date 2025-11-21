#!/usr/bin/env python3
"""
ClinVar Database Downloader

Downloads ClinVar VCF files from NCBI FTP server.
No authentication required - public data.

Usage:
    python download_clinvar.py
    python download_clinvar.py --assembly GRCh38
    python download_clinvar.py --output /path/to/output

Author: UIUC Cancer Research Team
License: MIT
"""

import argparse
import hashlib
import gzip
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime
import sys


# ClinVar FTP Configuration
CLINVAR_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar"


def calculate_checksum(file_path):
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_with_progress(url, output_path):
    """
    Download file with progress bar

    Args:
        url: URL to download
        output_path: Path to save file

    Returns:
        None
    """
    def report_progress(block_num, block_size, total_size):
        """Progress callback for urllib"""
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print()  # New line after progress


def download_clinvar(assembly='GRCh38', output_dir=None, include_papu=False):
    """
    Download ClinVar VCF files from NCBI FTP

    Args:
        assembly: Genome assembly (GRCh37 or GRCh38)
        output_dir: Output directory path
        include_papu: Whether to include PAPU (Prostate-specific) variants

    Returns:
        Dictionary with paths to downloaded files
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'variants'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"ClinVar Download Configuration")
    print(f"{'='*70}")
    print(f"Assembly:       {assembly}")
    print(f"Output dir:     {output_dir}")
    print(f"FTP base:       {CLINVAR_FTP_BASE}")
    print(f"{'='*70}\n")

    downloaded_files = {}

    # Main ClinVar VCF file
    vcf_url = f"{CLINVAR_FTP_BASE}/vcf_{assembly}/clinvar.vcf.gz"
    vcf_filename = f"clinvar_{assembly}.vcf.gz"
    vcf_path = output_dir / vcf_filename

    print(f"[1/3] Downloading main ClinVar VCF file...")
    print(f"URL: {vcf_url}")
    print(f"Destination: {vcf_path}\n")

    try:
        download_with_progress(vcf_url, vcf_path)
        print(f"✓ VCF file downloaded successfully\n")
        downloaded_files['vcf'] = vcf_path
    except Exception as e:
        print(f"\nERROR: Failed to download VCF file: {e}")
        sys.exit(1)

    # Download VCF index (.tbi) file
    tbi_url = f"{vcf_url}.tbi"
    tbi_filename = f"{vcf_filename}.tbi"
    tbi_path = output_dir / tbi_filename

    print(f"[2/3] Downloading VCF index file (.tbi)...")
    print(f"URL: {tbi_url}")
    print(f"Destination: {tbi_path}\n")

    try:
        download_with_progress(tbi_url, tbi_path)
        print(f"✓ Index file downloaded successfully\n")
        downloaded_files['tbi'] = tbi_path
    except Exception as e:
        print(f"\nWARNING: Failed to download index file: {e}")
        print("Index file is optional but recommended for faster access\n")

    # Download MD5 checksum file for verification
    md5_url = f"{vcf_url}.md5"
    md5_filename = f"{vcf_filename}.md5"
    md5_path = output_dir / md5_filename

    print(f"[3/3] Downloading MD5 checksum file...")
    print(f"URL: {md5_url}")
    print(f"Destination: {md5_path}\n")

    try:
        urllib.request.urlretrieve(md5_url, md5_path)
        print(f"✓ MD5 checksum file downloaded\n")
        downloaded_files['md5'] = md5_path

        # Verify checksum
        print("Verifying file integrity...")
        with open(md5_path, 'r') as f:
            expected_md5 = f.read().strip().split()[0]

        # Calculate actual MD5
        import hashlib
        md5_hash = hashlib.md5()
        with open(vcf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        actual_md5 = md5_hash.hexdigest()

        if actual_md5 == expected_md5:
            print(f"✓ Checksum verification PASSED\n")
        else:
            print(f"WARNING: Checksum verification FAILED")
            print(f"Expected: {expected_md5}")
            print(f"Actual:   {actual_md5}\n")

    except Exception as e:
        print(f"\nWARNING: Failed to download/verify MD5: {e}\n")

    # Calculate SHA256 for our own tracking
    print("Calculating SHA256 checksum for tracking...")
    sha256 = calculate_checksum(vcf_path)
    print(f"SHA256: {sha256}\n")

    # Get file size
    file_size_mb = vcf_path.stat().st_size / (1024 * 1024)

    # Optional: Decompress for easier processing
    print("Decompressing VCF file (this may take several minutes)...")
    uncompressed_path = vcf_path.with_suffix('')

    try:
        with gzip.open(vcf_path, 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Decompressed to: {uncompressed_path}\n")
        downloaded_files['vcf_uncompressed'] = uncompressed_path
    except Exception as e:
        print(f"WARNING: Decompression failed: {e}\n")

    # Log download metadata
    metadata = {
        'source': 'ClinVar',
        'assembly': assembly,
        'download_date': datetime.now().isoformat(),
        'vcf_file': str(vcf_path),
        'vcf_uncompressed': str(uncompressed_path) if uncompressed_path.exists() else 'N/A',
        'file_size_mb': f"{file_size_mb:.2f}",
        'sha256': sha256,
        'source_url': vcf_url
    }

    metadata_path = output_dir / f'clinvar_{assembly}_metadata.txt'
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    downloaded_files['metadata'] = metadata_path

    print(f"{'='*70}")
    print(f"SUCCESS: ClinVar data downloaded successfully!")
    print(f"{'='*70}")
    print(f"VCF file:         {vcf_path}")
    print(f"Uncompressed:     {uncompressed_path}")
    print(f"Index file:       {tbi_path}")
    print(f"Metadata:         {metadata_path}")
    print(f"File size:        {file_size_mb:.2f} MB")
    print(f"{'='*70}\n")

    return downloaded_files


def main():
    parser = argparse.ArgumentParser(
        description='Download ClinVar VCF files from NCBI FTP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download default (GRCh38)
    python download_clinvar.py

    # Download GRCh37
    python download_clinvar.py --assembly GRCh37

    # Download to custom directory
    python download_clinvar.py --output /path/to/dir

Notes:
    - No authentication required (public data)
    - Download size: ~500MB compressed, ~5GB uncompressed
    - Expected time: 5-15 minutes
    - Automatic decompression after download
    - Checksum verification included

Data Source:
    NCBI ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
    FTP: ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/
        """
    )

    parser.add_argument('--assembly', '-a',
                       default='GRCh38',
                       choices=['GRCh37', 'GRCh38'],
                       help='Genome assembly (default: GRCh38)')

    parser.add_argument('--output', '-o',
                       help='Output directory (default: data/raw/variants)')

    parser.add_argument('--include-papu',
                       action='store_true',
                       help='Include PAPU (population-specific) variants')

    args = parser.parse_args()

    download_clinvar(
        assembly=args.assembly,
        output_dir=args.output,
        include_papu=args.include_papu
    )


if __name__ == '__main__':
    main()
