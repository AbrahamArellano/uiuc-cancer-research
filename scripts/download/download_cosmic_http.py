#!/usr/bin/env python3
"""
COSMIC Database Downloader (HTTP API v1)

Downloads COSMIC data via the official API endpoint.
Based on documentation from: https://cancer.sanger.ac.uk/cosmic/download

API Endpoint: /api/mono/products/v1/downloads/scripted

Requires COSMIC account registration at:
    https://cancer.sanger.ac.uk/cosmic/register

Usage:
    python download_cosmic_http.py
    python download_cosmic_http.py --version v102 --assembly GRCh38

Author: UIUC Cancer Research Team
License: MIT
Date: November 2025
"""

import argparse
import hashlib
import tarfile
import base64
import time
import os
from pathlib import Path
from datetime import datetime
import sys
import json

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed")
    print("Please install: pip install requests")
    sys.exit(1)

# COSMIC API Configuration (CORRECT endpoint as of 2025)
COSMIC_API_BASE = 'https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted'
DOWNLOAD_TIMEOUT = 7200  # 2 hours for large files


def print_header(message):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70 + "\n")


def calculate_checksum(file_path):
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_download_url(email, password, file_path):
    """
    Get temporary download URL from COSMIC API.

    Args:
        email: COSMIC account email
        password: COSMIC account password
        file_path: Path to file (e.g., grch38/cosmic/v102/Cosmic_MutantCensus_Tsv_v102_GRCh38.tar)

    Returns:
        Temporary download URL (valid for 1 hour)
    """
    # Build API URL with query parameters
    url = f"{COSMIC_API_BASE}?path={file_path}&bucket=downloads"

    # Create Base64 auth header (with newline to match echo behavior)
    auth_string = f"{email}:{password}\n"
    auth_base64 = base64.b64encode(auth_string.encode()).decode()
    headers = {"Authorization": f"Basic {auth_base64}"}

    print(f"Requesting download URL from COSMIC API...")
    print(f"  File: {file_path}")

    response = requests.get(url, headers=headers, timeout=60)

    if response.status_code == 401:
        raise Exception("Authentication failed. Check COSMIC_EMAIL and COSMIC_PASSWORD")
    elif response.status_code != 200:
        raise Exception(f"API error: HTTP {response.status_code} - {response.text[:200]}")

    # Parse JSON response
    try:
        data = response.json()
        download_url = data.get('url')
        if not download_url:
            raise Exception(f"No URL in response: {response.text[:200]}")
        return download_url
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON response: {response.text[:200]}")


def download_file(download_url, output_path, max_retries=3):
    """
    Download file from temporary URL.

    Args:
        download_url: Temporary S3 download URL
        output_path: Local path to save file
        max_retries: Number of retry attempts

    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    filename = output_path.name

    for attempt in range(max_retries):
        try:
            print(f"\n[Attempt {attempt + 1}/{max_retries}] Downloading {filename}...")

            response = requests.get(download_url, stream=True, timeout=DOWNLOAD_TIMEOUT)

            if response.status_code != 200:
                raise Exception(f"Download failed: HTTP {response.status_code}")

            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            total_mb = total_size / (1024 * 1024)
            print(f"  File size: {total_mb:.1f} MB")

            # Download with progress
            downloaded = 0
            chunk_size = 8192

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            print(f"  Progress: {progress:.1f}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)", end='\r')

            print(f"\n✓ Successfully downloaded: {filename}")
            return output_path

        except Exception as e:
            print(f"\n✗ Error on attempt {attempt + 1}: {str(e)}")

            if attempt < max_retries - 1:
                retry_delay = 10 * (attempt + 1)
                print(f"  → Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")


def extract_tar(tar_path, output_dir):
    """Extract tar file and return path to TSV file"""
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)

    print(f"\nExtracting {tar_path.name}...")

    with tarfile.open(tar_path, 'r') as tar:
        # List contents
        members = tar.getnames()
        print(f"  Archive contains {len(members)} files:")
        for m in members[:5]:
            print(f"    - {m}")
        if len(members) > 5:
            print(f"    ... and {len(members) - 5} more")

        # Extract all
        tar.extractall(path=output_dir)

    print(f"✓ Extracted to: {output_dir}")

    # Find the main TSV file
    tsv_files = list(output_dir.glob("*.tsv"))
    if tsv_files:
        return tsv_files[0]
    return None


def save_metadata(output_dir, version, assembly, file_path, checksum):
    """Save download metadata for reproducibility"""
    metadata = {
        'download_date': datetime.now().isoformat(),
        'cosmic_version': version,
        'genome_assembly': assembly,
        'file_downloaded': file_path,
        'sha256_checksum': checksum,
        'api_endpoint': COSMIC_API_BASE
    }

    metadata_file = Path(output_dir) / 'cosmic_download_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved: {metadata_file.name}")


def download_cosmic(email, password, version='v102', assembly='GRCh38', output_dir=None):
    """
    Main download function for COSMIC MutantCensus data.

    Args:
        email: COSMIC account email
        password: COSMIC account password
        version: COSMIC version (e.g., 'v102')
        assembly: Genome assembly ('GRCh37' or 'GRCh38')
        output_dir: Output directory

    Returns:
        Path to downloaded/extracted file
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd() / 'data' / 'raw' / 'variants'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build file path (lowercase assembly for API)
    assembly_lower = assembly.lower()
    filename = f"Cosmic_MutantCensus_Tsv_{version}_{assembly}.tar"
    file_path = f"{assembly_lower}/cosmic/{version}/{filename}"

    print_header("COSMIC DOWNLOAD CONFIGURATION")
    print(f"API Endpoint:   {COSMIC_API_BASE}")
    print(f"Version:        {version}")
    print(f"Assembly:       {assembly}")
    print(f"File:           {filename}")
    print(f"Output Dir:     {output_dir}")
    print(f"Email:          {email}")

    # Check if already downloaded
    tar_output = output_dir / filename
    if tar_output.exists():
        print(f"\n✓ File already exists: {tar_output}")
        print("  Skipping download. Delete file to re-download.")
    else:
        # Step 1: Get download URL
        print_header("STEP 1: GET DOWNLOAD URL")
        download_url = get_download_url(email, password, file_path)
        print(f"✓ Got temporary download URL (valid for 1 hour)")

        # Step 2: Download file
        print_header("STEP 2: DOWNLOAD FILE")
        download_file(download_url, tar_output)

    # Step 3: Calculate checksum
    print_header("STEP 3: VERIFY INTEGRITY")
    print("Calculating checksum...")
    checksum = calculate_checksum(tar_output)
    print(f"SHA256: {checksum}")

    # Step 4: Extract tar
    print_header("STEP 4: EXTRACT ARCHIVE")
    tsv_file = extract_tar(tar_output, output_dir)

    # Step 5: Save metadata
    save_metadata(output_dir, version, assembly, file_path, checksum)

    return tsv_file or tar_output


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Download COSMIC MutantCensus data via API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using environment variables
  export COSMIC_EMAIL=your@email.com
  export COSMIC_PASSWORD=your_password
  python download_cosmic_http.py

  # Download specific version
  python download_cosmic_http.py --version v102 --assembly GRCh38

  # Download to specific directory
  python download_cosmic_http.py --output /path/to/output
        """
    )

    parser.add_argument('--email',
                       default=os.getenv('COSMIC_EMAIL'),
                       help='COSMIC account email (or set COSMIC_EMAIL env var)')
    parser.add_argument('--password',
                       default=os.getenv('COSMIC_PASSWORD'),
                       help='COSMIC account password (or set COSMIC_PASSWORD env var)')
    parser.add_argument('--version',
                       default='v102',
                       help='COSMIC version (default: v102)')
    parser.add_argument('--assembly',
                       default='GRCh38',
                       choices=['GRCh37', 'GRCh38'],
                       help='Genome assembly (default: GRCh38)')
    parser.add_argument('--output',
                       help='Output directory (default: data/raw/variants)')

    args = parser.parse_args()

    # Validate credentials
    if not args.email or not args.password:
        print("ERROR: COSMIC credentials required")
        print("\nProvide credentials via:")
        print("  1. Command-line: --email EMAIL --password PASSWORD")
        print("  2. Environment: export COSMIC_EMAIL=... COSMIC_PASSWORD=...")
        print("  3. .env file: set -a && source .env && set +a")
        print("\nRegister at: https://cancer.sanger.ac.uk/cosmic/register")
        sys.exit(1)

    try:
        print_header("COSMIC DATABASE DOWNLOADER")

        result = download_cosmic(
            email=args.email,
            password=args.password,
            version=args.version,
            assembly=args.assembly,
            output_dir=args.output
        )

        print_header("DOWNLOAD COMPLETE")
        print(f"✓ Output: {result}")
        if result:
            print(f"✓ Size: {result.stat().st_size / (1024*1024):.1f} MB")

    except Exception as e:
        print_header("ERROR")
        print(f"✗ {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
