#!/usr/bin/env python3
"""
TCGA-PRAD Data Downloader

Downloads TCGA-PRAD (The Cancer Genome Atlas - Prostate Adenocarcinoma) data
from the GDC (Genomic Data Commons) Data Portal.

Data includes:
- Masked Somatic Mutations (MAF files)
- Clinical supplements
- Biospecimen supplements

Usage:
    python download_tcga_prad.py
    python download_tcga_prad.py --token /path/to/gdc-token.txt
    python download_tcga_prad.py --data-types mutation clinical

Author: UIUC Cancer Research Team
License: MIT
"""

import argparse
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime
import sys
import time


# GDC API Configuration
GDC_API_BASE = "https://api.gdc.cancer.gov"
GDC_DATA_ENDPOINT = f"{GDC_API_BASE}/data"
GDC_FILES_ENDPOINT = f"{GDC_API_BASE}/files"


def calculate_checksum(file_path):
    """Calculate MD5 checksum of a file"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def query_gdc_files(project_id='TCGA-PRAD', data_type='Masked Somatic Mutation', max_results=10000):
    """
    Query GDC API for TCGA-PRAD files

    Args:
        project_id: TCGA project ID (default: TCGA-PRAD)
        data_type: Type of data to download
        max_results: Maximum number of results

    Returns:
        List of file metadata dictionaries
    """
    print(f"Querying GDC API for {project_id} files...")
    print(f"Data type: {data_type}\n")

    # Build query filters
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type]
                }
            }
        ]
    }

    # Build query parameters
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,md5sum,file_size,data_type,cases.submitter_id",
        "format": "JSON",
        "size": str(max_results)
    }

    try:
        response = requests.get(GDC_FILES_ENDPOINT, params=params)
        response.raise_for_status()

        data = response.json()
        files = data['data']['hits']

        print(f"✓ Found {len(files)} files")
        print(f"  Total data size: {sum(f['file_size'] for f in files) / (1024**3):.2f} GB\n")

        return files

    except Exception as e:
        print(f"ERROR: Failed to query GDC API: {e}")
        sys.exit(1)


def download_gdc_file(file_id, file_name, output_dir, token=None, verify_checksum=True, expected_md5=None):
    """
    Download a single file from GDC

    Args:
        file_id: GDC file UUID
        file_name: Name for the downloaded file
        output_dir: Output directory path
        token: GDC authentication token (optional)
        verify_checksum: Whether to verify MD5 checksum
        expected_md5: Expected MD5 checksum from metadata

    Returns:
        Path to downloaded file
    """
    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    output_path = output_dir / file_name

    # Skip if file already exists
    if output_path.exists():
        print(f"  File already exists, skipping: {file_name}")
        return output_path

    # Set up headers with token if provided
    headers = {}
    if token:
        headers['X-Auth-Token'] = token

    try:
        print(f"  Downloading: {file_name}")
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        # Download with progress
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r    Progress: {percent:.1f}% ({downloaded/(1024**2):.1f}/{total_size/(1024**2):.1f} MB)",
                              end='', flush=True)

        print()  # New line after progress

        # Verify checksum if requested
        if verify_checksum and expected_md5:
            print(f"  Verifying checksum...")
            actual_md5 = calculate_checksum(output_path)
            if actual_md5 == expected_md5:
                print(f"  ✓ Checksum verified")
            else:
                print(f"  WARNING: Checksum mismatch!")
                print(f"  Expected: {expected_md5}")
                print(f"  Actual:   {actual_md5}")

        return output_path

    except Exception as e:
        print(f"\n  ERROR: Failed to download {file_name}: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        raise


def download_tcga_prad(data_types=None, output_dir=None, token_path=None, max_files=None):
    """
    Download TCGA-PRAD data from GDC

    Args:
        data_types: List of data types to download (default: ['Masked Somatic Mutation'])
        output_dir: Output directory path
        token_path: Path to GDC token file (for controlled-access data)
        max_files: Maximum number of files to download (for testing)

    Returns:
        Dictionary with download statistics
    """
    if data_types is None:
        data_types = ['Masked Somatic Mutation']

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'TCGA-PRAD'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load token if provided
    token = None
    if token_path:
        token_path = Path(token_path)
        if token_path.exists():
            with open(token_path, 'r') as f:
                token = f.read().strip()
            print(f"✓ Loaded GDC authentication token\n")
        else:
            print(f"WARNING: Token file not found: {token_path}\n")

    print(f"{'='*70}")
    print(f"TCGA-PRAD Download Configuration")
    print(f"{'='*70}")
    print(f"Data types:     {', '.join(data_types)}")
    print(f"Output dir:     {output_dir}")
    print(f"Token:          {'Yes' if token else 'No (open-access only)'}")
    print(f"Max files:      {max_files if max_files else 'All'}")
    print(f"{'='*70}\n")

    all_downloads = []
    download_stats = {
        'total_files': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'total_size_gb': 0
    }

    for data_type in data_types:
        print(f"\n{'='*70}")
        print(f"Processing: {data_type}")
        print(f"{'='*70}\n")

        # Query for files
        files = query_gdc_files(project_id='TCGA-PRAD', data_type=data_type)

        if not files:
            print(f"No files found for {data_type}\n")
            continue

        # Limit files if requested
        if max_files:
            files = files[:max_files]
            print(f"Limiting to {max_files} files\n")

        # Create subdirectory for this data type
        type_dir = output_dir / data_type.replace(' ', '_').lower()
        type_dir.mkdir(parents=True, exist_ok=True)

        # Download each file
        for i, file_info in enumerate(files, 1):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            expected_md5 = file_info.get('md5sum')
            file_size = file_info['file_size']

            print(f"\n[{i}/{len(files)}] {file_name}")
            print(f"  File ID:  {file_id}")
            print(f"  Size:     {file_size/(1024**2):.2f} MB")

            try:
                download_path = download_gdc_file(
                    file_id=file_id,
                    file_name=file_name,
                    output_dir=type_dir,
                    token=token,
                    verify_checksum=True,
                    expected_md5=expected_md5
                )

                all_downloads.append({
                    'file_name': file_name,
                    'file_id': file_id,
                    'data_type': data_type,
                    'path': str(download_path),
                    'size_mb': file_size / (1024**2),
                    'md5': expected_md5
                })

                download_stats['successful'] += 1
                download_stats['total_size_gb'] += file_size / (1024**3)

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                download_stats['failed'] += 1

            download_stats['total_files'] += 1

            # Rate limiting to be polite to GDC servers
            time.sleep(0.5)

    # Save download manifest
    manifest_path = output_dir / f'download_manifest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(manifest_path, 'w') as f:
        json.dump(all_downloads, f, indent=2)

    # Save metadata
    metadata = {
        'project': 'TCGA-PRAD',
        'data_types': data_types,
        'download_date': datetime.now().isoformat(),
        'total_files': download_stats['total_files'],
        'successful_downloads': download_stats['successful'],
        'failed_downloads': download_stats['failed'],
        'total_size_gb': f"{download_stats['total_size_gb']:.2f}",
        'manifest_file': str(manifest_path),
        'source': 'GDC Data Portal (https://portal.gdc.cancer.gov/)'
    }

    metadata_path = output_dir / 'tcga_prad_metadata.txt'
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total files:      {download_stats['total_files']}")
    print(f"Successful:       {download_stats['successful']}")
    print(f"Failed:           {download_stats['failed']}")
    print(f"Total size:       {download_stats['total_size_gb']:.2f} GB")
    print(f"Output dir:       {output_dir}")
    print(f"Manifest:         {manifest_path}")
    print(f"Metadata:         {metadata_path}")
    print(f"{'='*70}\n")

    return download_stats


def main():
    parser = argparse.ArgumentParser(
        description='Download TCGA-PRAD data from GDC Data Portal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download somatic mutations (default)
    python download_tcga_prad.py

    # Download with authentication token (for controlled-access data)
    python download_tcga_prad.py --token /path/to/gdc-token.txt

    # Download multiple data types
    python download_tcga_prad.py --data-types mutation clinical biospecimen

    # Download to custom directory
    python download_tcga_prad.py --output /path/to/dir

    # Test with limited files
    python download_tcga_prad.py --max-files 5

Data Types Available:
    - Masked Somatic Mutation (MAF files)
    - Clinical Supplement
    - Biospecimen Supplement
    - Copy Number Segment
    - Gene Expression Quantification

Notes:
    - No authentication required for open-access data
    - Token required for controlled-access data (register at https://gdc.cancer.gov/)
    - Download size: ~1-5 GB depending on data types
    - Automatic checksum verification
    - Rate limiting included to be polite to GDC servers

GDC Data Portal:
    https://portal.gdc.cancer.gov/projects/TCGA-PRAD
        """
    )

    parser.add_argument('--data-types', '-d',
                       nargs='+',
                       default=['Masked Somatic Mutation'],
                       help='Data types to download (default: Masked Somatic Mutation)')

    parser.add_argument('--output', '-o',
                       help='Output directory (default: data/raw/TCGA-PRAD)')

    parser.add_argument('--token', '-t',
                       help='Path to GDC authentication token file')

    parser.add_argument('--max-files', '-m',
                       type=int,
                       help='Maximum number of files to download (for testing)')

    args = parser.parse_args()

    download_tcga_prad(
        data_types=args.data_types,
        output_dir=args.output,
        token_path=args.token,
        max_files=args.max_files
    )


if __name__ == '__main__':
    main()
