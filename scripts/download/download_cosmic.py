#!/usr/bin/env python3
"""
COSMIC Database Downloader

Downloads COSMIC Mutant Census data via SFTP.
Requires COSMIC account registration at: https://cancer.sanger.ac.uk/cosmic/register

Usage:
    python download_cosmic.py --email your@email.com --password your_token
    python download_cosmic.py --version v102 --assembly GRCh38

Author: UIUC Cancer Research Team
License: MIT
"""

import argparse
import hashlib
import gzip
import shutil
from pathlib import Path
from datetime import datetime
import sys

try:
    import paramiko
except ImportError:
    print("ERROR: paramiko library not installed")
    print("Please install: pip install paramiko")
    sys.exit(1)

# COSMIC SFTP Configuration
COSMIC_SFTP_HOST = 'sftp-cancer.sanger.ac.uk'
COSMIC_SFTP_PORT = 22


def calculate_checksum(file_path):
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_cosmic_sftp(email, password, version='v102', assembly='GRCh38', output_dir=None):
    """
    Download COSMIC Mutant Census via SFTP

    Args:
        email: COSMIC account email
        password: COSMIC account password/token
        version: COSMIC database version (default: v102)
        assembly: Genome assembly (default: GRCh38)
        output_dir: Output directory path

    Returns:
        Path to downloaded file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'variants'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct remote file path
    remote_path = f'/cosmic/{version}/MutantCensus/Cosmic_MutantCensus_{version}_{assembly}.tsv.gz'
    local_filename = f'Cosmic_MutantCensus_{version}_{assembly}.tsv.gz'
    local_path = output_dir / local_filename

    print(f"{'='*70}")
    print(f"COSMIC Download Configuration")
    print(f"{'='*70}")
    print(f"Host:           {COSMIC_SFTP_HOST}")
    print(f"Remote path:    {remote_path}")
    print(f"Local path:     {local_path}")
    print(f"Version:        {version}")
    print(f"Assembly:       {assembly}")
    print(f"{'='*70}\n")

    try:
        print("Connecting to COSMIC SFTP server...")
        transport = paramiko.Transport((COSMIC_SFTP_HOST, COSMIC_SFTP_PORT))
        transport.connect(username=email, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        print(f"Connected successfully!")
        print(f"\nDownloading {remote_path}...")
        print(f"This may take 10-30 minutes depending on your connection speed.\n")

        # Download with progress callback
        def progress_callback(transferred, total):
            percent = (transferred / total) * 100
            mb_transferred = transferred / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_transferred:.1f}/{mb_total:.1f} MB)", end='', flush=True)

        # Get file size
        file_attr = sftp.stat(remote_path)
        file_size_mb = file_attr.st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB\n")

        # Download file
        sftp.get(remote_path, str(local_path), callback=progress_callback)
        print("\n\nDownload completed!")

        sftp.close()
        transport.close()

        # Calculate checksum
        print(f"\nCalculating checksum...")
        checksum = calculate_checksum(local_path)
        print(f"SHA256: {checksum}")

        # Decompress if needed
        if local_path.suffix == '.gz':
            print(f"\nDecompressing file...")
            uncompressed_path = local_path.with_suffix('')
            with gzip.open(local_path, 'rb') as f_in:
                with open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed to: {uncompressed_path}")

            # Keep compressed file for storage, but also have uncompressed
            print(f"Compressed file retained: {local_path}")

        # Log download metadata
        metadata = {
            'source': 'COSMIC',
            'version': version,
            'assembly': assembly,
            'download_date': datetime.now().isoformat(),
            'file_path': str(local_path),
            'file_size_mb': file_size_mb,
            'sha256': checksum,
            'remote_url': f"sftp://{COSMIC_SFTP_HOST}{remote_path}"
        }

        metadata_path = output_dir / f'cosmic_{version}_metadata.txt'
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        print(f"\n{'='*70}")
        print(f"SUCCESS: COSMIC data downloaded successfully!")
        print(f"{'='*70}")
        print(f"File location: {local_path}")
        print(f"Metadata:      {metadata_path}")
        print(f"{'='*70}\n")

        return local_path

    except paramiko.AuthenticationException:
        print(f"\n{'='*70}")
        print(f"ERROR: Authentication failed")
        print(f"{'='*70}")
        print(f"Please check your COSMIC credentials.")
        print(f"Register at: https://cancer.sanger.ac.uk/cosmic/register")
        print(f"{'='*70}\n")
        sys.exit(1)

    except FileNotFoundError:
        print(f"\n{'='*70}")
        print(f"ERROR: File not found on COSMIC server")
        print(f"{'='*70}")
        print(f"Remote path: {remote_path}")
        print(f"Please check version ({version}) and assembly ({assembly})")
        print(f"{'='*70}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Download failed")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        print(f"{'='*70}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download COSMIC Mutant Census data via SFTP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download with credentials
    python download_cosmic.py --email user@example.com --password my_token

    # Download specific version
    python download_cosmic.py --email user@example.com --password my_token --version v102

    # Download to custom directory
    python download_cosmic.py --email user@example.com --password my_token --output /path/to/dir

Requirements:
    - COSMIC account (register at https://cancer.sanger.ac.uk/cosmic/register)
    - paramiko library (pip install paramiko)

Notes:
    - Download size: ~2GB compressed
    - Expected time: 10-30 minutes
    - Automatic decompression after download
        """
    )

    parser.add_argument('--email', '-e',
                       required=True,
                       help='COSMIC account email')

    parser.add_argument('--password', '-p',
                       required=True,
                       help='COSMIC account password/token')

    parser.add_argument('--version', '-v',
                       default='v102',
                       help='COSMIC version (default: v102)')

    parser.add_argument('--assembly', '-a',
                       default='GRCh38',
                       choices=['GRCh37', 'GRCh38'],
                       help='Genome assembly (default: GRCh38)')

    parser.add_argument('--output', '-o',
                       help='Output directory (default: data/raw/variants)')

    args = parser.parse_args()

    download_cosmic_sftp(
        email=args.email,
        password=args.password,
        version=args.version,
        assembly=args.assembly,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
