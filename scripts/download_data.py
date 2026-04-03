#!/usr/bin/env python3
"""
Download the Sen1Floods11 Hand-Labeled dataset from Google Cloud Storage.

The dataset (~4 GB) contains Sentinel-1 SAR imagery and binary flood masks
for 11 flood events worldwide.

Paper  : https://github.com/cloudtostreet/Sen1Floods11
Bucket : gs://sen1floods11/

Prerequisites (choose one):
  Option A — Google Cloud SDK (recommended, parallel download):
      https://cloud.google.com/sdk/docs/install
      > gcloud auth login          (or use service account)

  Option B — Python google-cloud-storage library (no auth for public bucket):
      pip install google-cloud-storage

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output data/raw --method gsutil
    python scripts/download_data.py --output data/raw --method python
"""

import os
import sys
import subprocess
import argparse


GCS_BUCKET       = 'gs://sen1floods11'
HANDLABELED_PATH = 'v1.1/data/flood_events/HandLabeled'
S1_DIR           = 'S1Hand'
LABEL_BUCKET_DIR = 'LabelHand'  # remote bucket name
LABEL_LOCAL_DIR  = 'S1HandLabels' # local folder name
PUBLIC_BUCKET    = 'sen1floods11'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_gsutil() -> bool:
    try:
        r = subprocess.run(['gsutil', 'version'], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def download_with_gsutil(out_dir: str):
    """Parallel download via gsutil -m cp."""
    s1_gcs    = f"{GCS_BUCKET}/{HANDLABELED_PATH}/{S1_DIR}/"
    label_gcs = f"{GCS_BUCKET}/{HANDLABELED_PATH}/{LABEL_BUCKET_DIR}/"
    s1_local    = os.path.join(out_dir, 'S1Hand')
    label_local = os.path.join(out_dir, LABEL_LOCAL_DIR)
    os.makedirs(s1_local,    exist_ok=True)
    os.makedirs(label_local, exist_ok=True)

    print(f"\n[1/2] Downloading Sentinel-1 images → {s1_local}")
    subprocess.run(['gsutil', '-m', 'cp', '-r', s1_gcs, s1_local], check=True)

    print(f"\n[2/2] Downloading flood labels → {label_local}")
    subprocess.run(['gsutil', '-m', 'cp', '-r', label_gcs, label_local], check=True)

    print("\n✓ Download complete!")


def download_with_python(out_dir: str):
    """Download using the google-cloud-storage Python library (no auth needed)."""
    try:
        from google.cloud import storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed.")
        print("Run:  pip install google-cloud-storage")
        print("\nAlternatively, install Google Cloud SDK and use --method gsutil")
        sys.exit(1)

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(PUBLIC_BUCKET)
    s1_local    = os.path.join(out_dir, 'S1Hand')
    label_local = os.path.join(out_dir, LABEL_LOCAL_DIR)
    os.makedirs(s1_local,    exist_ok=True)
    os.makedirs(label_local, exist_ok=True)

    def _dl_prefix(prefix: str, local_dir: str, tag: str):
        blobs = list(bucket.list_blobs(prefix=prefix))
        total = len(blobs)
        print(f"\n{tag}: {total} files")
        for i, blob in enumerate(blobs, 1):
            fname = os.path.basename(blob.name)
            if not fname:
                continue
            dst = os.path.join(local_dir, fname)
            if os.path.exists(dst):
                print(f"  [{i}/{total}] Skip (exists): {fname}")
            else:
                print(f"  [{i}/{total}] Downloading : {fname}")
                blob.download_to_filename(dst)
        print(f"  ✓ {tag} done.")

    _dl_prefix(f"{HANDLABELED_PATH}/{S1_DIR}/",    s1_local,    "Sentinel-1 images")
    _dl_prefix(f"{HANDLABELED_PATH}/{LABEL_BUCKET_DIR}/", label_local, "Flood label masks")
    print("\n✓ Download complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Download Sen1Floods11 dataset')
    parser.add_argument('--output', '-o', default='data/raw',
                        help='Local output directory (default: data/raw)')
    parser.add_argument('--method', choices=['gsutil', 'python', 'auto'],
                        default='auto', help='Download backend (default: auto)')
    args = parser.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Sen1Floods11 — Hand-Labeled Dataset Downloader")
    print("=" * 60)
    print(f"  Output     : {os.path.abspath(out_dir)}")
    print(f"  Approx size: ~4 GB (S1Hand + S1HandLabels)")
    print()

    method = args.method
    if method == 'auto':
        method = 'gsutil' if _check_gsutil() else 'python'
        print(f"  Auto-detected backend: {method}")

    if method == 'gsutil':
        if not _check_gsutil():
            print("ERROR: gsutil not found on PATH.")
            print("Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
            sys.exit(1)
        download_with_gsutil(out_dir)
    else:
        download_with_python(out_dir)


if __name__ == '__main__':
    main()
