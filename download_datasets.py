#!/usr/bin/env python3
"""
Download missing datasets for the multimodal student performance pipeline.

Datasets fetched:
  1. UCI Student Dropout (697)  → dropout/
  2. Student Depression (Kaggle) → already may be present, else skips
  3. UCI Student Academics Performance (467) → academics/
  4. xAPI already present in xAPI/

Usage:
    python download_datasets.py
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent


def download_uci(dataset_id: int, folder_name: str):
    dest = ROOT / folder_name
    dest.mkdir(exist_ok=True)
    # UCI ML Repo direct download via their data API
    url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv"
    target = dest / "data.csv"
    if target.exists():
        print(f"  [{folder_name}] Already downloaded → {target}")
        return
    print(f"  [{folder_name}] Downloading from UCI id={dataset_id} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as f:
            f.write(resp.read())
        print(f"  [{folder_name}] Saved to {target}")
    except Exception as e:
        print(f"  [{folder_name}] Direct CSV failed ({e}), trying zip ...")
        try:
            zip_url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/dataset.zip"
            req2 = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
            zip_path = dest / "dataset.zip"
            with urllib.request.urlopen(req2, timeout=120) as resp2, open(zip_path, "wb") as zf:
                zf.write(resp2.read())
            with zipfile.ZipFile(zip_path, "r") as zr:
                zr.extractall(dest)
            zip_path.unlink(missing_ok=True)
            print(f"  [{folder_name}] Extracted zip to {dest}")
        except Exception as e2:
            print(f"  [{folder_name}] FAILED: {e2}")
            print(f"  [{folder_name}] Please manually download from https://archive.uci.edu/dataset/{dataset_id}")


def check_existing():
    found = []
    checks = {
        "OULAD": ROOT / "OULAD",
        "xAPI": ROOT / "xAPI" / "xAPI-Edu-Data.csv",
        "student+performance": ROOT / "student+performance" / "student" / "student-mat.csv",
        "Student Mental Health": ROOT / "Student Mental health.csv",
        "Placement": ROOT / "Placement_Data_Full_Class.csv",
        "EdNet-KT2": ROOT / "EdNet-KT2" / "KT2",
    }
    print("\n=== Existing Datasets ===")
    for name, path in checks.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path}")
        if path.exists():
            found.append(name)
    return found


def main():
    print("=" * 60)
    print("  Student Dataset Downloader")
    print("=" * 60)

    check_existing()

    print("\n=== Downloading Missing Datasets ===")

    # UCI Predict Students' Dropout and Academic Success (697)
    dropout_csv = ROOT / "dropout" / "data.csv"
    if not dropout_csv.exists():
        download_uci(697, "dropout")
    else:
        print(f"  [dropout] Already exists → {dropout_csv}")

    # UCI Student Academics Performance (467)
    academics_csv = ROOT / "academics" / "data.csv"
    if not academics_csv.exists():
        download_uci(467, "academics")
    else:
        print(f"  [academics] Already exists → {academics_csv}")

    print("\n=== Summary ===")
    for folder in ["dropout", "academics"]:
        p = ROOT / folder
        csvs = list(p.glob("**/*.csv"))
        if csvs:
            print(f"  ✓ {folder}: {len(csvs)} CSV file(s) found")
        else:
            print(f"  ✗ {folder}: No CSV found — manual download may be needed")

    print("\nDone. Run unified_pipeline.py next.")


if __name__ == "__main__":
    main()
