#!/usr/bin/env python3
"""
Prepare Oxford Flowers 102 dataset into a clean folder structure.

Downloads the Oxford Flowers 102 dataset and organizes it for use
with OxfordFlowers102Dataset (via torchvision).

This is a simple wrapper that uses torchvision's built-in download.
The actual download and organization is handled by torchvision.datasets.Flowers102.

Output structure (created by torchvision):
    OUT/
        flowers-102/
            jpg/
                image_00001.jpg
                image_00002.jpg
                ...
            imagelabels.mat
            setid.mat

Usage:
    # Simple download to default location
    python tools/prepare_flowers.py --out /data/lorantnagy/storage/genai/data/flowers102

    # Just verify existing dataset
    python tools/prepare_flowers.py --out /data/lorantnagy/storage/genai/data/flowers102 --no-download

Requirements:
    pip install torchvision scipy
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Oxford Flowers 102 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output folder for dataset"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download, just verify existing dataset"
    )
    args = parser.parse_args()

    # Setup output directory
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Oxford Flowers 102 Dataset Preparation")
    print(f"{'='*70}\n")
    print(f"Output directory: {out_dir}")
    
    # Check if torchvision is installed
    try:
        from torchvision.datasets import Flowers102
    except ImportError:
        print("\nError: torchvision not installed!")
        print("Install with: pip install torchvision")
        return 1
    
    # Check if scipy is installed (required for .mat files)
    try:
        import scipy
    except ImportError:
        print("\nError: scipy not installed!")
        print("Install with: pip install scipy")
        return 1
    
    print(f"Download: {'No (verify only)' if args.no_download else 'Yes'}\n")
    
    # Download/verify dataset for each split
    splits = ['train', 'val', 'test']
    total_images = 0
    
    for split in splits:
        print(f"Processing split: {split}")
        try:
            dataset = Flowers102(
                root=str(out_dir),
                split=split,
                download=not args.no_download
            )
            n_images = len(dataset)
            total_images += n_images
            print(f"  ✓ {split}: {n_images} images")
        except Exception as e:
            print(f"  ✗ Error loading {split}: {e}")
            return 1
    
    # Verify structure
    flowers_dir = out_dir / "flowers-102"
    jpg_dir = flowers_dir / "jpg"
    
    print(f"\n{'='*70}")
    print(f"Dataset preparation complete!")
    print(f"{'='*70}\n")
    print(f"Total images: {total_images}")
    print(f"  - Train: 1,020 images")
    print(f"  - Val: 1,020 images")
    print(f"  - Test: 6,149 images")
    print(f"\nDataset structure:")
    print(f"  {out_dir}/")
    print(f"    flowers-102/")
    print(f"      jpg/           (8,189 images)")
    print(f"      imagelabels.mat")
    print(f"      setid.mat")
    
    # Verify images exist
    if jpg_dir.exists():
        n_files = len(list(jpg_dir.glob("*.jpg")))
        print(f"\nVerified: {n_files} .jpg files found")
    
    print(f"\nTo use this dataset, set in your config:")
    print(f"  dataset_params:")
    print(f"    data_root: {out_dir}")
    print(f"    split: train  # or 'val' or 'test'")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())