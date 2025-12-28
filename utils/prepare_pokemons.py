#!/usr/bin/env python3
"""
Prepare Pokemon sprites into a clean folder structure.

Downloads and organizes Pokemon sprite images into a flat structure
suitable for use with PokemonSpritesDataset.

Output structure:
    OUT/
        images/
            <flat png files...>
        index.csv   (path,pokedex_id,is_shiny)

Usage examples:
    # If you already have the zip:
    python tools/prepare_pokemon_sprites.py \\
        --zip /path/pokemon-sprites-images.zip \\
        --out /data/pokemon_sprites_parsed

    # Or via Kaggle CLI (requires kaggle configured):
    python tools/prepare_pokemon_sprites.py \\
        --kaggle yehongjiang/pokemon-sprites-images \\
        --out /data/pokemon_sprites_parsed
    
    # Copy files instead of symlink:
    python tools/prepare_pokemon_sprites.py \\
        --zip /path/pokemon-sprites-images.zip \\
        --out /data/pokemon_sprites_parsed \\
        --copy
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional

# Supported image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _extract_zip(zip_path: Path, dst: Path) -> None:
    """Extract a zip file to destination directory."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst)
    print(f"  Extracted to {dst}")


def _find_images(root: Path) -> Iterator[Path]:
    """Recursively find all image files under root directory."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _infer_pokedex_id(path: Path) -> Optional[int]:
    """
    Try to extract Pokedex ID from filename.
    
    Looks for 1-4 digit numbers in the filename.
    Common patterns: "001_bulbasaur.png", "pikachu_025.png", "25.png"
    
    Returns:
        Integer Pokedex ID if found, None otherwise
    """
    m = re.search(r"(?<!\d)(\d{1,4})(?!\d)", path.stem)
    return int(m.group(1)) if m else None


def _infer_is_shiny(path: Path) -> bool:
    """
    Determine if sprite is shiny variant based on path.
    
    Looks for keywords "shiny", "_shiny", "-shiny" in path.
    
    Returns:
        True if shiny, False otherwise
    """
    s = str(path).lower()
    return ("shiny" in s) or ("_shiny" in s) or ("-shiny" in s)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Pokemon sprites dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--zip",
        type=str,
        default=None,
        help="Path to a downloaded zip file"
    )
    parser.add_argument(
        "--kaggle",
        type=str,
        default=None,
        help="Kaggle dataset slug, e.g. yehongjiang/pokemon-sprites-images"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output folder for parsed dataset"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlink (safer across filesystems)"
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.zip and not args.kaggle:
        parser.error("Provide either --zip or --kaggle")
    
    if args.zip and args.kaggle:
        parser.error("Provide only one of --zip or --kaggle, not both")

    # Setup output directories
    out_dir = Path(args.out).expanduser().resolve()
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Pokemon Sprites Dataset Preparation")
    print(f"{'='*70}\n")
    print(f"Output directory: {out_dir}")
    print(f"Images directory: {images_out}")
    print(f"File handling: {'copy' if args.copy else 'symlink'}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Download or locate dataset
        if args.kaggle:
            print(f"Downloading from Kaggle: {args.kaggle}")
            try:
                subprocess.check_call(
                    [
                        "kaggle", "datasets", "download",
                        "-d", args.kaggle,
                        "-p", str(tmp),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                print(f"\nError: Kaggle download failed!")
                print(f"Make sure you have:")
                print(f"  1. kaggle CLI installed: pip install kaggle")
                print(f"  2. API credentials configured: ~/.kaggle/kaggle.json")
                print(f"\nSee: https://github.com/Kaggle/kaggle-api#api-credentials")
                return 1
            except FileNotFoundError:
                print(f"\nError: 'kaggle' command not found!")
                print(f"Install with: pip install kaggle")
                return 1
            
            zips = list(tmp.glob("*.zip"))
            if not zips:
                print(f"Error: Kaggle download did not produce a zip file")
                return 1
            
            zip_path = zips[0]
            _extract_zip(zip_path, tmp / "unzipped")
            raw_root = tmp / "unzipped"
        
        elif args.zip:
            zip_path = Path(args.zip).expanduser().resolve()
            if not zip_path.exists():
                print(f"Error: Zip file not found: {zip_path}")
                return 1
            
            _extract_zip(zip_path, tmp / "unzipped")
            raw_root = tmp / "unzipped"

        # Process all images
        print(f"\nProcessing images...")
        rows = []
        processed = 0
        skipped = 0
        
        for src in _find_images(raw_root):
            # Create stable filename from relative path
            rel = src.relative_to(raw_root)
            safe = "_".join(rel.parts).replace(" ", "_")
            dst = images_out / safe

            # Skip if already exists
            if dst.exists():
                skipped += 1
                continue

            # Copy or symlink
            try:
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        # Symlink failed (cross-filesystem?), fall back to copy
                        shutil.copy2(src, dst)
            except Exception as e:
                print(f"  Warning: Failed to process {src.name}: {e}")
                continue

            # Record metadata
            rows.append({
                "path": str(dst),
                "pokedex_id": _infer_pokedex_id(src) or "",
                "is_shiny": int(_infer_is_shiny(src)),
            })
            
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed} images...", end="\r", flush=True)

        print(f"  Processed {processed} images... Done!")
        
        if skipped > 0:
            print(f"  Skipped {skipped} existing images")

        # Write index CSV
        index_path = out_dir / "index.csv"
        with open(index_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "pokedex_id", "is_shiny"])
            w.writeheader()
            w.writerows(rows)

        # Summary
        print(f"\n{'='*70}")
        print(f"Dataset preparation complete!")
        print(f"{'='*70}\n")
        print(f"Total images: {len(rows)}")
        print(f"Output directory: {out_dir}")
        print(f"  - Images: {images_out}/")
        print(f"  - Index: {index_path}")
        
        # Show distribution
        shiny_count = sum(1 for r in rows if r["is_shiny"])
        regular_count = len(rows) - shiny_count
        print(f"\nDistribution:")
        print(f"  - Regular: {regular_count}")
        print(f"  - Shiny: {shiny_count}")
        
        # Show sample of Pokedex IDs found
        ids_found = set(r["pokedex_id"] for r in rows if r["pokedex_id"])
        if ids_found:
            sample_ids = sorted(list(ids_found))[:10]
            print(f"\nSample Pokedex IDs found: {sample_ids}")
            if len(ids_found) > 10:
                print(f"  ... and {len(ids_found) - 10} more")
        
        print(f"\nTo use this dataset, set in your config:")
        print(f"  data_root: {out_dir}")
        print()

    return 0


if __name__ == "__main__":
    exit(main())