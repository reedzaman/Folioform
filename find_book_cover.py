#!/usr/bin/env python3
"""
find_book_cover.py

Given a HEX color value as a parameter, returns a book cover location
from the closest matching color spectrum folder in 'book-covers-sorted-by-colors'.
If the matched folder contains multiple book covers, a random one is returned every time.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

# Reuse palette definition & perceptual CIELAB conversion from sort_book_covers
from sort_book_covers import build_color_spectrum_palette, rgb_to_lab
import numpy as np


def hex_to_rgb(hex_code: str):
    """
    Parses a 3-character or 6-character hex string (with or without '#') into an (R, G, B) tuple.
    """
    cleaned = hex_code.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(c * 2 for c in cleaned)
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color format: '{hex_code}'. Expected e.g. '#FF5733' or 'F00'.")
    try:
        return tuple(int(cleaned[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError as e:
        raise ValueError(f"Invalid hex color digits in '{hex_code}': {e}")


def get_book_cover_by_hex(
    hex_code: str,
    base_dir: str = "book-covers-sorted-by-colors",
    absolute: bool = False,
    return_meta: bool = False,
):
    """
    Finds the closest color spectrum folder containing images and selects a random book cover.

    Args:
        hex_code: Target color as HEX string (e.g. '#E63946' or 'E63946').
        base_dir: Path to directory containing sorted color spectrum folders.
        absolute: Whether to return an absolute file path.
        return_meta: If True, returns a dict with metadata (match details).

    Returns:
        str (path to chosen book cover) or dict if return_meta=True.
        Returns None if no matching book cover is found.
    """
    rgb = hex_to_rgb(hex_code)
    palette_names, palette_rgbs, palette_lab = build_color_spectrum_palette()

    # Perceptual CIELAB Delta E distance to all spectrum folders
    target_lab = rgb_to_lab([rgb])[0]
    distances = np.linalg.norm(palette_lab - target_lab, axis=1)
    ranked_indices = np.argsort(distances)

    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Color folders directory '{base_dir}' does not exist.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Search folders starting from closest until a non-empty one is found
    for idx in ranked_indices:
        folder_name = palette_names[idx]
        folder_path = base_path / folder_name

        if not folder_path.is_dir():
            continue

        images = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        if images:
            chosen = random.choice(images)
            resolved_path = chosen.resolve() if absolute else chosen

            if return_meta:
                return {
                    "input_hex": f"#{hex_code.strip().lstrip('#').upper()}",
                    "input_rgb": rgb,
                    "matched_folder": folder_name,
                    "folder_representative_hex": f"#{palette_names[idx].split('_')[-1]}",
                    "cielab_distance": round(float(distances[idx]), 3),
                    "total_covers_in_folder": len(images),
                    "cover_path": str(resolved_path),
                }
            return str(resolved_path)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find a book cover matching a given HEX color value (returns a random one from closest folder)."
    )
    parser.add_argument(
        "hex_color",
        help="Hex color string to search for (e.g. '#E63946', '3498DB', '#FFF')",
    )
    parser.add_argument(
        "-d", "--directory",
        default="book-covers-sorted-by-colors",
        help="Directory with sorted color folders (default: 'book-covers-sorted-by-colors')",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Return absolute file path instead of relative path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print match details (matched folder, distance, total covers available)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    try:
        if args.json or args.verbose:
            meta = get_book_cover_by_hex(
                args.hex_color,
                base_dir=args.directory,
                absolute=args.absolute,
                return_meta=True,
            )
            if not meta:
                print(f"Error: No book covers found in '{args.directory}'", file=sys.stderr)
                sys.exit(1)

            if args.json:
                print(json.dumps(meta, indent=2))
            else:
                print(f"Target Color:      {meta['input_hex']} (RGB: {meta['input_rgb']})")
                print(f"Matched Folder:    {meta['matched_folder']}")
                print(f"Color Distance:    ΔE {meta['cielab_distance']}")
                print(f"Covers in Folder:  {meta['total_covers_in_folder']}")
                print(f"Selected Cover:    {meta['cover_path']}")
        else:
            cover_path = get_book_cover_by_hex(
                args.hex_color,
                base_dir=args.directory,
                absolute=args.absolute,
                return_meta=False,
            )
            if not cover_path:
                print(f"Error: No book covers found in '{args.directory}'", file=sys.stderr)
                sys.exit(1)
            print(cover_path)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()

