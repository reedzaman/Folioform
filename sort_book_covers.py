#!/usr/bin/env python3
"""
sort_book_covers.py

Extracts the dominant color of book covers using K-Means clustering,
creates ~100 spectrum-based color folders inside the output directory,
and places each image into the folder of its closest color range.
Images with dominant color coverage below a threshold can be skipped/rejected.
"""

import os
import sys
import shutil
import argparse
import colorsys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress NumPy/SciPy version mismatch warning if present
warnings.filterwarnings("ignore", category=UserWarning)

# Limit OpenMP threads per worker to avoid thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 100-Color Spectrum Definition
# 18 Hues x 5 Variants (90 chromatic) + 10 Grayscale / Neutrals = 100 Bins
# ---------------------------------------------------------------------------
HUE_SPECTRUM = [
    (0, "red"),
    (20, "red-orange"),
    (40, "orange"),
    (60, "amber-gold"),
    (80, "yellow"),
    (100, "lime"),
    (120, "green"),
    (140, "spring-green"),
    (160, "teal-mint"),
    (180, "cyan"),
    (200, "sky-blue"),
    (220, "cobalt-blue"),
    (240, "blue"),
    (260, "indigo"),
    (280, "violet-purple"),
    (300, "magenta"),
    (320, "pink-rose"),
    (340, "crimson"),
]

VARIANTS = [
    ("pastel", 0.35, 0.92),
    ("vivid", 0.90, 0.92),
    ("muted", 0.45, 0.55),
    ("deep", 0.90, 0.55),
    ("dark", 0.75, 0.30),
]

GRAYSCALE_LEVELS = [
    ("black", 0.05),
    ("near-black", 0.15),
    ("charcoal", 0.25),
    ("dark-gray", 0.35),
    ("mid-dark-gray", 0.48),
    ("medium-gray", 0.60),
    ("light-gray", 0.72),
    ("silver", 0.82),
    ("off-white", 0.92),
    ("white", 0.99),
]


def rgb_to_lab(rgb_arr):
    """
    Converts RGB array [0..255] to CIELAB color space for perceptual distance.
    """
    rgb = np.array(rgb_arr, dtype=float) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # sRGB to XYZ (D65)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = np.dot(rgb, matrix.T)

    # Normalize by D65 reference white
    xyz_ref = np.array([0.95047, 1.00000, 1.08883])
    normalized = xyz / xyz_ref

    mask2 = normalized > 0.008856
    f = np.empty_like(normalized)
    f[mask2] = normalized[mask2] ** (1.0 / 3.0)
    f[~mask2] = (7.787 * normalized[~mask2]) + (16.0 / 116.0)

    l_channel = (116.0 * f[..., 1]) - 16.0
    a_channel = 500.0 * (f[..., 0] - f[..., 1])
    b_channel = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l_channel, a_channel, b_channel], axis=-1)


def build_color_spectrum_palette():
    """
    Constructs exactly 100 color bins covering the spectrum:
    - 90 chromatic bins (18 hues x 5 variants)
    - 10 neutral/grayscale bins
    Returns:
        palette_names: list of folder names
        palette_rgbs: list of (R, G, B) tuples
        palette_lab: numpy array of CIELAB values (100, 3)
    """
    palette_names = []
    palette_rgbs = []
    idx = 1

    # 1. Chromatic spectrum (90 bins)
    for deg, name in HUE_SPECTRUM:
        h = deg / 360.0
        for var_name, s, v in VARIANTS:
            r, g, b = [int(round(c * 255)) for c in colorsys.hsv_to_rgb(h, s, v)]
            folder = f"{idx:03d}_{name}_{var_name}_{r:02X}{g:02X}{b:02X}"
            palette_names.append(folder)
            palette_rgbs.append((r, g, b))
            idx += 1

    # 2. Neutrals / Grayscale (10 bins)
    for g_name, v in GRAYSCALE_LEVELS:
        c = int(round(v * 255))
        folder = f"{idx:03d}_neutral_{g_name}_{c:02X}{c:02X}{c:02X}"
        palette_names.append(folder)
        palette_rgbs.append((c, c, c))
        idx += 1

    palette_lab = rgb_to_lab(palette_rgbs)
    return palette_names, palette_rgbs, palette_lab


def create_spectrum_folders(output_dir, palette_names):
    """
    Creates all ~100 spectrum folders inside output_dir if they don't already exist.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for folder in palette_names:
        (out_path / folder).mkdir(exist_ok=True)


def get_dominant_color(image_path, clusters=5, resize_dim=100):
    """
    Extracts dominant color of an image using K-Means clustering.
    If resize_dim > 0, thumbnails the image first for significantly faster computation.
    """
    image = Image.open(image_path).convert("RGB")
    if resize_dim and (image.width > resize_dim or image.height > resize_dim):
        image.thumbnail((resize_dim, resize_dim))

    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=1,
        max_iter=25,
    )
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_index = np.argmax(counts)

    dominant_rgb = np.round(kmeans.cluster_centers_[dominant_index]).astype(int)
    hex_color = "#{:02X}{:02X}{:02X}".format(*dominant_rgb)

    return {
        "hex": hex_color,
        "rgb": tuple(dominant_rgb),
        "percentage": round(counts[dominant_index] / len(pixels) * 100, 2),
    }


def find_closest_spectrum_folder(rgb, palette_names, palette_lab):
    """
    Finds the closest color spectrum folder using CIELAB Euclidean distance (Delta E76).
    """
    lab = rgb_to_lab([rgb])[0]
    distances = np.linalg.norm(palette_lab - lab, axis=1)
    best_idx = int(np.argmin(distances))
    return palette_names[best_idx]


def _place_file(src_path, dest_path, action):
    """
    Places file at destination using the specified action (copy, move, or symlink).
    """
    if action == "copy":
        shutil.copy2(src_path, dest_path)
    elif action == "move":
        shutil.move(src_path, dest_path)
    elif action == "symlink":
        if dest_path.exists() or dest_path.is_symlink():
            dest_path.unlink()
        dest_path.symlink_to(os.path.abspath(src_path))


def _process_single_image(args_tuple):
    """
    Helper function for parallel processing of a single image.
    """
    (
        img_path,
        input_dir,
        output_dir,
        action,
        clusters,
        resize_dim,
        threshold,
        rejected_dir,
        palette_names,
        palette_lab,
    ) = args_tuple

    try:
        res = get_dominant_color(img_path, clusters=clusters, resize_dim=resize_dim)
        coverage = res["percentage"]

        # Check coverage threshold
        if threshold > 0 and coverage < threshold:
            if rejected_dir:
                rej_path = Path(rejected_dir)
                rej_path.mkdir(parents=True, exist_ok=True)
                rel_path = os.path.relpath(img_path, input_dir)
                safe_filename = rel_path.replace(os.sep, "_")
                _place_file(img_path, rej_path / safe_filename, action)
            return "skipped", img_path, f"Coverage {coverage}% < threshold {threshold}%", res["hex"]

        folder_name = find_closest_spectrum_folder(res["rgb"], palette_names, palette_lab)
        target_dir = Path(output_dir) / folder_name

        # Prevent name collisions across subcategories (e.g. Medical/0000001.jpg -> Medical_0000001.jpg)
        rel_path = os.path.relpath(img_path, input_dir)
        safe_filename = rel_path.replace(os.sep, "_")
        target_path = target_dir / safe_filename

        _place_file(img_path, target_path, action)

        return "sorted", img_path, folder_name, res["hex"]
    except Exception as e:
        return "error", img_path, str(e), None


def sort_book_covers(
    input_dir="book-covers",
    output_dir="book-covers-sorted-by-colors",
    action="copy",
    clusters=5,
    resize_dim=100,
    threshold=0.0,
    rejected_dir=None,
    workers=None,
    limit=None,
):
    """
    Processes all book cover images and sorts them into 100 color spectrum folders.
    """
    palette_names, _, palette_lab = build_color_spectrum_palette()

    print(f"Initializing {len(palette_names)} color spectrum folders in: {output_dir}")
    create_spectrum_folders(output_dir, palette_names)

    # Discover images
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    all_images = [
        str(p)
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    total_images = len(all_images)
    if limit and limit < total_images:
        all_images = all_images[:limit]
        print(f"Found {total_images} images. Processing limited set of {len(all_images)} images...")
    else:
        print(f"Found {total_images} images to sort.")

    if not all_images:
        print("No images found to process.")
        return

    if threshold > 0:
        rej_info = f", rejected saved to '{rejected_dir}'" if rejected_dir else ", rejected will be skipped"
        print(f"Rejection threshold: coverage < {threshold}%{rej_info}")

    num_workers = workers or os.cpu_count() or 4
    print(f"Processing with {num_workers} parallel workers (action='{action}')...\n")

    tasks = [
        (
            img,
            input_dir,
            output_dir,
            action,
            clusters,
            resize_dim,
            threshold,
            rejected_dir,
            palette_names,
            palette_lab,
        )
        for img in all_images
    ]

    processed = 0
    sorted_count = 0
    skipped_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_image, t) for t in tasks]
        for future in as_completed(futures):
            status, path, info, hex_col = future.result()
            processed += 1
            if status == "sorted":
                sorted_count += 1
            elif status == "skipped":
                skipped_count += 1
            elif status == "error":
                error_count += 1
                print(f"Failed [{path}]: {info}")

            if processed % 500 == 0 or processed == len(all_images):
                pct = (processed / len(all_images)) * 100
                status_line = f"[{processed}/{len(all_images)}] ({pct:.1f}%) - Sorted: {sorted_count}"
                if threshold > 0:
                    status_line += f", Skipped: {skipped_count}"
                print(status_line)

    print("\n" + "=" * 40)
    print("SORTING SUMMARY")
    print("=" * 40)
    print(f"Total processed: {processed}")
    print(f"Sorted:          {sorted_count}")
    if threshold > 0:
        print(f"Skipped:         {skipped_count} (below {threshold}% threshold)")
    print(f"Errors:          {error_count}")
    print(f"Output directory: {output_dir}")
    if rejected_dir and skipped_count > 0:
        print(f"Rejected folder:  {rejected_dir}")


def inspect_single_image(image_path, clusters=5, resize_dim=100, threshold=0.0):
    """
    Inspects an individual image and outputs dominant color, coverage, and assigned spectrum folder.
    """
    palette_names, _, palette_lab = build_color_spectrum_palette()
    result = get_dominant_color(image_path, clusters=clusters, resize_dim=resize_dim)
    folder = find_closest_spectrum_folder(result["rgb"], palette_names, palette_lab)

    print(f"Image:          {image_path}")
    print(f"Dominant Color: {result['hex']}")
    print(f"RGB:            {result['rgb']}")
    print(f"Coverage:       {result['percentage']}%")
    if threshold > 0:
        meets = result["percentage"] >= threshold
        status_str = "ACCEPTED" if meets else f"REJECTED (below {threshold}% threshold)"
        print(f"Threshold:      {threshold}% -> {status_str}")
    print(f"Closest Folder: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort book covers into ~100 spectrum-based color folders."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Path to an individual image (inspect mode) or an input directory. If omitted, sorts the full collection.",
    )
    parser.add_argument(
        "-i", "--input-dir",
        default="book-covers",
        help="Input directory containing book covers (default: 'book-covers')",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="book-covers-sorted-by-colors",
        help="Output directory for color folders (default: 'book-covers-sorted-by-colors')",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.0,
        help="Minimum dominant color coverage percentage (0-100). Images below this threshold are skipped (default: 0.0)",
    )
    parser.add_argument(
        "--rejected-dir",
        default=None,
        help="Optional directory to place skipped/rejected images instead of just ignoring them",
    )
    parser.add_argument(
        "--action",
        choices=["copy", "move", "symlink"],
        default="copy",
        help="Action to place images into color folders: 'copy', 'move', or 'symlink' (default: 'copy')",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of color clusters for K-Means (default: 5)",
    )
    parser.add_argument(
        "--resize-dim",
        type=int,
        default=100,
        help="Downscale image dimension before clustering for high performance (default: 100, set 0 to disable)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all available CPU cores)",
    )
    parser.add_argument(
        "--find-color",
        default=None,
        help="Given a HEX color, find and return a matching book cover location from sorted spectrum folders",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to sort (useful for test runs)",
    )

    args = parser.parse_args()

    # Determine mode: find by color HEX, inspect single image, OR sort directory
    if args.find_color:
        from find_book_cover import get_book_cover_by_hex
        cover = get_book_cover_by_hex(args.find_color, base_dir=args.output_dir, return_meta=True)
        if cover:
            print(f"Matched Cover: {cover['cover_path']}")
            print(f"Matched Folder: {cover['matched_folder']} (ΔE: {cover['cielab_distance']})")
        else:
            print(f"No matching cover found in '{args.output_dir}'")
    elif args.target and os.path.isfile(args.target):
        inspect_single_image(
            args.target,
            clusters=args.clusters,
            resize_dim=args.resize_dim,
            threshold=args.threshold,
        )
    else:
        input_directory = args.target if (args.target and os.path.isdir(args.target)) else args.input_dir
        sort_book_covers(
            input_dir=input_directory,
            output_dir=args.output_dir,
            action=args.action,
            clusters=args.clusters,
            resize_dim=args.resize_dim,
            threshold=args.threshold,
            rejected_dir=args.rejected_dir,
            workers=args.workers,
            limit=args.limit,
        )
