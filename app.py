#!/usr/bin/env python3
"""
app.py - Folioform Academic Color Spectrum Portal
By Reed Zaman

A single-page, light-themed academic archive interface:
- Search Page with slim header, always-visible color wheel, and strictly ranked matches.
- Transform Your Portrait Page:
  * Rectangular cells matching authentic book cover aspect ratio (~1:1.45)
  * Real book cover replacement with intelligent color matching and neighborhood randomization
  * High-resolution zoomable master image export (2K-3K master PNG)
  * Hardware-accelerated, lag-free smooth scrolling
"""

import os
import io
import json
import base64
import random
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template_string
from PIL import Image
import numpy as np

from sort_book_covers import build_color_spectrum_palette, rgb_to_lab
from find_book_cover import hex_to_rgb

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
SORTED_DIR = BASE_DIR / "book-covers-sorted-by-colors"
INDEX_PATH = BASE_DIR / "cover_colors_index.json"
MOSAIC_CACHE_DIR = BASE_DIR / "mosaic_cache"
MOSAIC_CACHE_DIR.mkdir(exist_ok=True)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Pre-build palette
PALETTE_NAMES, PALETTE_RGBS, PALETTE_LAB = build_color_spectrum_palette()


def load_cover_index():
    """Loads pre-extracted exact color index for sorted book covers."""
    if not INDEX_PATH.exists():
        return [], np.empty((0, 3))

    with open(INDEX_PATH, "r") as f:
        data = json.load(f)

    labs = np.array([item["lab"] for item in data], dtype=float)
    return data, labs


COVER_INDEX, INDEXED_LABS = load_cover_index()

# In-memory thumbnail cache for book covers to allow sub-second high-res mosaic assembly
THUMB_CACHE = {}


@app.route("/")
def index():
    if (BASE_DIR / "index.html").exists():
        return send_file(BASE_DIR / "index.html")
    return render_template_string(HTML_TEMPLATE)


@app.route("/cover_colors_index.json")
def serve_index_json():
    return send_file(INDEX_PATH, mimetype="application/json")


@app.route("/book-covers-sorted-by-colors/<path:filepath>")
def serve_sorted_cover(filepath):
    return send_from_directory(SORTED_DIR, filepath)


@app.route("/api/search")
def search_covers():
    hex_param = request.args.get("hex", "#A33327")
    limit = int(request.args.get("limit", 6))
    limit = max(1, min(limit, 48))
    strategy = request.args.get("strategy", "strict")  # 'strict', 'exact', 'spread'
    shuffle = request.args.get("shuffle", "false").lower() == "true"

    try:
        rgb = hex_to_rgb(hex_param)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    target_lab = rgb_to_lab([rgb])[0]

    # 1. Distances to all 100 spectrum centroid folders
    folder_distances = np.linalg.norm(PALETTE_LAB - target_lab, axis=1)
    ranked_folder_indices = np.argsort(folder_distances)
    closest_folder_idx = int(ranked_folder_indices[0])
    primary_folder_name = PALETTE_NAMES[closest_folder_idx]
    primary_folder_dist = float(folder_distances[closest_folder_idx])

    # 2. Individual cover distances
    if len(COVER_INDEX) > 0:
        cover_distances = np.linalg.norm(INDEXED_LABS - target_lab, axis=1)
    else:
        cover_distances = np.array([])

    results = []

    if strategy == "strict":
        candidates = []
        for i, item in enumerate(COVER_INDEX):
            if item["folder"] == primary_folder_name:
                candidates.append((float(cover_distances[i]), item))

        if len(candidates) < limit:
            for next_idx in ranked_folder_indices[1:]:
                next_folder = PALETTE_NAMES[next_idx]
                for i, item in enumerate(COVER_INDEX):
                    if item["folder"] == next_folder:
                        candidates.append((float(cover_distances[i]), item))
                if len(candidates) >= limit:
                    break

        candidates.sort(key=lambda x: x[0])

        if shuffle and len(candidates) > limit:
            pool_size = min(len(candidates), limit * 3)
            sampled = random.sample(candidates[:pool_size], min(limit, pool_size))
            sampled.sort(key=lambda x: x[0])
            chosen = sampled
        else:
            chosen = candidates[:limit]

        for rank, (dist, item) in enumerate(chosen):
            fname = item["filename"]
            folder = item["folder"]
            cat = fname.split("_", 1)[0].replace("-", " ") if "_" in fname else "General"
            results.append({
                "rank": rank + 1,
                "filename": fname,
                "folder": folder,
                "category": cat,
                "distance": round(dist, 2),
                "cover_hex": item.get("hex", f"#{folder.split('_')[-1]}"),
                "folder_hex": f"#{folder.split('_')[-1]}",
                "url": f"/covers/{folder}/{fname}",
            })

    elif strategy == "exact":
        all_candidates = [
            (float(cover_distances[i]), item) for i, item in enumerate(COVER_INDEX)
        ]
        all_candidates.sort(key=lambda x: x[0])

        if shuffle and len(all_candidates) > limit:
            pool = all_candidates[: limit * 3]
            chosen = random.sample(pool, limit)
            chosen.sort(key=lambda x: x[0])
        else:
            chosen = all_candidates[:limit]

        for rank, (dist, item) in enumerate(chosen):
            fname = item["filename"]
            folder = item["folder"]
            cat = fname.split("_", 1)[0].replace("-", " ") if "_" in fname else "General"
            results.append({
                "rank": rank + 1,
                "filename": fname,
                "folder": folder,
                "category": cat,
                "distance": round(dist, 2),
                "cover_hex": item.get("hex", f"#{folder.split('_')[-1]}"),
                "folder_hex": f"#{folder.split('_')[-1]}",
                "url": f"/covers/{folder}/{fname}",
            })

    else:
        for idx in ranked_folder_indices:
            f_name = PALETTE_NAMES[idx]
            folder_items = [
                (float(cover_distances[i]), item)
                for i, item in enumerate(COVER_INDEX)
                if item["folder"] == f_name
            ]
            if not folder_items:
                continue

            folder_items.sort(key=lambda x: x[0])
            picked = folder_items[0]
            dist, item = picked
            fname = item["filename"]
            cat = fname.split("_", 1)[0].replace("-", " ") if "_" in fname else "General"
            results.append({
                "rank": len(results) + 1,
                "filename": fname,
                "folder": f_name,
                "category": cat,
                "distance": round(dist, 2),
                "cover_hex": item.get("hex", f"#{f_name.split('_')[-1]}"),
                "folder_hex": f"#{f_name.split('_')[-1]}",
                "url": f"/covers/{f_name}/{fname}",
            })
            if len(results) >= limit:
                break

    return jsonify({
        "input_hex": f"#{hex_param.strip().lstrip('#').upper()}",
        "input_rgb": rgb,
        "results_count": len(results),
        "primary_match": {
            "folder": primary_folder_name,
            "hex": f"#{primary_folder_name.split('_')[-1]}",
            "distance": round(primary_folder_dist, 2),
        },
        "covers": results,
    })


@app.route("/api/generate_mosaic", methods=["POST"])
def generate_mosaic():
    """
    Transforms a cropped portrait image into a high-resolution book cover mosaic:
    - Rectangular book cells (aspect ratio 1 : 1.45)
    - Nearest perceptual color matching via CIELAB vector index
    - Randomization across candidate pool with neighborhood repeat-suppression
    - High-resolution zoomable output
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image", "")
    cols = int(data.get("cols", 28))
    cols = max(12, min(cols, 60))

    # Standard book cover aspect ratio: width : height = 1 : 1.45
    COVER_ASPECT_RATIO = 1.45
    rows = int(round(cols / COVER_ASPECT_RATIO))

    if not image_b64:
        return jsonify({"error": "No image payload provided"}), 400

    try:
        # Strip header if present
        if "base64," in image_b64:
            image_b64 = image_b64.split("base64,", 1)[1]
        raw_bytes = base64.b64decode(image_b64)
        input_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {e}"}), 400

    # Downsample source portrait to cols x rows matrix
    downsampled = input_image.resize((cols, rows), Image.Resampling.BILINEAR)
    pixels = np.array(downsampled)

    # High-resolution cell dimensions (crisp covers)
    cell_w = 60
    cell_h = int(round(cell_w * COVER_ASPECT_RATIO))  # 87px
    master_w = cols * cell_w
    master_h = rows * cell_h

    mosaic_master = Image.new("RGB", (master_w, master_h))

    # Keep track of recently used cover indices to prevent wallpaper-repeat
    recent_used = []
    matrix_info = []

    for r in range(rows):
        row_items = []
        for c in range(cols):
            rgb = pixels[r, c]
            t_lab = rgb_to_lab([rgb])[0]

            # Vectorized CIELAB distance across all covers
            dists = np.linalg.norm(INDEXED_LABS - t_lab, axis=1)

            # Pool of top 25 closest color matches
            top_k_indices = np.argsort(dists)[:25]

            # Avoid covers that were used in the immediate neighborhood (last 12 cells)
            candidates = [idx for idx in top_k_indices if idx not in recent_used[-12:]]
            if not candidates:
                candidates = list(top_k_indices)

            chosen_idx = random.choice(candidates)
            recent_used.append(chosen_idx)
            chosen_cover = COVER_INDEX[chosen_idx]

            # Cache thumbnail
            cache_key = (chosen_cover["folder"], chosen_cover["filename"])
            if cache_key not in THUMB_CACHE:
                cpath = SORTED_DIR / chosen_cover["folder"] / chosen_cover["filename"]
                if cpath.exists():
                    im = Image.open(cpath).convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR)
                    THUMB_CACHE[cache_key] = im
                else:
                    im = Image.new("RGB", (cell_w, cell_h), color=tuple(chosen_cover["rgb"]))
                    THUMB_CACHE[cache_key] = im
            else:
                im = THUMB_CACHE[cache_key]

            mosaic_master.paste(im, (c * cell_w, r * cell_h))
            row_items.append({
                "hex": chosen_cover.get("hex"),
                "folder": chosen_cover.get("folder"),
                "filename": chosen_cover.get("filename"),
            })

        matrix_info.append(row_items)

    # Save high-res master & web-optimized preview
    mosaic_id = str(uuid.uuid4())
    master_path = MOSAIC_CACHE_DIR / f"{mosaic_id}_master.png"
    preview_path = MOSAIC_CACHE_DIR / f"{mosaic_id}_preview.jpg"

    mosaic_master.save(master_path, format="PNG", optimize=True)

    # Web preview resized to max ~800px width for fast loading
    preview_w = min(800, master_w)
    preview_h = int(round(preview_w * (master_h / master_w)))
    preview_img = mosaic_master.resize((preview_w, preview_h), Image.Resampling.BILINEAR)
    preview_img.save(preview_path, format="JPEG", quality=88)

    return jsonify({
        "mosaic_id": mosaic_id,
        "cols": cols,
        "rows": rows,
        "total_covers": cols * rows,
        "preview_url": f"/api/mosaic_preview/{mosaic_id}",
        "download_url": f"/api/mosaic_download/{mosaic_id}",
        "master_dimensions": f"{master_w} × {master_h}",
    })


@app.route("/api/mosaic_preview/<mosaic_id>")
def mosaic_preview(mosaic_id):
    path = MOSAIC_CACHE_DIR / f"{mosaic_id}_preview.jpg"
    if not path.exists():
        return "Preview not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/mosaic_download/<mosaic_id>")
def mosaic_download(mosaic_id):
    path = MOSAIC_CACHE_DIR / f"{mosaic_id}_master.png"
    if not path.exists():
        return "File not found", 404
    return send_file(
        path,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"folioform-portrait-mosaic-{mosaic_id[:8]}.png",
    )


@app.route("/covers/<folder>/<filename>")
def serve_cover(folder, filename):
    folder_path = SORTED_DIR / folder
    return send_from_directory(folder_path, filename)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Folioform by Reed Zaman</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <!-- Cropper.js for standard 1:1 square image cropping -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.1/cropper.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.1/cropper.min.js"></script>

  <style>
    :root {
      --paper-base: #f6f4ee;
      --surface: #ffffff;
      --border-dark: #1a1a1a;
      --border-light: #dedbcf;
      --border-subtle: #eae7db;
      --text-primary: #121212;
      --text-secondary: #58554d;
      --text-muted: #858175;
      --mono-font: 'IBM Plex Mono', monospace;
      --serif-font: 'EB Garamond', Georgia, serif;
      --sans-font: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      border-radius: 0 !important;
    }

    /* Fixed hardware-accelerated background layer for butter-smooth 120 FPS scrolling */
    body::before {
      content: "";
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      pointer-events: none;
      z-index: -1;
      background-color: var(--paper-base);
      background-image: 
        radial-gradient(circle at 50% 50%, rgba(255,255,255,0.35) 0%, rgba(238,234,222,0.45) 100%),
        url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='paperNoise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23paperNoise)' opacity='0.035'/%3E%3C/svg%3E");
      transform: translateZ(0);
      will-change: transform;
    }

    html, body {
      height: 100vh;
      max-height: 100vh;
      overflow: hidden;
      background: transparent;
      color: var(--text-primary);
      font-family: var(--sans-font);
      font-size: 13px;
      line-height: 1.4;
      display: flex;
      flex-direction: column;
    }

    /* Top Bar */
    .topbar {
      height: 56px;
      min-height: 56px;
      background: var(--surface);
      border-bottom: 2px solid var(--border-dark);
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 28px;
      z-index: 20;
    }

    .brand-title {
      font-family: var(--serif-font);
      font-size: 2.1rem;
      font-weight: 600;
      letter-spacing: -0.015em;
      color: var(--text-primary);
      display: inline-flex;
      align-items: baseline;
      gap: 12px;
    }

    .brand-byline {
      font-family: var(--serif-font);
      font-size: 1.05rem;
      font-style: italic;
      font-weight: 400;
      color: var(--text-secondary);
    }

    .topbar-meta {
      font-family: var(--mono-font);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-muted);
    }

    /* Main Viewport Workspace */
    .workspace {
      flex: 1;
      display: flex;
      overflow: hidden;
      height: calc(100vh - 56px);
    }

    /* Narrow Clean Sidebar */
    .app-sidebar {
      width: 210px;
      min-width: 210px;
      max-width: 210px;
      background: #faf9f5;
      border-right: 1px solid var(--border-dark);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 20px 14px;
    }

    .sidebar-heading {
      font-family: var(--mono-font);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 10px;
      padding-left: 6px;
    }

    .sidebar-menu {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .nav-btn {
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: var(--mono-font);
      font-size: 12px;
      font-weight: 500;
      padding: 10px 12px;
      text-decoration: none;
      color: var(--text-primary);
      border: 1px solid transparent;
      background: transparent;
      cursor: pointer;
      text-align: left;
      transition: background 0.1s, border-color 0.1s;
    }

    .nav-btn:hover {
      background: #f1eee2;
    }

    .nav-btn.active {
      background: var(--surface);
      border-color: var(--border-dark);
      font-weight: 600;
      box-shadow: 2px 2px 0 rgba(0,0,0,0.06);
    }

    .nav-marker {
      width: 6px;
      height: 6px;
      background: var(--border-dark);
      display: inline-block;
      flex-shrink: 0;
    }

    .sidebar-footer {
      font-family: var(--mono-font);
      font-size: 10px;
      color: var(--text-muted);
      border-top: 1px solid var(--border-light);
      padding-top: 12px;
      line-height: 1.5;
    }

    /* Page Views Switcher */
    .page-view {
      display: none;
      flex: 1;
      height: 100%;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      transform: translateZ(0);
      scroll-behavior: smooth;
    }

    .page-view.active {
      display: flex;
      flex-direction: column;
    }

    /* ---------------------------------------------------- */
    /* PAGE 1: SEARCH PAGE STYLES                           */
    /* ---------------------------------------------------- */
    .search-view-inner {
      padding: 20px 32px 36px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .search-page-header {
      padding: 4px 0 12px;
      border-bottom: 1px solid var(--border-dark);
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 8px;
    }

    .header-pretitle {
      font-family: var(--mono-font);
      font-size: 10px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 3px;
    }

    .header-main-title {
      font-family: var(--serif-font);
      font-size: 1.7rem;
      font-weight: 500;
      letter-spacing: -0.01em;
      color: var(--text-primary);
      line-height: 1.15;
    }

    .header-caption {
      font-size: 12px;
      color: var(--text-secondary);
      max-width: 640px;
      line-height: 1.45;
    }

    .search-workbench {
      background: var(--surface);
      border: 1px solid var(--border-dark);
      box-shadow: 0 4px 16px rgba(0,0,0,0.04);
      flex-shrink: 0;
    }

    .workbench-header {
      background: #f1f0eb;
      border-bottom: 1px solid var(--border-dark);
      padding: 8px 16px;
      font-family: var(--mono-font);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .workbench-body {
      padding: 18px 20px;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 28px;
      align-items: center;
    }

    @media (max-width: 1100px) {
      .workbench-body {
        grid-template-columns: 1fr;
      }
    }

    .color-wheel-module {
      display: flex;
      align-items: center;
      gap: 16px;
      padding-right: 20px;
      border-right: 1px solid var(--border-light);
    }

    .canvas-wheel-wrap {
      position: relative;
      width: 120px;
      height: 120px;
      flex-shrink: 0;
    }

    #colorWheelCanvas {
      width: 120px;
      height: 120px;
      cursor: crosshair;
      display: block;
    }

    .wheel-crosshair {
      position: absolute;
      width: 10px;
      height: 10px;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 1px #111111;
      pointer-events: none;
      transform: translate(-50%, -50%);
      top: 50%;
      left: 50%;
    }

    .brightness-slider-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      height: 110px;
    }

    .brightness-slider-wrap input[type="range"] {
      writing-mode: bt-lr;
      -webkit-appearance: slider-vertical;
      width: 8px;
      height: 90px;
      margin: 0;
    }

    .slider-tiny-lbl {
      font-family: var(--mono-font);
      font-size: 9px;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .color-readout-col {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .swatch-box {
      width: 44px;
      height: 44px;
      border: 1px solid var(--border-dark);
      background: #A33327;
      cursor: pointer;
      position: relative;
    }

    .swatch-box input[type="color"] {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      opacity: 0;
      cursor: pointer;
    }

    .hex-field {
      font-family: var(--mono-font);
      font-size: 13px;
      font-weight: 600;
      padding: 6px 8px;
      border: 1px solid var(--border-dark);
      width: 90px;
      outline: none;
      background: #ffffff;
    }

    .rgb-readout {
      font-family: var(--mono-font);
      font-size: 10px;
      color: var(--text-secondary);
    }

    .controls-module {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .controls-upper-row {
      display: flex;
      align-items: flex-end;
      gap: 20px;
      flex-wrap: wrap;
    }

    .field-wrap {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .field-lbl {
      font-family: var(--mono-font);
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      display: flex;
      justify-content: space-between;
    }

    .field-val-counter {
      color: var(--text-primary);
      font-weight: 600;
      margin-left: 8px;
    }

    select.academic-select {
      font-family: var(--sans-font);
      font-size: 12px;
      padding: 6px 10px;
      border: 1px solid var(--border-dark);
      background: #ffffff;
      outline: none;
      height: 32px;
    }

    .slider-inline {
      width: 140px;
      height: 2px;
      background: var(--border-dark);
      outline: none;
      -webkit-appearance: none;
      margin: 8px 0;
    }

    .slider-inline::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 12px;
      height: 12px;
      background: var(--border-dark);
      cursor: pointer;
    }

    .btns-wrap {
      display: flex;
      gap: 6px;
      align-items: flex-end;
    }

    button.act-btn {
      font-family: var(--mono-font);
      font-size: 11px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      font-weight: 500;
      padding: 8px 14px;
      cursor: pointer;
      border: 1px solid var(--border-dark);
      background: #ffffff;
      color: var(--text-primary);
      transition: background 0.1s, color 0.1s;
      height: 32px;
      white-space: nowrap;
    }

    button.act-btn:hover {
      background: #111111;
      color: #ffffff;
    }

    button.act-btn.btn-main {
      background: #111111;
      color: #ffffff;
      font-weight: 600;
    }

    button.act-btn.btn-main:hover {
      background: #333333;
    }

    .standards-row {
      display: flex;
      align-items: center;
      gap: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--border-subtle);
      flex-wrap: wrap;
    }

    .standards-label {
      font-family: var(--mono-font);
      font-size: 10px;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
    }

    .standards-chips {
      display: flex;
      gap: 5px;
      flex-wrap: wrap;
    }

    .std-chip {
      width: 18px;
      height: 18px;
      border: 1px solid rgba(0,0,0,0.3);
      cursor: pointer;
      transition: outline 0.1s;
    }

    .std-chip:hover {
      outline: 2px solid var(--border-dark);
      outline-offset: 1px;
    }

    .ledger-banner {
      background: #ffffff;
      border: 1px solid var(--border-dark);
      padding: 8px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono-font);
      font-size: 11px;
      flex-wrap: wrap;
      gap: 12px;
    }

    .ledger-cell {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .ledger-key {
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .ledger-val {
      font-weight: 600;
      color: var(--text-primary);
    }

    .ledger-dot {
      width: 10px;
      height: 10px;
      display: inline-block;
      border: 1px solid rgba(0,0,0,0.3);
    }

    .gallery-section {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .gallery-bar {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid var(--border-dark);
      padding-bottom: 6px;
    }

    .gallery-heading {
      font-family: var(--serif-font);
      font-size: 1.25rem;
      font-weight: 600;
    }

    .gallery-note {
      font-family: var(--mono-font);
      font-size: 11px;
      color: var(--text-muted);
    }

    .covers-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 14px;
      align-content: start;
    }

    .specimen-card {
      background: #ffffff;
      border: 1px solid var(--border-light);
      padding: 6px;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      position: relative;
      transition: border-color 0.15s ease, transform 0.1s ease;
    }

    .specimen-card:hover {
      border-color: var(--border-dark);
      transform: translateY(-2px);
    }

    .rank-tag {
      position: absolute;
      top: 10px;
      left: 10px;
      background: #111111;
      color: #ffffff;
      font-family: var(--mono-font);
      font-size: 9px;
      font-weight: 600;
      padding: 2px 5px;
      z-index: 2;
    }

    .rank-tag.top-match {
      background: #8b0000;
    }

    .specimen-frame {
      width: 100%;
      height: 175px;
      background: #eeece5;
      border: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      margin-bottom: 6px;
    }

    .specimen-frame img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .specimen-meta {
      font-size: 10px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .discipline-tag {
      font-family: var(--sans-font);
      font-size: 10px;
      font-weight: 600;
      color: var(--text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      text-transform: capitalize;
    }

    .specimen-stats {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono-font);
      font-size: 9px;
      color: var(--text-muted);
    }

    .color-swatch-mini {
      width: 8px;
      height: 8px;
      display: inline-block;
      border: 1px solid rgba(0,0,0,0.2);
      margin-right: 3px;
      vertical-align: middle;
    }

    .delta-rating {
      font-weight: 600;
      color: var(--text-primary);
    }

    .empty-state {
      grid-column: 1 / -1;
      padding: 56px 16px;
      text-align: center;
      background: #ffffff;
      border: 1px dashed var(--border-light);
      font-family: var(--mono-font);
      font-size: 12px;
      color: var(--text-secondary);
    }

    /* ---------------------------------------------------- */
    /* PAGE 2: TRANSFORM PORTRAIT STYLES                    */
    /* ---------------------------------------------------- */
    .portrait-view-inner {
      padding: 24px 36px 60px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
      max-width: 920px;
      margin: 0 auto;
      width: 100%;
    }

    .portrait-header-box {
      width: 100%;
      text-align: center;
      border-bottom: 1px solid var(--border-dark);
      padding-bottom: 14px;
    }

    .portrait-title {
      font-family: var(--serif-font);
      font-size: 2.1rem;
      font-weight: 500;
      letter-spacing: -0.015em;
      margin-bottom: 4px;
    }

    .portrait-desc {
      font-size: 13px;
      color: var(--text-secondary);
      max-width: 640px;
      margin: 0 auto;
    }

    /* Central Blank Canvas / Mosaic Matrix Box */
    .mosaic-canvas-card {
      background: var(--surface);
      border: 2px solid var(--border-dark);
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-shadow: 0 8px 24px rgba(0,0,0,0.06);
      width: 100%;
      max-width: 540px;
    }

    .canvas-plate-header {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono-font);
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-light);
      padding-bottom: 8px;
      margin-bottom: 12px;
    }

    .canvas-viewport {
      width: 460px;
      height: 460px;
      max-width: 100%;
      background: #eeece5;
      border: 1px solid var(--border-dark);
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    #portraitDisplayCanvas {
      width: 100%;
      height: 100%;
      display: none;
      object-fit: contain;
    }

    .canvas-vacant-placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 24px;
      gap: 12px;
      color: var(--text-muted);
      font-family: var(--mono-font);
    }

    .vacant-crossmark {
      font-size: 28px;
      font-weight: 300;
      color: var(--text-secondary);
    }

    .vacant-title {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-primary);
    }

    .vacant-sub {
      font-size: 11px;
      max-width: 260px;
      line-height: 1.5;
    }

    /* Config Bar */
    .portrait-config-bar {
      width: 100%;
      max-width: 540px;
      background: var(--surface);
      border: 1px solid var(--border-dark);
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .config-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    .config-label-cell {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .config-title {
      font-family: var(--mono-font);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .config-subtext {
      font-size: 11px;
      color: var(--text-muted);
    }

    .res-slider-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
      justify-content: flex-end;
    }

    .res-display-pill {
      font-family: var(--mono-font);
      font-size: 11px;
      font-weight: 600;
      background: #faf9f5;
      border: 1px solid var(--border-dark);
      padding: 4px 8px;
      min-width: 140px;
      text-align: center;
    }

    .res-slider {
      width: 140px;
      height: 2px;
      background: var(--border-dark);
      outline: none;
      -webkit-appearance: none;
    }

    .res-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 12px;
      height: 12px;
      background: var(--border-dark);
      cursor: pointer;
    }

    .view-mode-toggle {
      display: flex;
      gap: 6px;
    }

    .view-mode-btn {
      font-family: var(--mono-font);
      font-size: 10px;
      text-transform: uppercase;
      padding: 5px 10px;
      border: 1px solid var(--border-dark);
      background: #ffffff;
      cursor: pointer;
    }

    .view-mode-btn.active {
      background: #111111;
      color: #ffffff;
    }

    /* Upload Portal Box */
    .upload-portal-box {
      width: 100%;
      max-width: 540px;
      border: 1px dashed var(--border-dark);
      background: var(--surface);
      padding: 22px;
      text-align: center;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease, border-color 0.15s ease;
    }

    .upload-portal-box:hover {
      background: #faf8f0;
      border-style: solid;
    }

    .upload-portal-box.dragover {
      background: #f1eee2;
      border-style: solid;
      border-color: #000000;
    }

    .upload-icon {
      font-family: var(--mono-font);
      font-size: 20px;
      font-weight: 600;
    }

    .upload-title {
      font-family: var(--mono-font);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-primary);
    }

    .upload-meta {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* Actions: Download Button */
    .portrait-actions-row {
      display: flex;
      gap: 12px;
      width: 100%;
      max-width: 540px;
      justify-content: center;
    }

    #loadingIndicator {
      display: none;
      font-family: var(--mono-font);
      font-size: 11px;
      color: var(--text-primary);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-top: 4px;
    }

    /* Cropper Modal */
    .cropper-modal-overlay {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(18, 18, 18, 0.78);
      z-index: 2000;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .cropper-modal-overlay.active {
      display: flex;
    }

    .cropper-modal-box {
      background: #ffffff;
      border: 2px solid var(--border-dark);
      max-width: 620px;
      width: 100%;
      box-shadow: 0 16px 40px rgba(0,0,0,0.35);
      display: flex;
      flex-direction: column;
    }

    .cropper-modal-header {
      background: #f1f0eb;
      border-bottom: 1px solid var(--border-dark);
      padding: 10px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono-font);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .cropper-work-area {
      width: 100%;
      height: 380px;
      background: #111111;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .cropper-work-area img {
      max-width: 100%;
      max-height: 100%;
      display: block;
    }

    .cropper-modal-footer {
      padding: 14px 20px;
      background: #ffffff;
      border-top: 1px solid var(--border-dark);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .crop-notice {
      font-family: var(--mono-font);
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .crop-btn-group {
      display: flex;
      gap: 8px;
    }

    /* Lightbox Modal for Covers */
    .academic-modal {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(18, 18, 18, 0.75);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .academic-modal.active {
      display: flex;
    }

    .modal-box {
      background: #ffffff;
      border: 2px solid var(--border-dark);
      max-width: 820px;
      width: 100%;
      display: grid;
      grid-template-columns: 1fr 1fr;
      box-shadow: 0 16px 36px rgba(0,0,0,0.3);
    }

    @media (max-width: 768px) {
      .modal-box {
        grid-template-columns: 1fr;
      }
    }

    .modal-image-side {
      background: #eeece5;
      padding: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-right: 1px solid var(--border-dark);
    }

    .modal-image-side img {
      max-height: 480px;
      max-width: 100%;
      border: 1px solid var(--border-dark);
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    .modal-details-side {
      padding: 28px 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .modal-header-meta {
      font-family: var(--mono-font);
      font-size: 10px;
      text-transform: uppercase;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-dark);
      padding-bottom: 6px;
      margin-bottom: 12px;
    }

    .modal-title {
      font-family: var(--serif-font);
      font-size: 22px;
      margin-bottom: 16px;
      line-height: 1.25;
      color: var(--text-primary);
    }

    .modal-meta-table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--mono-font);
      font-size: 11px;
      margin-bottom: 20px;
    }

    .modal-meta-table td {
      padding: 7px 0;
      border-bottom: 1px solid var(--border-light);
    }

    .modal-meta-table td:first-child {
      color: var(--text-muted);
      text-transform: uppercase;
      width: 40%;
    }

    .modal-close-btn {
      font-family: var(--mono-font);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 8px 14px;
      border: 1px solid var(--border-dark);
      background: var(--border-dark);
      color: #ffffff;
      cursor: pointer;
      align-self: flex-start;
    }

    .modal-close-btn:hover {
      background: #333333;
    }
  </style>
</head>
<body>

  <!-- Top Bar -->
  <header class="topbar">
    <div class="brand-title">
      Folioform
      <span class="brand-byline">by Reed Zaman</span>
    </div>

    <div class="topbar-meta">
      CIELAB ΔE Chromatic Catalogue · 2,919 Accessions
    </div>
  </header>

  <!-- Workspace Container -->
  <main class="workspace">
    <!-- Narrow Sidebar with Page Navigation -->
    <aside class="app-sidebar">
      <div>
        <div class="sidebar-heading">Navigation</div>
        <nav class="sidebar-menu">
          <button class="nav-btn active" id="navPortraitBtn" data-target="pagePortrait">
            <span class="nav-marker"></span>
            <span>Transform Portrait</span>
          </button>
          <button class="nav-btn" id="navSearchBtn" data-target="pageSearch">
            <span class="nav-marker"></span>
            <span>Search & Index</span>
          </button>
        </nav>
      </div>

      <div class="sidebar-footer">
        Collection Index<br>
        2,919 / 32,580 covers<br>
        100 Spectral Bins
      </div>
    </aside>

    <!-- ==================================================== -->
    <!-- VIEW 1: SEARCH PAGE                                  -->
    <!-- ==================================================== -->
    <div class="page-view" id="pageSearch">
      <div class="search-view-inner">

        <!-- Slimmer Header on Search Page -->
        <header class="search-page-header">
          <div>
            <div class="header-pretitle">Taxonomic Archive / Cover Art Repository</div>
            <h1 class="header-main-title">Chromatic Index of Book Covers</h1>
          </div>
          <p class="header-caption">
            A computational catalog classifying 32,580 book covers across a 100-interval color spectrum. Matches computed via CIELAB Euclidean distance (ΔE<sub>76</sub>).
          </p>
        </header>

        <!-- Color Searching Section (Workbench in Main Body) -->
        <section class="search-workbench">
          <div class="workbench-header">
            <span>Chromatic Calibration & Spectral Query</span>
            <span>CIE 1976 (L*, a*, b*)</span>
          </div>

          <div class="workbench-body">
            <!-- Always Visible Canvas Color Wheel & Direct Inputs -->
            <div class="color-wheel-module">
              <div class="canvas-wheel-wrap">
                <canvas id="colorWheelCanvas" width="120" height="120"></canvas>
                <div class="wheel-crosshair" id="wheelCrosshair"></div>
              </div>

              <!-- Value / Lightness Slider -->
              <div class="brightness-slider-wrap">
                <span class="slider-tiny-lbl">Light</span>
                <input type="range" id="brightnessSlider" min="10" max="100" value="85" title="Brightness / Lightness">
                <span class="slider-tiny-lbl">Dark</span>
              </div>

              <!-- Direct Swatch & Hex input -->
              <div class="color-readout-col">
                <div class="swatch-box" id="swatchBox" title="Click for native color picker">
                  <input type="color" id="nativeColorPicker" value="#A33327">
                </div>
                <input type="text" id="hexInput" class="hex-field" value="#A33327" maxlength="7">
                <div class="rgb-readout" id="rgbReadout">R:163 G:51 B:39</div>
              </div>
            </div>

            <!-- Controls: Configurable Limit & Strict Strategy -->
            <div class="controls-module">
              <div class="controls-upper-row">
                <!-- Match Count Slider (Default: 6) -->
                <div class="field-wrap">
                  <div class="field-lbl">
                    <span>Match Count</span>
                    <span class="field-val-counter" id="limitDisplay">6</span>
                  </div>
                  <input type="range" id="limitSlider" class="slider-inline" min="2" max="24" step="2" value="6">
                </div>

                <!-- Strategy Selector (Default: Strict) -->
                <div class="field-wrap">
                  <div class="field-lbl">
                    <span>Strategy</span>
                  </div>
                  <select id="strategySelect" class="academic-select">
                    <option value="strict" selected>Strict (Closest Interval)</option>
                    <option value="exact">Exact (True Lowest ΔE)</option>
                    <option value="spread">Spectral Spread (Adjacent)</option>
                  </select>
                </div>

                <!-- Action Buttons -->
                <div class="btns-wrap">
                  <button class="act-btn btn-main" id="btnSearch">Retrieve</button>
                  <button class="act-btn" id="btnRandom">Random</button>
                  <button class="act-btn" id="btnShuffle" title="Draw another random sample">Re-sample</button>
                </div>
              </div>

              <!-- Reference Standards Swatches -->
              <div class="standards-row">
                <span class="standards-label">Standard Presets:</span>
                <div class="standards-chips">
                  <div class="std-chip" style="background: #A33327;" data-hex="#A33327" title="Deep Venetian Red"></div>
                  <div class="std-chip" style="background: #C4623B;" data-hex="#C4623B" title="Burnt Ochre"></div>
                  <div class="std-chip" style="background: #D9A74A;" data-hex="#D9A74A" title="Raw Sienna"></div>
                  <div class="std-chip" style="background: #E8D382;" data-hex="#E8D382" title="Pale Amber"></div>
                  <div class="std-chip" style="background: #5B8254;" data-hex="#5B8254" title="Sage Green"></div>
                  <div class="std-chip" style="background: #2C5E4A;" data-hex="#2C5E4A" title="Forest Green"></div>
                  <div class="std-chip" style="background: #3B728C;" data-hex="#3B728C" title="Slate Cyan"></div>
                  <div class="std-chip" style="background: #244673;" data-hex="#244673" title="Prussian Blue"></div>
                  <div class="std-chip" style="background: #513B6B;" data-hex="#513B6B" title="Imperial Violet"></div>
                  <div class="std-chip" style="background: #873D5E;" data-hex="#873D5E" title="Madder Crimson"></div>
                  <div class="std-chip" style="background: #1C1C1C;" data-hex="#1C1C1C" title="Carbon Black"></div>
                  <div class="std-chip" style="background: #808080;" data-hex="#808080" title="Neutral Gray"></div>
                  <div class="std-chip" style="background: #C8C5BB;" data-hex="#C8C5BB" title="Parchment Gray"></div>
                  <div class="std-chip" style="background: #FDFDFB;" data-hex="#FDFDFB" title="Vellum White"></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Ledger Summary Banner -->
        <div class="ledger-banner">
          <div class="ledger-cell">
            <span class="ledger-key">Query:</span>
            <span class="ledger-val" id="ledgerHex">#A33327</span>
          </div>
          <div class="ledger-cell">
            <span class="ledger-key">Interval:</span>
            <span class="ledger-dot" id="ledgerDot" style="background:#A33327;"></span>
            <span class="ledger-val" id="ledgerFolder">—</span>
          </div>
          <div class="ledger-cell">
            <span class="ledger-key">Folder Distance:</span>
            <span class="ledger-val" id="ledgerDelta">—</span>
          </div>
          <div class="ledger-cell">
            <span class="ledger-key">Retrieved:</span>
            <span class="ledger-val" id="ledgerCount">6 specimens</span>
          </div>
        </div>

        <!-- Results Gallery Section -->
        <section class="gallery-section">
          <div class="gallery-bar">
            <h2 class="gallery-heading">Matched Book Covers</h2>
            <span class="gallery-note" id="galleryNote">Ranked in strictly ascending ΔE order</span>
          </div>

          <div class="covers-grid" id="coversGrid">
            <div class="empty-state">Querying chromatic database...</div>
          </div>
        </section>

      </div>
    </div>

    <!-- ==================================================== -->
    <!-- VIEW 2: TRANSFORM YOUR PORTRAIT PAGE                 -->
    <!-- ==================================================== -->
    <div class="page-view active" id="pagePortrait">
      <div class="portrait-view-inner">

        <!-- Page Header -->
        <div class="portrait-header-box">
          <div class="header-pretitle">Discretized Chromatics · Book Cover Photomosaic</div>
          <h1 class="portrait-title">Transform Your Portrait</h1>
          <p class="portrait-desc">
            Convert a portrait into a mosaic composed entirely of matching book covers.
            Each cell maintains the authentic rectangular aspect ratio (1 : 1.45), with intelligent color clustering and neighbor-repeat suppression.
          </p>
        </div>

        <!-- 1. The Central Canvas Box -->
        <div class="mosaic-canvas-card">
          <div class="canvas-plate-header">
            <span>Specimen Plate · Book Cover Matrix</span>
            <span id="canvasPlateMeta">Status: Vacant</span>
          </div>

          <div class="canvas-viewport" id="viewportWrap">
            <!-- Blank Vacant Placeholder -->
            <div class="canvas-vacant-placeholder" id="canvasPlaceholder">
              <div class="vacant-crossmark">[ + ]</div>
              <div class="vacant-title">Vacant Specimen Box</div>
              <div class="vacant-sub">Upload and square-crop your portrait below to generate the book cover mosaic.</div>
            </div>

            <!-- Active Canvas / Image Element -->
            <img id="portraitDisplayCanvas" src="" alt="Book Cover Mosaic">
          </div>
        </div>

        <!-- 2. Configurable Density Controls -->
        <div class="portrait-config-bar">
          <div class="config-row">
            <div class="config-label-cell">
              <span class="config-title">Grid Density (Book Aspect 1 : 1.45)</span>
              <span class="config-subtext">Restricted range: 16 to 60 columns</span>
            </div>
            <div class="res-slider-wrap">
              <input type="range" id="resSlider" class="res-slider" min="16" max="60" step="4" value="28">
              <span class="res-display-pill" id="resDisplay">28 cols × 19 rows (532 covers)</span>
            </div>
          </div>
        </div>

        <!-- 3. Image Uploading Portal (Below Canvas Box) -->
        <div class="upload-portal-box" id="uploadPortal">
          <input type="file" id="portraitFileInput" accept="image/jpeg,image/png,image/webp" style="display:none;">
          <div class="upload-icon">+</div>
          <div class="upload-title">Select Portrait Image to Upload</div>
          <div class="upload-meta">Drag & drop image file here, or click to browse (1:1 square crop required)</div>
        </div>

        <!-- Loading Indicator -->
        <div id="loadingIndicator">Generating High-Res Book Cover Mosaic...</div>

        <!-- 4. Actions: Download Button -->
        <div class="portrait-actions-row">
          <button class="act-btn btn-main" id="btnDownloadPortrait" disabled style="opacity: 0.5; cursor: not-allowed; width: 100%; height:38px;">
            Download High-Res Master Mosaic (PNG)
          </button>
        </div>

      </div>
    </div>

  </main>

  <!-- ==================================================== -->
  <!-- MODAL: FORCED 1:1 SQUARE CROPPING                    -->
  <!-- ==================================================== -->
  <div class="cropper-modal-overlay" id="cropperModal">
    <div class="cropper-modal-box">
      <div class="cropper-modal-header">
        <span>Specimen Calibration · Standard 1:1 Square Crop</span>
        <span style="color:var(--text-muted);">Required Aspect Ratio</span>
      </div>

      <div class="cropper-work-area">
        <img id="imageToCrop" src="" alt="Source for Cropping">
      </div>

      <div class="cropper-modal-footer">
        <span class="crop-notice">Drag handles to compose square portrait framing</span>
        <div class="crop-btn-group">
          <button class="act-btn" id="btnCancelCrop">Cancel</button>
          <button class="act-btn btn-main" id="btnApplyCrop">Apply Crop & Generate Mosaic</button>
        </div>
      </div>
    </div>
  </div>

  <!-- ==================================================== -->
  <!-- MODAL: SPECIMEN INSPECTION LIGHTBOX (COVERS)         -->
  <!-- ==================================================== -->
  <div class="academic-modal" id="specimenModal">
    <div class="modal-box">
      <div class="modal-image-side">
        <img id="modalImg" src="" alt="Specimen Large">
      </div>
      <div class="modal-details-side">
        <div>
          <div class="modal-header-meta">Archival Record · Provenance</div>
          <h2 class="modal-title" id="modalCategory">Category</h2>
          <table class="modal-meta-table">
            <tr>
              <td>Accession File</td>
              <td id="modalFilename" style="word-break: break-all;">—</td>
            </tr>
            <tr>
              <td>Spectrum Class</td>
              <td id="modalSpectrum">—</td>
            </tr>
            <tr>
              <td>Cover Dominant Color</td>
              <td id="modalHex">—</td>
            </tr>
            <tr>
              <td>Perceptual Distance</td>
              <td id="modalDelta">—</td>
            </tr>
          </table>
        </div>
        <button class="modal-close-btn" id="modalClose">Close Record</button>
      </div>
    </div>
  </div>

  <script>
    // ----------------------------------------------------
    // NAVIGATION CONTROLLER
    // ----------------------------------------------------
    const navSearchBtn = document.getElementById('navSearchBtn');
    const navPortraitBtn = document.getElementById('navPortraitBtn');
    const pageSearch = document.getElementById('pageSearch');
    const pagePortrait = document.getElementById('pagePortrait');

    function switchView(target) {
      if (target === 'pageSearch') {
        navSearchBtn.classList.add('active');
        navPortraitBtn.classList.remove('active');
        pageSearch.classList.add('active');
        pagePortrait.classList.remove('active');
      } else {
        navPortraitBtn.classList.add('active');
        navSearchBtn.classList.remove('active');
        pagePortrait.classList.add('active');
        pageSearch.classList.remove('active');
      }
    }

    navSearchBtn.addEventListener('click', () => switchView('pageSearch'));
    navPortraitBtn.addEventListener('click', () => switchView('pagePortrait'));

    // ----------------------------------------------------
    // SEARCH & COLOR WHEEL CONTROLLER
    // ----------------------------------------------------
    const wheelCanvas = document.getElementById('colorWheelCanvas');
    const wheelCrosshair = document.getElementById('wheelCrosshair');
    const brightnessSlider = document.getElementById('brightnessSlider');
    const swatchBox = document.getElementById('swatchBox');
    const nativeColorPicker = document.getElementById('nativeColorPicker');
    const hexInput = document.getElementById('hexInput');
    const rgbReadout = document.getElementById('rgbReadout');
    const limitSlider = document.getElementById('limitSlider');
    const limitDisplay = document.getElementById('limitDisplay');
    const strategySelect = document.getElementById('strategySelect');
    const btnSearch = document.getElementById('btnSearch');
    const btnRandom = document.getElementById('btnRandom');
    const btnShuffle = document.getElementById('btnShuffle');
    const coversGrid = document.getElementById('coversGrid');

    const ledgerHex = document.getElementById('ledgerHex');
    const ledgerDot = document.getElementById('ledgerDot');
    const ledgerFolder = document.getElementById('ledgerFolder');
    const ledgerDelta = document.getElementById('ledgerDelta');
    const ledgerCount = document.getElementById('ledgerCount');

    const specimenModal = document.getElementById('specimenModal');
    const modalImg = document.getElementById('modalImg');
    const modalCategory = document.getElementById('modalCategory');
    const modalFilename = document.getElementById('modalFilename');
    const modalSpectrum = document.getElementById('modalSpectrum');
    const modalHex = document.getElementById('modalHex');
    const modalDelta = document.getElementById('modalDelta');
    const modalClose = document.getElementById('modalClose');

    let currentH = 0.015;
    let currentS = 0.76;
    let currentV = 0.64;
    let isDraggingWheel = false;

    function hsvToRgb(h, s, v) {
      let r, g, b;
      let i = Math.floor(h * 6);
      let f = h * 6 - i;
      let p = v * (1 - s);
      let q = v * (1 - f * s);
      let t = v * (1 - (1 - f) * s);
      switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
      }
      return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    }

    function rgbToHsv(r, g, b) {
      r /= 255; g /= 255; b /= 255;
      let max = Math.max(r, g, b), min = Math.min(r, g, b);
      let h, s, v = max;
      let d = max - min;
      s = max === 0 ? 0 : d / max;
      if (max === min) {
        h = 0;
      } else {
        switch (max) {
          case r: h = (g - b) / d + (g < b ? 6 : 0); break;
          case g: h = (b - r) / d + 2; break;
          case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
      }
      return [h, s, v];
    }

    function rgbToHex(r, g, b) {
      return "#" + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('').toUpperCase();
    }

    function hexToRgb(hex) {
      hex = hex.replace('#', '');
      if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
      let num = parseInt(hex, 16);
      return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
    }

    function drawWheel() {
      const ctx = wheelCanvas.getContext('2d');
      const w = wheelCanvas.width;
      const h = wheelCanvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const r = cx - 1;

      const imgData = ctx.createImageData(w, h);
      const data = imgData.data;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let dx = x - cx;
          let dy = y - cy;
          let dist = Math.sqrt(dx * dx + dy * dy);
          let idx = (y * w + x) * 4;

          if (dist <= r) {
            let angle = Math.atan2(dy, dx) * (180 / Math.PI);
            if (angle < 0) angle += 360;
            let sat = Math.min(1, dist / r);
            let [red, green, blue] = hsvToRgb(angle / 360, sat, currentV);
            data[idx] = red;
            data[idx + 1] = green;
            data[idx + 2] = blue;
            data[idx + 3] = 255;
          } else {
            data[idx + 3] = 0;
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = '#1a1a1a';
      ctx.lineWidth = 1;
      ctx.stroke();

      let angleRad = currentH * 2 * Math.PI;
      let distPx = currentS * r;
      let px = cx + Math.cos(angleRad) * distPx;
      let py = cy + Math.sin(angleRad) * distPx;
      wheelCrosshair.style.left = px + 'px';
      wheelCrosshair.style.top = py + 'px';
    }

    function handleWheelPick(e) {
      const rect = wheelCanvas.getBoundingClientRect();
      const cx = rect.width / 2;
      const cy = rect.height / 2;
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const r = cx - 1;

      let angle = Math.atan2(dy, dx) * (180 / Math.PI);
      if (angle < 0) angle += 360;

      currentH = angle / 360;
      currentS = Math.min(1, dist / r);

      let [red, green, blue] = hsvToRgb(currentH, currentS, currentV);
      let hex = rgbToHex(red, green, blue);

      applyColor(hex, false);
      drawWheel();
      debouncedRetrieve();
    }

    wheelCanvas.addEventListener('mousedown', (e) => {
      isDraggingWheel = true;
      handleWheelPick(e);
    });

    window.addEventListener('mousemove', (e) => {
      if (isDraggingWheel) handleWheelPick(e);
    });

    window.addEventListener('mouseup', () => {
      if (isDraggingWheel) isDraggingWheel = false;
    });

    brightnessSlider.addEventListener('input', (e) => {
      currentV = e.target.value / 100;
      let [red, green, blue] = hsvToRgb(currentH, currentS, currentV);
      let hex = rgbToHex(red, green, blue);
      applyColor(hex, false);
      drawWheel();
      debouncedRetrieve();
    });

    function applyColor(hex, updateHsv = true) {
      if (!hex.startsWith('#')) hex = '#' + hex;
      swatchBox.style.backgroundColor = hex;
      nativeColorPicker.value = hex;
      hexInput.value = hex.toUpperCase();

      let [r, g, b] = hexToRgb(hex);
      rgbReadout.textContent = `R:${r} G:${g} B:${b}`;

      if (updateHsv) {
        let [h, s, v] = rgbToHsv(r, g, b);
        currentH = h;
        currentS = s;
        currentV = Math.max(0.1, v);
        brightnessSlider.value = Math.round(currentV * 100);
        drawWheel();
      }
    }

    nativeColorPicker.addEventListener('input', (e) => {
      applyColor(e.target.value, true);
      debouncedRetrieve();
    });

    hexInput.addEventListener('change', (e) => {
      let val = e.target.value.trim();
      if (!val.startsWith('#')) val = '#' + val;
      if (/^#[0-9A-Fa-f]{6}$/.test(val)) {
        applyColor(val, true);
        debouncedRetrieve(true);
      }
    });

    limitSlider.addEventListener('input', (e) => {
      limitDisplay.textContent = e.target.value;
    });
    limitSlider.addEventListener('change', () => debouncedRetrieve(true));
    strategySelect.addEventListener('change', () => debouncedRetrieve(true));

    document.querySelectorAll('.std-chip').forEach(sw => {
      sw.addEventListener('click', () => {
        applyColor(sw.dataset.hex, true);
        debouncedRetrieve(true);
      });
    });

    btnSearch.addEventListener('click', () => debouncedRetrieve(true));
    btnShuffle.addEventListener('click', () => debouncedRetrieve(true, true));

    btnRandom.addEventListener('click', () => {
      const randomHex = '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0').toUpperCase();
      applyColor(randomHex, true);
      debouncedRetrieve(true);
    });

    let debounceTimer = null;
    let currentAbortController = null;

    function debouncedRetrieve(immediate = false, shuffle = false) {
      if (debounceTimer) clearTimeout(debounceTimer);
      if (immediate) {
        retrieveCatalogue(shuffle);
      } else {
        debounceTimer = setTimeout(() => retrieveCatalogue(shuffle), 130);
      }
    }

    async function retrieveCatalogue(shuffle = false) {
      const hex = hexInput.value.trim();
      const limit = limitSlider.value;
      const strategy = strategySelect.value;

      ledgerHex.textContent = hex;

      if (currentAbortController) currentAbortController.abort();
      currentAbortController = new AbortController();

      try {
        const url = `/api/search?hex=${encodeURIComponent(hex)}&limit=${limit}&strategy=${strategy}&shuffle=${shuffle}`;
        const res = await fetch(url, { signal: currentAbortController.signal });
        const data = await res.json();

        if (data.error) {
          coversGrid.innerHTML = `<div class="empty-state">System Error: ${data.error}</div>`;
          return;
        }

        if (data.primary_match) {
          ledgerDot.style.backgroundColor = data.primary_match.hex;
          ledgerFolder.textContent = data.primary_match.folder.split('_').slice(1, -1).join(' ');
          ledgerDelta.textContent = `ΔE ${data.primary_match.distance}`;
        }
        ledgerCount.textContent = `${data.results_count} specimens`;

        if (!data.covers || data.covers.length === 0) {
          coversGrid.innerHTML = '<div class="empty-state">No specimens recorded within this interval.</div>';
          return;
        }

        coversGrid.innerHTML = data.covers.map((c, i) => `
          <div class="specimen-card"
               data-url="${c.url}"
               data-category="${c.category}"
               data-filename="${c.filename}"
               data-folder="${c.folder}"
               data-hex="${c.cover_hex}"
               data-delta="${c.distance}">
            <div class="rank-tag ${i === 0 ? 'top-match' : ''}">#${c.rank}${i === 0 ? ' BEST' : ''}</div>
            <div class="specimen-frame">
              <img src="${c.url}" alt="${c.filename}" loading="lazy">
            </div>
            <div class="specimen-meta">
              <div class="discipline-tag">${c.category}</div>
              <div class="specimen-stats">
                <span>
                  <span class="color-swatch-mini" style="background:${c.cover_hex}"></span>
                  ${c.cover_hex}
                </span>
                <span class="delta-rating">ΔE ${c.distance}</span>
              </div>
            </div>
          </div>
        `).join('');

        document.querySelectorAll('.specimen-card').forEach(card => {
          card.addEventListener('click', () => {
            modalImg.src = card.dataset.url;
            modalCategory.textContent = card.dataset.category;
            modalFilename.textContent = card.dataset.filename;
            modalSpectrum.textContent = card.dataset.folder;
            modalHex.textContent = card.dataset.hex;
            modalDelta.textContent = `ΔE ${card.dataset.delta}`;
            specimenModal.classList.add('active');
          });
        });

      } catch (err) {
        if (err.name !== 'AbortError') {
          coversGrid.innerHTML = `<div class="empty-state">Network exception: ${err.message}</div>`;
        }
      }
    }

    modalClose.addEventListener('click', () => specimenModal.classList.remove('active'));
    specimenModal.addEventListener('click', (e) => {
      if (e.target === specimenModal) specimenModal.classList.remove('active');
    });

    // ----------------------------------------------------
    // TRANSFORM YOUR PORTRAIT CONTROLLER
    // ----------------------------------------------------
    const uploadPortal = document.getElementById('uploadPortal');
    const portraitFileInput = document.getElementById('portraitFileInput');
    const cropperModal = document.getElementById('cropperModal');
    const imageToCrop = document.getElementById('imageToCrop');
    const btnCancelCrop = document.getElementById('btnCancelCrop');
    const btnApplyCrop = document.getElementById('btnApplyCrop');

    const portraitDisplayCanvas = document.getElementById('portraitDisplayCanvas');
    const canvasPlaceholder = document.getElementById('canvasPlaceholder');
    const canvasPlateMeta = document.getElementById('canvasPlateMeta');
    const resSlider = document.getElementById('resSlider');
    const resDisplay = document.getElementById('resDisplay');
    const btnDownloadPortrait = document.getElementById('btnDownloadPortrait');
    const loadingIndicator = document.getElementById('loadingIndicator');

    let cropper = null;
    let croppedBase64 = null;
    let currentMosaicData = null;

    // Open file chooser
    uploadPortal.addEventListener('click', () => {
      portraitFileInput.click();
    });

    uploadPortal.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadPortal.classList.add('dragover');
    });

    uploadPortal.addEventListener('dragleave', () => {
      uploadPortal.classList.remove('dragover');
    });

    uploadPortal.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadPortal.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    portraitFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });

    function handleFileSelected(file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file (JPEG, PNG, WEBP).');
        return;
      }

      const reader = new FileReader();
      reader.onload = (evt) => {
        imageToCrop.src = evt.target.result;
        cropperModal.classList.add('active');

        if (cropper) {
          cropper.destroy();
        }

        // Initialize Cropper strictly locked to 1:1 square crop
        cropper = new Cropper(imageToCrop, {
          aspectRatio: 1,
          viewMode: 1,
          dragMode: 'move',
          autoCropArea: 0.92,
          restore: false,
          guides: true,
          center: true,
          highlight: false,
          cropBoxMovable: true,
          cropBoxResizable: true,
          toggleDragModeOnDblclick: false,
        });
      };
      reader.readAsDataURL(file);
    }

    btnCancelCrop.addEventListener('click', () => {
      cropperModal.classList.remove('active');
      if (cropper) {
        cropper.destroy();
        cropper = null;
      }
      portraitFileInput.value = '';
    });

    btnApplyCrop.addEventListener('click', () => {
      if (!cropper) return;

      const canvas = cropper.getCroppedCanvas({
        width: 600,
        height: 600,
        imageSmoothingEnabled: true,
        imageSmoothingQuality: 'high',
      });

      croppedBase64 = canvas.toDataURL('image/jpeg', 0.92);

      cropperModal.classList.remove('active');
      cropper.destroy();
      cropper = null;
      portraitFileInput.value = '';

      canvasPlaceholder.style.display = 'none';
      portraitDisplayCanvas.style.display = 'block';

      // Generate the mosaic
      generateCoverMosaic();
    });

    resSlider.addEventListener('input', (e) => {
      const cols = parseInt(e.target.value, 10);
      const rows = Math.round(cols / 1.45);
      resDisplay.textContent = `${cols} cols × ${rows} rows (${cols * rows} covers)`;
    });

    resSlider.addEventListener('change', () => {
      if (croppedBase64) {
        generateCoverMosaic();
      }
    });

    async function generateCoverMosaic() {
      if (!croppedBase64) return;

      const cols = parseInt(resSlider.value, 10);
      const rows = Math.round(cols / 1.45);

      loadingIndicator.style.display = 'block';
      loadingIndicator.textContent = `Assembling ${cols * rows} rectangular book covers...`;
      canvasPlateMeta.textContent = `Processing ${cols}×${rows} Matrix...`;

      // Render local rectangular pixel grid preview immediately
      renderLocalPixelPreview(cols, rows);

      try {
        const res = await fetch('/api/generate_mosaic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: croppedBase64,
            cols: cols
          })
        });

        const data = await res.json();
        if (data.error) {
          alert('Error generating mosaic: ' + data.error);
          loadingIndicator.style.display = 'none';
          return;
        }

        currentMosaicData = data;
        loadingIndicator.style.display = 'none';
        canvasPlateMeta.textContent = `Master Mosaic: ${data.cols}×${data.rows} (${data.total_covers} covers · ${data.master_dimensions})`;

        portraitDisplayCanvas.src = data.preview_url + '?t=' + Date.now();

        // Enable download button
        btnDownloadPortrait.disabled = false;
        btnDownloadPortrait.style.opacity = '1';
        btnDownloadPortrait.style.cursor = 'pointer';
        btnDownloadPortrait.textContent = `Download High-Res Master Mosaic (${data.master_dimensions} PNG)`;

      } catch (err) {
        loadingIndicator.style.display = 'none';
        alert('Failed to connect to mosaic generator: ' + err.message);
      }
    }

    function renderLocalPixelPreview(cols, rows) {
      const img = new Image();
      img.onload = () => {
        // Create an offscreen canvas to sample down to cols x rows
        const off = document.createElement('canvas');
        off.width = cols;
        off.height = rows;
        const octx = off.getContext('2d');
        octx.drawImage(img, 0, 0, cols, rows);

        // Scale up to rectangular cells (1 : 1.45 aspect ratio)
        const cellW = 16;
        const cellH = Math.round(cellW * 1.45);
        const prevCanvas = document.createElement('canvas');
        prevCanvas.width = cols * cellW;
        prevCanvas.height = rows * cellH;
        const pctx = prevCanvas.getContext('2d');
        pctx.imageSmoothingEnabled = false;
        pctx.drawImage(off, 0, 0, prevCanvas.width, prevCanvas.height);

        portraitDisplayCanvas.src = prevCanvas.toDataURL('image/png');
      };
      img.src = croppedBase64;
    }

    // Trigger download of master high-resolution PNG
    btnDownloadPortrait.addEventListener('click', () => {
      if (!currentMosaicData || !currentMosaicData.download_url) return;
      window.location.href = currentMosaicData.download_url;
    });

    // Initialize Wheel & First Search
    drawWheel();
    applyColor('#A33327', true);
    debouncedRetrieve(true);
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Folioform Web Portal")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web portal on (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Starting Folioform Web Portal on http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
