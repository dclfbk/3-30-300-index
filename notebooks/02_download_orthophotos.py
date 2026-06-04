"""
download_nir_tiles.py  —  WMS NIR tile downloader
Usage:
    python download_nir_tiles.py --city bologna
    python download_nir_tiles.py --city bologna --gsd 0.20 --tile-px 2048 --overlap-px 64
"""
import argparse
import io
import math
import os
import sys

import numpy as np
import rasterio
from PIL import Image
from owslib.wms import WebMapService
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from tqdm import tqdm

from config.config import CITIES, connect_db, get_city_cfg, load_table, ROOT_DIR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--city",       required=True)
p.add_argument("--gsd",        type=float, default=0.20,  help="Ground sampling distance in metres")
p.add_argument("--tile-px",    type=int,   default=2048,  help="Tile size in pixels (square)")
p.add_argument("--overlap-px", type=int,   default=64,    help="Overlap in pixels on each edge")
p.add_argument("--save-crs",   default="EPSG:32632",      help="Output CRS (metric)")
p.add_argument("--layer",       default=None,              help="WMS layer name (default: first available)")
p.add_argument("--wms-version",default="1.3.0")
p.add_argument("--fmt",        default="image/png",       help="WMS image format")
args = p.parse_args()

CITY = args.city

# ---------------------------------------------------------------------------
# Validate city has WMS
# ---------------------------------------------------------------------------
cfg = get_city_cfg(CITY)
if "wms" not in cfg:
    wms_cities = [c for c, v in CITIES.items() if v and "wms" in v]
    sys.exit(f"No WMS configured for '{CITY}'. Cities with WMS: {', '.join(wms_cities)}")

WMS_URL  = cfg["wms"]
TILE_DIR = os.path.join(ROOT_DIR, "data", CITY, f"{CITY}_nir_tiles")
os.makedirs(TILE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
DEG_PER_M   = 1 / 111_320               # degrees per metre (approx, sufficient at IT latitudes)
GSD_DEG     = args.gsd * DEG_PER_M
TILE_DEG    = args.tile_px    * GSD_DEG
OVERLAP_DEG = args.overlap_px * GSD_DEG

# ---------------------------------------------------------------------------
# Extent from DB boundary
# ---------------------------------------------------------------------------
con = connect_db()
xmin, ymin, xmax, ymax = (
    load_table(con, "raw", f"{CITY}_boundary")
    .to_crs("EPSG:4326")
    .total_bounds
)
con.close()

# ---------------------------------------------------------------------------
# WMS connection + layer/band inspection
# ---------------------------------------------------------------------------

# Known band layouts keyed by substrings found in layer names/titles.
# Order is (band_1, band_2, band_3) matching RGB channels of the PNG response.
BAND_HINTS: dict[str, tuple[str, str, str]] = {
    "nir":   ("NIR",   "Red",   "Green"),   # CIR composite
    "cir":   ("NIR",   "Red",   "Green"),
    "false": ("NIR",   "Red",   "Green"),   # "false colour"
    "ir":    ("NIR",   "Red",   "Green"),
    "rgb":   ("Red",   "Green", "Blue"),
    "ortho": ("Red",   "Green", "Blue"),
    "foto":  ("Red",   "Green", "Blue"),
    "photo": ("Red",   "Green", "Blue"),
    "true":  ("Red",   "Green", "Blue"),    # "true colour"
}
DEFAULT_BANDS = ("Red", "Green", "Blue")   # fallback if no hint matches

def detect_bands(layer_name: str, layer_title: str) -> tuple[str, str, str]:
    combined = (layer_name + " " + (layer_title or "")).lower()
    for keyword, bands in BAND_HINTS.items():
        if keyword in combined:
            return bands
    return DEFAULT_BANDS

print(f"[{CITY}] Connecting to WMS …")
wms    = WebMapService(WMS_URL, version=args.wms_version, timeout=120)
layers = list(wms.contents.keys())

# Let the user override the layer via --layer; otherwise pick the first
LAYER = args.layer if args.layer else layers[0]
if LAYER not in wms.contents:
    sys.exit(f"Layer '{LAYER}' not found. Available layers:\n  " + "\n  ".join(layers))

layer_title = getattr(wms.contents[LAYER], "title", "")
BANDS       = detect_bands(LAYER, layer_title)

print(f"  Layer : {LAYER}  ('{layer_title}')")
print(f"  Bands : 1={BANDS[0]}  2={BANDS[1]}  3={BANDS[2]}"
      + ("  ← NIR detected" if BANDS[0] == "NIR" else "  ← RGB (no NIR)"))
print(f"  All layers available: {layers}")
print(f"  Extent: ({xmin:.4f}, {ymin:.4f}) → ({xmax:.4f}, {ymax:.4f})")

# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------
cols = math.ceil((xmax - xmin) / TILE_DEG)
rows = math.ceil((ymax - ymin) / TILE_DEG)
print(f"  Grid  : {cols} cols × {rows} rows = {cols * rows} tiles\n")

transformer = Transformer.from_crs("EPSG:4326", args.save_crs, always_xy=True)

# ---------------------------------------------------------------------------
# Download loop
# ---------------------------------------------------------------------------
failed = []

for row in tqdm(range(rows), desc="Rows"):
    for col in range(cols):

        tile_path = os.path.join(TILE_DIR, f"tile_{row:04d}_{col:04d}.tif")
        if os.path.exists(tile_path):
            continue

        # Core tile bounds (WGS84)
        tx0 = xmin + col * TILE_DEG
        ty0 = ymin + row * TILE_DEG
        tx1 = min(tx0 + TILE_DEG, xmax)
        ty1 = min(ty0 + TILE_DEG, ymax)

        # Expanded bounds with overlap, clamped to study area
        ex0 = max(tx0 - OVERLAP_DEG, xmin)
        ey0 = max(ty0 - OVERLAP_DEG, ymin)
        ex1 = min(tx1 + OVERLAP_DEG, xmax)
        ey1 = min(ty1 + OVERLAP_DEG, ymax)

        px_w = max(1, round((ex1 - ex0) / GSD_DEG))
        px_h = max(1, round((ey1 - ey0) / GSD_DEG))

        try:
            resp = wms.getmap(
                layers=[LAYER],
                srs="EPSG:4326",
                bbox=(ex0, ey0, ex1, ey1),
                size=(px_w, px_h),
                format=args.fmt,
                transparent=False,
            )
            arr = np.array(Image.open(io.BytesIO(resp.read())).convert("RGB"))

            # Metric bounds for GeoTIFF transform
            mx0, my0 = transformer.transform(ex0, ey0)
            mx1, my1 = transformer.transform(ex1, ey1)

            # True (non-overlapping) tile bounds in metric, stored as tags
            tmx0, tmy0 = transformer.transform(tx0, ty0)
            tmx1, tmy1 = transformer.transform(tx1, ty1)

            with rasterio.open(
                tile_path, "w",
                driver="GTiff",
                height=px_h, width=px_w,
                count=3,
                dtype=arr.dtype,
                crs=CRS.from_string(args.save_crs),
                transform=from_bounds(mx0, my0, mx1, my1, px_w, px_h),
                compress="lzw",
                tiled=True,
                blockxsize=256, blockysize=256,
            ) as dst:
                dst.write(arr[:, :, 0], 1)
                dst.write(arr[:, :, 1], 2)
                dst.write(arr[:, :, 2], 3)
                dst.update_tags(
                    band_1=BANDS[0], band_2=BANDS[1], band_3=BANDS[2],
                    overlap_px=args.overlap_px,
                    true_xmin=tmx0, true_ymin=tmy0,
                    true_xmax=tmx1, true_ymax=tmy1,
                )

        except Exception as e:
            tqdm.write(f"  FAILED ({row},{col}): {e}")
            failed.append((row, col))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = rows * cols
print(f"\nDone: {total - len(failed)}/{total} tiles → {TILE_DIR}")
if failed:
    print(f"Failed ({len(failed)} tiles — re-run to retry): {failed}")
print(f"Bands: 1={BANDS[0]}  2={BANDS[1]}  3={BANDS[2]}  |  CRS: {args.save_crs}")