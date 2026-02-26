"""
Downloads the Emilia-Romagna RGB orthophoto WMS in tiles
and merges them into a single GeoTIFF ready for the tree pipeline.
WS endpoint:
    https://servizigis.regione.emilia-romagna.it/wms/comune_bo_2023_rgb
"""

import os
import io
import math
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge
from rasterio.crs import CRS
import rasterio.io
from owslib.wms import WebMapService
from tqdm import tqdm
import glob
import rasterio
from rasterio.merge import merge

# ── CONFIG ────────────────────────────────────────────────────

WMS_URL    = "https://servizigis.regione.emilia-romagna.it/wms/comune_bo_2023_rgb"
OUTPUT_DIR = "./bologna_ortho"
OUT_FILE   = os.path.join(OUTPUT_DIR, "bologna_rgb_2023.tif")

# Bologna municipality bounding box — EPSG:32632 (UTM 32N), metres
# Roughly: W=679000, S=4919000, E=695000, N=4932000
# Adjust to zoom in on a smaller area for testing
BBOX   = (679000, 4919000, 695000, 4932000)   # (xmin, ymin, xmax, ymax)
CRS_CODE = "EPSG:32632"

# Tile settings
TILE_PX   = 2048      # pixels per tile request — bigger = fewer requests, more RAM
GSD_M     = 0.20      # desired ground resolution in metres (20 cm = native resolution)
            # increase to e.g. 0.5 for a quick test run

WMS_VERSION = "1.3.0"
IMAGE_FMT   = "image/png"   # or "image/jpeg" — PNG keeps quality, JPEG is smaller
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Connect to WMS
print(f"Connecting to WMS: {WMS_URL}")
wms = WebMapService(WMS_URL, version=WMS_VERSION, timeout=120)

# Auto-detect layer name
layers = list(wms.contents.keys())
print(f"Available layers: {layers}")
LAYER = layers[0]   # use first layer — change if needed
print(f"Using layer: {LAYER}")

# ── Compute tile grid
xmin, ymin, xmax, ymax = BBOX
tile_m = TILE_PX * GSD_M    # real-world size of each tile in metres

cols = math.ceil((xmax - xmin) / tile_m)
rows = math.ceil((ymax - ymin) / tile_m)
total = cols * rows
print(f"\nGrid: {cols} cols × {rows} rows = {total} tiles")
print(f"Tile size: {tile_m:.0f}m × {tile_m:.0f}m  ({TILE_PX}×{TILE_PX} px)")
print(f"Total area: {(xmax-xmin)/1000:.1f} × {(ymax-ymin)/1000:.1f} km\n")

tile_paths = []

for row in tqdm(range(rows), desc="Downloading rows"):
    for col in range(cols):
        tx_min = xmin + col * tile_m
        ty_min = ymin + row * tile_m
        tx_max = min(tx_min + tile_m, xmax)
        ty_max = min(ty_min + tile_m, ymax)

        # Actual pixel size for edge tiles
        actual_px_w = max(1, round((tx_max - tx_min) / GSD_M))
        actual_px_h = max(1, round((ty_max - ty_min) / GSD_M))

        tile_path = os.path.join(OUTPUT_DIR, f"tile_{row:04d}_{col:04d}.tif")
        tile_paths.append(tile_path)

        if os.path.exists(tile_path):
            continue  # resume support

        try:
            # WMS 1.3.0 uses (miny, minx, maxy, maxx) for geographic CRS
            # but for projected CRS (EPSG:32632) it's (xmin, ymin, xmax, ymax)
            response = wms.getmap(
                layers=[LAYER],
                srs=CRS_CODE,
                bbox=(tx_min, ty_min, tx_max, ty_max),
                size=(actual_px_w, actual_px_h),
                format=IMAGE_FMT,
                transparent=False,
            )

            img_bytes = response.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            arr = np.array(img)   # (H, W, 3)

            transform = from_bounds(tx_min, ty_min, tx_max, ty_max,
                                    actual_px_w, actual_px_h)

            with rasterio.open(
                tile_path, "w",
                driver="GTiff",
                height=actual_px_h,
                width=actual_px_w,
                count=3,
                dtype=arr.dtype,
                crs=CRS.from_string(CRS_CODE),
                transform=transform,
                compress="lzw",
            ) as dst:
                dst.write(arr[:, :, 0], 1)
                dst.write(arr[:, :, 1], 2)
                dst.write(arr[:, :, 2], 3)

        except Exception as e:
            print(f"\n  ⚠ Tile ({row},{col}) failed: {e}")
            tile_paths.remove(tile_path)

# ── Merge tiles into single GeoTIFF
print(f"merging {len(tile_paths)} tiles into {OUT_FILE}...")

datasets = [rasterio.open(p) for p in tile_paths if os.path.exists(p)]
mosaic, mosaic_transform = merge(datasets)

meta = datasets[0].meta.copy()
meta.update({
    "driver":    "GTiff",
    "height":    mosaic.shape[1],
    "width":     mosaic.shape[2],
    "transform": mosaic_transform,
    "compress":  "lzw",
    "tiled":     True,
    "blockxsize": 512,
    "blockysize": 512,
})

with rasterio.open(OUT_FILE, "w", **meta) as dst:
    dst.write(mosaic)

for ds in datasets:
    ds.close()

print(f"Saved: {OUT_FILE}")
print(f"Size: {mosaic.shape[2]:,} × {mosaic.shape[1]:,} px")
print(f"IMAGE_PATH = '{OUT_FILE}' in bologna_tree_pipeline.py")

TILE_DIR = "./bologna_ortho"                          # .tif files
OUT_FILE = "./bologna_ortho/bologna_rgb_2023.tif"     # output mosaic path

# Find all tiles
tile_paths = sorted(glob.glob(os.path.join(TILE_DIR, "tile_*.tif")))
print(f"Found {len(tile_paths)} tiles in {TILE_DIR}")

if len(tile_paths) == 0:
    exit()

# Check for missing tiles (gaps from failed downloads)
# Infer expected grid from filenames
rows_found = sorted(set(int(os.path.basename(p).split("_")[1]) for p in tile_paths))
cols_found = sorted(set(int(os.path.basename(p).split("_")[2].replace(".tif","")) for p in tile_paths))
expected_count = (max(rows_found)+1) * (max(cols_found)+1)
if len(tile_paths) < expected_count:
    print(f"Expected ~{expected_count} tiles, found {len(tile_paths)} — {expected_count - len(tile_paths)} missing.")
    print("Merging anyway — missing tiles will appear as black holes in the mosaic.")
else:
    print(f"All tiles present")

# ── Merge
print(f"Merging:")
print(f" checking progress of the merge process: {len(tile_paths)} tiles to merge...")
datasets = [rasterio.open(p) for p in tile_paths]

mosaic, mosaic_transform = merge(datasets)
print(f"Mosaic size: {mosaic.shape[2]:,} × {mosaic.shape[1]:,} px")

meta = datasets[0].meta.copy()
meta.update({
    "driver":     "GTiff",
    "height":     mosaic.shape[1],
    "width":      mosaic.shape[2],
    "transform":  mosaic_transform,
    "compress":   "lzw",
    "tiled":      True,
    "blockxsize": 512,
    "blockysize": 512,
    "BIGTIFF":    "YES",
})

with rasterio.open(OUT_FILE, "w", **meta) as dst:
    dst.write(mosaic)

del mosaic  # free RAM
for ds in datasets:
    ds.close()

file_size_gb = os.path.getsize(OUT_FILE) / (1024 ** 3)
print(f"Saved: {OUT_FILE}  ({file_size_gb:.1f} GB)")