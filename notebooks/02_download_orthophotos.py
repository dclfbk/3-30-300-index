

import os
import io
import math
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer
from owslib.wms import WebMapService
from tqdm import tqdm
from config.config import load_table, connect_db, get_city_cfg, get_output_paths
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WMS_URL     = "https://servizigis.regione.emilia-romagna.it/wms/comune_bo_2023_nir"
TILE_DIR = "/Users/terezasaskova/Library/CloudStorage/GoogleDrive-tsaskova@fbk.eu/My Drive/3-30-300/data/Italy/Bologna/bologna_nir_tiles"
CRS_CODE    = "EPSG:4326"
SAVE_CRS    = "EPSG:32632"           # reproject to metric CRS when saving

con = connect_db()
CITY = "bologna"

# Full Bologna extent from boundingBoxWGS84
boundary_gdf = load_table(con, "raw", f"{CITY}_boundary")
xmin, ymin, xmax, ymax = boundary_gdf.to_crs("EPSG:4326").total_bounds

GSD_DEG     = 0.20 / 111320          # 20cm in degrees (approx at Bologna latitude)
TILE_PX     = 2048
OVERLAP_PX  = 64
WMS_VERSION = "1.3.0"
IMAGE_FMT   = "image/png"

os.makedirs(TILE_DIR, exist_ok=True)

# Transformer WGS84 => UTM32N for saving metric tiles - this means the tile bounds will be slightly off from the original WGS84 bounds, but it's a necessary step to get accurate area measurements and simplify geometry later on.
transformer = Transformer.from_crs("EPSG:4326", SAVE_CRS, always_xy=True)

# Connect + detect layer
print(f"Connecting: {WMS_URL}")
wms = WebMapService(WMS_URL, version=WMS_VERSION, timeout=120)
LAYER = list(wms.contents.keys())[0]
print(f"Using layer: {LAYER}\n")

# Tile grid in degrees
xmin, ymin, xmax, ymax = BBOX
tile_deg    = TILE_PX * GSD_DEG
overlap_deg = OVERLAP_PX * GSD_DEG

cols = math.ceil((xmax - xmin) / tile_deg)
rows = math.ceil((ymax - ymin) / tile_deg)
print(f"Grid: {cols} cols × {rows} rows = {cols * rows} tiles\n")

# Download
failed = []

for row in tqdm(range(rows), desc="Rows"):
    for col in range(cols):

        # True tile bounds (no overlap) in WGS84
        tx_min = xmin + col * tile_deg
        ty_min = ymin + row * tile_deg
        tx_max = min(tx_min + tile_deg, xmax)
        ty_max = min(ty_min + tile_deg, ymax)

        # Expanded bounds with overlap (clamped to BBOX)
        ex_min = max(tx_min - overlap_deg, xmin)
        ey_min = max(ty_min - overlap_deg, ymin)
        ex_max = min(tx_max + overlap_deg, xmax)
        ey_max = min(ty_max + overlap_deg, ymax)

        px_w = max(1, round((ex_max - ex_min) / GSD_DEG))
        px_h = max(1, round((ey_max - ey_min) / GSD_DEG))

        tile_path = os.path.join(TILE_DIR, f"tile_{row:04d}_{col:04d}.tif")
        if os.path.exists(tile_path):
            continue

        try:
            resp = wms.getmap(
                layers=[LAYER],
                srs=CRS_CODE,
                bbox=(ex_min, ey_min, ex_max, ey_max),
                size=(px_w, px_h),
                format=IMAGE_FMT,
                transparent=False,
            )
            arr = np.array(Image.open(io.BytesIO(resp.read())).convert("RGB"))

            # Convert bounds to metric for saving
            mx_min, my_min = transformer.transform(ex_min, ey_min)
            mx_max, my_max = transformer.transform(ex_max, ey_max)
            # True tile bounds in metric
            tmx_min, tmy_min = transformer.transform(tx_min, ty_min)
            tmx_max, tmy_max = transformer.transform(tx_max, ty_max)

            transform = from_bounds(mx_min, my_min, mx_max, my_max, px_w, px_h)

            with rasterio.open(
                tile_path, "w",
                driver="GTiff",
                height=px_h,
                width=px_w,
                count=3,
                dtype=arr.dtype,
                crs=CRS.from_string(SAVE_CRS),
                transform=transform,
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                dst.write(arr[:, :, 0], 1)  # NIR
                dst.write(arr[:, :, 1], 2)  # Red
                dst.write(arr[:, :, 2], 3)  # Green
                dst.update_tags(
                    true_xmin=tmx_min, true_ymin=tmy_min,
                    true_xmax=tmx_max, true_ymax=tmy_max,
                    overlap_px=OVERLAP_PX,
                    band_1="NIR", band_2="Red", band_3="Green",
                )

        except Exception as e:
            print(f"\n  FAILED tile ({row},{col}): {e}")
            failed.append((row, col))

total   = rows * cols
success = total - len(failed)
print(f"\nDone: {success}/{total} tiles => {TILE_DIR}/")

if failed:
    print(f"Failed: {failed} = re-run to retry")

print(f"\nBand order: 1=NIR  2=Red  3=Green")
print(f"Saved CRS: {SAVE_CRS}")
