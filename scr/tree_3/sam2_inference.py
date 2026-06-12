import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.windows import Window
from rasterio.features import shapes as rio_shapes
from pathlib import Path
from shapely.geometry import shape
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("Imports OK")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT     = "/workspace/3-30-300-index/data/sam2.1_hiera_large.pt"
TIF_PATH      = "/workspace/3-30-300-index/data/bologna_rgb_2023.tif"
RAW_PREDS_CSV = "/workspace/3-30-300-index/data/raw_predictions_v2.csv"
OUTPUT_GPKG   = "/workspace/3-30-300-index/data/tree_detections_sam2_v4.gpkg"

SAM2_CHIP_SIZE   = 512
BOX_PAD_PX       = 20
SAM2_MIN_QUALITY = 0.55
CRS_WGS          = "EPSG:4326"

# crown area filter in m² (CRS is UTM EPSG:32632)
MIN_CROWN_M2 = 1.0
MAX_CROWN_M2 = 450

sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE)
predictor  = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded")

boxes = pd.read_csv(RAW_PREDS_CSV)
print(f"Processing {len(boxes):,} DeepForest proposals with SAM2...")


def read_chip(src, cx, cy, chip_size):
    half    = chip_size // 2
    col_off = max(0, min(int(cx) - half, src.width  - chip_size))
    row_off = max(0, min(int(cy) - half, src.height - chip_size))
    win  = Window(col_off, row_off, chip_size, chip_size)
    data = src.read([1, 2, 3], window=win)
    return np.moveaxis(data, 0, -1), col_off, row_off


def mask_to_polygon(mask, col_off, row_off, transform):
    mask_u8 = mask.astype(np.uint8)
    chip_transform = transform * rasterio.transform.Affine.translation(col_off, row_off)
    polys = [
        shape(geom) for geom, val
        in rio_shapes(mask_u8, mask=(mask_u8 == 1), transform=chip_transform)
        if val == 1
    ]
    return max(polys, key=lambda p: p.area) if polys else None


results = []
skipped_score = 0
skipped_area  = 0
CHIP = SAM2_CHIP_SIZE
half = CHIP // 2

with rasterio.open(TIF_PATH) as src:
    transform    = src.transform
    crs          = src.crs
    img_w, img_h = src.width, src.height

    boxes["cx"]   = (boxes["xmin"] + boxes["xmax"]) / 2
    boxes["cy"]   = (boxes["ymin"] + boxes["ymax"]) / 2
    boxes["area"] = (boxes["xmax"] - boxes["xmin"]) * (boxes["ymax"] - boxes["ymin"])

    GRID = CHIP // 2
    boxes["grid_cx"] = (boxes["cx"] // GRID * GRID + GRID // 2).clip(half, img_w - half).astype(int)
    boxes["grid_cy"] = (boxes["cy"] // GRID * GRID + GRID // 2).clip(half, img_h - half).astype(int)

    chip_groups = boxes.groupby(["grid_cx", "grid_cy"])
    n_chips     = len(chip_groups)
    print(f"{len(boxes):,} boxes → {n_chips:,} chips")

    t_start = time.time()
    for chip_i, ((gcx, gcy), chip_rows) in enumerate(chip_groups):
        if chip_i % 200 == 0:
            elapsed = (time.time() - t_start) / 60
            eta     = (elapsed / max(chip_i, 1)) * (n_chips - chip_i)
            print(f"  Chip {chip_i:,}/{n_chips:,}  elapsed={elapsed:.1f}min  ETA={eta:.1f}min"
                  f"  kept={len(results):,}  skip_score={skipped_score:,}  skip_area={skipped_area:,}",
                  flush=True)

        rgb, col_off, row_off = read_chip(src, gcx, gcy, CHIP)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)

            for _, box_row in chip_rows.iterrows():
                lx0 = np.clip(box_row["xmin"] - col_off - BOX_PAD_PX, 0, CHIP - 1)
                ly0 = np.clip(box_row["ymin"] - row_off - BOX_PAD_PX, 0, CHIP - 1)
                lx1 = np.clip(box_row["xmax"] - col_off + BOX_PAD_PX, 0, CHIP - 1)
                ly1 = np.clip(box_row["ymax"] - row_off + BOX_PAD_PX, 0, CHIP - 1)

                box_arr = np.array([[lx0, ly0, lx1, ly1]], dtype=np.float32)
                masks, sam_scores, _ = predictor.predict(box=box_arr, multimask_output=False)
                if masks.ndim == 4:     masks      = masks[:, 0]
                if sam_scores.ndim == 2: sam_scores = sam_scores[:, 0]
                mask, score = masks[0], sam_scores[0]

                if score < SAM2_MIN_QUALITY:
                    skipped_score += 1
                    continue

                poly = mask_to_polygon(mask, col_off, row_off, transform)
                if poly is None or poly.is_empty:
                    continue

                area_m2 = poly.area
                if area_m2 < MIN_CROWN_M2 or area_m2 > MAX_CROWN_M2:
                    skipped_area += 1
                    continue

                results.append({
                    "df_score":  float(box_row["score"]),
                    "sam_score": float(score),
                    "area_m2":   round(area_m2, 2),
                    "geometry":  poly,
                })

print(f"\nDone. {len(results):,} crown polygons from {len(boxes):,} proposals")
print(f"  skipped low score: {skipped_score:,}  skipped bad area: {skipped_area:,}")

results_gdf = gpd.GeoDataFrame(results, crs=crs)
results_gdf.to_crs(CRS_WGS).to_file(OUTPUT_GPKG, driver="GPKG", layer="tree_crowns")
print(f"Saved {len(results_gdf):,} polygons → {OUTPUT_GPKG}")