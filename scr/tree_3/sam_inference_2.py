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
from shapely.geometry import shape, box as shapely_box
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
OUTPUT_GPKG   = "/workspace/3-30-300-index/data/tree_detections_sam2_v5.gpkg"

SAM2_CHIP_SIZE   = 512
SAM2_MIN_QUALITY = 0.78          # raised from 0.55
PAD_FRAC         = 0.15          # adaptive padding
PAD_MIN_PX       = 4
PAD_MAX_PX       = 25
MAX_MASK_BOX_RATIO = 2.5         # reject if mask bbox area > ratio * input box area
MIN_IOU_WITH_BOX   = 0.15        # reject if mask barely overlaps input box
CRS_WGS          = "EPSG:4326"

# crown area filter in m² (CRS is UTM EPSG:32632)
MIN_CROWN_M2 = 1.0
MAX_CROWN_M2 = 600

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


def mask_bbox_local(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def select_best_mask(masks, scores, input_box):
    """masks: (N,H,W), scores: (N,), input_box: (x0,y0,x1,y1) local coords.
    Returns (mask, score) or (None, None) if all rejected."""
    box_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])
    best = None
    for m, s in zip(masks, scores):
        if s < SAM2_MIN_QUALITY:
            continue
        mbbox = mask_bbox_local(m)
        if mbbox is None:
            continue
        mbox_area = (mbbox[2] - mbbox[0]) * (mbbox[3] - mbbox[1])
        if box_area > 0 and mbox_area > MAX_MASK_BOX_RATIO * box_area:
            continue
        ov = iou_xyxy(mbbox, input_box)
        if ov < MIN_IOU_WITH_BOX:
            continue
        # score, prefer higher SAM score and higher box overlap
        rank = s + 0.2 * ov
        if best is None or rank > best[2]:
            best = (m, s, rank)
    if best is None:
        return None, None
    return best[0], best[1]


results = []
skipped_score   = 0
skipped_area    = 0
skipped_geom    = 0
CHIP = SAM2_CHIP_SIZE
half = CHIP // 2

with rasterio.open(TIF_PATH) as src:
    transform    = src.transform
    crs          = src.crs
    img_w, img_h = src.width, src.height

    boxes["cx"]   = (boxes["xmin"] + boxes["xmax"]) / 2
    boxes["cy"]   = (boxes["ymin"] + boxes["ymax"]) / 2

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
                  f"  kept={len(results):,}  skip_score={skipped_score:,}"
                  f"  skip_area={skipped_area:,}  skip_geom={skipped_geom:,}",
                  flush=True)

        rgb, col_off, row_off = read_chip(src, gcx, gcy, CHIP)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)

            for _, box_row in chip_rows.iterrows():
                bw = box_row["xmax"] - box_row["xmin"]
                bh = box_row["ymax"] - box_row["ymin"]
                pad_x = float(np.clip(PAD_FRAC * bw, PAD_MIN_PX, PAD_MAX_PX))
                pad_y = float(np.clip(PAD_FRAC * bh, PAD_MIN_PX, PAD_MAX_PX))

                lx0 = np.clip(box_row["xmin"] - col_off - pad_x, 0, CHIP - 1)
                ly0 = np.clip(box_row["ymin"] - row_off - pad_y, 0, CHIP - 1)
                lx1 = np.clip(box_row["xmax"] - col_off + pad_x, 0, CHIP - 1)
                ly1 = np.clip(box_row["ymax"] - row_off + pad_y, 0, CHIP - 1)

                input_box_local = (lx0, ly0, lx1, ly1)
                box_arr = np.array([[lx0, ly0, lx1, ly1]], dtype=np.float32)

                masks, sam_scores, _ = predictor.predict(box=box_arr, multimask_output=True)
                if masks.ndim == 4:
                    masks      = masks[0]      # (3,H,W)
                    sam_scores = sam_scores[0] # (3,)

                mask, score = select_best_mask(masks, sam_scores, input_box_local)
                if mask is None:
                    skipped_score += 1
                    continue

                poly = mask_to_polygon(mask, col_off, row_off, transform)
                if poly is None or poly.is_empty:
                    skipped_geom += 1
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
print(f"  skipped low score/rejected: {skipped_score:,}  skipped bad area: {skipped_area:,}  skipped empty geom: {skipped_geom:,}")

results_gdf = gpd.GeoDataFrame(results, crs=crs)
results_gdf.to_crs(CRS_WGS).to_file(OUTPUT_GPKG, driver="GPKG", layer="tree_crowns")
print(f"Saved {len(results_gdf):,} polygons → {OUTPUT_GPKG}")





# ── Post-processing after SAM2 ───────────────────────────────────────────
# Input:  `results` list produced by the SAM2 loop (cell 5)
#         Each entry: {'df_score': float, 'sam_score': float, 'geometry': Polygon}
# Output: gdf_clean  — filtered, deduplicated crown polygons ready for export
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
from pathlib import Path

# ── 0. Build raw GeoDataFrame from SAM2 results ───────────────────────────
with rasterio.open(TIF_PATH) as src:
    crs = src.crs

gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=crs)
print(f"Raw SAM2 polygons:   {len(gdf):,}")

# ── 1. Fill holes & keep only the largest part of multi-polygons ──────────
# SAM2 sometimes produces crowns with internal holes (shadowed centre,
# occluded branch) or small disconnected fragments.  Filling holes prevents
# underestimates in canopy area; keeping only the largest ring removes noise.
def fill_holes(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)          # drop all interior rings
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda p: p.area)
        return Polygon(largest.exterior)
    return geom                                # LineString artefact — leave as-is

gdf["geometry"] = gdf.geometry.apply(fill_holes)
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
print(f"After hole-fill:     {len(gdf):,}")

# ── 2. Simplify polygon boundaries ───────────────────────────────────────
# rasterio.features.shapes produces staircase outlines at pixel resolution.
# tolerance=0.3 m smooths without visibly distorting crown shapes.
# A small buffer–unbuffer also rounds sharp corners (optional visual polish).
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.3, preserve_topology=True)
# gdf["geometry"] = gdf.geometry.buffer(0.2).buffer(-0.2)   # uncomment for smoother crowns

# ── 3. Area, compactness & aspect-ratio filters ───────────────────────────
# Now using real polygon area (not bbox estimate).
gdf["area_m2"]     = gdf.geometry.area
gdf["perimeter_m"] = gdf.geometry.length
# Polsby-Popper compactness: 1.0 = perfect circle, <0.05 = wire-thin artefact
gdf["compactness"] = (4 * np.pi * gdf["area_m2"]) / (gdf["perimeter_m"] ** 2).replace(0, np.nan)

# Minimum bounding rectangle for aspect ratio
def bbox_aspect(geom):
    if geom is None or geom.is_empty:
        return np.nan
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    if min(w, h) == 0:
        return np.nan
    return max(w, h) / min(w, h)

gdf["aspect_ratio"] = gdf.geometry.apply(bbox_aspect)

mask_area       = gdf["area_m2"].between(MIN_AREA_M2, MAX_AREA_M2)
mask_compact    = gdf["compactness"] > 0.05     # removes road/edge artefacts
mask_aspect     = gdf["aspect_ratio"] <= MAX_ASPECT

gdf_filtered    = gdf[mask_area & mask_compact & mask_aspect].copy()
gdf_rejected    = gdf[~(mask_area & mask_compact & mask_aspect)].copy()
print(f"After shape filters: {len(gdf_filtered):,}  (rejected {len(gdf_rejected):,})")

# ── 4. Spatial-index IoU deduplication ───────────────────────────────────
# Two SAM2 prompts from overlapping DeepForest boxes can produce near-
# identical polygons.  Greedy dedup: sort by sam_score DESC, then for each
# polygon discard any later polygon whose IoU with it exceeds iou_thresh.
# Uses an R-tree spatial index to skip non-overlapping pairs (O(n log n)).
IOU_DEDUP_THRESH = 0.50

def dedup_by_iou(gdf_in, iou_thresh):
    gdf_s  = gdf_in.sort_values("sam_score", ascending=False).reset_index(drop=True)
    sindex = gdf_s.sindex
    keep   = np.ones(len(gdf_s), dtype=bool)

    for i, row in gdf_s.iterrows():
        if not keep[i]:
            continue
        # Only check candidates whose bounding box intersects
        candidates = list(sindex.intersection(row.geometry.bounds))
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            geom_j = gdf_s.at[j, "geometry"]
            inter  = row.geometry.intersection(geom_j).area
            if inter == 0:
                continue
            union  = row.geometry.union(geom_j).area
            if union > 0 and inter / union > iou_thresh:
                keep[j] = False   # lower-score duplicate → drop

    result = gdf_s[keep].reset_index(drop=True)
    return result

gdf_dedup = dedup_by_iou(gdf_filtered, IOU_DEDUP_THRESH)
print(f"After IoU dedup:     {len(gdf_dedup):,}  (removed {len(gdf_filtered) - len(gdf_dedup):,} duplicates)")

# ── 5. Crown-type labelling ───────────────────────────────────────────────
# Classify detections by size, matching the cat_id=1/2 split from training.
# Thresholds can be tuned from the area histogram in cell 7.
gdf_dedup["crown_type"] = np.where(
    gdf_dedup["area_m2"] < 25, "individual", "canopy_patch"
)
print(f"  individual (<25 m²): {(gdf_dedup['crown_type'] == 'individual').sum():,}")
print(f"  canopy patch:        {(gdf_dedup['crown_type'] == 'canopy_patch').sum():,}")

# ── 6. Final clean GeoDataFrame ───────────────────────────────────────────
gdf_clean = gdf_dedup[[
    "df_score", "sam_score", "area_m2", "compactness", "aspect_ratio",
    "crown_type", "geometry"
]].copy()

# ── 7. Export to GeoPackage (two layers) ─────────────────────────────────
GPKG_OUT = "/workspace/3-30-300-index/data/bologna_trees_sam2.gpkg"
gdf_clean.to_file(GPKG_OUT, driver="GPKG", layer="tree_crowns_clean")
gdf_rejected.to_file(GPKG_OUT, driver="GPKG", layer="tree_crowns_rejected")

print(f"\nSaved → {GPKG_OUT}")
print(f"  layer=tree_crowns_clean    {len(gdf_clean):,} features")
print(f"  layer=tree_crowns_rejected {len(gdf_rejected):,} features")
print(f"\nArea stats (clean):")
print(gdf_clean["area_m2"].describe().round(2).to_string())
