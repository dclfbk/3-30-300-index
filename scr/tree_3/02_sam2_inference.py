#before running this script:
#run 01_deep_forest.py to get the raw tree boxes 
#download sam2 model weights and place them in the sam2 folder (more in the readme)
#firstly run --test (adjust exg threshold if needed)

"""
SAM 2 MODEl - precise detection trees based on Deep Forest boxes
Usage:
    python 02_sam2.py --city verona
    python 02_sam2.py --city bolzano_rgb --test
"""

import argparse
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
import os
import sys

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT_PATH = "/workspace/3-30-300-index/"
sys.path.insert(0, str(ROOT_PATH))
os.chdir(ROOT_PATH)


from config.config import (
    connect_db_with_retry, gdf_to_duckdb, CRS_METRIC, CRS_WGS,
    DATA_DIR,
)



_parser = argparse.ArgumentParser()
_parser.add_argument("--city", required=True,
                      help="City key, e.g. 'verona' or 'bolzano_rgb'. "
                           "Determines ALL input/output paths for this run.")
_parser.add_argument("--test", action="store_true",
                      help="Test mode: first TEST_TILE_LIMIT tiles-with-boxes "
                           "only, output filenames get a _TEST suffix.")
_args = _parser.parse_args()


CITY = _args.city
TEST_MODE = _args.test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"City: {CITY}  |  test_mode: {TEST_MODE}")

# ── city / paths ─────────────────────────────────────────────────────────
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = str(Path(ROOT_PATH) / "sam2" / "sam2.1_hiera_large.pt")
TILES_DIR = DATA_DIR / CITY / f"{CITY}_ortho_tiles"

# IMPORTANT: point this at the zeroshot/rescaled boxes CSV, not the
# finetuned model's output.
RAW_PREDS_CSV = DATA_DIR / CITY / f"{CITY}_tree_detections_ZEROSHOT_RESSCALED.csv"
# ── test mode ───────────────────────────────────────────────────────────
TEST_TILE_LIMIT = 5
TEST_BOX_LIMIT_PER_TILE = 4000

_suffix = "_TEST" if TEST_MODE else ""
OUTPUT_BASE = f"tree_detections_zeroshot_{CITY}{_suffix}"
OUTPUT_DIR = DATA_DIR / CITY
CSV_PATH = OUTPUT_DIR / f"{OUTPUT_BASE}.csv"
GPKG_PATH = OUTPUT_DIR / f"{OUTPUT_BASE}.gpkg"
SAM2_CHIP_SIZE = 512
PAD_FRAC = 0.4      # was 0.15
PAD_MIN_PX = 3
PAD_MAX_PX = 60      # was 20

MAX_MASK_BOX_RATIO = 6.0   # geometric sanity only, not a quality gate:
                            # reject a candidate that's obviously wrong
                            # shape (e.g. SAM2 segments a huge chunk of
                            # surrounding forest instead of this one
                            # tree). Loose on purpose — this catches
                            # gross errors, not marginal cases; ExG
                            # below is now the real acceptance gate.
MIN_IOU_WITH_BOX   = 0.05   # geometric sanity only: reject a candidate
                            # that barely overlaps the input box at all
                            # (SAM2 snapped to the wrong nearby object).
MIN_CROWN_M2 = 1.0
GREENNESS_MIN = 0.10   # was 0.04 — bring in line with the loosest canopy-extraction
MAX_CROWN_M2 = 400      

sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded")

boxes_all = pd.read_csv(RAW_PREDS_CSV)
if "source_tile" not in boxes_all.columns:
    raise ValueError(
        "RAW_PREDS_CSV has no 'source_tile' column — boxes can't be matched "
        "back to individual tile files."
    )
print(f"Loaded {len(boxes_all):,} proposals across "
      f"{boxes_all['source_tile'].nunique():,} tiles")
if TEST_MODE:
    print(f"[TEST_MODE] limiting to first {TEST_TILE_LIMIT} tiles-with-boxes, "
          f"max {TEST_BOX_LIMIT_PER_TILE} boxes/tile")

def read_chip(src, cx, cy, chip_size):
    half = chip_size // 2
    col_off = max(0, min(int(cx) - half, src.width - chip_size))
    row_off = max(0, min(int(cy) - half, src.height - chip_size))
    win = Window(col_off, row_off, chip_size, chip_size)
    data = src.read([1, 2, 3], window=win)
    return np.moveaxis(data, 0, -1), col_off, row_off


def clip_mask_to_box(mask, box_local, pad_px, chip_size):
    mask = mask.astype(bool)  # defensive: caller should already cast, but
                               # a non-bool mask here silently breaks any
                               # later rgb_chip[mask]-style fancy indexing
    x0, y0, x1, y1 = box_local
    xi0 = max(0, int(x0 - pad_px)); yi0 = max(0, int(y0 - pad_px))
    xi1 = min(chip_size, int(x1 + pad_px)); yi1 = min(chip_size, int(y1 + pad_px))
    clipped = np.zeros_like(mask)
    clipped[yi0:yi1, xi0:xi1] = mask[yi0:yi1, xi0:xi1]
    return clipped


def box_mask(box_local, chip_size):
    """Boolean mask covering just the (unpadded) input box — used to
    compute greenness for fallback-box detections that have no SAM2 mask."""
    x0, y0, x1, y1 = box_local
    m = np.zeros((chip_size, chip_size), dtype=bool)
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(chip_size, int(np.ceil(x1))), min(chip_size, int(np.ceil(y1)))
    m[yi0:yi1, xi0:xi1] = True
    return m


def mean_exg(rgb_chip, mask):
    if not mask.any():
        return -1.0
    px = rgb_chip[mask].astype(np.float32) / 255.0
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    return float(np.mean(2 * g - r - b))


def mask_to_polygon(mask, col_off, row_off, transform):
    mask_u8 = mask.astype(np.uint8)
    chip_transform = transform * rasterio.transform.Affine.translation(col_off, row_off)
    polys = [
        shape(geom) for geom, val
        in rio_shapes(mask_u8, mask=(mask_u8 == 1), transform=chip_transform)
        if val == 1
    ]
    return max(polys, key=lambda p: p.area) if polys else None


def pixel_box_to_world(xmin, ymin, xmax, ymax, transform):
    """Convert a pixel-space DeepForest box straight to a georeferenced
    polygon — used for the dense-canopy fallback."""
    x0, y0 = transform * (xmin, ymin)
    x1, y1 = transform * (xmax, ymax)
    minx, maxx = sorted([x0, x1])
    miny, maxy = sorted([y0, y1])
    return shapely_box(minx, miny, maxx, maxy)


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


def select_best_mask(masks, scores, input_box, max_mask_box_ratio, min_iou_with_box):
    """Simplified: no quality-score rejection. Every candidate is
    considered; a candidate is only geometrically disqualified if it
    blows up far past the input box (wrong shape) or barely overlaps it
    at all (wrong location) — both loose, sanity-only checks, not
    quality gates. Among surviving candidates, picks the one with the
    highest SAM2 score. If ALL 3 candidates fail the geometric sanity
    checks (rare — usually means the box itself was on a genuine
    non-object), returns None so the caller falls back to the raw
    DeepForest box, still subject to the ExG greenness gate."""
    box_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])
    best = None
    for m, s in zip(masks, scores):
        mbbox = mask_bbox_local(m)
        if mbbox is None:
            continue
        mbox_area = (mbbox[2] - mbbox[0]) * (mbbox[3] - mbbox[1])
        if box_area > 0 and mbox_area > max_mask_box_ratio * box_area:
            continue
        ov = iou_xyxy(mbbox, input_box)
        if ov < min_iou_with_box:
            continue
        if best is None or s > best[1]:
            best = (m, s)
    if best is None:
        return None, None
    return best[0], best[1]


def process_tile(tif_path, tile_boxes, chip_size):
    results = []
    counters = {
        "skip_reject": 0,   # SAM2 rejected all candidate masks, no fallback available/used
        "skip_veg": 0,      # accepted shape (mask or fallback box) failed greenness check
        "skip_area": 0,
        "skip_geom": 0,
        "kept": 0,
        "n_fallback": 0,
        "n_dense_chip_boxes": 0,
    }
    half = chip_size // 2

    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs
        img_w, img_h = src.width, src.height

        b = tile_boxes.copy()
        b["cx"] = (b["xmin"] + b["xmax"]) / 2
        b["cy"] = (b["ymin"] + b["ymax"]) / 2
        grid = chip_size // 2
        b["grid_cx"] = (b["cx"] // grid * grid + grid // 2).clip(half, max(half, img_w - half)).astype(int)
        b["grid_cy"] = (b["cy"] // grid * grid + grid // 2).clip(half, max(half, img_h - half)).astype(int)

        for (gcx, gcy), chip_rows in b.groupby(["grid_cx", "grid_cy"]):
            rgb, col_off, row_off = read_chip(src, gcx, gcy, chip_size)

            # Chip grouping kept purely for computational efficiency
            # (amortising SAM2's image-encoding cost across nearby boxes)
            # — no longer used for threshold selection, since there's
            # only one threshold set now.
            chip_density = len(chip_rows)
            counters["n_dense_chip_boxes"] += chip_density if chip_density >= 6 else 0

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)

                for _, box_row in chip_rows.iterrows():
                    bw = box_row["xmax"] - box_row["xmin"]
                    bh = box_row["ymax"] - box_row["ymin"]
                    pad_x = float(np.clip(PAD_FRAC * bw, PAD_MIN_PX, PAD_MAX_PX))
                    pad_y = float(np.clip(PAD_FRAC * bh, PAD_MIN_PX, PAD_MAX_PX))

                    lx0 = np.clip(box_row["xmin"] - col_off - pad_x, 0, chip_size - 1)
                    ly0 = np.clip(box_row["ymin"] - row_off - pad_y, 0, chip_size - 1)
                    lx1 = np.clip(box_row["xmax"] - col_off + pad_x, 0, chip_size - 1)
                    ly1 = np.clip(box_row["ymax"] - row_off + pad_y, 0, chip_size - 1)
                    box_local = (lx0, ly0, lx1, ly1)
                    box_arr = np.array([[lx0, ly0, lx1, ly1]], dtype=np.float32)

                    masks, sam_scores, _ = predictor.predict(box=box_arr, multimask_output=True)
                    if masks.ndim == 4:
                        masks = masks[0]
                        sam_scores = sam_scores[0]

                    mask, sam_score = select_best_mask(
                        masks, sam_scores, box_local,
                        max_mask_box_ratio=MAX_MASK_BOX_RATIO,
                        min_iou_with_box=MIN_IOU_WITH_BOX,
                    )

                    source = "sam2"
                    if mask is not None:
                        mask = mask.astype(bool)
                        veg_mask = mask
                    else:
                        # SAM2 produced nothing geometrically sane for this
                        # box (rare). Fall back to the raw DeepForest box —
                        # applied universally now, not just in dense chips,
                        # since ExG below is the real acceptance gate either way.
                        veg_mask = box_mask(box_local, chip_size)
                        source = "box_fallback"
                        sam_score = np.nan
                        counters["n_fallback"] += 1

                    exg = mean_exg(rgb, veg_mask)
                    if exg < GREENNESS_MIN:
                        counters["skip_veg"] += 1
                        continue

                    if source == "sam2":
                        poly = mask_to_polygon(mask, col_off, row_off, transform)
                    else:
                        poly = pixel_box_to_world(
                            box_row["xmin"], box_row["ymin"],
                            box_row["xmax"], box_row["ymax"], transform,
                        )

                    if poly is None or poly.is_empty:
                        counters["skip_geom"] += 1
                        continue

                    area_m2 = poly.area
                    if area_m2 < MIN_CROWN_M2 or area_m2 > MAX_CROWN_M2:
                        counters["skip_area"] += 1
                        continue

                    counters["kept"] += 1
                    results.append({
                        "df_score": float(box_row["score"]),
                        "sam_score": sam_score,
                        "exg": round(exg, 3),
                        "area_m2": round(area_m2, 2),
                        "source": source,
                        "chip_density": chip_density,
                        "source_tile": tif_path.stem,
                        "geometry": poly,
                    })

    return results, crs, counters


# ── incremental writers ──────────────────────────────────────────────────
_csv_header_written = CSV_PATH.exists()
_gpkg_initialized = GPKG_PATH.exists()


def save_tile_results(tile_results, crs):
    """Incremental per-tile checkpoint: CSV + GPKG only. The shared
    master DuckDB is written ONCE at the very end of the whole run (see
    bottom of file) via config.gdf_to_duckdb, not per-tile — a single
    shared connection/file across the whole project is a poor fit for
    frequent small incremental writes during an hours-long run, and
    doing it once at the end means a DuckDB problem can never affect
    anything mid-run; GPKG is the durable, crash-safe checkpoint
    throughout, exactly as before."""
    global _csv_header_written, _gpkg_initialized

    if not tile_results:
        return

    gdf = gpd.GeoDataFrame(tile_results, crs=crs).to_crs(CRS_WGS)
    gdf["geom_wkt"] = gdf.geometry.apply(lambda g: g.wkt)

    df_out = gdf.drop(columns="geometry")
    df_out.to_csv(CSV_PATH, mode="a", header=not _csv_header_written, index=False)
    _csv_header_written = True

    try:
        gdf.drop(columns="geom_wkt").to_file(
            GPKG_PATH, driver="GPKG", layer="tree_crowns",
            mode="a" if _gpkg_initialized else "w",
        )
        _gpkg_initialized = True
    except Exception as e:
        print(f"   [WARN] GPKG append failed for this tile (data still safe in CSV+DuckDB): {e}")


if not TILES_DIR.exists():
    raise FileNotFoundError(f"Tiles directory not found: {TILES_DIR}")

tif_files = sorted(TILES_DIR.glob("*.tif"))
print(f"Found {len(tif_files)} tile files in {TILES_DIR}")

out_crs = None
totals = {
    "skip_reject": 0, "skip_veg": 0, "skip_area": 0, "skip_geom": 0,
    "kept": 0, "n_fallback": 0, "n_dense_chip_boxes": 0,
}
t_start = time.time()
tiles_processed = 0

for i, tif_path in enumerate(tif_files):
    tile_boxes = boxes_all[boxes_all["source_tile"] == tif_path.stem]
    if tile_boxes.empty:
        continue

    if TEST_MODE and tiles_processed >= TEST_TILE_LIMIT:
        print(f"[TEST_MODE] reached tile limit ({TEST_TILE_LIMIT}), stopping early")
        break
    if TEST_MODE and len(tile_boxes) > TEST_BOX_LIMIT_PER_TILE:
        tile_boxes = tile_boxes.iloc[:TEST_BOX_LIMIT_PER_TILE]

    elapsed = (time.time() - t_start) / 60
    print(f"[{i+1}/{len(tif_files)}] {tif_path.name}  ({len(tile_boxes):,} boxes)  "
          f"elapsed={elapsed:.1f}min  kept_so_far={totals['kept']:,}", flush=True)

    try:
        tile_results, crs, counters = process_tile(tif_path, tile_boxes, SAM2_CHIP_SIZE)
    except Exception as e:
        print(f"   [ERROR] {tif_path.name}: {e}")
        continue

    out_crs = out_crs or crs
    try:
        save_tile_results(tile_results, crs)
    except Exception as e:
        # save_tile_results already wraps its own GPKG write, so this is
        # a last-resort catch for anything unexpected (e.g. the
        # GeoDataFrame construction itself) — one bad tile should never
        # take down the rest of a multi-hour run.
        print(f"   [ERROR] save_tile_results failed for {tif_path.name}: {e}")
    del tile_results

    for k, v in counters.items():
        totals[k] += v
    tiles_processed += 1

print(f"\nDone. {totals['kept']:,} crown polygons kept across {tiles_processed} tiles processed"
      f"{' (TEST_MODE)' if TEST_MODE else ''}")
print(f"  kept={totals['kept']:,}  "
      f"skip_veg={totals['skip_veg']:,}  skip_area={totals['skip_area']:,}  skip_geom={totals['skip_geom']:,}")
print(f"  dense-chip boxes seen: {totals['n_dense_chip_boxes']:,}  "
      f"box_fallback used: {totals['n_fallback']:,} "
      f"({100 * totals['n_fallback'] / max(totals['kept'], 1):.1f}% of kept)")
print(f"\nSaved incrementally throughout the run:")
print(f"  CSV    → {CSV_PATH}")
print(f"  GPKG   → {GPKG_PATH}")

try:
    print(f"\nWriting to shared master DuckDB...")
    final_gdf = gpd.read_file(GPKG_PATH, layer="tree_crowns").to_crs(CRS_METRIC)
    table_name = f"tree_polygons_{CITY}{_suffix}"
    con = connect_db_with_retry(read_only=False)
    gdf_to_duckdb(con, final_gdf, f"preprocessing.{table_name}")
    con.close()
    print(f"  DuckDB → preprocessing.{table_name} ({len(final_gdf):,} rows)")
except Exception as e:
    print(f"  [WARN] Shared DuckDB write failed — data remains fully safe "
          f"in GPKG at {GPKG_PATH}: {e}")