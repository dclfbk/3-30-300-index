"""
batch_zeroshot_inference.py — batch zero-shot DeepForest inference across
cities, with resolution-aware patch sizing and an integrated boundary
filter.
"""
import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from torchvision.ops import nms
from shapely.ops import unary_union
from deepforest import main as df_main

ROOT_PATH = Path("/workspace/3-30-300-index/")
sys.path.insert(0, str(ROOT_PATH))
os.chdir(ROOT_PATH)
from config.config import (
    connect_db_with_retry, gdf_to_duckdb, load_table, strip_source_suffix,
    CRS_METRIC, ROOT_PATH, DATA_DIR, CITIES,
    DB_LOCK_MAX_RETRIES, DB_LOCK_BASE_WAIT_S, DB_LOCK_MAX_WAIT_S,
)



TARGET_PATCH_METERS = 80.0



EFFECTIVE_GSD = {
    "genova": 0.1430,
    "torino_rgb": 0.30,   # updated: now AGEA 2021 leaf-on imagery, was 0.1447 (leaf-off coto_ortofoto_2023)
    "trento": 0.1338,
}

DEFAULT_GSD = 0.20
PATCH_OVERLAP       = 0.2
IOU_THRESHOLD        = 0.45
INFER_SCORE_THRESH  = 0.22
DEDUPE_IOU_INTRA = 0.3
DEDUPE_IOU_CROSS = 0.3
TEST_TILE_LIMIT = 5


def patch_size_for_city(city):
    gsd = EFFECTIVE_GSD.get(city, DEFAULT_GSD)
    size = max(64, round(TARGET_PATCH_METERS / gsd))
    return size, gsd


def dedupe_boxes(preds, iou_thresh=0.3, cols=("xmin", "ymin", "xmax", "ymax")):
    if preds.empty:
        return preds
    boxes = torch.tensor(preds[list(cols)].values, dtype=torch.float32)
    scores = torch.tensor(preds["score"].values, dtype=torch.float32)
    keep = nms(boxes, scores, iou_thresh)
    return preds.iloc[keep.numpy()].reset_index(drop=True)


def pixel_boxes_to_geo(df, transform):
    xmin = df["xmin"].values
    ymin = df["ymin"].values
    xmax = df["xmax"].values
    ymax = df["ymax"].values

    gx0, gy0 = transform * (xmin, ymax)
    gx1, gy1 = transform * (xmax, ymin)

    df = df.copy()
    df["geo_xmin"] = np.minimum(gx0, gx1)
    df["geo_xmax"] = np.maximum(gx0, gx1)
    df["geo_ymin"] = np.minimum(gy0, gy1)
    df["geo_ymax"] = np.maximum(gy0, gy1)
    return df


def filter_to_boundary(city_df, city, con):
    """Drop detections whose centroid falls outside the city's
    administrative boundary polygon (raw.{city}_boundary in DuckDB)."""
    lookup_city = strip_source_suffix(city)
    try:
        boundary_gdf = load_table(con, "raw", f"{lookup_city}_boundary").to_crs(CRS_METRIC)
        boundary_geom = unary_union(boundary_gdf.geometry.values)
    except Exception as e:
        print(f"[WARN] Could not load boundary for '{lookup_city}': {e}. Skipping boundary filter.")
        return city_df

    cx = (city_df["geo_xmin"] + city_df["geo_xmax"]) / 2
    cy = (city_df["geo_ymin"] + city_df["geo_ymax"]) / 2
    points = gpd.GeoDataFrame(
        city_df, geometry=gpd.points_from_xy(cx, cy), crs=CRS_METRIC
    )

    inside_mask = points.geometry.within(boundary_geom)
    n_before = len(points)
    filtered = points[inside_mask].drop(columns="geometry").reset_index(drop=True)
    n_after = len(filtered)

    print(f"[BOUNDARY] {city}: {n_before:,} -> {n_after:,} "
          f"({n_before - n_after:,} removed, {100 * (n_before - n_after) / max(n_before, 1):.1f}%)")
    return filtered


def load_zeroshot_model(device):
    print("\n[INIT] Loading zero-shot DeepForest model (base weecology/deepforest-tree)...")
    m = df_main.deepforest()
    m.load_model(model_name="weecology/deepforest-tree", revision="main")

    m.config["score_thresh"] = INFER_SCORE_THRESH
    if hasattr(m.model, "score_thresh"):
        m.model.score_thresh = INFER_SCORE_THRESH

    m.config["batch_size"] = 2
    m.config["workers"] = 0
    m.create_trainer()
    print("[INIT] Zero-shot model ready.\n")
    return m


def process_single_city(city, tile_dir, model, tile_limit=None):
    patch_size, gsd_used = patch_size_for_city(city)
    print(f"\n>>> Processing City: {city.upper()}")
    print(f"   Tile Directory: {tile_dir}")
    print(f"   [RESOLUTION] effective_gsd={gsd_used}m/px -> patch_size={patch_size}px "
          f"(~{patch_size * gsd_used:.0f}m ground coverage per patch)")

    if not tile_dir.exists():
        print(f"   [SKIP] Directory not found.")
        return []

    tif_files = sorted(tile_dir.glob("*.tif"))
    if tile_limit:
        random.seed(42)
        tif_files = random.sample(tif_files, min(tile_limit * 3, len(tif_files)))
        tif_files = sorted(tif_files)
        print(f"   [TEST_MODE] randomly sampling up to {tile_limit} valid tiles")

    print(f"   Found {len(tif_files)} tiles.")

    city_predictions = []
    processed = 0

    for i, tif_path in enumerate(tif_files):
        if tif_path.stat().st_size < 50_000:
            continue

        if tile_limit and processed >= tile_limit:
            print(f"   [TEST_MODE] reached tile limit, stopping early")
            break

        try:
            print(f"   [{i+1}/{len(tif_files)}] Running: {tif_path.name} ...", end=" ", flush=True)

            preds = model.predict_tile(
                path=str(tif_path),
                patch_size=patch_size,
                patch_overlap=PATCH_OVERLAP,
                iou_threshold=IOU_THRESHOLD,
                dataloader_strategy="window",
            )

            if not preds.empty:
                raw_count = len(preds)
                preds = preds[preds["score"] >= INFER_SCORE_THRESH]

                if not preds.empty:
                    preds = dedupe_boxes(preds, iou_thresh=DEDUPE_IOU_INTRA)

                    with rasterio.open(tif_path) as src:
                        transform = src.transform
                    preds = pixel_boxes_to_geo(preds, transform)

                    preds["source_tile"] = tif_path.stem
                    preds["city"] = city

                    city_predictions.append(preds)
                    print(f"OK ({raw_count} -> {len(preds)} after score filter + dedupe)")
                else:
                    print(f"OK ({raw_count} -> 0, all below score_thresh={INFER_SCORE_THRESH})")
            else:
                print("OK (0)")

            processed += 1

        except Exception as e:
            print(f"ERROR: {str(e)[:80]}...")
            continue

    return city_predictions


def main():
    parser = argparse.ArgumentParser(description="Batch zero-shot tree detection across cities")
    parser.add_argument("--city", type=str, default="all",
                         help="Specific city name (e.g., 'milano') or 'all' to run all configured cities.")
    parser.add_argument("--test", action="store_true",
                         help="Test mode: first N tiles per city only (default TEST_TILE_LIMIT=5 "
                              "unless --n-tiles overrides it), output filenames get a _TEST suffix.")
    parser.add_argument("--n-tiles", type=int, default=None,
                         help="Limit to this many randomly-sampled tiles per city for a quick test "
                              "run — implies test mode even without --test. Overrides TEST_TILE_LIMIT.")
    parser.add_argument("--skip-boundary-filter", action="store_true",
                         help="Disable the automatic boundary filter (on by default).")
    args = parser.parse_args()

    tile_limit = args.n_tiles if args.n_tiles is not None else (TEST_TILE_LIMIT if args.test else None)
    test_mode = tile_limit is not None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Set up: IOU_THRESHOLD={IOU_THRESHOLD}, INFER_SCORE_THRESH={INFER_SCORE_THRESH}, "
          f"DEDUPE_IOU_INTRA={DEDUPE_IOU_INTRA}, DEDUPE_IOU_CROSS={DEDUPE_IOU_CROSS}")

    print(f"Using Device: {device}")
    if test_mode:
        print(f"[TEST_MODE] enabled — tile_limit={tile_limit}")

    model = load_zeroshot_model(device)

    if args.city == "all":
        cities_to_run = list(CITIES.keys())
    else:
        target = args.city.lower()
        if target not in CITIES:
            print(f"Error: '{args.city}' not in configured cities. Options: {list(CITIES.keys())}")
            sys.exit(1)
        cities_to_run = [target]

    start_time = time.time()
    suffix = "_TEST" if test_mode else ""

    for city in cities_to_run:
        city_dir = DATA_DIR / city
        tile_dir = city_dir / f"{city}_ortho_tiles"

        if not tile_dir.exists():
            print(f"\n[WARN] No tiles found for {city} (expected at {tile_dir}). Skipping.")
            continue

        preds = process_single_city(city, tile_dir, model, tile_limit=tile_limit)
        if not preds:
            continue

        city_df = pd.concat(preds, ignore_index=True)
        pre_cross_count = len(city_df)
        city_df = dedupe_boxes(
            city_df,
            iou_thresh=DEDUPE_IOU_CROSS,
            cols=("geo_xmin", "geo_ymin", "geo_xmax", "geo_ymax"),
        )
        print(f"[DEDUPE] {city}: {pre_cross_count} -> {len(city_df)} after cross-tile dedupe")

        con = connect_db_with_retry()

        if not args.skip_boundary_filter:
            city_df = filter_to_boundary(city_df, city, con)

        city_dir.mkdir(parents=True, exist_ok=True)
        city_csv = city_dir / f"{city}_tree_detections_ZEROSHOT_RESSCALED{suffix}.csv"
        city_df.to_csv(city_csv, index=False)
        print(f"[SAVED] {city}: {len(city_df):,} detections -> {city_csv}")

        try:
            if hasattr(city_df, 'geometry'):
                table_name = f"inference.detectors_zeroshot_{city}_{time.strftime('%Y%m%d_%H%M%S')}"
                gdf_to_duckdb(con, city_df, table_name)
                print(f"[SUCCESS] Uploaded to DB: {table_name}")
            else:
                print("[INFO] Dataframe has no geometry column. Skipping DB upload.")
        except Exception as db_err:
            print(f"[WARN] DB upload failed for {city}: {db_err}")
        finally:
            con.close()

    print(f"\nTotal Runtime: {(time.time() - start_time)/60:.1f} minutes")


if __name__ == "__main__":
    main()