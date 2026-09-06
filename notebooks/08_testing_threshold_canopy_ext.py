#randomly sample tiles for ANY city se OSM building footprints as ground truth for "definitely
#rooftop, not canopy," compare against likely-canopy pixels, and
# auto-suggest starting threshold values from the CANOPY EXTRACTION.

"""
Usage:
    python 3-30-300-index/notebooks/08_testing_threshold_canopy_ext.py --city milano --n-tiles 15
    python 3-30-300-index/notebooks/08_testing_threshold_canopy_ext.py --city all --n-tiles 10
    python 3-30-300-index/notebooks/08_testing_threshold_canopy_ext.py --city genova --n-tiles 15 --out-csv suggestions.csv

    with the thresholds you want to use in cofig.yaml:
        city:
            canopy_extraction:
                exg_thresh: xy => this is the main threshold for excess green (ExG) to detect vegetation
                hue_range: [lo, hi] => this is range to filter out non-green hues (in HSV color space)
                sat_min: xy => this is the minimum saturation to filter out low-saturation pixels (in HSV color space)
                brightness_gate: xy => this is the minimum brightness to filter out dark pixels (in HSV color space)
                closing_enabled: True/False => this enables closing operation to bridge small gaps in detected canopy regions
    
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from skimage.color import rgb2hsv
from shapely.geometry import box
from shapely.ops import unary_union
import geopandas as gpd
import osmnx as ox
from osmnx._errors import InsufficientResponseError

DATA_ROOT = Path("/workspace/3-30-300-index/data")
CRS_WGS = "EPSG:4326"

ALL_CITIES = ["venezia",
              "verona", "perugia", "trento",
              "bologna_rgb", "bolzano_rgb", "torino_rgb", "firenze_rgb"]

# Same TILE_DIR_PATTERNS convention as canopy_extract_rgb.py
TILE_DIR_PATTERNS = ["{city}/{city}_ortho_tiles"]

LOOSE_EXG_FLOOR = 0.10   # candidate-selection floor only, not a real threshold
OSM_FETCH_MAX_RETRIES = 3
OSM_FETCH_SLEEP_BETWEEN_TILES = 1.0   # be gentler on Overpass across many tiles/cities


def find_tile_dir(city: str) -> Path:
    base_city = city.replace("_rgb", "")
    for pat in TILE_DIR_PATTERNS:
        for candidate in [
            DATA_ROOT / pat.format(city=city),
            DATA_ROOT / base_city / pat.format(city=city),
            DATA_ROOT / base_city / pat.format(city=base_city),
        ]:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No tile directory found for city='{city}'")


def get_building_mask(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)

    tile_box = box(*bounds)
    envelope_wgs = gpd.GeoSeries([tile_box], crs=crs).to_crs(CRS_WGS).iloc[0].buffer(0.0005)

    for attempt in range(OSM_FETCH_MAX_RETRIES):
        try:
            buildings = ox.features_from_polygon(envelope_wgs, tags={"building": True})
            buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if buildings.empty:
                return None
            buildings = buildings.to_crs(crs)
            building_geom = unary_union(buildings.geometry.values)
            mask = ~geometry_mask([building_geom], out_shape=shape, transform=transform, invert=False)
            return mask
        except InsufficientResponseError:
            # Query succeeded, genuinely zero buildings in this tile's
            # small envelope — expected and common, not a failure. Return
            # immediately, no retry, no warning, no wasted backoff time.
            return None
        except Exception as e:
            if attempt < OSM_FETCH_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [warn] OSM building fetch failed for {tif_path.name} "
                      f"after {OSM_FETCH_MAX_RETRIES} attempts: {str(e)[:80]}")
                return None


def compute_spectral(r, g, b):
    total = r + g + b
    with np.errstate(invalid="ignore", divide="ignore"):
        r_n = np.where(total > 0, r / total, 0.0)
        g_n = np.where(total > 0, g / total, 0.0)
        b_n = np.where(total > 0, b / total, 0.0)
    exg = 2 * g_n - r_n - b_n
    rgb = np.stack([r / 255.0, g / 255.0, b / 255.0], axis=-1)
    hsv = rgb2hsv(rgb)
    return total, exg, hsv[..., 0], hsv[..., 1]


def summarize(arr):
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def sample_one_city(city: str, n_tiles: int, seed: int, skip_osm: bool = False):
    try:
        tile_dir = find_tile_dir(city)
    except FileNotFoundError as e:
        print(f"[{city}] SKIP: {e}")
        return None

    all_tiles = sorted(tile_dir.glob("tile_*.tif"))
    if not all_tiles:
        print(f"[{city}] SKIP: no tiles found in {tile_dir}")
        return None

    rng = random.Random(seed)
    sampled_tiles = rng.sample(all_tiles, min(n_tiles, len(all_tiles)))
    print(f"\n[{city}] Sampling {len(sampled_tiles)} random tiles from {len(all_tiles)} total")

    rooftop_exg, rooftop_hue, rooftop_sat = [], [], []
    canopy_exg, canopy_hue, canopy_sat = [], [], []
    n_building_fetch_ok = 0

    for i, tif_path in enumerate(sampled_tiles):
        with rasterio.open(tif_path) as src:
            data = src.read([1, 2, 3], out_dtype=np.float32)
        r, g, b = data[0], data[1], data[2]
        total, exg, hue, sat = compute_spectral(r, g, b)

        building_mask = None if skip_osm else get_building_mask(tif_path)
        if building_mask is not None and building_mask.any():
            n_building_fetch_ok += 1
            rooftop_exg.append(exg[building_mask])
            rooftop_hue.append(hue[building_mask])
            rooftop_sat.append(sat[building_mask])

        candidate_mask = exg > LOOSE_EXG_FLOOR
        if building_mask is not None:
            candidate_mask &= ~building_mask
        if candidate_mask.any():
            canopy_exg.append(exg[candidate_mask])
            canopy_hue.append(hue[candidate_mask])
            canopy_sat.append(sat[candidate_mask])

        if not skip_osm:
            time.sleep(OSM_FETCH_SLEEP_BETWEEN_TILES)

    print(f"  [{city}] OSM building fetch succeeded on {n_building_fetch_ok}/{len(sampled_tiles)} tiles")

    result = {"city": city, "n_tiles_sampled": len(sampled_tiles),
              "n_building_fetch_ok": n_building_fetch_ok}

    if canopy_exg:
        c_exg = np.concatenate(canopy_exg)
        c_hue = np.concatenate(canopy_hue)
        c_sat = np.concatenate(canopy_sat)
        exg_s, hue_s, sat_s = summarize(c_exg), summarize(c_hue), summarize(c_sat)
        result.update({
            "canopy_n_px": len(c_exg),
            "canopy_exg": exg_s, "canopy_hue": hue_s, "canopy_sat": sat_s,
            # Auto-suggested starting values — sanity-check before trusting,
            # these are a starting point from measured data, not gospel.
            "suggested_exg_thresh": round(exg_s["p25"], 3),
            "suggested_hue_range": (round(hue_s["p10"], 3), round(hue_s["p90"], 3)),
            "suggested_sat_min": round(sat_s["p10"], 3),
        })
        print(f"  [{city}] canopy ExG  p10={exg_s['p10']:.3f} p50={exg_s['p50']:.3f} p90={exg_s['p90']:.3f}")
        print(f"  [{city}] canopy hue  p10={hue_s['p10']:.3f} p50={hue_s['p50']:.3f} p90={hue_s['p90']:.3f}")
        print(f"  [{city}] canopy sat  p10={sat_s['p10']:.3f} p50={sat_s['p50']:.3f} p90={sat_s['p90']:.3f}")
        print(f"  [{city}] SUGGESTED: exg_thresh={result['suggested_exg_thresh']}, "
              f"hue_range={result['suggested_hue_range']}, sat_min={result['suggested_sat_min']}")
    else:
        print(f"  [{city}] No canopy candidates found — check tiles have real vegetation")

    if rooftop_exg:
        r_exg = np.concatenate(rooftop_exg)
        r_s = summarize(r_exg)
        result["rooftop_n_px"] = len(r_exg)
        result["rooftop_exg"] = r_s
        print(f"  [{city}] rooftop ExG p10={r_s['p10']:.3f} p50={r_s['p50']:.3f} p90={r_s['p90']:.3f}")
        if r_s["p90"] > result.get("suggested_exg_thresh", 999):
            print(f"  [{city}] [WARN] rooftop p90 ExG ({r_s['p90']:.3f}) exceeds suggested "
                  f"exg_thresh ({result.get('suggested_exg_thresh')}) — spectral overlap exists, "
                  f"exclusion (buildings) matters more than threshold tuning for this city")
    else:
        reason = "--skip-osm was set" if skip_osm else "OSM fetch failed or no buildings in sampled tiles"
        print(f"  [{city}] No rooftop ground truth ({reason})")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help="City name, or 'all'")
    ap.add_argument("--n-tiles", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default=None, help="Optional path to write a summary CSV")
    ap.add_argument("--skip-osm", action="store_true",
                     help="Skip OSM building fetch entirely for fast iteration. "
                          "Canopy-candidate spectral stats are unaffected and "
                          "still computed normally; you just lose the rooftop "
                          "ground-truth comparison for this run.")
    args = ap.parse_args()

    cities = ALL_CITIES if args.city == "all" else [args.city]

    results = []
    for city in cities:
        r = sample_one_city(city, args.n_tiles, args.seed, skip_osm=args.skip_osm)
        if r:
            results.append(r)

    print(f"\n{'='*70}")
    print("SUMMARY — suggested starting thresholds per city")
    print(f"{'='*70}")
    rows = []
    for r in results:
        if "suggested_exg_thresh" not in r:
            continue
        print(f"  {r['city']:<15} exg_thresh={r['suggested_exg_thresh']:<7} "
              f"hue_range={r['suggested_hue_range']}  sat_min={r['suggested_sat_min']}  "
              f"(building fetch ok: {r['n_building_fetch_ok']}/{r['n_tiles_sampled']})")
        rows.append({
            "city": r["city"],
            "suggested_exg_thresh": r["suggested_exg_thresh"],
            "suggested_hue_min": r["suggested_hue_range"][0],
            "suggested_hue_max": r["suggested_hue_range"][1],
            "suggested_sat_min": r["suggested_sat_min"],
            "n_building_fetch_ok": r["n_building_fetch_ok"],
            "n_tiles_sampled": r["n_tiles_sampled"],
        })

    if args.out_csv and rows:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\nWrote summary CSV: {args.out_csv}")

    print(f"\nThese are STARTING points derived from measured percentiles, not final "
          f"values — sanity-check qualitatively in QGIS before committing to a full run, "
          f"especially for any city where 'building fetch ok' is low (means the rooftop "
          f"comparison wasn't reliable for that city — the canopy numbers are still valid "
          f"since they don't depend on OSM at all).")


if __name__ == "__main__":
    main()