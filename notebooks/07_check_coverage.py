# downloaded tile footprints vs. city boundary - check progress for gievn city, possibly brokem wms etc
# IMPORTANT: run this script each time after downloading tiles
# sometimes municipality bondary is not fully covered by downloaded tiles and provided orthophotos 
"""
Usage:
    python 3-30-300-index/notebooks/07_check_coverage.py --city bari
    python 3-30-300-index/notebooks/07_check_coverage.py --city firenze_rgb --boundary-city firenze
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box as shapely_box
from shapely.prepared import prep
from pathlib import Path
from config.config import ROOT_PATH, connect_db, load_table
from shapely.ops import unary_union


def load_boundary(city, ref_crs, boundary_path=None):
    """Load city boundary, auto-stripping _rgb/_nir/_cir suffixes."""
    import geopandas as gpd
    
    if boundary_path:
        gdf = gpd.read_file(boundary_path).to_crs(ref_crs)
    else:
        con = connect_db(read_only=True)
        
        # Strip suffixes to get base city name for DB lookup
        lookup_city = city.replace("_rgb", "").replace("_nir", "").replace("_cir", "")
        
        gdf = load_table(con, "raw", f"{lookup_city}_boundary").to_crs(ref_crs)
        con.close()
    
    return gdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, help="City key, e.g. 'bari' or 'firenze_rgb'")
    p.add_argument("--boundary-city", default=None,
                    help="Boundary table to use if different from --city (e.g. base city for *_rgb variants)")
    p.add_argument("--boundary-path", default=None,
                    help="Load boundary from a file (shp/geojson) instead of the DB")
    p.add_argument("--tiles-dir", default=None, help="Default: data/<city>/<city>_ortho_tiles")
    p.add_argument("--out", default=None, help="Default: data/<city>/<city>_coverage_check.png")
    args = p.parse_args()

    boundary_city = args.boundary_city or args.city
    tiles_dir = args.tiles_dir or os.path.join("data", args.city, f"{args.city}_ortho_tiles")
    out_path = args.out or os.path.join("data", args.city, f"{args.city}_coverage_check.png")

    tif_paths = sorted(glob.glob(os.path.join(tiles_dir, "tile_*.tif")))
    print(f"[{args.city}] {len(tif_paths)} tiles found in {tiles_dir}")
    if not tif_paths:
        sys.exit("No tiles found — check --tiles-dir, or run the downloader first.")

    with rasterio.open(tif_paths[0]) as src:
        ref_crs = src.crs
    print(f"[{args.city}] plotting in tile CRS: {ref_crs}")

    gdf = load_boundary(boundary_city, ref_crs, args.boundary_path)
    boundary_prepared = prep(gdf.geometry.union_all())

    boxes, skipped_crs, skipped_corrupt = [], 0, 0
    for tp in tif_paths:
        try:
            with rasterio.open(tp) as src:
                if src.crs != ref_crs:
                    skipped_crs += 1
                    continue
                boxes.append(tuple(src.bounds))  # (left, bottom, right, top)
        except Exception as e:
            # Corrupted TIFF, incomplete download, or other read error
            skipped_corrupt += 1
            print(f"  [SKIP] {os.path.basename(tp)}: {type(e).__name__}")
            continue

    if skipped_crs:
        print(
            f"  WARNING: {skipped_crs} tiles have a different CRS than {ref_crs} "
            f"and were skipped — that mismatch alone could explain missing coverage."
        )
    if skipped_corrupt:
        print(
            f"  WARNING: {skipped_corrupt} tiles are corrupted or unreadable "
            f"(truncated downloads, incomplete writes, etc). Delete and re-download them."
        )

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.boundary.plot(ax=ax, color="red", linewidth=1.5, zorder=3)

    n_outside = 0
    for left, bottom, right, top in boxes:
        inside = boundary_prepared.intersects(shapely_box(left, bottom, right, top))
        if not inside:
            n_outside += 1
        color = "steelblue" if inside else "crimson"
        ax.add_patch(Rectangle(
            (left, bottom), right - left, top - bottom,
            facecolor=color, edgecolor=color,
            alpha=0.25 if inside else 0.55, linewidth=0.3, zorder=2,
        ))

    ax.set_aspect("equal")
    ax.set_title(f"{args.city}: {len(boxes)} valid tiles vs. boundary ({n_outside} outside)")
    ax.set_xlabel(f"easting ({ref_crs})")
    ax.set_ylabel("northing")
    ax.legend(handles=[
        Line2D([0], [0], color="red", linewidth=1.5, label="city boundary"),
        Patch(facecolor="steelblue", edgecolor="steelblue", alpha=0.25, label="tile overlaps boundary"),
        Patch(facecolor="crimson", edgecolor="crimson", alpha=0.55, label="tile has NO overlap"),
    ], loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[{args.city}] {n_outside}/{len(boxes)} valid tiles don't overlap the boundary")
    print(f"[{args.city}] saved -> {out_path}")


if __name__ == "__main__":
    main()