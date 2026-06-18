"""
plot_city_coverage.py — Visual QA: downloaded tile footprints vs. city boundary.

For a given city, opens every tile_*.tif on disk, reads its real bounds/CRS
straight from the GeoTIFF header (no pixel decoding, so this stays fast even
at thousands of tiles), reprojects the city boundary polygon to match, and
draws both. Tiles that don't actually overlap the boundary (e.g. they landed
in the sea, or a CRS mismatch put them somewhere wrong) are drawn in red
instead of blue — so this exact class of bug shows up in the figure itself,
not just as a failure count in a log.

Usage:
    python plot_city_coverage.py --city bari
    python plot_city_coverage.py --city firenze_rgb --boundary-city firenze
    python plot_city_coverage.py --city bari --tiles-dir /custom/path --out /tmp/bari_qa.png

Run with PYTHONPATH=/workspace/3-30-300-index already set (same as the rest
of the pipeline) so `from config.config import ...` resolves.

Note: this only plots tiles that exist on disk. Tiles that failed to
download (e.g. blank-response failures) never got written, so they won't
appear here at all — they'll just show as a gap. If you want failed (row,
col) cells reconstructed and overlaid too, that needs the exact --gsd /
--tile-px / --overlap-px used in that download run, since the grid is
anchored to them; ask and I can add it once you confirm those values.
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
ROOT_PATH = "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index/"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)
from config.config import connect_db, load_table



def load_boundary(city, target_crs, boundary_path=None):
    if boundary_path:
        import geopandas as gpd
        return gpd.read_file(boundary_path).to_crs(target_crs)
    con = connect_db(read_only=True)
    try:
        gdf = load_table(con, "raw", f"{city}_boundary").to_crs(target_crs)
    except Exception as e:
        sys.exit(
            f"Couldn't load '{city}_boundary' from the DB ({e}). "
            f"Pass --boundary-city if this city's tiles use a different base "
            f"city's boundary (e.g. firenze_rgb -> firenze), or --boundary-path "
            f"to load from a file instead."
        )
    finally:
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
    boundary_prepared = prep(gdf.geometry.unary_union)

    boxes, skipped_crs = [], 0
    for tp in tif_paths:
        with rasterio.open(tp) as src:
            if src.crs != ref_crs:
                skipped_crs += 1
                continue
            boxes.append(tuple(src.bounds))  # (left, bottom, right, top)

    if skipped_crs:
        print(
            f"  WARNING: {skipped_crs} tiles have a different CRS than {ref_crs} "
            f"and were skipped — that mismatch alone could explain missing coverage."
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
    ax.set_title(f"{args.city}: {len(boxes)} downloaded tiles vs. boundary ({n_outside} outside)")
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

    print(f"[{args.city}] {n_outside}/{len(boxes)} downloaded tiles don't overlap the boundary")
    print(f"[{args.city}] saved -> {out_path}")


if __name__ == "__main__":
    main()