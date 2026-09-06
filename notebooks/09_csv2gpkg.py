
"""
Raw zeroshot box CSV -> GPKG, for visual inspection in QGIS = used for checking raw zeroshot boxes

Goal: see the DeepForest boxes themselves (before SAM2 ever touches them),
so you can tell whether poor SAM2 results trace back to bad boxes (e.g.
rooftops) already present at this stage, vs. something SAM2 introduced.

Usage:
    =>set up CSV_PATH, GPKG_PATH, TILES_DIR
    python 3-30-300-index/notebooks/09_csv2gpkg.py

"""

import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

CSV_PATH = "3-30-300-index/data/trento/trento_tree_detections_ZEROSHOT_RESSCALED.csv"
GPKG_PATH = "3-30-300-index/data/trento/trento_tree_detections_ZEROSHOT_RESSCALED.gpkg"


TILES_DIR = "3-30-300-index/data/trento/trento_ortho_tiles"
FALLBACK_CRS = "EPSG:32632" 

def build_from_pixels(df, tiles_dir):
    """Fallback: build box polygons from xmin/ymin/xmax/ymax per tile,
    using that tile's own affine transform to georeference."""
    import rasterio
    import rasterio.transform

    tiles_dir = Path(tiles_dir)
    geoms = [None] * len(df)
    df = df.reset_index(drop=True)

    for tile_stem, idxs in df.groupby("source_tile").groups.items():
        tif_path = tiles_dir / f"{tile_stem}.tif"
        if not tif_path.exists():
            print(f"  [WARN] tile not found, skipping: {tif_path.name}")
            continue
        with rasterio.open(tif_path) as src:
            transform = src.transform
            for idx in idxs:
                row = df.loc[idx]
                x0, y0 = transform * (row["xmin"], row["ymin"])
                x1, y1 = transform * (row["xmax"], row["ymax"])
                geoms[idx] = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    return pd.Series(geoms)


def main():
    csv_path = Path(CSV_PATH)
    gpkg_path = Path(GPKG_PATH)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows loaded, columns: {list(df.columns)}")

    print("Building georeferenced boxes from pixel coords via each tile's own transform ...")
    geoms = build_from_pixels(df, TILES_DIR)
    used_crs = FALLBACK_CRS

    df = df.copy()
    df["geom"] = geoms
    before = len(df)
    df = df[df["geom"].notna()]
    dropped = before - len(df)
    if dropped:
        print(f"  [WARN] dropped {dropped:,} rows (tile file not found)")

    keep_cols = [c for c in ["score", "label", "source_tile", "city"] if c in df.columns]
    gdf = gpd.GeoDataFrame(df[keep_cols], geometry=df["geom"], crs=used_crs)

    print(f"Writing GPKG ({used_crs}) → {gpkg_path} ...")
    if gpkg_path.exists():
        gpkg_path.unlink()
    gdf.to_file(gpkg_path, driver="GPKG", layer="zeroshot_boxes")
    print(f"  OK — {gpkg_path.stat().st_size / 1e6:.1f} MB, {len(gdf):,} boxes")
    print(f"\nOpen in QGIS and compare against your SAM2 output layer to see "
          f"whether bad detections (rooftops etc) are already present at the "
          f"raw DeepForest stage.")


if __name__ == "__main__":
    main()