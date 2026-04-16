# download.py
"""
    python download.py milano
    python download.py all
"""

import sys
import os
import yaml
import osmnx as ox
import geopandas as gpd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_PATH = Path(__file__).resolve().parent
os.chdir(ROOT_PATH)

with open(ROOT_PATH / "config" / "config.yaml") as f:
    CITIES = yaml.safe_load(f)

# Filter out empty entries (aosta, venezia, etc.)
CITIES = {k: v for k, v in CITIES.items() if v}


def download_city(city: str) -> None:
    cfg = CITIES[city]
    query = cfg["query"]
    out_dir = ROOT_PATH / "data" / city
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_path = out_dir / f"{city}_boundary.gpkg"
    if boundary_path.exists():
        print(f"[{city}] boundary exists, skipping")
        boundary = gpd.read_file(boundary_path)
    else:
        print(f"[{city}] downloading boundary...")
        boundary = ox.geocode_to_gdf(query, which_result=1)
        boundary.to_file(boundary_path, driver="GPKG")
        print(f"[{city}] saved {boundary_path}")

    buildings_path = out_dir / f"{city}_buildings.gpkg"
    if buildings_path.exists():
        print(f"[{city}] buildings exist, skipping")
    else:
        print(f"[{city}] downloading OSM buildings...")
        boundary_wgs84 = boundary.to_crs("EPSG:4326").geometry.union_all()
        gdf = ox.features_from_polygon(boundary_wgs84, tags={"building": True})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        keep = [c for c in ["geometry", "height", "building:levels", "building"] if c in gdf.columns]
        gdf[keep].reset_index(drop=True).to_file(buildings_path, driver="GPKG")
        print(f"[{city}] saved {buildings_path} ({len(gdf):,} buildings)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python download.py <city|all>")
        print(f"Cities: {', '.join(CITIES)}")
        sys.exit(1)

    arg = sys.argv[1].lower()
    cities = list(CITIES.keys()) if arg == "all" else [arg]

    for c in cities:
        if c not in CITIES:
            print(f"Unknown city '{c}'. Available: {', '.join(CITIES)}")
            sys.exit(1)

    # Parallel downloads
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(download_city, c): c for c in cities}
        for f in as_completed(futures):
            c = futures[f]
            if f.exception():
                print(f"[{c}] ERROR: {f.exception()}")


if __name__ == "__main__":
    main()