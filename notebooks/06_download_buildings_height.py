import mercantile, requests, gzip, json
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
from pathlib import Path
import os
import sys
import yaml
from pathlib import Path

ROOT_PATH = Path("/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index")
sys.path.insert(0, str(ROOT_PATH))
os.chdir(ROOT_PATH)

from config.config import (
    DB_PATH, get_city_cfg, CRS_WGS, CRS_METRIC,
    connect_db, gdf_to_duckdb, load_table,get_output_paths,
)

con = connect_db()

with open(ROOT_PATH / "config/config.yaml") as f:
    CITIES = yaml.safe_load(f)

# Filter out empty entries (aosta, venezia, etc.)
CITIES = {k: v for k, v in CITIES.items() if v}

# adjust schema/table to where your boundaries live
boundaries = {}
for city in CITIES:
    gdf = load_table(con, "raw", f"{city}_boundary")  # adjust schema/table name
    boundaries[city] = gdf.to_crs("EPSG:4326").total_bounds  # (minx, miny, maxx, maxy)


# download dataset-links.csv once
links_url = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
links_df  = pd.read_csv(links_url)
italy_links = links_df[links_df["Location"] == "Italy"]

def get_quadkeys_for_bbox(boundary, zoom=9):
    tiles = mercantile.tiles(*boundary, zooms=zoom)
    return [mercantile.quadkey(t) for t in tiles]

def download_city_buildings(con, city, boundary, out_dir="data"):

    Path(out_dir).mkdir(exist_ok=True)
    qks = get_quadkeys_for_bbox(boundary)
    
    matched = italy_links[italy_links["QuadKey"].astype(str).apply(
        lambda q: any(qk.startswith(q) or q.startswith(qk) for qk in qks)
    )]
    
    gdfs = []
    for _, row in matched.iterrows():
        r = requests.get(row["Url"], stream=True)
        lines = gzip.decompress(r.content).decode().strip().split("\n")
        features = [json.loads(l) for l in lines]
        gdf = gpd.GeoDataFrame(
            [f["properties"] for f in features],
            geometry=[shape(f["geometry"]) for f in features],
            crs="EPSG:4326"
        )
        # clip to bbox
        from shapely.geometry import box
        gdf = gdf[gdf.geometry.intersects(box(*boundary))]
        gdfs.append(gdf)
    
    if not gdfs:
        print(f"{city}: no data found")
        return None
    
    result = gpd.GeoDataFrame(pd.concat(gdfs), crs="EPSG:4326").to_crs("EPSG:32632")
    result.to_file(f"{out_dir}/{city}_buildings.gpkg", driver="GPKG")
    print(f"{city}: {len(result):,} buildings saved")
    gdf_to_duckdb(con, result.reset_index(drop=True), f"raw.{city}_buildings")

    return result

# run all cities
for city, boundary in boundaries.items():
    download_city_buildings(con, city, boundary)

con.close()
