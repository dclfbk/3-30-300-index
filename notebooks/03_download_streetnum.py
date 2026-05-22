import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings, os, sys
warnings.filterwarnings("ignore")
from pathlib import Path

CITY = "laquila"

ROOT_PATH = "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index"
sys.path.insert(0, str(ROOT_PATH))
os.chdir(ROOT_PATH)

from config.config import (
    EUCLID_CUTOFF, connect_db, gdf_to_duckdb, load_table, gpkg_to_duckdb,
    get_city_cfg, get_output_paths, DB_PATH, GH_PORT, CRS_WGS, CRS_METRIC, WALKING_DISTANCE,
)

con   = connect_db()
cfg   = get_city_cfg(CITY)
paths = get_output_paths(CITY)

INPUT_CSV   = "/Users/terezasaskova/Downloads/INDIR_MOLI_20260406.csv"
OUTPUT_GPKG = f"data/{CITY}/{CITY}_civici.gpkg"

# --- boundary from DuckDB
boundary     = load_table(con, "raw", f"{CITY}_boundary").to_crs(CRS_WGS)
print(f"Boundary bounds: {boundary.total_bounds}")

# --- load + parse csv
df = pd.read_csv(INPUT_CSV, sep=";", dtype=str, on_bad_lines="skip")
print(df.columns.tolist())
df = df.dropna(subset=["COORD_X_COMUNE", "COORD_Y_COMUNE"])
df["lon"] = pd.to_numeric(df["COORD_X_COMUNE"].str.replace(",", "."), errors="coerce")
df["lat"] = pd.to_numeric(df["COORD_Y_COMUNE"].str.replace(",", "."), errors="coerce")
df = df.dropna(subset=["lon", "lat"])

gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(x, y) for x, y in zip(df["lon"], df["lat"])],
    crs=CRS_WGS,
)

# --- filter to boundary
gdf_filtered = gpd.sjoin(gdf, boundary[["geometry"]], how="inner", predicate="within")
gdf_filtered = gdf_filtered.drop(columns=["index_right"])

print(f"Total:    {len(gdf):,}")
print(f"Within:   {len(gdf_filtered):,}")
print(f"Dropped:  {len(gdf) - len(gdf_filtered):,}")

os.makedirs(os.path.dirname(OUTPUT_GPKG), exist_ok=True)
gdf_filtered.to_file(OUTPUT_GPKG, driver="GPKG", layer="civici")
print(f"Saved → {OUTPUT_GPKG}")