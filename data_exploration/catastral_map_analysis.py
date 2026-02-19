import requests
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import time

BASE = "https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/c_a944ctc_edifici_pl/records"

# ── 1. Total count ────────────────────────────────────────────────────────────
r = requests.get(BASE, params={"limit": 1})
total = r.json()["total_count"]
print(f"Total buildings in Bologna: {total:,}")

# ── 2. Download all (paginated) ───────────────────────────────────────────────
records = []
limit = 100
for offset in range(0, total, limit):
    for attempt in range(3):   # retry up to 3x
        try:
            r = requests.get(BASE, params={"limit": limit, "offset": offset}, timeout=30)
            data = r.json()
            if "results" in data:
                records.extend(data["results"])
                break
            else:
                print(f"\n  No 'results' at offset {offset}, response: {data}")
                time.sleep(2)
        except Exception as e:
            print(f"\n  Error at offset {offset}: {e}, retrying...")
            time.sleep(2)
    print(f"  fetched {min(offset+limit, total)}/{total}", end="\r")
    time.sleep(0.2)   # be polite to the API

print(f"\nDownloaded {len(records):,} records")

# ── 3. Geometry ───────────────────────────────────────────────────────────────
df = pd.json_normalize(records)

def parse_geom(row):
    try:
        return shape({
            "type": row["geo_shape.geometry.type"],
            "coordinates": row["geo_shape.geometry.coordinates"]
        })
    except:
        return None

df["geometry"] = df.apply(parse_geom, axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf_proj = gdf.to_crs("EPSG:32632")
gdf_proj["area_m2"] = gdf_proj.geometry.area

print(f"Valid geometries: {gdf.geometry.notna().sum():,} / {len(gdf):,}")

# ── 4. Save checkpoint as you go (in case of crash) ──────────────────────────
save_cols = [c for c in gdf_proj.columns if c != "geo_shape.geometry.coordinates"]
gdf_proj[save_cols].to_file("bologna_buildings_full.gpkg", driver="GPKG")
print("Saved: bologna_buildings_full.gpkg")