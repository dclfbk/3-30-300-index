# extract_boundaries.py
import geopandas as gpd
import yaml
from pathlib import Path

with open("config/config.yaml") as f:
    CITIES = yaml.safe_load(f)
CITIES = {k: v for k, v in CITIES.items() if v}

CITY_NAME_MAP = {
    "bologna": "Bologna",
    "bari": "Bari",
    "bolzano": "Bolzano",        # may appear as "Bolzano/Bozen"
    "firenze": "Firenze",
    "genova": "Genova",
    "milano": "Milano",
    "napoli": "Napoli",
    "palermo": "Palermo",
    "roma": "Roma",
    "torino": "Torino",
    "trento": "Trento",
    "verona": "Verona",
}

# Load with explicit encoding fix
src = Path("/Users/terezasaskova/Downloads/Limiti01012026_g/Com01012026_g")  # <-- update this
gdf = gpd.read_file(src)
gdf["COMUNE"] = gdf["COMUNE"].str.encode("latin-1").str.decode("utf-8", errors="replace")

for city_key, city_name in CITY_NAME_MAP.items():
    row = gdf[gdf["COMUNE"] == city_name]
    if row.empty:
        # fuzzy fallback — catches "Bolzano/Bozen" etc.
        row = gdf[gdf["COMUNE"].str.startswith(city_name, na=False)]
    if row.empty:
        print(f"[{city_key}] NOT FOUND")
        continue

    out_path = Path(f"data/{city_key}/{city_key}_boundary.gpkg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row.to_crs("EPSG:4326").to_file(out_path, driver="GPKG")
    print(f"[{city_key}] saved -> {out_path}  ({len(row)} row)")