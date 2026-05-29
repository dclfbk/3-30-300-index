import geopandas as gpd
from pathlib import Path
import rasterio
from config.config import CRS_METRIC, CRS_WGS, get_output_paths, CITY, connect_db, get_city_cfg, DB_PATH, load_table, CRS_WGS, CRS_METRIC
import matplotlib.pyplot as plt

CITY = "bologna"
con = connect_db()
cfg = get_city_cfg(CITY)
boundary  = load_table(con, "raw", f"{CITY}_boundary")
tile_dir = Path("~/Downloads/bologna_nir_tiles_local/").expanduser()
tile_files = sorted(tile_dir.glob("*.tif"))

#=========================================================================

# get bounds of each tile
records = []
bad_tiles = []

for f in tile_files:
    try:
        with rasterio.open(f) as src:
            bounds = src.bounds
            records.append({"file": f.name, "geometry": plt.box()})
    except Exception:
        bad_tiles.append(f.name)

print(f"Good: {len(records)}, Bad: {len(bad_tiles)}")
print(bad_tiles)

# save bad list to file
with open("/tmp/bad_tiles.txt", "w") as f:
    for t in bad_tiles:
        f.write(t + "\n")