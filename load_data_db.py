import sys
import os
import duckdb
from config import DB_PATH, get_city_cfg

CITY = sys.argv[1]
cfg  = get_city_cfg(CITY)
src  = cfg["data_sources"]

con = duckdb.connect(str(DB_PATH))
con.execute("LOAD spatial;")

def load(table: str, path: str) -> bool:
    """Load any vector file into raw schema. Returns False if file not found."""
    if not os.path.exists(path):
        print(f"  SKIP raw.{table}: file not found: {path}")
        return False
    print(f"  loading raw.{table} = {path}")
    con.execute(f"""
        CREATE OR REPLACE TABLE raw.{table} AS
        SELECT '{CITY}' AS city, *
        FROM ST_Read('{path}');
    """)
    return True

#shared
load(f"{CITY}_boundary",  src["boundary"])
load(f"{CITY}_buildings", src["buildings_sn"])

if src.get("zones"):
    load(f"{CITY}_zones", src["zones"])
else:
    print(f"  SKIP {CITY}_zones: not defined in config")

#300m: one table per raw park source
for name, path in cfg["park_layers"].items():
    load(f"{CITY}_parks_{name}", path)

#30%: canopy raster stays on disk 
#3-trees: orthophoto + tree_polygons loaded after DeepForest runs

con.close()
print(f"\n {CITY}: raw schema loaded (skipped files listed above)")