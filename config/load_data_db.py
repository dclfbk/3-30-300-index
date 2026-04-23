"""
load_data_db.py  —  seed DuckDB raw schema for one city
========================================================
Usage:
    python load_data_db.py milano
    python load_data_db.py all          # loop every city in config

What it does
------------
For each city it loads all vector layers from data_sources (skipping rasters)
plus all park_layers into DuckDB schema = raw:

    raw.{city}_boundary
    raw.{city}_buildings
    raw.{city}_buildings_sn
    raw.{city}_parks_gardens
    raw.{city}_zones
    raw.{city}_tree_polygons
    raw.{city}_parks_{name}     (one table per park_layers entry)

Every table is:
  • reprojected to EPSG:32632 (CRS_METRIC)
  • stored with geometry as DuckDB GEOMETRY type (WKB round-trip via ST_GeomFromWKB)
  • tagged with a `city` column
  • idempotent — CREATE OR REPLACE, safe to re-run

Adding a new vector layer to config.py data_sources automatically
loads it — no changes needed here.

Notes on compatibility
----------------------
- Requires pandas >= 2.0, geopandas >= 0.14, duckdb >= 0.10 with spatial ext.
- gdf_to_duckdb() in config.py handles Arrow-backed StringDtype introduced in
  pandas 3 / geopandas 1.x — no dtype coercion is needed here.
"""

import sys
import os
import shutil
import pandas as pd

# ── Root path ──────────────────────────────────────────────────────────────────

ROOT = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index"
)
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Flush stale config cache (iCloud / __pycache__ safe)
if "config" in sys.modules:
    del sys.modules["config"]
pycache = os.path.join(ROOT, "__pycache__")
if os.path.exists(pycache):
    shutil.rmtree(pycache)

# ── Imports ────────────────────────────────────────────────────────────────────

import duckdb
import geopandas as gpd
from shapely.validation import make_valid

from config.config import DB_PATH, CITIES, get_city_cfg, gdf_to_duckdb

# ── Keys to skip (non-vector raster files) ────────────────────────────────────

RASTER_KEYS = {"canopy_raster", "orthophoto"}

# ── Core loader ───────────────────────────────────────────────────────────────

def load_vector(con, city, table_name, path, cfg=None):
    """
    Read a vector file, reproject to CRS_METRIC, validate geometries,
    tag with city column, write to raw schema in DuckDB.

    Returns True on success, False if file is missing or has no valid geometries.
    All dtype coercion is delegated to gdf_to_duckdb() in config.py.
    """
    schema_table = f"raw.{table_name}"

    if not os.path.exists(path):
        print(f"  SKIP  {schema_table}  (not found: {path})")
        return False

    print(f"  load  {schema_table}  <- {path}")

    try:
        gdf = gpd.read_file(path).to_crs("EPSG:32632")
    except Exception as e:
        print(f"  ERROR {schema_table}: {e}")
        return False

    # Drop null/empty, repair invalid geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    if gdf.empty:
        print(f"  SKIP  {schema_table}  (no valid geometries after cleaning)")
        return False

    gdf["city"] = city
    
    #Stable unique building_id (composite key, set once at ingestion)
    if "buildings_sn" in table_name:
        id_col = cfg.get("building_id_col", "NUMEROCOMPLETO")
        if id_col in gdf.columns and "CODICE_VIA" in gdf.columns:
            gdf["building_id"] = (
                gdf["CODICE_VIA"].astype(str).str.strip() + "_"
                + gdf[id_col].astype(str).str.strip() + "_"
                + gdf.get("LETTERA", pd.Series("", index=gdf.index)).fillna("").astype(str).str.strip()
            )
        else:
            gdf["building_id"] = gdf.index.astype(str)

    # gdf_to_duckdb handles all StringDtype / ArrowDtype coercion + WKB serialisation
    gdf_to_duckdb(con, gdf, schema_table)
    return True


def load_city(con: duckdb.DuckDBPyConnection, city: str) -> None:
    """Load all raw vector layers for one city into DuckDB."""
    cfg = get_city_cfg(city)

    print(f"\n{'='*60}")
    print(f"CITY: {city.upper()}")
    print(f"{'='*60}")

    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    loaded, skipped = 0, 0

    # ── data_sources: load every vector key, skip rasters ─────────────────────
    for key, path in cfg["data_sources"].items():
        if key in RASTER_KEYS:
            print(f"  SKIP  raw.{city}_{key}  (raster — not loaded into DuckDB)")
            skipped += 1
            continue
        ok = load_vector(con, city, f"{city}_{key}", path, cfg=cfg)

        loaded += ok
        skipped += not ok

    # ── park_layers: one table per entry ──────────────────────────────────────
    for layer_name, layer_path in cfg["park_layers"].items():
        ok = load_vector(con, city, f"{city}_parks_{layer_name}", layer_path)
        loaded += ok
        skipped += not ok

    # ── Summary ───────────────────────────────────────────────────────────────
    all_tables = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'raw'
        ORDER BY table_name
    """).df()["table_name"].tolist()

    city_tables = [t for t in all_tables if t.startswith(city)]

    print(f"\n  Summary for {city}: {loaded} loaded, {skipped} skipped")
    print(f"  raw schema tables ({len(city_tables)}):")
    for t in city_tables:
        n = con.execute(f"SELECT COUNT(*) FROM raw.{t}").fetchone()[0]
        print(f"    raw.{t}: {n:,} rows")
        
def get_building_id_expr(city_key: str) -> str:
    """Returns a SQL expression for a stable unique building_id."""
    cfg = CITIES[city_key]
    cols = cfg.get("building_id_composite", [cfg.get("building_id_col", "NUMEROCOMPLETO")])
    return " || '_' || ".join([f"COALESCE(CAST({c} AS VARCHAR), '')" for c in cols])

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python load_data_db.py <city>")
        print("       python load_data_db.py all")
        print(f"Cities: {', '.join(CITIES)}")
        sys.exit(1)

    arg = sys.argv[1].lower()
    cities = list(CITIES.keys()) if arg == "all" else [arg]

    for city in cities:
        if city not in CITIES:
            print(f"Unknown city '{city}'. Available: {', '.join(CITIES)}")
            sys.exit(1)

    con = duckdb.connect(str(DB_PATH))
    con.execute("INSTALL spatial; LOAD spatial;")

    for city in cities:
        load_city(con, city)

    con.close()
    print(f"\nDone. DB: {DB_PATH}")


if __name__ == "__main__":
    main()