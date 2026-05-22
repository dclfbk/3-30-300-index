import geopandas as gpd
import pandas as pd
import duckdb
import yaml

import sys
from pathlib import Path

project_root = Path("/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

ROOT_PATH = Path(__file__).resolve().parent.parent
ROOT_DIR  = ROOT_PATH
DATA_DIR  = ROOT_DIR / "data"
LOGS_DIR  = ROOT_DIR / "logs"

DB_PATH = "/Users/terezasaskova/3_30_300_master.duckdb"
CRS_WGS  = "EPSG:4326"
CRS_METRIC   = "EPSG:32632"

GH_PORT  = 8989
WALKING_DISTANCE = 300   # metres
EUCLID_CUTOFF    = 420   # metres — Euclidean pre-filter before point-to-point routing


for d in [DATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CRS_WGS    = "EPSG:4326"
CRS_METRIC = "EPSG:32632"

#GraphHopper 
CITY  = "milano"
GH_PORT = 8989

WALKING_DISTANCE = 300
EUCLID_CUTOFF = 420


_CITIES_FILE = Path(__file__).parent / "config.yaml"

with open(_CITIES_FILE) as f:
    CITIES = yaml.safe_load(f)


def get_city_cfg(city_key: str) -> dict:
    if city_key not in CITIES:
        raise ValueError(f"Unknown city '{city_key}'. Available: {', '.join(CITIES)}")
    cfg = CITIES[city_key]
    if cfg is None:
        raise ValueError(f"City '{city_key}' has no config defined yet.")
    return {"key": city_key, **cfg}


def get_output_paths(city_key: str) -> dict:
    base = ROOT_DIR / "results" / city_key
    base.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "figures" / city_key).mkdir(parents=True, exist_ok=True)
    return {
        "gpkg":    str(base / f"{city_key}_3_30_300_results.gpkg"),
        "csv":     str(base / f"{city_key}_3_30_300_results.csv"),
        "parquet": str(base / f"{city_key}_3_30_300_results.parquet"),
        "fig_dir": str(ROOT_DIR / "figures" / city_key),
    }

def get_produced_paths(city_key: str) -> dict:
    base = DATA_DIR / city_key
    return {
        "tree_polygons": str(base / "3trees" / f"{city_key}_tree_polygons.gpkg"),
        "parks_gardens": str(base / "300m"   / f"{city_key}_greenery.gpkg"),
    }

def connect_db() -> duckdb.DuckDBPyConnection:
    """
    Open the shared DuckDB file, load the spatial extension,
    and ensure all project schemas exist.
    Use this instead of bare duckdb.connect() in every notebook.
    """
    con = duckdb.connect(str(DB_PATH))
    con.execute("LOAD spatial;")
    for schema in ["raw", "preprocessing", "processing", "rule_300m", "raw_2", "preprocessing_2", "rule_300m_2"]:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return con


def gdf_to_duckdb(con, gdf: gpd.GeoDataFrame, full_table: str):
    schema, table = full_table.split(".", 1)
    df = gdf.copy()
    df["geometry"] = gdf.geometry.to_wkb().apply(
        lambda x: bytes(x) if x is not None else None
    )
    for col in df.columns:
        if col == "geometry":
            continue
        try:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        except Exception:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)

    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    con.register("_tmp", df)
    con.execute(f"""
        CREATE OR REPLACE TABLE {full_table} AS 
        SELECT * EXCLUDE geometry,
               ST_GeomFromWKB(geometry) AS geometry
        FROM _tmp
    """)
    con.unregister("_tmp")
    print(f"{full_table}: {len(df)} rows written")

def load_table(con, schema, table, crs=CRS_METRIC):
    # Check if geometry is already GEOMETRY type or raw BLOB
    col_types = {r[0]: r[1] for r in con.execute(f"DESCRIBE {schema}.{table}").fetchall()}
    
    if col_types.get("geometry") == "GEOMETRY":
        geom_expr = "ST_AsWKB(geometry) AS geometry"
    else:
        # Already raw WKB bytes stored as BLOB
        geom_expr = "geometry"
    
    df = con.execute(f"""
        SELECT * EXCLUDE geometry, {geom_expr}
        FROM {schema}.{table}
    """).df()
    
    df["geometry"] = df["geometry"].apply(bytes)
    gdf = gpd.GeoDataFrame(
        df.drop(columns="geometry"),
        geometry=gpd.array.from_wkb(df["geometry"]),
        crs=crs
    )
    return gdf

def gpkg_to_duckdb(con: duckdb.DuckDBPyConnection,
                   path: str,
                   schema_table: str,
                   layer=None) -> gpd.GeoDataFrame:
    """
    Load a GeoPackage (or any GDAL-readable file), reproject to CRS_METRIC,
    write to DuckDB, and return the GeoDataFrame for immediate use.

    Usage:
        gdf = gpkg_to_duckdb(con, "path/to/file.gpkg", "schema.tablename")
    """
    gdf = gpd.read_file(path, layer=layer).to_crs(CRS_METRIC)
    gdf_to_duckdb(con, gdf, schema_table)
    return gdf

