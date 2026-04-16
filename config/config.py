from pathlib import Path
import geopandas as gpd
import pandas as pd
import duckdb
from shapely import wkt as shapely_wkt
import yaml


ROOT_PATH = Path(__file__).resolve().parent.parent  
ROOT_DIR  = ROOT_PATH  

DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
DB_PATH  = ROOT_DIR / "3_30_300.duckdb"

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
        raise ValueError(
            f"Unknown city '{city_key}'. Available: {', '.join(CITIES)}"
        )
    return {"key": city_key, **CITIES[city_key]}


def get_output_paths(city_key: str) -> dict:
    base = ROOT_DIR / "output" / city_key
    base.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "fig" / city_key).mkdir(parents=True, exist_ok=True)
    return {
        "gpkg":    str(base / f"{city_key}_3_30_300_results.gpkg"),
        "csv":     str(base / f"{city_key}_3_30_300_results.csv"),
        "geojson": str(base / f"{city_key}_3_30_300_results.geojson"),
        "parquet": str(base / f"{city_key}_3_30_300_results.parquet"),
        "fig_dir": str(ROOT_DIR / "fig" / city_key),
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
    for schema in ["raw", "preprocessing", "processing", "rule_300m"]:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return con



def gdf_to_duckdb(con, gdf, table_name):
    """
    Saves a GeoDataFrame to DuckDB using native spatial types.
    """
    if "." in table_name:
        schema = table_name.split(".")[0]
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    
    #Convert geometry to WKB which DuckDB Spatial understands natively
    temp_df = gdf.copy()
    temp_df['geometry'] = temp_df.geometry.to_wkb()
    
    # Use DuckDB to register the dataframe and create a table
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS "
                f"SELECT * EXCLUDE geometry, ST_GeomFromWKB(geometry) AS geometry "
                f"FROM temp_df")
    
    print(f"{table_name}: {len(gdf)} rows written")

def load_table(con, schema, table, crs=CRS_METRIC):
    """
    Loads a DuckDB table into a GeoPandas GeoDataFrame natively.
    """
    # Fetch as a dataframe, converting spatial type to WKB in the query
    df = con.execute(f"""
        SELECT * EXCLUDE geometry, 
               ST_AsWKB(geometry) AS geometry 
        FROM {schema}.{table}
    """).df()
    df["geometry"] = df["geometry"].apply(bytes)

    # GeoPandas can load the whole column from WKB in one go (vectorized)
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

