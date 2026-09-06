# config.py
import geopandas as gpd
import pandas as pd
import duckdb
import yaml
import sys
from pathlib import Path
from typing import Optional
import time
import osmnx as ox
from osmnx._errors import InsufficientResponseError
from shapely.ops import unary_union

ROOT_PATH = Path(__file__).resolve().parent.parent
ROOT_DIR  = ROOT_PATH
DATA_DIR  = ROOT_DIR / "data"
LOGS_DIR  = ROOT_DIR / "logs"
SUMMARY_CSV = DATA_DIR / "rule30_compliance_summary.csv"

DB_PATH = ROOT_PATH / "3_30_300_master.duckdb"
DB_LOCK_MAX_RETRIES = 12
DB_LOCK_BASE_WAIT_S = 15   # grows per attempt, capped below
DB_LOCK_MAX_WAIT_S  = 120

CRS_WGS    = "EPSG:4326"
CRS_METRIC = "EPSG:32632"

GH_PORT          = 8989
WALKING_DISTANCE = 300
EUCLID_CUTOFF    = 420

CITY = "milano"  # vestigial — confirm unused before removing

for d in [DATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_CITIES_FILE = Path(__file__).parent / "config.yaml"
with open(_CITIES_FILE) as f:
    CITIES = yaml.safe_load(f)

TILE_DIR_PATTERNS = [
    "{city}/{city}_ortho_tiles",
]

# --- Canopy pipeline constants -------------------------------------------
# Moved from canopy_extract_rgb.py. WATER_BUFFER_M was previously imported
# here AND redefined locally in that script (local value silently won) —
# this is now the single source of truth.
OSM_BUFFER_M        = 2.0
WATER_BUFFER_M      = 6.0
TEST_OSM_BUFFER_DEG = 0.001

def get_city_cfg(city_key: str) -> dict:
    if city_key not in CITIES:
        raise ValueError(f"Unknown city '{city_key}'. Available: {', '.join(CITIES)}")
    cfg = CITIES[city_key]
    if cfg is None:
        raise ValueError(f"City '{city_key}' has no config defined yet.")
    return {"key": city_key, **cfg}

def find_tile_dir(city: str, base_city: str) -> Path:
    """Resolve the ortho-tile directory for a city, falling back to the
    base city name for RGB/NIR variants (e.g. 'torino_rgb' -> 'torino')."""
    for candidate in [
        DATA_DIR / TILE_DIR_PATTERNS[0].format(city=city),
        DATA_DIR / base_city / TILE_DIR_PATTERNS[0].format(city=city),
        DATA_DIR / base_city / TILE_DIR_PATTERNS[0].format(city=base_city),
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No tile directory found for city='{city}'")

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
        "buildings": str(base / "buildings" / f"{city_key}_buildings.gpkg"),
        "civici": str(base / "civici" / f"{city_key}_civici.gpkg"),
        "canopy": str(base / "30%" / f"{city_key}_canopy.gpkg"),
    }

def strip_source_suffix(city: str) -> str:
    """Strip RGB/NIR/CIR source-variant suffixes to get the base city key
    (e.g. 'torino_rgb' -> 'torino')."""
    for suffix in ("_rgb", "_nir", "_cir"):
        city = city.removesuffix(suffix)
    return city

def connect_db(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Open the shared DuckDB file, load the spatial extension,
    and ensure all project schemas exist.
    Use this instead of bare duckdb.connect() in every notebook.
    """
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute("LOAD spatial;")
    if not read_only:
        for schema in ["raw", "preprocessing", "processing", "rule_300m", "raw_2", "preprocessing_2", "rule_300m_2"]:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return con

def connect_db_with_retry(read_only: bool = False):
    """Wraps connect_db() with a polite retry loop for the common
    "Conflicting lock is held" error — this happens whenever another
    city-processing script (or a manual DB check) currently holds the
    write lock on the shared master DB. DuckDB is correctly REFUSING a
    second writer here, not corrupting anything — but crashing outright
    on that refusal throws away everything already computed and sitting
    in memory (hundreds of thousands of detections, hours of GPU time),
    for a condition that usually resolves itself in minutes once the
    other process finishes. Retries with a growing wait instead.
    """
    for attempt in range(1, DB_LOCK_MAX_RETRIES + 1):
        try:
            return connect_db(read_only=read_only)
        except Exception as e:
            msg = str(e)
            if "Conflicting lock" in msg or "Could not set lock" in msg:
                wait = min(DB_LOCK_BASE_WAIT_S * attempt, DB_LOCK_MAX_WAIT_S)
                print(f"[DB LOCK] Another process holds the write lock on the shared DB — "
                      f"retrying in {wait}s (attempt {attempt}/{DB_LOCK_MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Could not acquire DB lock after {DB_LOCK_MAX_RETRIES} attempts. "
        f"Check for a stuck/zombie process still holding the lock (ps aux | grep python) "
        f"before giving up — the computed results for this run are still safe in memory "
        f"only up until this process exits, so don't kill this one without saving them first."
    )
    


def gdf_to_duckdb(con, gdf: gpd.GeoDataFrame, full_table: str):
    schema, table = full_table.split(".", 1)
    df = pd.DataFrame(gdf.copy())
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

def load_boundary(lookup_city: str, boundary_path: Optional[str] = None):
    if boundary_path and Path(boundary_path).exists():
        gdf = gpd.read_file(boundary_path)
    else:
        gpkg = DATA_DIR / f"{lookup_city}_boundary.gpkg"
        if gpkg.exists():
            gdf = gpd.read_file(str(gpkg))
        else:
            con = connect_db(read_only=True)
            gdf = load_table(con, "raw", f"{lookup_city}_boundary")
            con.close()
    return unary_union(gdf.to_crs(CRS_METRIC).geometry.values)

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


OSM_EXCLUDE_TAGS = {
    "leisure":  ["pitch", "track", "golf_course", "stadium"],
    "landuse":  ["agriculture", "crop", "farmland", "farmyard", "meadow", "grass",
                 "recreation_ground", "village_green", "orchard", "vineyard", "allotments"],
    "natural":  ["grassland", "fell", "heath", "scrub", "water", "wetland", "beach",
                 "bay", "strait", "coastline"],
    "landcover": ["crops", "grass", "field", "meadow", "vineyard", "orchard", "scrub", "heath", "bare_rock"],
    "water":    ["river", "canal", "lake", "reservoir", "pond", "sea", "wetland", "ocean"],
    "place":    ["sea"],
    "building": True,
}

WATER_TAG_KEYS = {
    "natural": {"water", "wetland", "beach", "bay", "strait", "coastline"},
    "water":   {"river", "canal", "lake", "reservoir", "pond", "sea", "wetland", "ocean"},
    "place":   {"sea"},
}

OSM_INCLUDE_TAGS = {
    "landuse": ["forest", "wood", "woodland", "scrub", "orchard"]
}

OSM_GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve"],
    "landuse": ["village_green"],
}
PRIVATE_ACCESS = {"private", "no", "restricted", "permit", "customers"}

# --- OSM exclusion/inclusion fetch cluster --------------------------------
# Moved from canopy_extract_rgb.py as a set (see that file's own comment
# on build_osm_inclusion — a prior partial reorder attempt broke this
# before by splitting these across files). Keep them together.

def _osm_cache_path(city: str, key: str, value, scope: str) -> Path:
    safe_val = str(value).replace("/", "_")
    return DATA_DIR / "osm_cache" / f"{city}__{key}__{safe_val}__{scope}.gpkg"


def _fetch_osm_tag(city: str, key: str, value, query_geom, scope: str) -> Optional[gpd.GeoDataFrame]:
    cache = _osm_cache_path(city, key, value, scope)
    if cache.exists():
        return gpd.read_file(str(cache))
    try:
        gdf = ox.features_from_polygon(query_geom, tags={key: value})
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].copy()
        if polys.empty:
            return None
        polys["osm_tag"] = f"{key}={value}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        polys.to_file(str(cache), driver="GPKG")
        return polys
    except InsufficientResponseError:
        # Query succeeded, genuinely zero features here — expected, not a failure.
        return None
    except Exception:
        return None


def _is_water_tag(key: str, value) -> bool:
    return value in WATER_TAG_KEYS.get(key, set())


def build_osm_exclusion(city: str, query_geom, scope: str) -> Optional[gpd.GeoDataFrame]:
    water_gdfs, other_gdfs = [], []
    for key, values in OSM_EXCLUDE_TAGS.items():
        tag_list = values if isinstance(values, list) else [values]
        for value in tag_list:
            gdf = _fetch_osm_tag(city, key, value, query_geom, scope)
            if gdf is None:
                continue
            (water_gdfs if _is_water_tag(key, value) else other_gdfs).append(gdf)

    if not water_gdfs and not other_gdfs:
        return None

    parts = []
    if water_gdfs:
        water = gpd.GeoDataFrame(pd.concat(water_gdfs, ignore_index=True), crs=CRS_WGS).to_crs(CRS_METRIC)
        water["geometry"] = water.geometry.buffer(-WATER_BUFFER_M)
        parts.append(water[~water.geometry.is_empty].copy())
    if other_gdfs:
        other = gpd.GeoDataFrame(pd.concat(other_gdfs, ignore_index=True), crs=CRS_WGS).to_crs(CRS_METRIC)
        other["geometry"] = other.geometry.buffer(-OSM_BUFFER_M)
        parts.append(other[~other.geometry.is_empty].copy())

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS_METRIC)


def build_osm_inclusion(city: str, query_geom, scope: str) -> Optional[gpd.GeoDataFrame]:
    gdfs = []
    for key, values in OSM_INCLUDE_TAGS.items():
        for value in values:
            gdf = _fetch_osm_tag(city, key, value, query_geom, scope)
            if gdf is not None:
                gdfs.append(gdf)
    if not gdfs:
        return None
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=CRS_WGS).to_crs(CRS_METRIC)

