from pathlib import Path
import geopandas as gpd
import pandas as pd
import duckdb
from shapely import wkt as shapely_wkt

ROOT_PATH = ""

try:
    ROOT_DIR = Path(__file__).parent.resolve()
except NameError:
    ROOT_DIR = Path.cwd()

DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
DB_PATH  = ROOT_DIR / "3_30_300.duckdb"

for d in [DATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CRS_WGS    = "EPSG:4326"
CRS_METRIC = "EPSG:32632"

#GraphHopper 
CITY    = "milano"
GH_PORT = 8989

WALKING_DISTANCE = 300
EUCLID_CUTOFF    = 420


#City configs 
CITIES: dict[str, dict] = {

    "bologna": {
        "query":             "Bologna, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": "quartiere",
        "park_layers": {
            "pe_unis": "data/bologna/300m/parks_raw_pe_unis.gpkg",
            "osm":     "data/bologna/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/bologna/3trees/bologna_cadastral.geojson",
            "gba":       "data/bologna/3trees/bologna_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/bologna/bologna_boundary.gpkg",
            "buildings_sn":  "data/bologna/bologna_buildings_streetnum.gpkg",
            "buildings":     "data/bologna/bologna_buildings.gpkg",
            "parks_gardens": "data/bologna/300m/bologna_greenery.gpkg",
            "zones":         "data/bologna/300m/bologna_aree_statistiche.geojson",
            "canopy_raster": "data/bologna/30/bologna_ndvi.tif",
            "orthophoto":    "data/bologna/3trees/bologna_orthophoto.tif",
            "tree_polygons": "data/bologna/3trees/bologna_tree_polygons.gpkg",
        },
    },

    "bari": {
        "query":             "Bari, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers": {
            "municipal": "data/bari/300m/parks_raw_municipal.gpkg",
            "osm":       "data/bari/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/bari/3trees/bari_cadastral.geojson",
            "gba":       "data/bari/3trees/bari_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/bari/bari_boundary.gpkg",
            "buildings_sn":  "data/bari/bari_buildings_streetnum.gpkg",
            "buildings":     "data/bari/bari_buildings.gpkg",
            "parks_gardens": "data/bari/300m/bari_greenery.gpkg",
            "zones":         "data/bari/300m/bari_zones.geojson",
            "canopy_raster": "data/bari/30/bari_ndvi.tif",
            "orthophoto":    "data/bari/3trees/bari_orthophoto.tif",
            "tree_polygons": "data/bari/3trees/bari_tree_polygons.gpkg",
        },
    },

    "bolzano": {
        "query":             "Bolzano, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers": {
            "municipal": "data/bolzano/300m/parks_raw_municipal.gpkg",
            "osm":       "data/bolzano/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/bolzano/3trees/bolzano_cadastral.geojson",
            "gba":       "data/bolzano/3trees/bolzano_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/bolzano/bolzano_boundary.gpkg",
            "buildings_sn":  "data/bolzano/bolzano_buildings_streetnum.gpkg",
            "buildings":     "data/bolzano/bolzano_buildings.gpkg",
            "parks_gardens": "data/bolzano/300m/bolzano_greenery.gpkg",
            "zones":         "data/bolzano/300m/bolzano_zones.geojson",
            "canopy_raster": "data/bolzano/30/bolzano_ndvi.tif",
            "orthophoto":    "data/bolzano/3trees/bolzano_orthophoto.tif",
            "tree_polygons": "data/bolzano/3trees/bolzano_tree_polygons.gpkg",
        },
    },

    "firenze": {
        "query":             "Florence, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": "quartiere",
        "park_layers": {
            "pe_unis": "data/firenze/300m/parks_raw_pe_unis.gpkg",
            "osm":     "data/firenze/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/firenze/3trees/firenze_cadastral.geojson",
            "gba":       "data/firenze/3trees/firenze_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/firenze/firenze_boundary.gpkg",
            "buildings_sn":  "data/firenze/firenze_buildings_streetnum.gpkg",
            "buildings":     "data/firenze/firenze_buildings.gpkg",
            "parks_gardens": "data/firenze/300m/firenze_greenery.gpkg",
            "zones":         "data/firenze/300m/firenze_zones.geojson",
            "canopy_raster": "data/firenze/30/firenze_ndvi.tif",
            "orthophoto":    "data/firenze/3trees/firenze_orthophoto.tif",
            "tree_polygons": "data/firenze/3trees/firenze_tree_polygons.gpkg",
        },
    },

    "genova": {
        "query":             "Genoa, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers": {
            "pe_unis":    "data/genova/300m/parks_raw_pe_unis.gpkg",
            "area_verde": "data/genova/300m/parks_raw_area_verde.gpkg",
            "osm":        "data/genova/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/genova/3trees/genova_cadastral.geojson",
            "gba":       "data/genova/3trees/genova_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/genova/genova_boundary.gpkg",
            "buildings_sn":  "data/genova/genova_buildings_streetnum.gpkg",
            "buildings":     "data/genova/genova_buildings.gpkg",
            "parks_gardens": "data/genova/300m/genova_greenery.gpkg",
            "zones":         "data/genova/300m/genova_zones.geojson",
            "canopy_raster": "data/genova/30/genova_ndvi.tif",
            "orthophoto":    "data/genova/3trees/genova_orthophoto.tif",
            "tree_polygons": "data/genova/3trees/genova_tree_polygons.gpkg",
        },
    },

    "milano": {
        "query":             "Milan, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": "NIL",
        "park_layers": {
            "municipal": "data/milano/300m/milano_municipality_parks.gpkg",
            "osm":       "data/milano/300m/milano_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/milano/3trees/milano_cadastral.gpkg",
            "gba":       "data/milano/3trees/milano_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/milano/milano_boundary.gpkg",
            "buildings_sn":  "data/milano/300m/milano_buildings_streetnum.gpkg",
            "buildings":     "data/milano/milano_buildings.gpkg",
            "parks_gardens": "data/milano/300m/milano_greenery.gpkg",
            "zones":         "data/milano/300m/milano_zones.geojson",
            "canopy_raster": "data/milano/30/milano_ndvi.tif",
            "orthophoto":    "data/milano/3trees/milano_orthophoto.tif",
            "tree_polygons": "data/milano/3trees/milano_tree_polygons.gpkg",
        },
    },

    "napoli": {
        "query":             "Naples, Italy",
        "building_id_col":   "ID",
        "neighbourhood_col": None,
        "park_layers": {
            "pe_unis": "data/napoli/300m/parks_raw_pe_unis.gpkg",
            "osm":     "data/napoli/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/napoli/3trees/napoli_cadastral.geojson",
            "gba":       "data/napoli/3trees/napoli_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/napoli/napoli_boundary.gpkg",
            "buildings_sn":  "data/napoli/napoli_buildings_streetnum_osm.geojson",
            "buildings":     "data/napoli/napoli_buildings.gpkg",
            "parks_gardens": "data/napoli/300m/napoli_greenery.gpkg",
            "zones":         "data/napoli/300m/napoli_zones.geojson",
            "canopy_raster": "data/napoli/30/napoli_ndvi.tif",
            "orthophoto":    "data/napoli/3trees/napoli_orthophoto.tif",
            "tree_polygons": "data/napoli/3trees/napoli_tree_polygons.gpkg",
        },
    },

    "palermo": {
        "query":             "Palermo, Italy",
        "building_id_col":   "ID",
        "neighbourhood_col": None,
        "park_layers": {
            "pe_unis": "data/palermo/300m/parks_raw_pe_unis.gpkg",
            "osm":     "data/palermo/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/palermo/3trees/palermo_cadastral.geojson",
            "gba":       "data/palermo/3trees/palermo_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/palermo/palermo_boundary.gpkg",
            "buildings_sn":  "data/palermo/palermo_buildings_streetnum_osm.geojson",
            "buildings":     "data/palermo/palermo_buildings.gpkg",
            "parks_gardens": "data/palermo/300m/palermo_greenery.gpkg",
            "zones":         "data/palermo/300m/palermo_zones.geojson",
            "canopy_raster": "data/palermo/30/palermo_ndvi.tif",
            "orthophoto":    "data/palermo/3trees/palermo_orthophoto.tif",
            "tree_polygons": "data/palermo/3trees/palermo_tree_polygons.gpkg",
        },
    },

    "roma": {
        "query":             "Rome, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": "municipio",
        "park_layers": {
            "municipal": "data/roma/300m/parks_raw_municipal.gpkg",
            "osm":       "data/roma/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/roma/3trees/roma_cadastral.geojson",
            "gba":       "data/roma/3trees/roma_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/roma/roma_boundary.gpkg",
            "buildings_sn":  "data/roma/roma_buildings_streetnum.gpkg",
            "buildings":     "data/roma/roma_buildings.gpkg",
            "parks_gardens": "data/roma/300m/roma_greenery.gpkg",
            "zones":         "data/roma/300m/roma_zones.geojson",
            "canopy_raster": "data/roma/30/roma_ndvi.tif",
            "orthophoto":    "data/roma/3trees/roma_orthophoto.tif",
            "tree_polygons": "data/roma/3trees/roma_tree_polygons.gpkg",
        },
    },

    "torino": {
        "query":             "Turin, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": "circoscrizione",
        "park_layers": {
            "pe_unis":    "data/torino/300m/parks_raw_pe_unis.gpkg",
            "area_verde": "data/torino/300m/parks_raw_area_verde.gpkg",
            "osm":        "data/torino/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/torino/3trees/torino_cadastral.geojson",
            "gba":       "data/torino/3trees/torino_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/torino/torino_boundary.gpkg",
            "buildings_sn":  "data/torino/torino_buildings_streetnum.gpkg",
            "buildings":     "data/torino/torino_buildings.gpkg",
            "parks_gardens": "data/torino/300m/torino_greenery.gpkg",
            "zones":         "data/torino/300m/torino_zones.geojson",
            "canopy_raster": "data/torino/30/torino_ndvi.tif",
            "orthophoto":    "data/torino/3trees/torino_orthophoto.tif",
            "tree_polygons": "data/torino/3trees/torino_tree_polygons.gpkg",
        },
    },

    "trento": {
        "query":             "Trento, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers": {
            "municipal": "data/trento/300m/parks_raw_municipal.gpkg",
            "osm":       "data/trento/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/trento/3trees/trento_cadastral.geojson",
            "gba":       "data/trento/3trees/trento_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/trento/trento_boundary.gpkg",
            "buildings_sn":  "data/trento/trento_buildings_streetnum.gpkg",
            "buildings":     "data/trento/trento_buildings.gpkg",
            "parks_gardens": "data/trento/300m/trento_greenery.gpkg",
            "zones":         "data/trento/300m/trento_zones.geojson",
            "canopy_raster": "data/trento/30/trento_ndvi.tif",
            "orthophoto":    "data/trento/3trees/trento_orthophoto.tif",
            "tree_polygons": "data/trento/3trees/trento_tree_polygons.gpkg",
        },
    },

    "verona": {
        "query":             "Verona, Italy",
        "building_id_col":   "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers": {
            "pe_unis": "data/verona/300m/parks_raw_pe_unis.gpkg",
            "osm":     "data/verona/300m/parks_raw_osm.gpkg",
        },
        "building_layers": {
            "cadastral": "data/verona/3trees/verona_cadastral.geojson",
            "gba":       "data/verona/3trees/verona_gba.geojson",
        },
        "data_sources": {
            "boundary":      "data/verona/verona_boundary.gpkg",
            "buildings_sn":  "data/verona/verona_buildings_streetnum.gpkg",
            "buildings":     "data/verona/verona_buildings.gpkg",
            "parks_gardens": "data/verona/300m/verona_greenery.gpkg",
            "zones":         "data/verona/300m/verona_zones.geojson",
            "canopy_raster": "data/verona/30/verona_ndvi.tif",
            "orthophoto":    "data/verona/3trees/verona_orthophoto.tif",
            "tree_polygons": "data/verona/3trees/verona_tree_polygons.gpkg",
        },
    },
}


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

