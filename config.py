"""
config.py - for the multi-city 300m rule.
"""

import os
from pathlib import Path

#Paths 
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "data" / "300m"
LOGS_DIR   = ROOT_DIR / "logs"

for d in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#CRS
CRS_WGS    = "EPSG:4326"
CRS_METRIC = "EPSG:32632"   

WALKING_DISTANCE = 300      # metres:  3-30-300 rule threshold
EUCLID_CUTOFF    = 420      # metres: pre-filter before GH routing

GH_BASE_PORT = 8989         # first city gets 8989, next 8990, etc.

# ─── Cities ───────────────────────────────────────────────────────────────────

# park_layers:  ordered list of keys from data_sources to load as parks.
#               Each key must point to a .gpkg file saved by greenery_data_exploration.ipynb.
#               Layers read from each file: ["osm_public", "municipality", "dsbn_pe_unis", "dsbn_ar_vrd"] (skipped if absent).
#
# building_id_col: column in buildings_sn file used as the unique address key.
# Common values: "NUMEROCOMPLETO" (Milano, Bologna), "ID" (OSM-based files), needs to be explored otherwise.
#
# neighbourhood_col: column to use for neighbourhood grouping in stats notebook.
#                    Set to None if the source file has no neighbourhood field. - should be explore otherwise

CITIES: dict[str, dict] = {
    "bologna": {
        "query":            "Bologna, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": "quartiere",
        "park_layers":      ["parks_dsnb"],   # greenery .gpkg produced by exploration nb
        "data_sources": {
            "boundary":      "data/bologna_boundary.gpkg",
            "buildings_sn":  "data/bologna_buildings_streetnum.gpkg",
            "parks_all":    "data/300m/bologna_greenery.gpkg",
        },
    },
    "milano": {
        "query":            "Milan, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": "NIL",          # Milano neighbourhood column
        "park_layers":      ["parks_municipal"],
        "data_sources": {
            "boundary":      "data/milano_boundary.gpkg",
            "buildings_sn":  "data/300m/milano_buildings_streetnum.gpkg",
            "parks_municipal": "data/300m/milano_greenery.gpkg",
        },
    },
    "bari": {
        "query":            "Bari, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers":      ["parks_municipal"],
        "data_sources": {
            "boundary":      "data/bari_boundary.gpkg",
            "buildings_sn":  "data/bari_buildings_streetnum.gpkg",
            "parks_municipal":    "data/300m/bari_greenery.gpkg",
        },
    },
    "bolzano": {
        "query":            "Bolzano, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers":      ["parks_municipal"],
        "data_sources": {
            "boundary":      "data/bolzano_boundary.gpkg",
            "buildings_sn":  "data/bolzano_buildings_streetnum.gpkg",
            "parks_municipal": "data/300m/bolzano_municipality_parks.gpkg",
        },
    },
    "firenze": {
        "query":            "Florence, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": "quartiere",
        "park_layers":      ["pe_unis"],
        "data_sources": {
            "boundary":      "data/firenze_boundary.gpkg",
            "buildings_sn":  "data/firenze_buildings_streetnum.gpkg",
            "pe_unis":    "data/300m/firenze_pe_uins.gpkg",
        },
    },
    "genova": {
        "query":            "Genoa, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers":      ["pe_unis", "area_verde"],  # two DSNB files
        "data_sources": {
            "boundary":      "data/genova_boundary.gpkg",
            "buildings_sn":  "data/genova_buildings_streetnum.gpkg",
            "pe_unis":    "data/300m/genova_pe_unis.gpkg",
            "area_verde": "data/300m/genova_area_verde.gpkg",
        },
    },
    "napoli": {
        "query":            "Naples, Italy",
        "building_id_col":  "ID",            # OSM-based file
        "neighbourhood_col": None,
        "park_layers":      ["pe_unis"],
        "data_sources": {
            "boundary":      "data/napoli_boundary.gpkg",
            "buildings_sn":  "data/napoli_buildings_streetnum_osm.geojson",
            "pe_unis":    "data/300m/napoli_pe_unis.gpkg",
        },
    },
    "palermo": {
        "query":            "Palermo, Italy",
        "building_id_col":  "ID",
        "neighbourhood_col": None,
        "park_layers":      ["pe_unis"],
        "data_sources": {
            "boundary":      "data/palermo_boundary.gpkg",
            "buildings_sn":  "data/palermo_buildings_streetnum_osm.geojson",
            "pe_unis":    "data/300m/palermo_pe_unis.gpkg",
        },
    },
    "roma": {
        "query":            "Rome, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": "municipio",
        "park_layers":      ["parks_municipal"],
        "data_sources": {
            "boundary":      "data/roma_boundary.gpkg",
            "buildings_sn":  "data/roma_buildings_streetnum.gpkg",
            "parks_municipal": "data/300m/roma_municipality_parks.gpkg",
        },
    },
    "torino": {
        "query":            "Turin, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": "circoscrizione",
        "park_layers":      ["pe_unis", "parks_municipal"],
        "data_sources": {
            "boundary":      "data/torino_boundary.gpkg",
            "buildings_sn":  "data/torino_buildings_streetnum.gpkg",
            "pe_unis":    "data/300m/torino_pe_unis.gpkg",
            "area_verde": "data/300m/torino_area_verde.gpkg",
        },
    },
    "trento": {
        "query":            "Trento, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers":      ["parks_municipal"],
        "data_sources": {
            "boundary":      "data/trento_boundary.gpkg",
            "buildings_sn":  "data/trento_buildings_streetnum.gpkg",
            "parks_municipal": "data/300m/trento_municipality_parks.gpkg",
        },
    },
    "verona": {
        "query":            "Verona, Italy",
        "building_id_col":  "NUMEROCOMPLETO",
        "neighbourhood_col": None,
        "park_layers":      ["pe_unis"],
        "data_sources": {
            "boundary":      "data/verona_boundary.gpkg",
            "buildings_sn":  "data/verona_buildings_streetnum.gpkg",
            "pe_unis":    "data/300m/verona_pe_unis.gpkg",
        },
    },
}


def get_city_cfg(city_key: str) -> dict:
    """Return config dict for a city, raising a clear error if unknown."""
    if city_key not in CITIES:
        available = ", ".join(CITIES.keys())
        raise ValueError(f"Unknown city '{city_key}'. Available: {available}")
    return {"key": city_key, **CITIES[city_key]}

def get_gh_port(city_key: str) -> int:
    """Stable per-city GraphHopper port derived from insertion order."""
    idx = list(CITIES.keys()).index(city_key)
    return GH_BASE_PORT + idx

def get_output_paths(city_key: str) -> dict:
    """Return all output file paths for a city as a dict."""
    base = OUTPUT_DIR / city_key
    base.mkdir(parents=True, exist_ok=True)
    return {
        "gpkg":    str(base / f"{city_key}_walkability_map.gpkg"),
        "csv":     str(base / f"{city_key}_buildings_300m_indicator.csv"),
        "geojson": str(base / f"{city_key}_buildings_300m_indicator.geojson"),
        "fig_dir": str(ROOT_DIR / "fig" / city_key),
    }