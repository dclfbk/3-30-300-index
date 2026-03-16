# GRAPHHOPPER SETUP - run before executing this script
# ===============================================================================
# Requires Java - check with: java -version
# If not installed: brew install openjdk
#
# In a separate terminal, navigate to the directory containing:
#   - graphhopper-web-11.0.jar
#   - config_bologna.yml  
#   - Bologna.osm.pbf
#
# Then run:
#   rm -rf graph-cache
#   java -jar graphhopper-web-11.0.jar server config_bologna.yml
#
# Wait for "Started server at HTTP 8989" then run this script
# ===============================================================================

import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from shapely.geometry import Point
from shapely.strtree import STRtree
import os, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

#CONFIG
#===========================================================

CRS_WGS = "EPSG:4326"
CRS_METRIC  = "EPSG:32632"
WALKING_DISTANCE  = 300 #metres - zaezr we might apply a deviation from the hard threshold...
MIN_PARK_AREA_KM2 = 0.01 #given from many previous studies, can be skipped 
EUCLID_CUTOFF = 420 #euclidian distance buffer, to prefilter an make the whole computation faster
N_WORKERS = 12   

#DATA
#===============================================================================
#1. boundary
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
bologna_wgs84 = bologna_boundary.to_crs(CRS_WGS).geometry.union_all()  

#2. buildings from open data commune bologna
buildings = (gpd.read_file("data/bologna_buildings_num.gpkg")
             .to_crs(CRS_METRIC)
             .loc[lambda df: df.geometry.notna()]
             .reset_index(drop=True))

print(f"{len(buildings):,} buildings")

# 3. parks - read ALL layers together from the start
parks = (
    pd.concat([
        gpd.read_file("data/greenery__all_size.gpkg", layer=l)
        for l in ["ar_vrd_all", "pe_unis_all", "osm_all_public"]
    ], ignore_index=True)
    .to_crs(CRS_METRIC)
    .reset_index(drop=True)
    .assign(park_id=lambda df: df.index)
)

print(f"{len(parks):,} parks total before filtering")
print(parks["layer_type"].value_counts())

# filter ar_vrd using building footprints
building_footprints = gpd.read_file("data/bologna_buildings_final.gpkg").to_crs(CRS_METRIC)
building_footprints = building_footprints.loc[
    building_footprints.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

footprint_tree = STRtree(building_footprints.geometry.values)

def buildings_inside(park_geom):
    candidates = footprint_tree.query(park_geom, predicate="intersects")
    return len(candidates)

# compute only for ar_vrd, fill 0 for others
ar_vrd_mask = parks["layer_type"] == "ar_vrd_all"
parks.loc[ar_vrd_mask, "n_buildings_inside"] = (
    parks.loc[ar_vrd_mask, "geometry"].apply(buildings_inside)
)
parks["n_buildings_inside"] = parks["n_buildings_inside"].fillna(0)

# apply filter only to ar_vrd
parks = parks[
    ~ar_vrd_mask | (parks["n_buildings_inside"] <= 6)
].reset_index(drop=True)

# reassign park_id after filtering
parks["park_id"] = parks.index

print(f"{len(parks):,} parks after filtering")
print(parks["layer_type"].value_counts())

# 4. walk network - osm networktype = walk 
G = ox.graph_from_polygon(bologna_wgs84, network_type="walk")
G = ox.project_graph(G, to_crs=CRS_METRIC)
_, edges = ox.graph_to_gdfs(G)
edges = edges.reset_index(drop=True)
print(f"{len(edges):,} edges")


#extracting points
def extract_points(geom) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection": #run along part of the boundary (LineString) and cross it at a point (Point)
        pts = []
        for g in geom.geoms:
            pts.extend(extract_points(g))
        return pts
    return []

edges_geom = edges.geometry.values
tree = STRtree(edges_geom) #spatial idx over all bounding boxes

#6.PARK ENTRANCE
#boundaries of park polygon x  walking path
#parks crossings = outer loop, edges - STRtree
rows = []
for _, park in tqdm(parks.iterrows(), total=len(parks), desc="park entrances"):
    boundary = park.geometry.boundary
    for idx in tree.query(boundary, predicate="intersects"):
        inter = boundary.intersection(edges_geom[idx])
        for pt in extract_points(inter):
            rows.append({"park_id": park["park_id"], "geometry": pt})

#create package with entrances
entrances = gpd.GeoDataFrame(rows, crs=CRS_METRIC)
print(f"{len(entrances):,} raw entrance points")

#num of parks with no entrances - assuming these are private gardens
#limitation of this approach - not all private gardens are filltered this way
missing = set(parks["park_id"]) - set(entrances["park_id"])
if missing:
    print(f"{len(missing)} parks with no entrances: {missing}")

#deduplicate — if multiple possible points of entrances are very close
#distnace set to 20 - this is not backup by literature 
def deduplicate_points(group, min_dist=20):
    kept = []
    for pt in group.geometry:
        if all(pt.distance(k) >= min_dist for k in kept):
            kept.append(pt)
    return gpd.GeoDataFrame({"geometry": kept}, crs=group.crs)

entrances = (entrances
             .groupby("park_id", group_keys=True)
             .apply(deduplicate_points)
             .reset_index(level=0)
             .reset_index(drop=True)
             .pipe(gpd.GeoDataFrame, crs=CRS_METRIC)) 

print(f"{len(entrances):,} entrance points after deduplication")

#DISTANCE COMPUTATION
#===============================================================================

#batch: nearest park entrance per building 
#spatial pre-filter: only query GH for entrances within 600m straight line
#2× walking target — avoids sending 10k useless requests

results = []

# Pre-extract ALL arrays once outside threading
buildings["building_id"] = buildings["indirizzo_completo"]

building_ids       = buildings["building_id"].values
buildings_m_arr    = buildings.geometry.values                # already points, no centroid needed
buildings_w_arr    = [(p.y, p.x) for p in buildings.geometry.to_crs(CRS_WGS)]
buildings_tree     = STRtree(buildings_m_arr)

entrances_geom_arr = entrances.geometry.values
entrances_park_ids = entrances["park_id"].values
entrances_w_arr      = [(p.y, p.x) for p in entrances.geometry.to_crs(CRS_WGS)]

#WALK DISTANCE COMPUTATION
#===============================================================================

#run graphopper - take lon, lat adn compute shortest wlaking path
# REPLACE the graphhopper function + route_one + ThreadPoolExecutor block
# with this:

ISO_URL = "http://localhost:8989/isochrone"

def get_isochrone(lat, lon, distance_m=300):
    params = {
        "point": f"{lat},{lon}",
        "profile": "foot",
        "distance_limit": distance_m,
        "type": "json",
    }
    try:
        r = requests.get(ISO_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()["polygons"][0]
    except Exception:
        return None

# One isochrone per entrance
from shapely.geometry import shape

iso_rows = []
for i in tqdm(range(len(entrances)), desc="isochrones"):
    lat, lon = entrances_w_arr[i]
    result   = get_isochrone(lat, lon, distance_m=WALKING_DISTANCE)
    if result:
        iso_rows.append({
            "park_id":      entrances_park_ids[i],
            "entrance_lat": lat,
            "entrance_lon": lon,
            "geometry":     shape(result["geometry"])
        })

isochrones = gpd.GeoDataFrame(iso_rows, crs=CRS_WGS).to_crs(CRS_METRIC)

# Dissolve per park — union all entrance isochrones for same park
park_reach = isochrones.dissolve(by="park_id").reset_index()

# Spatial join — which buildings fall inside any park's 300m reach?
buildings_gdf = gpd.GeoDataFrame(
    {"building_id": building_ids},
    geometry=buildings.geometry,
    crs=CRS_METRIC
)

joined = gpd.sjoin(buildings_gdf, park_reach, how="left", predicate="within")

# if isochrones overlap, keep the park with lowest park_id per building
joined = (joined
          .sort_values("park_id")
          .drop_duplicates(subset="building_id", keep="first"))

routing_df = joined[["building_id", "park_id", "entrance_lat", "entrance_lon"]].copy()
routing_df["within_300m"] = routing_df["park_id"].notna()

# merge with buildings
buildings_routed = buildings.merge(routing_df, on="building_id", how="left")

# fill within_300m for buildings that got no match (left join → NaN)
buildings_routed["within_300m"] = buildings_routed["within_300m"].fillna(False)

# merge park attributes
name_col = next((c for c in ["nome", "name", "descrizione"] if c in parks.columns), None)
keep = ["park_id", "area_ha", "source", "layer_type"]
if name_col:
    keep.append(name_col)
park_attrs = parks[keep].copy()
buildings_routed = buildings_routed.merge(park_attrs, on="park_id", how="left")

# rename
rename_dict = {
    "civico":             "street_number",
    "indirizzo_completo": "address",
    "layer_type":         "park_type",
    "area_ha":            "park_area_ha",
    "source":             "park_source",
}
if name_col:
    rename_dict[name_col] = "park_name"
buildings_routed = buildings_routed.rename(columns=rename_dict)

# save
reachable_park_ids = buildings_routed.loc[buildings_routed["within_300m"], "park_id"].dropna().unique()
parks_within_300m  = parks[parks["park_id"].isin(reachable_park_ids)]

buildings_routed.to_file("data/buildings_routed.gpkg", driver="GPKG", layer="buildings")
parks_within_300m.to_file("data/buildings_routed.gpkg", driver="GPKG", layer="parks")

print(f"buildings within {WALKING_DISTANCE}m walk of a park: "
      f"{buildings_routed['within_300m'].sum():,} "
      f"({buildings_routed['within_300m'].mean()*100:.1f}%)")
