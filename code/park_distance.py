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
GH_URL = "http://localhost:8989/route"
EUCLID_CUTOFF = 420 #euclidian distance buffer, to prefilter an make the whoel computation faster
N_WORKERS = 12   
CHECKPOINT_EVERY = 5000

#DATA
#===============================================================================
#1. boundary
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
bologna_wgs84 = bologna_boundary.to_crs(CRS_WGS).geometry.union_all()  

#2. buildings
buildings = (gpd.read_file("data/bologna_buildings_final.gpkg")
             .to_crs(CRS_METRIC)
             .loc[lambda df: df.geometry.notna()]
             .loc[lambda df: df.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
             .reset_index(drop=True))

buildings["building_id"] = buildings.index
print(f"{len(buildings):,} buildings")

#3. parks
parks = (gpd.read_file("bologna_parks_03_ha.gpkg")
         .to_crs(CRS_METRIC)
         .reset_index(drop=True)
         .assign(park_id=lambda df: df.index))


# 4. walk network
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

#6. park entrances
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

entrances_wgs = entrances.to_crs(CRS_WGS)
buildings_wgs = buildings.to_crs(CRS_WGS)
entrance_tree = STRtree(entrances.geometry.values)
results = []

#pre-convert ALL entrances to WGS84 once 
entrances_latlon = (
    entrances.geometry
    .to_crs(CRS_WGS)
    .apply(lambda p: (p.y, p.x))
    .values
)

#CENTROIDS OF THE BUILDINGS
#===============================================================================
#Pre-extract centroids once
centroids_metric = buildings.geometry.centroid
centroids_wgs = centroids_metric.to_crs(CRS_WGS)


#Pre-extract ALL arrays once outside threading
building_ids       = buildings["building_id"].values
centroids_m_arr    = centroids_metric.values
centroids_w_arr    = [(p.y, p.x) for p in centroids_wgs]
entrances_geom_arr = entrances.geometry.values        # for distance computation
entrances_park_ids = entrances["park_id"].values      # for park_id lookup
entrance_tree      = STRtree(entrances_geom_arr)

#WALK DISTANCE COMPUTATION
#===============================================================================

#run graphopper - take lon, lat adn compute shortest wlaking path
def graphhopper (orig_lat, orig_lon, dest_lat, dest_lon) -> float | None:
    params = {
        "point": [f"{orig_lat},{orig_lon}", f"{dest_lat},{dest_lon}"],
        "profile": "foot",
        "calc_points": "false",
        "instructions": "false",
    }
    try:
        r = requests.get(GH_URL, params=params, timeout=5)
        r.raise_for_status()
        return r.json()["paths"][0]["distance"]
    except Exception:
        return None

def route_one(i):
    building_id = building_ids[i]
    centroid_m = centroids_m_arr[i]
    orig_lat, orig_lon = centroids_w_arr[i]

    candidates_idx = entrance_tree.query(centroid_m.buffer(EUCLID_CUTOFF))
    if len(candidates_idx) == 0:
        return {"building_id": building_id, "walk_m": None, "park_id": None}  

    candidates = sorted([(centroid_m.distance(entrances_geom_arr[j]), j) for j in candidates_idx])  

    best_dist, best_park = None, None

    for eucl_dist, idx in candidates:
        if best_dist is not None and eucl_dist >= best_dist:
            break
        dest_lat, dest_lon = entrances_latlon[idx]
        dist = graphhopper(orig_lat, orig_lon, dest_lat, dest_lon)
        if dist is not None and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_park = entrances_park_ids[idx] 

    return {"building_id": building_id, "walk_m": best_dist, "park_id": best_park}  

#parallelize the computation
results = [None] * len(buildings)


with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(route_one, i): i for i in range(len(buildings))}
    completed = 0
    for future in tqdm(as_completed(futures), total=len(buildings), desc="routing"):
        i = futures[future]
        try:
            results[i] = future.result()
        except Exception as e:
            bldg = buildings.iloc[i]
            results[i] = {"building_id": bldg["building_id"], "walk_m": None, "park_id": None}
        
        completed += 1
        if completed % CHECKPOINT_EVERY == 0:
            pd.DataFrame([r for r in results if r is not None]).to_pickle(
                f"data/routing_checkpoint_{completed}.pkl"
            )
            print(f"  checkpoint saved at {completed}")

#save
routing_df = pd.DataFrame(results)
buildings_routed = buildings.merge(routing_df, on="building_id")
buildings_routed["within_300m"] = buildings_routed["walk_m"].fillna(float("inf")) <= WALKING_DISTANCE
# Get park_ids reachable within 300m
reachable_park_ids = buildings_routed.loc[buildings_routed["within_300m"], "park_id"].dropna().unique()

# Filter parks
parks_within_300m = parks[parks["park_id"].isin(reachable_park_ids)]

# Save both
buildings_routed.to_file("data/buildings_routed.gpkg", driver="GPKG", layer="buildings")
parks_within_300m.to_file("data/buildings_routed.gpkg", driver="GPKG", layer="parks")


print(f"buildings within {WALKING_DISTANCE}m walk of a park: "
      f"{buildings_routed['within_300m'].sum():,} "
      f"({buildings_routed['within_300m'].mean()*100:.1f}%)")