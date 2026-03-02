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

# ── config ────────────────────────────────────────────────────────────────────
CRS_WGS = "EPSG:4326"
CRS_METRIC  = "EPSG:32632"
VALHALLA  = "https://valhalla1.openstreetmap.de"
WALKING_DISTANCE  = 300        # metres
MIN_PARK_AREA_KM2 = 0.01
DBSN_GDB_PATH  = "/Users/terezasaskova/Downloads/BO_dbsn_2025-07-28/Bologna_dbsn.gdb"

# 1. boundary
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
bologna_wgs84    = bologna_boundary.to_crs(CRS_WGS).geometry.union_all()  # for OSMnx

# 2. buildings
buildings = (gpd.read_file("data/bologna_buildings_final.gpkg")
             .to_crs(CRS_METRIC)
             .loc[lambda df: df.geometry.notna()]
             .loc[lambda df: df.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
             .reset_index(drop=True))
buildings["building_id"] = buildings.index
print(f"{len(buildings):,} buildings")

# 3. parks
parks = (gpd.read_file(DBSN_GDB_PATH, layer="ar_vrd")
         .to_crs(CRS_METRIC)
         .query('ar_vrd_ty == "01"')
         .assign(area_km2=lambda df: df.geometry.area / 1e6)
         .loc[lambda df: df.area_km2 >= MIN_PARK_AREA_KM2]
         .pipe(gpd.clip, bologna_boundary)
         .reset_index(drop=True)
         .assign(park_id=lambda df: df.index))
print(f"{len(parks)} parks >= {MIN_PARK_AREA_KM2} km²")

# 4. walk network
G = ox.graph_from_polygon(bologna_wgs84, network_type="walk")
G = ox.project_graph(G, to_crs=CRS_METRIC)
_, edges = ox.graph_to_gdfs(G)
edges = edges.reset_index(drop=True)
print(f"{len(edges):,} edges")

#5. park entrances
#park entrances — where park boundaries cross walk network edges
def extract_points(geom) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    if geom.geom_type == "LineString":
        c = list(geom.coords)
        return [Point(c[0]), Point(c[-1])]
    if geom.geom_type == "GeometryCollection":
        pts = []
        for g in geom.geoms:
            pts.extend(extract_points(g))
        return pts
    return []

edges_geom = edges.geometry.values
tree = STRtree(edges_geom)

rows = []
for _, park in tqdm(parks.iterrows(), total=len(parks), desc="park entrances"):
    boundary = park.geometry.boundary
    for idx in tree.query(boundary):
        edge = edges_geom[idx]
        if edge is None or edge.is_empty:
            continue
        inter = boundary.intersection(edge)
        for pt in extract_points(inter):
            rows.append({"park_id": park["park_id"], "geometry": pt})

entrances = gpd.GeoDataFrame(rows, crs=CRS_METRIC)
print(f"{len(entrances):,} raw entrance points")

#num of parks with no entrances - just to check the info
missing = set(parks["park_id"]) - set(entrances["park_id"])
if missing:
    print(f"{len(missing)} parks with no entrances: {missing}")

#deduplicate — if multiple entrances are very close 
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

#4. compute paths
#Valhalla isochrones (300m pedestrian) 
#=> already computes shortest path along the walk network 
print("computing isochrones via Valhalla:") 

# Valhalla API call - to get 300m walking isochrone polygon for each entrance point
#park entrance => Valhalla isochrone => 300m polygon => which buildings fall inside
def get_isochrone(lon, lat, distance_m=WALKING_DISTANCE):
    body = {
        "locations": [{"lon": lon, "lat": lat}],
        "costing":   "pedestrian",
        "contours":  [{"distance": distance_m / 1000}],  # km
        "polygons":  True,
        "shortest":  True,
    }
    r = requests.post(f"{VALHALLA}/isochrone", json=body, timeout=120)
    r.raise_for_status()
    return gpd.GeoDataFrame.from_features(
        r.json()["features"], crs=CRS_WGS
    ).geometry.iloc[0]

CACHE_FILE = "isochrone_cache.pkl"
cache = pickle.load(open(CACHE_FILE, "rb")) if os.path.exists(CACHE_FILE) else {}

entrances_wgs = entrances.to_crs(CRS_WGS)
iso_polys = []

CACHE_FILE = "isochrone_cache.pkl"
cache = pickle.load(open(CACHE_FILE, "rb")) if os.path.exists(CACHE_FILE) else {}

entrances_wgs = entrances.to_crs(CRS_WGS)
iso_polys = []

for _, row in tqdm(entrances_wgs.iterrows(), total=len(entrances_wgs), desc="isochrones"):
    key = (round(row.geometry.x, 6), round(row.geometry.y, 6))
    
    if key in cache:
        iso_polys.append({"park_id": row["park_id"], "geometry": cache[key]})
        continue                         #no sleep, just read from disk
    try:
        poly = get_isochrone(row.geometry.x, row.geometry.y)
        cache[key] = poly
        pickle.dump(cache, open(CACHE_FILE, "wb"))
        iso_polys.append({"park_id": row["park_id"], "geometry": poly})
        time.sleep(1.5)                  # sleep for real API calls
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("rate limited, waiting 30s")
            time.sleep(20)              #back off and the loop will retry next run
        else:
            print(f"  failed ({row.geometry.x:.5f}, {row.geometry.y:.5f}): {e}")

iso_gdf = gpd.GeoDataFrame(iso_polys, crs=CRS_WGS).to_crs(CRS_METRIC)
park_iso = gpd.overlay(parks[["park_id", "geometry"]], iso_gdf, how="intersection")
print(f"{len(park_iso):,} park isochrone polygons")

#building centroids 
print("assigning buildings")
bldg_centroids = gpd.GeoDataFrame(
    {"building_id": buildings["building_id"],
     "geometry":    buildings.geometry.centroid},
    crs=CRS_METRIC
)

#spatial join 
joined = gpd.sjoin(bldg_centroids, park_iso, how="left", predicate="within")

#resolve buildings inside multiple park isochrones → keep nearest park
dupes_mask = joined.index.duplicated(keep=False)
clean      = joined[~dupes_mask]
dupes      = joined[dupes_mask]

if not dupes.empty:
    park_centroids = parks.set_index("park_id").geometry.centroid
    def pick_nearest(grp):
        pt = grp.geometry.iloc[0]
        return grp.loc[[
            grp["park_id"].map(lambda pid: pt.distance(park_centroids[pid])).idxmin()
        ]]
    resolved = dupes.groupby(level=0, group_keys=False).apply(pick_nearest)
    joined   = pd.concat([clean, resolved])

#attach results to buildings 
park_map = joined.set_index("building_id")["park_id"]
buildings["within_300m"]     = buildings["building_id"].isin(park_map[park_map.notna()].index)
buildings["nearest_park_id"] = buildings["building_id"].map(park_map)

pct = buildings["within_300m"].mean() * 100
print(f"within 300 m: {buildings['within_300m'].sum():,}  ({pct:.1f} %)")


#plot
fig, ax = plt.subplots(figsize=(14, 12))
buildings[~buildings.within_300m].plot(ax=ax, color="lightgrey", linewidth=0, label="beyond 300m")
buildings[buildings.within_300m].plot(ax=ax, color="#2ecc71", linewidth=0, alpha=0.7, label="within 300m")
park_iso.boundary.plot(ax=ax, color="darkgreen", linewidth=0.6, alpha=0.7)
parks.plot(ax=ax, color="green", alpha=0.3, zorder=5)
entrances.plot(ax=ax, color="black", markersize=3, zorder=6, label="entrances")
ax.set_title(f"{buildings.within_300m.mean()*100:.1f}% of buildings within 300m walking distance of a park")
ax.axis("off")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("bologna_park_access.png", dpi=150, bbox_inches="tight")
plt.show()

