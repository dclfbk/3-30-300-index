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

# ── config ────────────────────────────────────────────────────────────────────
CRS_WGS           = "EPSG:4326"
CRS_METRIC        = "EPSG:32632"
VALHALLA          = "https://valhalla1.openstreetmap.de"
WALKING_DISTANCE  = 300        # metres
MIN_PARK_AREA_KM2 = 0.01
DBSN_GDB_PATH     = "/Users/terezasaskova/Downloads/BO_dbsn_2025-07-28/Bologna_dbsn.gdb"

# ── data ──────────────────────────────────────────────────────────────────────
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
buildings        = gpd.read_file("data/bologna_buildings_final.gpkg").to_crs(CRS_METRIC)
bologna_wgs84    = bologna_boundary.to_crs(CRS_WGS).geometry.union_all()

buildings = (buildings[buildings.geometry.notna()].copy()
             .pipe(lambda df: df[df.geometry.geom_type.isin(["Polygon", "MultiPolygon"])])
             .reset_index(drop=True))
buildings["building_id"] = buildings.index
print(f"{len(buildings):,} buildings")

# ── parks (DBSN ar_vrd, type 01 = public green area) ─────────────────────────
print("loading parks from DBSN...")
parks = (gpd.read_file(DBSN_GDB_PATH, layer="ar_vrd")
           .to_crs(CRS_METRIC)
           .pipe(lambda df: df[df["ar_vrd_ty"] == "01"].copy()))
parks["area_km2"] = parks.geometry.area / 1e6
parks = parks[parks["area_km2"] >= MIN_PARK_AREA_KM2].reset_index(drop=True)
parks = gpd.clip(parks, bologna_boundary).reset_index(drop=True)
parks["park_id"] = parks.index
print(f"  {len(parks)} parks >= {MIN_PARK_AREA_KM2} km²")

#walk network (OSMnx)
print("fetching walk network...")
G = ox.project_graph(
    ox.graph_from_polygon(bologna_wgs84, network_type="walk"),
    to_crs=CRS_METRIC
)
_, edges = ox.graph_to_gdfs(G)
edges = edges.reset_index(drop=True)
edges = edges.to_crs(CRS_METRIC)
print(f"  {len(edges):,} edges")

print("extracting park entrances...") #can be more then one per park 
park_boundaries = parks[["park_id", "geometry"]].copy()
park_boundaries["geometry"] = parks.geometry.boundary

intersected = gpd.overlay(park_boundaries, edges[["geometry"]], how="intersection")

def extract_points(geom):
    if geom.geom_type == "Point":
        return [geom]
    elif geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    elif geom.geom_type == "LineString":
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])]
    return []

#build spatial index for edges
#use strtree for faster spatial queries instead of overlay
edges_geom = edges.geometry.values
tree = STRtree(edges_geom)

rows = []
for _, park in tqdm(parks.iterrows(), total=len(parks), desc="parks"):
    boundary = park.geometry.boundary
    # find candidate edges that touch the boundary bbox
    candidates_idx = tree.query(boundary)
    for idx in candidates_idx:
        inter = boundary.intersection(edges_geom[idx])
        if inter.is_empty:
            continue
        for pt in extract_points(inter):
            rows.append({"park_id": park["park_id"], "geometry": pt})

# create GeoDataFrame of entrances - all points where park boundaries intersect walk network edges
entrances = gpd.GeoDataFrame(rows, crs=CRS_METRIC)
print(f"  {len(entrances):,} entrance points before deduplication")

# deduplicate entrances that are within 20m of each other (to avoid redundant isochrones for nearby points)
def deduplicate_points(group, min_dist=20):
    kept = []
    for pt in group.geometry:
        if all(pt.distance(k) >= min_dist for k in kept):
            kept.append(pt)
    return gpd.GeoDataFrame(
        {"geometry": kept},
        crs=group.crs
    )

entrances_dedup = (entrances.to_crs(CRS_WGS)
                   .groupby("park_id", group_keys=True)
                   .apply(deduplicate_points)
                   .reset_index(level=0)   # brings park_id back as column
                   .reset_index(drop=True))
entrances = gpd.GeoDataFrame(entrances_dedup, crs=CRS_WGS)

#Valhalla isochrones (300m pedestrian) => already computes shortest path along the walk network 
print("computing isochrones via Valhalla:") 

# Valhalla API call - to get 300m walking isochrone polygon for each entrance point
#park entrance → Valhalla isochrone → 300m polygon → which buildings fall inside
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

iso_polys = []

#loop through entrances and compute isochrones, with a small delay 
for _, row in tqdm(entrances.iterrows(), total=len(entrances), desc="isochrones"):
    try:
        poly = get_isochrone(row.geometry.x, row.geometry.y)
        iso_polys.append({"park_id": row["park_id"], "geometry": poly})
        time.sleep(0.7)  
    except Exception as e:
        print(f"  failed ({row.geometry.x:.4f}, {row.geometry.y:.4f}): {e}")

iso_gdf = gpd.GeoDataFrame(iso_polys, crs=CRS_WGS).to_crs(CRS_METRIC)
print(f"  {len(iso_gdf):,} isochrones computed")

# merge multiple entrance isochrones into one polygon per park
park_iso = (iso_gdf.dissolve(by="park_id")
                   .reset_index()[["park_id", "geometry"]])

#assign buildings 
print("assigning buildings:")
bldg_centroids = buildings[["building_id", "geometry"]].copy()
bldg_centroids["geometry"] = buildings.geometry.centroid

joined = gpd.sjoin(bldg_centroids, park_iso, how="left", predicate="within")
joined = joined[~joined.index.duplicated(keep="first")]  # keep first park if overlapping

buildings["within_300m"] = buildings["building_id"].isin(
    joined[joined["park_id"].notna()]["building_id"]
)
buildings["nearest_park_id"] = buildings["building_id"].map(
    joined.set_index("building_id")["park_id"]
)

print(f"within 300m: {buildings.within_300m.sum():,} "
      f"({buildings.within_300m.mean()*100:.1f}%)")

# save results
def main():
    buildings[["building_id", "within_300m", "nearest_park_id", "geometry"]]\
        .to_file("building_park_access.gpkg", driver="GPKG")
    park_iso.to_file("park_isochrones.gpkg", driver="GPKG")
    print("saved: building_park_access.gpkg + park_isochrones.gpkg")

if __name__ == "__main__":
    main()

#plot
fig, ax = plt.subplots(figsize=(14, 12))
buildings[~buildings.within_300m].plot(ax=ax, color="lightgrey", linewidth=0, label="beyond 300m")
buildings[buildings.within_300m].plot(ax=ax, color="#2ecc71", linewidth=0, alpha=0.7, label="within 300m")
park_iso.boundary.plot(ax=ax, color="darkgreen", linewidth=0.6, alpha=0.7)
parks.plot(ax=ax, color="green", alpha=0.3, zorder=5)
entrances.to_crs(CRS_METRIC).plot(ax=ax, color="black", markersize=3, zorder=6, label="entrances")
ax.set_title(f"{buildings.within_300m.mean()*100:.1f}% of buildings within 300m walking distance of a park")
ax.axis("off")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("bologna_park_access.png", dpi=150, bbox_inches="tight")
plt.show()

