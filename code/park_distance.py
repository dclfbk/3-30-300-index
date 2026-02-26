"""
1. Euclidean pre-filter (skip buildings definitely > 300m)
2. multi-source Dijkstra with 300m (one sweep from ALL parks simultaneously)
3. lookup per building from Dijkstra result dict
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

CRS_WGS    = "EPSG:4326"
CRS_METRIC = "EPSG:32632"
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
buildings    = gpd.read_file("data/bologna_buildings_final.gpkg").to_crs(CRS_METRIC)
bologna_wgs84 = bologna_boundary.to_crs(CRS_WGS).geometry.union_all()

#1. ACCESS TO PARK: gate or boundary sampling
print("Park gates and entrances from osm:")

gate_tag_list = [
    {"barrier": "gate"},
    {"entrance": "main"},
    {"entrance": "yes"},
    {"entrance": "secondary"},
    {"amenity": "park_entrance"},
]

gate_gdfs = []
for tags in gate_tag_list:
    try:
        gdf = ox.features_from_polygon(bologna_wgs84, tags=tags)
        gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
        if len(gdf) > 0:
            gate_gdfs.append(gdf[["geometry"]])
            print(f"  {tags} → {len(gdf)} points")
    except Exception as e:
        print(f"  {tags} → no results ({e})")

if gate_gdfs:
    gates = pd.concat(gate_gdfs).drop_duplicates().reset_index(drop=True)
    gates = gpd.GeoDataFrame(gates, crs=CRS_WGS).to_crs(CRS_METRIC)
    print(f"Total unique gate/entrance nodes: {len(gates)}")
else:
    gates = gpd.GeoDataFrame(geometry=[], crs=CRS_METRIC)
    print("No gate data found — will use boundary sampling.")

#2. GET PARK OR GARDEN => from osm, about 0,1km2 + public access 
print("osm parks and gardens:")
osm_gdfs = []
for val in ["park", "garden"]:
    try:
        gdf = ox.features_from_polygon(bologna_wgs84, tags={"leisure": val})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf["green_type"] = f"leisure={val}"
        cols = ["geometry", "green_type", "name"]
        if "access" in gdf.columns:
            cols.append("access")
        osm_gdfs.append(gdf[cols])
        print(f"  leisure={val}: {len(gdf)}")
    except Exception as e:
        print(f"  leisure={val}: failed ({e})")

osm = pd.concat(osm_gdfs).reset_index(drop=True)
osm = gpd.GeoDataFrame(osm, crs=CRS_WGS)

if "access" in osm.columns:
    osm = osm[~osm["access"].isin(["private", "no"])]

osm = osm.to_crs(CRS_METRIC)
osm["area_km2"] = osm.geometry.area / 1e6
osm = osm[osm["area_km2"] >= 0.01].reset_index(drop=True)
osm["source"]  = "OSM"
osm["park_id"] = osm.index
print(f"{len(osm)} parks/gardens >= 0.01 km²")

#3. ASSIGN GATES TO PARKS (if gate data exists)
if len(gates) > 0:
    gates_in_parks = gpd.sjoin(
        gates, osm[["park_id", "name", "geometry"]],
        how="left", predicate="within"
    )
    gates_in_parks = gates_in_parks[~gates_in_parks["park_id"].isna()]
    parks_with_gates = gates_in_parks["park_id"].nunique()
    print(f"\nParks with ≥1 gate: {parks_with_gates}/{len(osm)} "
          f"({parks_with_gates/len(osm)*100:.0f}%)")
    gate_counts = gates_in_parks.groupby("park_id").size().reset_index(name="n_gates")
    osm = osm.merge(gate_counts, on="park_id", how="left")
    osm["n_gates"] = osm["n_gates"].fillna(0).astype(int)
    print(f"Parks with 0 gates: {(osm['n_gates']==0).sum()} → boundary sampling")
else:
    osm["n_gates"] = 0
    gates_in_parks = gpd.GeoDataFrame()
    print("No gate data → all parks use boundary sampling.")

#4. BUILDINGS 
print("Buildings:")

buildings = buildings[buildings.geometry.notna()].copy()
buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
buildings = buildings.reset_index(drop=True)
buildings["building_id"] = buildings.index
buildings["centroid_x"]  = buildings.geometry.centroid.x
buildings["centroid_y"]  = buildings.geometry.centroid.y
print(f"  {len(buildings):,} building footprints")

#5 NETWORK DISTANCE: 
#network type: walk
#intermediate points: handled by Dijkstra
#create undirected graph for Dijkstra (since parks can be accessed from any direction)
#cKDTree for fast nearest node lookup (for both park access points and building centroids)
#pre-filter: Euclidean < 300m to any park

print("Street network from osm:")
G = ox.graph_from_polygon(bologna_wgs84, network_type="walk", simplify=True)
G = ox.project_graph(G, to_crs=CRS_METRIC)
G_undirected = G.to_undirected()
nodes_gdf, _ = ox.graph_to_gdfs(G)
nodes_xy = np.array([[n.x, n.y] for n in nodes_gdf.geometry])
node_ids= nodes_gdf.index.to_numpy()
node_kdtree= cKDTree(nodes_xy)
print(f"{len(G.nodes):,} nodes, {len(G.edges):,} edges")

#6 ACCESS POINTS: either gates or boundary sampling
print("Building park access points:")

def get_park_access_points(park_geom, park_gates_gdf=None, n_sample=8):
    if park_gates_gdf is not None and len(park_gates_gdf) > 0:
        return [(g.x, g.y) for g in park_gates_gdf.geometry]
    boundary = park_geom.boundary
    return [
        boundary.interpolate(i / n_sample, normalized=True).coords[0]
        for i in range(n_sample)
    ]

all_access_pts  = []
all_access_pids = []

for _, park_row in osm.iterrows():
    pid = park_row["park_id"]
    pg  = gates_in_parks[gates_in_parks["park_id"] == pid] \
          if (park_row["n_gates"] > 0 and len(gates_in_parks) > 0) else None
    pts = get_park_access_points(park_row.geometry, pg)
    all_access_pts.extend(pts)
    all_access_pids.extend([pid] * len(pts))

all_access_pts  = np.array(all_access_pts)
all_access_pids = np.array(all_access_pids)
print(f"{len(all_access_pts)} total access points across {len(osm)} parks")

#7 EUCLIDEAN PRE-FILTER 
print("Euclidean pre-filter(300m):")
park_kdtree = cKDTree(all_access_pts)
building_centroids = np.column_stack([
    buildings["centroid_x"].values,
    buildings["centroid_y"].values,
])
euc_dists, _ = park_kdtree.query(building_centroids, k=1, workers=-1)
candidate_mask = euc_dists < 300
print(f"Skipped:{(~candidate_mask).sum():,} ({(~candidate_mask).mean()*100:.1f}%)")
candidates = buildings[candidate_mask].copy()

#8 SNAP ACCESS POINTS TO NEAREST NODES
_, nearest_idx   = node_kdtree.query(all_access_pts, k=1)
snapped_node_ids = node_ids[nearest_idx]
source_nodes     = list(set(snapped_node_ids.tolist()))

node_to_park = {}
for node_id, pid in zip(snapped_node_ids, all_access_pids):
    if node_id not in node_to_park:
        node_to_park[node_id] = pid

print(f"{len(source_nodes)} unique network source nodes")

#9 DIJKSTRA
# #in order to get the nearest park for each node
#Dijkstra create a dict with node as key and distance to nearest park as value
#path_to_park dict has node as key and the path to nearest park as value (list of nodes)

print("Multi-source Dijkstra (cutoff=300m):")
dist_to_park, path_to_park = nx.multi_source_dijkstra(
    G_undirected, sources=source_nodes, weight="length", cutoff=300,
)
nearest_park_source = {node: path[0] for node, path in path_to_park.items()}
print(f"  {len(dist_to_park):,} nodes reachable within 300m")

#assign distances to buildings:
print("assigning distances to candidate buildings:")

def snap_building_to_network(geom, node_kdtree, node_ids, nodes_xy):
    boundary = geom.boundary
    n_pts = min(max(4, int(boundary.length / 5)), 16)
    pts = np.array([
        boundary.interpolate(i / n_pts, normalized=True).coords[0]
        for i in range(n_pts)
    ])
    _, idx = node_kdtree.query(pts, k=1)
    dists  = np.linalg.norm(pts - nodes_xy[idx], axis=1)
    return node_ids[idx[np.argmin(dists)]]

building_results = []

for _, bldg in tqdm(candidates.iterrows(), total=len(candidates), desc="candidates"):
    bldg_node = snap_building_to_network(
        bldg.geometry, node_kdtree, node_ids, nodes_xy)
    net_dist = dist_to_park.get(bldg_node)

    if net_dist is not None:
        src_node  = nearest_park_source.get(bldg_node)
        park_id   = node_to_park.get(src_node)
        park_info = osm[osm["park_id"] == park_id].iloc[0] if park_id is not None else None
        park_name = park_info["name"] if park_info is not None else None
    else:
        park_id = park_name = None

    building_results.append({
        "building_id":        bldg["building_id"],
        "geometry":           bldg.geometry,
        "nearest_park_id":    park_id,
        "nearest_park_name":  park_name,
        "network_dist_m":     round(net_dist, 1) if net_dist is not None else None,
        "within_300m":        net_dist is not None and net_dist <= 300,
        "euclidean_filtered": True,
    })

for _, bldg in buildings[~candidate_mask].iterrows():
    building_results.append({
        "building_id":        bldg["building_id"],
        "geometry":           bldg.geometry,
        "nearest_park_id":    None,
        "nearest_park_name":  None,
        "network_dist_m":     None,
        "within_300m":        False,
        "euclidean_filtered": False,
    })

results_gdf = gpd.GeoDataFrame(building_results, crs=CRS_METRIC)

#reuslts summary 
print(f"Results: {len(results_gdf):,} buildings")
print(f"Within 300m:  {results_gdf['within_300m'].sum():>8,} ({results_gdf['within_300m'].mean()*100:.1f}%)")
print(f"beyond 300m:  {(~results_gdf['euclidean_filtered']).sum():>8,} ({(~results_gdf['euclidean_filtered']).mean()*100:.1f}%)")
valid = results_gdf["network_dist_m"].dropna()
print(f"Median dist:  {valid.median():>7.0f} m")
print(f"Mean dist:    {valid.mean():>7.0f} m")
results_gdf.to_file("bologna_park_access.geojson", driver="GeoJSON")
results_gdf.drop(columns="geometry").to_csv("bologna_park_access.csv", index=False)
print("Saved: bologna_park_access.geojson + bologna_park_access.csv")


#fig
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle("Network Distance to Nearest Park — Bologna", fontsize=13)
ax = axes[0]
results_gdf.plot(
    ax=ax, column="network_dist_m", cmap="RdYlGn_r", legend=True,
    missing_kwds={"color": "lightgrey", "label": "> 300m or no path"},
    legend_kwds={"label": "Network distance (m)", "shrink": 0.6},
    vmin=0, vmax=300,
)
osm.plot(ax=ax, color="green", alpha=0.5, zorder=5)
ax.set_title("Network distance to nearest park\n(grey = beyond 300m)")
ax.axis("off")

ax2 = axes[1]
ax2.hist(valid, bins=50, color="#3498db", alpha=0.8, edgecolor="white")
ax2.axvline(300, color="red", linestyle="--", linewidth=2, label="300m threshold")
ax2.axvline(valid.median(), color="orange", linestyle="--",
            linewidth=1.5, label=f"Median {valid.median():.0f}m")
ax2.set_xlabel("Network distance to nearest park (m)")
ax2.set_ylabel("Number of buildings")
ax2.set_title("Distance distribution")
ax2.legend()
within_pct = results_gdf["within_300m"].mean() * 100
ax2.text(0.97, 0.95, f"{within_pct:.1f}% of all buildings\nwithin 300m",
         transform=ax2.transAxes, ha="right", va="top", fontsize=11, color="green",
         bbox=dict(facecolor="white", edgecolor="green", alpha=0.8))

plt.tight_layout()
plt.savefig("bologna_park_network_distance.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved: bologna_park_network_distance.png")