import warnings
warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd

CRS_WGS = "EPSG:4326"
CRS_METRIC = "EPSG:32632"
MIN_PARK_AREA_KM2 = 0.01
DBSN_GDB_PATH = "/Users/terezasaskova/Downloads/BO_dbsn_2025-07-28/Bologna_dbsn.gdb"
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
boundary_area_km2 = bologna_boundary.geometry.area.sum() / 1e6

#1. DBSN parks 
#================================================================================
print("DBSN")
ar_vrd = gpd.read_file(DBSN_GDB_PATH, layer="ar_vrd").to_crs(CRS_METRIC)
print(f"ar_vrd layer: {ar_vrd.shape[0]:,} features, columns: {ar_vrd.columns.tolist()}")
print("ar_vrd_ty value counts:")
print(ar_vrd["ar_vrd_ty"].value_counts().to_string())

parks_dbsn = (
    ar_vrd
    .query('ar_vrd_ty == "01" or ar_vrd_ty == "95"')
    .copy()
    .assign(area_km2=lambda df: df.geometry.area / 1e6)
    .loc[lambda df: df.area_km2 >= MIN_PARK_AREA_KM2]
    .pipe(gpd.clip, bologna_boundary)
    .reset_index(drop=True)
    .assign(
        park_id=lambda df: df.index,
        source="DBSN",
    )
)

parks_all = (
    ar_vrd
    .query('ar_vrd_ty == "01" or ar_vrd_ty == "95"')
    .copy()
    .assign(area_km2=lambda df: df.geometry.area / 1e6)
    .pipe(gpd.clip, bologna_boundary)
    .reset_index(drop=True)
    .assign(
        park_id=lambda df: df.index,
        source="DBSN",
    )
)

total_dbsn_bigparks_km2 = parks_dbsn.area_km2.sum()
total_dbsn_smallparks_km2 = parks_all.area_km2.sum()

print(f"DBSN parks (≥ {MIN_PARK_AREA_KM2} km², clipped to boundary): {len(parks_dbsn)}")
print(f"total green area - parks above 0,01 km2 : {total_dbsn_bigparks_km2:.2f} km²")
print(f"total green area  - all parks : {total_dbsn_smallparks_km2:.2f} km²")
print(f"coverage  - parks above 0.01 km2 : {total_dbsn_bigparks_km2 / boundary_area_km2 * 100:.1f}% of Bologna")
print(f"area stats (km²) - parks above 0.01 km2:\n{parks_dbsn.area_km2.describe().round(4).to_string()}")
print(f"coverage  - all parks: {total_dbsn_smallparks_km2 / boundary_area_km2 * 100:.1f}% of Bologna")
print(f"area stats (km²)  - all parks:\n{parks_all.area_km2.describe().round(4).to_string()}")

#2. OSM 
#======================================================================
#filter osm for leasure = park, garden + first without filter , later filter for public only parks, gardens....

tags = {"leisure": ["park", "garden"]}
parks_osm_raw = ox.features_from_place("Bologna, Italy", tags=tags)

parks_osm = (
    parks_osm_raw[parks_osm_raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    .copy()
    .to_crs(CRS_METRIC)
    .assign(area_km2=lambda df: df.geometry.area / 1e6)
    .pipe(gpd.clip, bologna_boundary)
    .reset_index(drop=True)
    .assign(
        park_id=lambda df: df.index,
        source="OSM",
    )
)

parks_osm_filtered = parks_osm.loc[lambda df: df.area_km2 >= MIN_PARK_AREA_KM2]

total_osm_all_km2 = parks_osm.area_km2.sum()
total_osm_big_km2 = parks_osm_filtered.area_km2.sum()
# public OSM filter
public_park_garden_osm = parks_osm_filtered.loc[
    lambda df: df["access"].isna() | ~df["access"].isin(["private", "no", "restricted"])
]

print(f"OSM parks+gardens (all): {len(parks_osm)}")
print(f"OSM parks+gardens (≥ {MIN_PARK_AREA_KM2} km²): {len(parks_osm_filtered)}")
print(f"total green area - all OSM: {total_osm_all_km2:.2f} km²")
print(f"total green area - OSM ≥ 0.01 km²: {total_osm_big_km2:.2f} km²")
print(f"coverage - all OSM: {total_osm_all_km2 / boundary_area_km2 * 100:.1f}% of Bologna")
print(f"coverage - OSM ≥ 0.01 km²: {total_osm_big_km2 / boundary_area_km2 * 100:.1f}% of Bologna")
print(f"area stats (km²) - all OSM:\n{parks_osm.area_km2.describe().round(4).to_string()}")
print(f"public only OSM parks and gardens: {len(public_park_garden_osm)}")
print("DBSN vs OSM comparison")

#COMPARE
#================================================================
comparison = pd.DataFrame({
    "source": ["DBSN", "DBSN", "OSM", "OSM"],
    "filter": [f"≥{MIN_PARK_AREA_KM2} km²", "all", f"≥{MIN_PARK_AREA_KM2} km²", "all"],
    "count": [len(parks_dbsn), len(parks_all), len(parks_osm_filtered), len(parks_osm)],
    "total_km2": [total_dbsn_bigparks_km2, total_dbsn_smallparks_km2, total_osm_big_km2, total_osm_all_km2],
    "pct_bologna": [
        total_dbsn_bigparks_km2 / boundary_area_km2 * 100,
        total_dbsn_smallparks_km2 / boundary_area_km2 * 100,
        total_osm_big_km2 / boundary_area_km2 * 100,
        total_osm_all_km2 / boundary_area_km2 * 100,
    ]
}).round(2)
print(comparison.to_string(index=False))

#plot
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (gdf, title) in zip(axes, [
    (parks_dbsn, f"DBSN parks (≥{MIN_PARK_AREA_KM2} km²)\nn={len(parks_dbsn)}"),
    (parks_osm_filtered, f"OSM parks+gardens (≥{MIN_PARK_AREA_KM2} km²)\nn={len(parks_osm_filtered)}"),
]):
    bologna_boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
    gdf.plot(ax=ax, color="#4CAF50", alpha=0.6, edgecolor="darkgreen", linewidth=0.3)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()

plt.suptitle("Green Space Coverage — Bologna", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("bologna_parks_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Map saved to bologna_parks_comparison.png")

# save
parks_all.to_file("parks_dbsn.gpkg", driver="GPKG")
print(f"Saved park layer: {len(parks_all)} parks ({total_dbsn_smallparks_km2:.2f} km²)")

