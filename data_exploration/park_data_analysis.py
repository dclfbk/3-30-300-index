import pandas as pd
import geopandas as gpd

DBSN_GDB_PATH ="/Users/terezasaskova/Downloads/BO_dbsn_2025-07-28/Bologna_dbsn.gdb"
print("Loading parks from DBSN:")
CRS_WGS    = "EPSG:4326"
CRS_METRIC = "EPSG:32632"

VALHALLA = "https://valhalla1.openstreetmap.de"
WALKING_DISTANCE = 300
MIN_PARK_AREA_KM2 = 0.01
DBSN_GDB_PATH ="/Users/terezasaskova/Downloads/BO_dbsn_2025-07-28/Bologna_dbsn.gdb"

bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs(CRS_METRIC)
buildings        = gpd.read_file("data/bologna_buildings_final.gpkg").to_crs(CRS_METRIC) #from global atlas 
bologna_wgs84    = bologna_boundary.to_crs(CRS_WGS).geometry.union_all()


parks = gpd.read_file(DBSN_GDB_PATH).to_crs("EPSG:32632")
parks.head()


ar_vrd = gpd.read_file(DBSN_GDB_PATH, layer="ar_vrd")
print(ar_vrd.shape)
print(ar_vrd.columns.tolist())
print(ar_vrd.dtypes)
print(ar_vrd.head())

# see what type codes exist
for col in ar_vrd.columns:
    if ar_vrd[col].dtype == object or ar_vrd[col].nunique() < 30:
        print(f"\n{col}:\n{ar_vrd[col].value_counts()}")


print("Loading parks from DBSN:")
parks_raw = gpd.read_file(DBSN_GDB_PATH, layer="ar_vrd").to_crs(CRS_METRIC)#from ar_vrd - which is the land use layer, we will filter for parks (01)

parks = parks_raw[parks_raw["ar_vrd_ty"] == "01"].copy() #01 = park
parks["area_km2"] = parks.geometry.area / 1e6
parks = parks[parks["area_km2"] >= MIN_PARK_AREA_KM2].reset_index(drop=True)
parks = parks.to_crs(CRS_METRIC)           # reproject first
parks["area_km2"] = parks.geometry.area / 1e6   # then compute area
parks = parks[parks["area_km2"] >= MIN_PARK_AREA_KM2].reset_index(drop=True)
parks = gpd.clip(parks, bologna_boundary).reset_index(drop=True)
parks["park_id"] = parks.index

print(f"{len(parks)} parks from DBSN with area >= {MIN_PARK_AREA_KM2} km²")
