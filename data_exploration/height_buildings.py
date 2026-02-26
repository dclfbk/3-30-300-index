# Compare building heights from cadastral map vs OSM, and gap-fill missing/zero heights in cadastral via OSM where possible.
import geopandas as gpd
import pandas as pd
import numpy as np

print("load files:")
bologna_boundary = gpd.read_file("data/bologna_boundary.gpkg").to_crs("EPSG:32632")
global_altas          = gpd.read_file("data/global_atlas_height.geojson").to_crs("EPSG:32632")
cadastral              = gpd.read_file("data/catastral_map_with_heights.geojson").to_crs("EPSG:32632")
print(f"Boundary:   {bologna_boundary.geometry.area.sum() / 1e6:.1f} km²")
print(f"Global altas height:        {len(global_altas):,} buildings")
print(f"Cadastral:  {len(cadastral):,} buildings")

gdf = gpd.clip(global_altas, bologna_boundary)
print(f"After boundary clip: {len(gdf):,} global atlas buildings")

gdf["area_m2"] = gdf.geometry.area
gdf["height_filtered"] = gdf["height"].where(gdf["area_m2"] > 5, 0)
print(f"  height > 0: {(gdf['height_filtered'] > 0).sum():,}")
print(f"  height = 0: {(gdf['height_filtered'] == 0).sum():,}")


print(f"Final result:")
print(f"Total buildings:    {len(gdf):,}")
print(f"With height > 0:    {(gdf['height_filtered'] > 0).sum():,}")
print(f"Still zero/missing: {(gdf['height_filtered'] <= 0).sum():,}")
print(gdf["height_filtered"].describe())

gdf.to_file("bologna_buildings_final.gpkg", driver="GPKG")
print("saved: bologna_buildings_final.gpkg")