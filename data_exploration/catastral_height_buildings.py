#compute height of buildings from catastral map data
#altezza_gr = height of the building (if available)
#quota_piede = elevation of the ground floor (where the building meets the ground)
#quota_gronda = elevation of the roofline/eaves (top of the last floor, where the roof begins)


import geopandas as gpd
import pandas as pd
import numpy as np

#original catastral data
gdf = gpd.read_file("data/catastral_map.geojson")
gdf = gdf.to_crs("EPSG:32632")
gdf["area_m2"] = gdf.geometry.area
#compute height from columns - read columns first
print(gdf.head())

#print columns
print(gdf.columns)

#compute height of buldigns from quota_gron and quota_pied 
gdf["height"] = gdf["altezza_gr"].fillna(gdf["quota_gron"] - gdf["quota_pied"])

#save to new file
gdf.to_file("data/catastral_map_with_heights.geojson", driver="GeoJSON")
print("Saved: data/catastral_map_with_heights.geojson")

