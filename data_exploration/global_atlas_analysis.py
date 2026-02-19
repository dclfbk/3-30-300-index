import geopandas as gpd
from shapely.geometry import box

BOLOGNA_BBOX = (11.20, 44.45, 11.45, 44.58)
BOLOGNA_BOX  = box(*BOLOGNA_BBOX)

gdf = gpd.read_file("./gba_data/Polygon/europe/e010_n45_e015_n40.geojson")

#eda results to csv

print(f"Total features: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(f"Bounds: {gdf.total_bounds}")  # (minx, miny, maxx, maxy)
print(f"\nFirst few rows:")
print(gdf.head())
print(f"\nSample geometry: {gdf.geometry.iloc[0]}")

with open("buildings_eda.csv", "w") as f:
    f.write("Total features,CRS,Bounds\n")
    f.write(f"{len(gdf)},{gdf.crs},{gdf.total_bounds}\n")
    f.write("\nFirst few rows:\n")
    f.write(gdf.head().to_csv(index=False))
    
save_path = "buildings_eda.csv"
print(f"EDA results saved to {save_path}")
