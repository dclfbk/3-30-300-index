import osmnx as ox
import geopandas as gpd

#always simply change the name, when testing new city
city = "milano"

print(f"{city} boundary:")
boundary= ox.geocode_to_gdf("Milano, Italy")
boundary.to_file(f"data/{city}_boundary.gpkg", driver="GPKG")
print(f"saved - data/{city}_boundary.gpkg")

print("osm buildings:")
boundary_wgs84 = boundary.to_crs("EPSG:4326").geometry.union_all()
gdf_osm = ox.features_from_polygon(boundary_wgs84, tags={"building": True})
gdf_osm = gdf_osm[gdf_osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

# Keep only useful columns to reduce file size
keep_cols = ["geometry", "height", "building:levels", "building"]
keep_cols = [c for c in keep_cols if c in gdf_osm.columns]
gdf_osm = gdf_osm[keep_cols].reset_index(drop=True)

gdf_osm.to_file(f"data/osm_buildings_{city}.gpkg", driver="GPKG")
print(f"saved: data/osm_buildings_{city}.gpkg  ({len(gdf_osm):,} buildings)")
