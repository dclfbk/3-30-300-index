import osmnx as ox
import geopandas as gpd

print("bologna boundary:")
bologna_boundary = ox.geocode_to_gdf("Bologna, Italy")
bologna_boundary.to_file("data/bologna_boundary.gpkg", driver="GPKG")
print(f"saved - data/bologna_boundary.gpkg")

print("osm buildings:")
bologna_wgs84 = bologna_boundary.to_crs("EPSG:4326").geometry.union_all()
gdf_osm = ox.features_from_polygon(bologna_wgs84, tags={"building": True})
gdf_osm = gdf_osm[gdf_osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

# Keep only useful columns to reduce file size
keep_cols = ["geometry", "height", "building:levels", "building"]
keep_cols = [c for c in keep_cols if c in gdf_osm.columns]
gdf_osm = gdf_osm[keep_cols].reset_index(drop=True)

gdf_osm.to_file("data/osm_buildings_bologna.gpkg", driver="GPKG")
print(f"saved: data/osm_buildings_bologna.gpkg  ({len(gdf_osm):,} buildings)")
