import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox

# ── load catastral ────────────────────────────────────────────────
gdf = gpd.read_file("data/catastral_map_with_heights.geojson")
gdf = gdf.to_crs("EPSG:32632")
gdf["area_m2"] = gdf.geometry.area
# fillter only buildings with area > 5 m² to avoid noise in height calculation
gdf["height_filtered"] = gdf["height"].where(gdf["area_m2"] > 5, 0)
print(f"Buildings with height > 0: {len(gdf[gdf['height_filtered'] > 0]):,}")
print(f"Buildings with height = 0: {len(gdf[gdf['height_filtered'] == 0]):,}")

# ── OSM fetch (fixed API) ─────────────────────────────────────────
gdf_osm = ox.features_from_bbox(bbox=(11.20, 44.45, 11.45, 44.58), tags={"building": True})

# keep only polygons (not points/lines)
gdf_osm = gdf_osm[gdf_osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
print(f"OSM buildings: {len(gdf_osm):,}")

# parse heights
if "height" in gdf_osm.columns:
    gdf_osm["height_osm"] = pd.to_numeric(gdf_osm["height"], errors="coerce")
else:
    gdf_osm["height_osm"] = np.nan

if "building:levels" in gdf_osm.columns:
    gdf_osm["height_from_levels"] = pd.to_numeric(gdf_osm["building:levels"], errors="coerce") * 3.0
else:
    gdf_osm["height_from_levels"] = np.nan

gdf_osm["height_osm_best"] = gdf_osm["height_osm"].fillna(gdf_osm["height_from_levels"])
print(f"With explicit height: {gdf_osm['height_osm'].notna().sum():,}")
print(f"With levels only:     {gdf_osm['height_from_levels'].notna().sum():,}")
print(f"With any height:      {gdf_osm['height_osm_best'].notna().sum():,}")

# ── gap-fill zero-height catastral buildings ──────────────────────
gdf_zero = gdf[gdf["height_filtered"] <= 0].copy()
print(f"\nBuildings needing gap-fill: {len(gdf_zero):,}")

gdf_osm_proj = gdf_osm[["height_osm_best", "geometry"]].to_crs("EPSG:32632")
gdf_zero_proj = gdf_zero.to_crs("EPSG:32632")
gdf_zero_proj = gdf_zero_proj.set_geometry(gdf_zero_proj.geometry.centroid)

joined = gpd.sjoin(gdf_zero_proj, gdf_osm_proj, how="left", predicate="within")
joined = joined[joined["height_osm_best"].notna()]
print(f"Gap-filled via OSM: {len(joined):,}")

# ── apply fills ───────────────────────────────────────────────────
osm_fill = joined[["codice_ogg", "height_osm_best"]].drop_duplicates("codice_ogg")
gdf = gdf.merge(osm_fill, on="codice_ogg", how="left")

# priority: 1) catastral, 2) OSM, 3) volume/area
mask_osm = (gdf["height_filtered"] <= 0) & gdf["height_osm_best"].notna()
gdf.loc[mask_osm, "height_filtered"] = gdf.loc[mask_osm, "height_osm_best"]

mask_vol = (gdf["height_filtered"] <= 0) & (gdf["area_m2"] > 0)


print(f"\nFinal result:")
print(f"  With height > 0:    {(gdf['height_filtered'] > 0).sum():,}")
print(f"  Still zero/missing: {(gdf['height_filtered'] <= 0).sum():,}")
print(gdf["height_filtered"].describe())

gdf.to_file("bologna_buildings_final.gpkg", driver="GPKG")
print("\nSaved: bologna_buildings_final.gpkg")