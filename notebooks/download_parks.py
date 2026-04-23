import requests 
from shapely.validation import make_valid

base = "https://geoservices3.civis.bz.it/geoserver/ows"

urban_green_url = (
    f"{base}?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=gvcc-BasicCore:VegeUrbGreen-Polygon"
    "&outputFormat=application/json&srsName=EPSG:4326"
)
r = requests.get(urban_green_url, verify=False)
gdf_green = gpd.read_file(io.BytesIO(r.content))
gdf_green["geometry"] = gdf_green["geometry"].apply(make_valid)
gdf_green = gdf_green[gdf_green.geometry.notna() & ~gdf_green.geometry.is_empty]

gdf_green_bz = gpd.clip(gdf_green, boundary_wgs84)
print(gdf_green.shape)
print(gdf_green.columns.tolist())
print(f"{gdf_green.to_crs('EPSG:32632').area.sum() / 10000:.1f} ha")

boundary_gdf = load_table(con, "raw", f"{CITY}_boundary")
boundary_wgs84 = boundary_gdf.to_crs(CRS_WGS).geometry.union_all()
print(boundary_gdf)

# clip to city
gdf_green_bz = gpd.clip(gdf_green, boundary_wgs84)
gdf_green_bz.columns = gdf_green_bz.columns.str.lower()
gdf_green_bz.to_file('data/bolzano/bolzano_municipality_parks.gpkg', driver='GPKG')
print(gdf_green_bz.shape)
print(f"{gdf_green_bz.to_crs('EPSG:32632').area.sum() / 10000:.1f} ha")
print(gdf_green_bz["TYPE_IT"].value_counts())