#==============================
CITY = "bologna"
#==============================

import warnings
warnings.filterwarnings("ignore")
import os, sys, tempfile, subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box, mapping
from shapely.strtree import STRtree
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

RESULTS_DIR = Path("/tmp/viewshed_chunks")
RESULTS_DIR.mkdir(exist_ok=True)
CHUNK_SIZE_M = 500

ROOT_PATH = "/workspace/3-30-300-index/"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

from config.config import (
    EUCLID_CUTOFF, connect_db, gdf_to_duckdb, load_table, gpkg_to_duckdb,
    get_city_cfg, get_output_paths, DB_PATH, GH_PORT, CRS_WGS, CRS_METRIC, WALKING_DISTANCE,
)
con = connect_db()
cfg   = get_city_cfg(CITY)
paths = get_output_paths(CITY)
CRS   = CRS_METRIC

MIN_SCORE  = 0.3 #min probability from deep forest outputs to consider 
FLOOR_HEIGHT_M  = 3.0 #average floor height, used to compute compliance for each floor x civic number 
OBSERVER_HEIGHT_M = 1.6 
DEFAULT_TREE_HEIGHT_M = cfg.get("avg_tree_height_m", 5.0) # average tree height from ETH 2020 dataset 
BUFFER_M = 30.0 # buffer around building footprints to consider for canopy extraction
COMPLIANCE_THRESHOLD  = 3 
RESOLUTION_M  = 0.5 
AVG_BUILDING_HEIGHT_M = 10.0

print(f"City: {CITY}  |  DB: {DB_PATH}")

boundary  = load_table(con, "raw", f"{CITY}_boundary")
buildings = load_table(con, "raw", f"{CITY}_buildings")   # needs height column
civici    = load_table(con, "raw", f"{CITY}_buildings_sn")
trees     = load_table(con, "raw", f"tree_polygons_{CITY}")  # crown polygons + score + height_m

buildings = buildings.to_crs(CRS) #we do project each dataset to CRS in this case eu italy: EPSG:32632
civici    = civici.to_crs(CRS)
trees     = trees.to_crs(CRS)

#filter trees - deeppforest get polygons w. give probability score 
trees["source"] = "deepforest"
trees = trees[trees["score"] >= MIN_SCORE].reset_index(drop=True)
trees["centroid"] = trees.geometry.centroid
trees["tree_id"]  = trees.index

tree_centroids_gs = gpd.GeoSeries(trees["centroid"], crs=CRS)
tree_array        = tree_centroids_gs.values #we need to get the underlying numpy array of shapely geometries for STRtree
tree_ids          = trees["tree_id"].values # we will use this to map back from STRtree results to tree IDs

#STR is needed to efficiently query which trees fall within each viewshed polygon
tree_idx = STRtree(tree_array) 

print(f"buildings: {len(buildings):,}  |  civici: {len(civici):,}  |  trees: {len(trees):,}")

#transform building footprints with height attributes into a raster DSM (digital surface model) 
#where pixel values represent height. We will use this DSM to query building heights when computing viewsheds later on.
#in our "aritfical DSM" we only have building heights
# the rest of the pixels (non-building areas) will be 0 = simplification

def buildings_to_dsm(buildings: gpd.GeoDataFrame, output_path: str, resolution: float = RESOLUTION_M) -> str | None:
    buildings = buildings[buildings.geometry.notna() & buildings.geometry.is_valid]
    if buildings.empty:
        return None


    minx, miny, maxx, maxy = buildings.total_bounds
    pad = 50 #ensure we capture the full extent of buildings at the edges
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    width  = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = (
        (mapping(geom), h)
        for geom, h in zip(buildings.geometry, buildings["height"])
        if geom is not None and h > 0
    )
    dsm = rasterize(shapes, out_shape=(height, width), transform=transform,
                    fill=0.0, dtype="float32",
                    merge_alg=rasterio.enums.MergeAlg.replace)

    with rasterio.open(output_path, "w", driver="GTiff",
                       height=height, width=width, count=1,
                       dtype="float32", crs=buildings.crs, transform=transform) as dst:
        dst.write(dsm, 1)

    print(f"DSM written: {output_path}  ({width}×{height} px)")
    return output_path


#using gdal func => meaning we need DSM raster data
#also note that gdal_viewshed only accepts a single observer point
#it returns a raster where pixel values indicate visibility (255 = visible, 0 = not visible).
#we sample this raster at the tree centroid locations to determine which trees are visible from the observer point.
#final output is a boolean mask (numpy array) where True means the tree is visible from the observer point,
#we transfer it back to gdf - this is the core function we will call for each observer point (civico) to determine which trees are visible from that point, given the DSM of the buildings.

def gdal_viewshed_visible_trees(
    observer_x: float,
    observer_y: float,
    dsm_path: str,
    tree_centroids: gpd.GeoSeries,
    observer_z: float = OBSERVER_HEIGHT_M,
    target_height: float = OBSERVER_HEIGHT_M,
    max_distance: float = BUFFER_M,
) -> np.ndarray:
    """Return bool mask — True where tree centroid is visible from observer."""
    out_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    try:
        cmd = [
            "gdal_viewshed",
            "-ox", str(observer_x), "-oy", str(observer_y),
            "-oz", str(observer_z), "-tz", str(target_height),
            "-md", str(max_distance),
            "-f", "GTiff", "-vv", "255", "-iv", "0", "-ov", "0",
            dsm_path, out_tif,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True) #add subprocess to run gdal_viewshed with the specified parameters, outputting to a temporary GeoTIFF
        except subprocess.CalledProcessError:
            return np.zeros(len(tree_centroids), dtype=bool) # if gdal_viewshed fails, we assume no trees are visible

        coords = [(pt.x, pt.y) for pt in tree_centroids]
        with rasterio.open(out_tif) as vs:
            # vectorised sample — replaces the Python loop => we sample the viewshed raster at all tree centroid locations in one go, getting an array of visibility values (255 or 0)
            samples = np.fromiter((v[0] for v in vs.sample(coords)), dtype=np.uint8, count=len(coords))

        return samples == 255

    finally:
        if os.path.exists(out_tif):
            os.remove(out_tif)

print("Viewshed helper ready")

# 1. Perform the spatial join
civici = civici.drop(columns=["index_right"], errors="ignore")
civici = gpd.sjoin(civici, buildings[["height", "geometry"]], how="left", predicate="within")
civici = civici.drop_duplicates(subset="civkey")

# 2. Normalize the height column name BEFORE defining keep_cols
# After sjoin, if 'buildings' had 'height', the result usually has 'height' unless there's a conflict.
if "height_right" in civici.columns:
    civici.rename(columns={"height_right": "height"}, inplace=True)
elif "height_left" in civici.columns:
    civici.rename(columns={"height_left": "height"}, inplace=True)
    
    # Fallback if somehow neither exists (shouldn't happen based on your sjoin args)
    if "height" not in civici.columns:
        print(f"Warning: No height column found after join. Available: {list(civici.columns)}")
        civici["height"] = AVG_BUILDING_HEIGHT_M 

# 3. Define keep_cols with the CORRECTED name ("height", not "height_right")
keep_cols = ["civkey", "indirizzo_completo", "building_id", "geometry", "height"]

# 4. Select columns
civici = civici[keep_cols]

# 5. Finalize calculations
civici["height"] = civici["height"].fillna(AVG_BUILDING_HEIGHT_M).replace(0, AVG_BUILDING_HEIGHT_M)
civici["floor_count"] = (civici["height"] / FLOOR_HEIGHT_M).astype(int).clip(lower=1)
civici["observer_z"] = OBSERVER_HEIGHT_M 


print(f"Civici ready: {len(civici):,}")

def process_civico(civ: dict, tree_idx, tree_array, tree_ids, dsm_path):
    """civ is a plain dict with keys: civkey, indirizzo_completo, building_id,
    floor_count, x, y, geometry."""
    ox, oy         = civ["x"], civ["y"] #observer coordinates 
    geom           = civ["geometry"] #the observer point geometry 
    candidate_idxs = tree_idx.query(geom.buffer(BUFFER_M)) #query the STRtree spatial index to get indices of trees whose centroids fall within the buffer around the observer point
    n_nearby       = len(candidate_idxs) # of trees near enough to potentially be visible 

    base = {
        "civkey":             civ["civkey"],
        "indirizzo_completo": civ["indirizzo_completo"],
        "building_id":        civ["building_id"],
        "geometry":           geom,
    }

    if n_nearby == 0:
        return [{**base, "floor": f, "observer_z_m": (f * FLOOR_HEIGHT_M) + 1.5,
                 "n_trees_nearby": 0, "n_trees_visible": 0,
                 "visible_tree_ids": [], "compliant": False}
                for f in range(civ["floor_count"])]

    candidate_centroids = gpd.GeoSeries(tree_array[candidate_idxs], crs=CRS)
    candidate_tree_ids  = tree_ids[candidate_idxs]

    rows = []
    for floor_idx in range(civ["floor_count"]):
        observer_z   = (floor_idx * FLOOR_HEIGHT_M) + 1.5
        visible_mask = gdal_viewshed_visible_trees(
            observer_x=ox, observer_y=oy, observer_z=observer_z,
            tree_centroids=candidate_centroids, dsm_path=dsm_path,
        )
        visible_ids = candidate_tree_ids[visible_mask].tolist()
        rows.append({**base, "floor": floor_idx, "observer_z_m": observer_z,
                     "n_trees_nearby": n_nearby, "n_trees_visible": int(visible_mask.sum()),
                     "visible_tree_ids": visible_ids,
                     "compliant": int(visible_mask.sum()) >= COMPLIANCE_THRESHOLD})
    return rows


def process_chunk_worker(civici_dicts, tile_dsm, tree_idx, tree_array, tree_ids):
    """Receives plain dicts — fully picklable across processes."""
    rows = []
    for civ in civici_dicts:
        rows.extend(process_civico(civ, tree_idx, tree_array, tree_ids, tile_dsm))
    return rows

print("Workers defined")


minx, miny, maxx, maxy = civici.total_bounds
chunk_id = 0

N_WORKERS = min(os.cpu_count(), 4)  # cap: each worker spawns gdal_viewshed subprocesses

from concurrent.futures import ThreadPoolExecutor, as_completed
      

for x0 in tqdm(np.arange(minx, maxx, CHUNK_SIZE_M), desc="Chunks"):
    for y0 in np.arange(miny, maxy, CHUNK_SIZE_M):
        tile           = box(x0, y0, x0 + CHUNK_SIZE_M, y0 + CHUNK_SIZE_M)
        civici_tile    = civici[civici.geometry.within(tile)]
        buildings_tile = buildings[buildings.geometry.intersects(tile.buffer(BUFFER_M))]

        if civici_tile.empty:
            continue

        tile_dsm = f"/tmp/dsm_{chunk_id}.tif"
        if buildings_to_dsm(buildings_tile, tile_dsm) is None:
            chunk_id += 1
            continue

        # convert to plain dicts — itertuples namedtuples are not picklable
        tile_records = civici_tile[[
            "civkey", "indirizzo_completo", "building_id", "floor_count"
        ]].copy()
        tile_records["x"] = civici_tile.geometry.x
        tile_records["y"] = civici_tile.geometry.y
        tile_records["geometry"] = civici_tile.geometry.values
        tile_dicts   = tile_records.to_dict(orient="records")
        batch_size   = max(1, len(tile_dicts) // N_WORKERS)
        batches      = [tile_dicts[i:i+batch_size] for i in range(0, len(tile_dicts), batch_size)]

        all_rows = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [
                pool.submit(process_chunk_worker, batch, tile_dsm, tree_idx, tree_array, tree_ids)
                for batch in batches
            ]
            for f in as_completed(futures):
                all_rows.extend(f.result())

        if all_rows:
            chunk_gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs=CRS)
            chunk_gdf.to_parquet(RESULTS_DIR / f"chunk_{chunk_id:04d}.parquet")

        os.remove(tile_dsm)
        chunk_id += 1

print(f"Done — {chunk_id} chunks written")

all_chunks = sorted(RESULTS_DIR.glob("*.parquet"))
result_gdf = gpd.GeoDataFrame(
    pd.concat([gpd.read_parquet(f) for f in all_chunks]),
    geometry="geometry", crs=CRS_METRIC,
)

# per-civico: compliant if any floor is compliant
civico_compliance = (
    result_gdf.groupby("civkey")["compliant"]
    .any()
    .rename("any_floor_compliant")
    .reset_index()
)
result_gdf = result_gdf.merge(civico_compliance, on="civkey", how="left")

rename = {"indirizzo_completo": "address"}
if cfg.get("neighbourhood_col") and cfg["neighbourhood_col"] in result_gdf.columns:
    rename[cfg["neighbourhood_col"]] = "neighbourhood"
result_gdf = result_gdf.rename(columns=rename)

# WGS84 output
out_wgs = result_gdf.to_crs(CRS_WGS)
out_wgs["longitude"] = out_wgs.geometry.x
out_wgs["latitude"]  = out_wgs.geometry.y

final_cols = [
    "civkey", "address", "longitude", "latitude", "building_id",
    "floor", "observer_z_m", "n_trees_nearby", "n_trees_visible",
    "visible_tree_ids", "compliant", "any_floor_compliant", "geometry",
]
out_wgs = out_wgs[[c for c in final_cols if c in out_wgs.columns]]

# ── persist ───────────────────────────────────────────────────────────────────
con = connect_db()
gdf_to_duckdb(con, result_gdf, f"rule_3.{CITY}_results")

gpkg_path    = paths["gpkg"].replace(".gpkg", "_rule3.gpkg")
parquet_path = paths["parquet"].replace(".parquet", "_rule3.parquet")

result_gdf.to_file(gpkg_path, driver="GPKG", layer="civici")
out_wgs.to_parquet(parquet_path, index=False)

con.execute(f"""
    CREATE OR REPLACE VIEW rule_3.{CITY}_geoparquet AS
    SELECT * FROM read_parquet('{parquet_path}')
""")
print(f"{CITY} rule_3 saved => DuckDB, GeoPackage, Parquet")
con.close()





