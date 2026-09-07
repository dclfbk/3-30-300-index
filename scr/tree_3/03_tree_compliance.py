import argparse
import warnings
warnings.filterwarnings("ignore")
import os, sys, tempfile, subprocess, shutil, time
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.config import (
    connect_db_with_retry, ROOT_PATH, gdf_to_duckdb, load_table,
    get_city_cfg, get_output_paths, DB_PATH, CRS_METRIC, CRS_WGS,
    DATA_DIR, strip_source_suffix,
)

RESULTS_DIR = Path("/tmp/viewshed_chunks")
CHUNK_SIZE_M = 500 #this is set to 500m to match the DSM tile size, but could be larger 
CSV_PATH = DATA_DIR / "rule_3_compliance_summary.csv"

sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

FLOOR_HEIGHT_M  = 3.0 #taking avg building floor height as 3m, which is a common assumption 
OBSERVER_HEIGHT_M = 1.6 #needed for gdal_viewshed 
BUFFER_M = 30.0   #buffer around each civico to search for nearby trees - taken from 3-30-300 index handbook recommendation
COMPLIANCE_THRESHOLD  = 3 #3 trees visible within 30m of a civico is considered compliant - filter in final gkpg
RESOLUTION_M  = 0.5 #needed for gdal_viewshed = it is the resolution of the DSM raster
AVG_BUILDING_HEIGHT_M = 10.0 #if height from GBA is not available 
AVG_TREE_HEIGHT_M = 12.2 #avg from all GEE canopy height data  

def _load_and_prepare_trees(con, city: str, city_metric_crs: str, force_reload: bool = False) -> gpd.GeoDataFrame:
    table_name = f"tree_polygons_{city}"
    if not force_reload:
        try:
            existing = load_table(con, "preprocessing", table_name)
            print(f"[trees] preprocessing.{table_name} found in DB ({len(existing):,} rows) — using it")
            if "height_m" not in existing.columns:
                print(f"[trees] Cached table missing 'height_m' — backfilling")
                existing = _assign_tree_height(existing, city)
                con.execute(f"CREATE SCHEMA IF NOT EXISTS preprocessing")
                gdf_to_duckdb(con, existing, f"preprocessing.{table_name}")
            return existing
        except Exception:
            pass
        try:
            trees = load_table(con, "raw", table_name)
            print(f"[trees] raw.{table_name} found in DB ({len(trees):,} rows)")
        except Exception:
            trees = None
    else:
        trees = None

    if trees is None:
        sam2_gpkg = DATA_DIR / city / f"tree_detections_zeroshot_{city}.gpkg"
        if not sam2_gpkg.exists():
            raise FileNotFoundError(
                f"No tree data found for '{city}': neither preprocessing.{table_name} "
                f"nor raw.{table_name} exist in the DB, and {sam2_gpkg} doesn't exist "
                f"either. Run sam_3_version2.py --city {city} first."
            )
        print(f"[trees] raw.{table_name} not found — loading from SAM2 gpkg: {sam2_gpkg}")
        trees = gpd.read_file(sam2_gpkg, layer="tree_crowns")
        if "df_score" in trees.columns and "score" not in trees.columns:
            trees = trees.rename(columns={"df_score": "score"})
        trees = _assign_tree_height(trees, city)

    if trees.crs is None:
        trees = trees.set_crs(CRS_WGS)
    trees = trees.to_crs(city_metric_crs)

    con.execute(f"CREATE SCHEMA IF NOT EXISTS preprocessing")
    gdf_to_duckdb(con, trees, f"preprocessing.{table_name}")
    return trees


def _assign_tree_height(trees: gpd.GeoDataFrame, city: str) -> gpd.GeoDataFrame:
    height_csv = DATA_DIR / "city_avg_tree_heights.csv"
    base_city = strip_source_suffix(city)
    heights = pd.read_csv(height_csv).set_index("city")["mean_tree_height_m"]
    height_m = float(heights.get(base_city, heights.get(city)))
    trees["height_m"] = height_m
    print(f"[trees] Using height_m={height_m:.2f} for {city}")
    return trees

_GDAL_ERROR_PRINTED = [False] 

def gdal_viewshed_visible_trees(
    observer_x: float, observer_y: float, dsm_path: str,
    tree_centroids: gpd.GeoSeries, target_heights: np.ndarray, observer_z: float,
    max_distance: float = BUFFER_M,
) -> np.ndarray:
    """target_heights = per-tree. gdal_viewshed only supports one target height per invocation, 
    so trees are grouped by distinct height value (just 1 anyways, since height_m is a uniform per-city value) and one pass runs per group."""

    coords = [(pt.x, pt.y) for pt in tree_centroids] 
    visible = np.zeros(len(tree_centroids), dtype=bool)
    unique_heights = np.unique(np.round(target_heights, 1))

    for h in unique_heights:
        height_mask = np.isclose(np.round(target_heights, 1), h) #target height is linked to avg tree height
        sub_coords = [coords[i] for i in np.where(height_mask)[0]]
        if not sub_coords:
            continue
        
        #cmd is the main gdal_viewshed command that is run for each height group. 
        #it generates a temporary raster file with the visibility results, which is then read back into memory to update the visible mask for this height group.
        out_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        try: #it takes the observer location, DSM path, target height, and other parameters to generate the viewshed raster
            cmd = [ 
                "gdal_viewshed", "-ox", str(observer_x), "-oy", str(observer_y),
                "-oz", str(observer_z), "-tz", str(float(h)), "-md", str(max_distance),
                "-f", "GTiff", "-vv", "255", "-iv", "0", "-ov", "0", dsm_path, out_tif,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                if not _GDAL_ERROR_PRINTED[0]:
                    print(f"\n[gdal_viewshed ERROR] cmd: {' '.join(cmd)}")
                    print(f"[gdal_viewshed ERROR] stderr: "
                          f"{e.stderr.decode(errors='replace') if e.stderr else '(none)'}")
                    print(f"[gdal_viewshed ERROR] (further identical errors this run "
                          f"will be suppressed)\n")
                    _GDAL_ERROR_PRINTED[0] = True
                continue  # leave this height group as not-visible (False)

            with rasterio.open(out_tif) as vs:
                sub_samples = np.fromiter(
                    (v[0] for v in vs.sample(sub_coords)), dtype=np.uint8, count=len(sub_coords)
                )
            visible[np.where(height_mask)[0]] = (sub_samples == 255)
        finally:
            if os.path.exists(out_tif):
                os.remove(out_tif)

    return visible



def process_civico(civ: dict, tree_idx, tree_array, tree_ids, tree_heights, dsm_path):
    ox, oy = civ["x"], civ["y"] #civico coordinates in metric CRS
    geom = civ["geometry"] #civico geometry in metric CRS
    candidate_idxs = tree_idx.query(geom.buffer(BUFFER_M)) #find trees within buffer of civico
    n_nearby = len(candidate_idxs) 

    #base is crucial = it contains the civico's unique key, full address, building ID, and geometry. This base information is used to construct the output rows for each floor of the building associated with the civico. Each row will include the floor number, observer height, number of nearby trees, number of visible trees, IDs of visible trees, and whether the civico is compliant based on the visibility threshold.
    base = {
        "civkey": civ["civkey"], "indirizzo_completo": civ["indirizzo_completo"],
        "building_id": civ["building_id"], "geometry": geom,
    }

    if n_nearby == 0:
        return [{**base, "floor": f, "observer_z_m": (f * FLOOR_HEIGHT_M) + 1.5,
                 "n_trees_nearby": 0, "n_trees_visible": 0, "visible_tree_ids": [], "compliant": False}
                for f in range(civ["floor_count"])]

    candidate_centroids = gpd.GeoSeries(tree_array[candidate_idxs], crs=CRS_METRIC)
    candidate_tree_ids = tree_ids[candidate_idxs]
    candidate_heights = tree_heights[candidate_idxs]

    #always computes the visibility of trees for each floor of the building associated with the civico. 
    #the observer height is calculated based on the floor index and the average floor height, with an additional 1.5 meters to account for the observer's eye level.
    #each civico floor is taken as a separete observation point for compliance computation
    rows = []
    for floor_idx in range(civ["floor_count"]):
        observer_z = (floor_idx * FLOOR_HEIGHT_M) + 1.5
        visible_mask = gdal_viewshed_visible_trees(
            ox, oy, dsm_path, candidate_centroids, candidate_heights, observer_z
        )
        visible_ids = candidate_tree_ids[visible_mask].tolist()
        rows.append({**base, "floor": floor_idx, "observer_z_m": observer_z,
                     "n_trees_nearby": n_nearby, "n_trees_visible": int(visible_mask.sum()),
                     "visible_tree_ids": visible_ids,
                     "compliant": int(visible_mask.sum()) >= COMPLIANCE_THRESHOLD})
    return rows

def process_chunk_worker(civici_dicts, tile_dsm, tree_idx, tree_array, tree_ids, tree_heights):
    rows = []
    for civ in civici_dicts:
        rows.extend(process_civico(civ, tree_idx, tree_array, tree_ids, tree_heights, tile_dsm))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--force-reload", action="store_true")
    args = ap.parse_args()
    CITY = args.city

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    con = connect_db_with_retry()
    CITY = args.city
    base_city_for_cfg = strip_source_suffix(CITY)

    cfg = get_city_cfg(base_city_for_cfg)
    paths = get_output_paths(CITY)   

    CITY_METRIC_CRS = cfg.get("crs_metric", cfg.get("crs", CRS_METRIC))
    print(f"Running 3 trees in 30m from a street number visibility compliance for City: {CITY} | DB: {DB_PATH} | Metric CRS: {CITY_METRIC_CRS}")

    buildings = load_table(con, "raw", f"{base_city_for_cfg}_buildings")
    civici = load_table(con, "raw", f"{base_city_for_cfg}_buildings_sn")
    trees = _load_and_prepare_trees(con, CITY, CITY_METRIC_CRS, force_reload=args.force_reload)

    print(f"Loaded raw datasets -> Buildings: {len(buildings):,}, Civici: {len(civici):,}, Trees: {len(trees):,}")

    # load_table() always tags results with a CRS on read (defaults to
    # CRS_METRIC), so a None-CRS case in practice only shows up for data
    # loaded directly via gpd.read_file() — already handled explicitly
    # in _load_and_prepare_trees's SAM2-gpkg fallback above. No separate
    # stabilization pass needed here (confirmed: earlier per-city bugs
    # traced to this were NOT actually CRS mismatches).
    buildings = buildings.to_crs(CITY_METRIC_CRS)
    civici = civici.to_crs(CITY_METRIC_CRS)
    trees = trees.to_crs(CITY_METRIC_CRS)

    if "height" not in buildings.columns:
        buildings["height"] = AVG_BUILDING_HEIGHT_M
    buildings["height"] = buildings["height"].fillna(AVG_BUILDING_HEIGHT_M).replace(0, AVG_BUILDING_HEIGHT_M)

    trees["centroid"] = trees.geometry.centroid
    trees["tree_id"] = trees.index
    tree_centroids_gs = gpd.GeoSeries(trees["centroid"], crs=CITY_METRIC_CRS)
    tree_array = tree_centroids_gs.values
    tree_ids = trees["tree_id"].values
    tree_heights = trees["height_m"].values
    tree_idx = STRtree(tree_array)

    civici = civici.drop(columns=["index_right"], errors="ignore")
    civici = gpd.sjoin(civici, buildings[["height", "geometry"]], how="left", predicate="within")

    # FIX: civkey can be entirely NULL for some cities (confirmed for
    # Bari: 58,496/58,496 rows null). drop_duplicates(subset="civkey")
    # on an all-null column treats every row as a duplicate of every
    # other (pandas: NaN == NaN for this purpose) and silently collapses
    # the ENTIRE dataset down to ~1 row — exactly what produced the
    # single-point total_bounds crash. Validate civkey first; derive a
    # real composite key from reliably-populated columns when it isn't
    # usable, with an object_id/id tiebreaker as a last resort so this
    # can never silently collapse the dataset again.
    civkey_valid = (
        "civkey" in civici.columns
        and civici["civkey"].notna().all()
        and civici["civkey"].nunique() == len(civici)
    )
    if civkey_valid:
        civici["civkey"] = civici["civkey"].astype(str)
    else:
        # Check for a ready-made unique ID before composing one — some
        # civic-address schemas (e.g. ISTAT/ANNCSU-style, seen on Genova)
        # already carry a genuinely unique national/local identifier.
        ready_made_candidates = [
            cfg.get("civico_id_col"), "PROGRESSIVO_NAZIONALE", "progressivo_nazionale",
            "object_id", "id",
        ]
        # Case-insensitive lookup — Roma's actual unique column is 'ID'
        # (uppercase), which a case-sensitive check against lowercase
        # 'id' would silently miss.
        lower_lookup_ready = {c.lower(): c for c in civici.columns}
        ready_made_col = None
        for col in ready_made_candidates:
            if not col:
                continue
            actual_col = lower_lookup_ready.get(col.lower())
            if actual_col and civici[actual_col].notna().all() and civici[actual_col].nunique() == len(civici):
                ready_made_col = actual_col
                break
 
        if ready_made_col:
            civici["civkey"] = civici[ready_made_col].astype(str)
            print(f"[civkey] Using ready-made unique column '{ready_made_col}' as civkey for {CITY}")
        else:
            # No pre-existing unique column either — derive a composite key.
            # Search several plausible column-name variants per role instead
            # of a single hardcoded default, since this varies genuinely
            # per city (confirmed: Bari uses cod_via/numero/esponente/ncivsub,
            # Genova uses ODONIMO/CIVICO/ESPONENTE/PROGRESSIVO_SNC).
            def _find_col(explicit_cfg_key, candidates):
                explicit = cfg.get(explicit_cfg_key)
                if explicit and explicit in civici.columns:
                    return explicit
                # Case-insensitive match — Milano's NUMERO, Verona's
                # COD_VIA, etc. would otherwise be missed purely on
                # casing despite being the exact right column.
                lower_lookup = {c.lower(): c for c in civici.columns}
                for c in candidates:
                    if c.lower() in lower_lookup:
                        return lower_lookup[c.lower()]
                return None
 
            via_key_col = _find_col("civico_via_col",
                ["cod_via", "CODICE_VIA", "ODONIMO", "denom_via", "via", "strada",
                 "codvia", "CODVIA", "COD_VIA", "CODICESTRA"])
            num_key_col = _find_col("address_num_col",
                ["numero", "CIVICO", "civico", "num", "NUM_CIV", "civico_num", "NUMEROCIVI"])
            suffix_col  = _find_col("address_suffix_col",
                ["esponente", "ESPONENTE", "ESP", "civico_let", "subciv", "LETTERA", "BISTER"])
            sub_col     = _find_col("civico_sub_col",
                ["ncivsub", "PROGRESSIVO_SNC", "progressivo_snc", "PROGRESSIVO_ACCESSO"])
 
            missing = [name for name, col in [("via", via_key_col), ("num", num_key_col)] if col is None]
            if missing:
                print(f"[CRITICAL] Can't derive civkey for {CITY}: no column found for "
                      f"role(s) {missing}. Available: {list(civici.columns)}")
                con.close()
                return
            civici["civkey"] = (
                civici[via_key_col].astype(str).str.strip() + "_"
                + civici[num_key_col].astype(str).str.strip() + "_"
                + (civici[suffix_col].astype("string").fillna("") if suffix_col else "") + "_"
                + (civici[sub_col].astype("string").fillna("") if sub_col else "")
            )
            n_dupe = len(civici) - civici["civkey"].nunique()
            if n_dupe > 0:
                id_col = next((c for c in ("object_id", "id", "PROGRESSIVO_NAZIONALE",
                                            "CODICE_COMUNALE_ACCESSO") if c in civici.columns), None)
                if id_col:
                    print(f"[civkey] WARNING: derived civkey still had {n_dupe} duplicate(s) for "
                          f"{CITY} — appending '{id_col}' as a tiebreaker.")
                    civici["civkey"] = civici["civkey"] + "_" + civici[id_col].astype(str)
                else:
                    # No plausible ID column exists at all — fall back to the
                    # DataFrame's own row index, which is guaranteed unique.
                    print(f"[civkey] WARNING: derived civkey still had {n_dupe} duplicate(s) for "
                          f"{CITY}, and no id-like column exists — appending row index as a tiebreaker.")
                    civici["civkey"] = civici["civkey"] + "_" + civici.index.astype(str)
            print(f"[civkey] Derived from {via_key_col}+{num_key_col}+{suffix_col}+{sub_col} "
                  f"for {CITY} ({civici['civkey'].nunique():,} unique / {len(civici):,} rows)")
 
    civici = civici.drop_duplicates(subset="civkey")
 
    if "height_right" in civici.columns:
        civici.rename(columns={"height_right": "height"}, inplace=True)
    elif "height_left" in civici.columns:
        civici.rename(columns={"height_left": "height"}, inplace=True)
 
    if "height" not in civici.columns:
        civici["height"] = AVG_BUILDING_HEIGHT_M
 
    civici["height"] = civici["height"].fillna(AVG_BUILDING_HEIGHT_M).replace(0, AVG_BUILDING_HEIGHT_M)
    civici["floor_count"] = (civici["height"] / FLOOR_HEIGHT_M).astype(int).clip(lower=1)
 
    via_col = cfg.get("address_via_col", "denom_via") if cfg.get("address_via_col", "denom_via") in civici.columns else None
    num_col = cfg.get("address_num_col", "numero") if cfg.get("address_num_col", "numero") in civici.columns else None
 
    if not via_col:
        for possible_col in ["strada", "via", "indirizzo", "display_name"]:
            if possible_col in civici.columns:
                via_col = possible_col
                break
    if not num_col:
        for possible_col in ["civico", "num", "civ"]:
            if possible_col in civici.columns:
                num_col = possible_col
                break
 
    if via_col and num_col:
        civici["indirizzo_completo"] = civici[via_col].astype(str).str.strip() + " " + civici[num_col].astype(str).str.strip()
    elif "display_name" in civici.columns:
        civici["indirizzo_completo"] = civici["display_name"]
    else:
        civici["indirizzo_completo"] = "Unknown Address " + civici["civkey"].astype(str)
 
    civici = civici[civici.geometry.notna() & ~civici.geometry.is_empty]
    
    if civici.empty:
        print("[ERROR] Civici dataset is empty after spatial processing and cleaning!")
        con.close()
        return
 
    minx, miny, maxx, maxy = civici.total_bounds
    print(f"Processing geographic bounds: MinX={minx:.1f}, MinY={miny:.1f}, MaxX={maxx:.1f}, MaxY={maxy:.1f}")
    
    if minx == maxx or miny == maxy:
        print("[CRITICAL] Spatial dataset coordinates are collapsed to a single point! Verify source tables.")
        con.close()
        return

    N_WORKERS = min(os.cpu_count(), 4)
    chunk_id = 0

    for x0 in tqdm(np.arange(minx, maxx, CHUNK_SIZE_M), desc="Chunks"):
        for y0 in np.arange(miny, maxy, CHUNK_SIZE_M):
            tile = box(x0, y0, x0 + CHUNK_SIZE_M, y0 + CHUNK_SIZE_M)
            civici_tile = civici[civici.geometry.within(tile)]
            buildings_tile = buildings[buildings.geometry.intersects(tile.buffer(BUFFER_M))]

            if civici_tile.empty:
                continue

            tile_dsm = f"/tmp/dsm_{chunk_id}.tif"
            # DSM extent always covers every civico's position + BUFFER_M search radius, regardless of where buildings actually are.
            civici_bounds = civici_tile.total_bounds
            b_bounds = buildings_tile.total_bounds if not buildings_tile.empty else civici_bounds
            pad = 50
            b_minx = min(b_bounds[0], civici_bounds[0] - BUFFER_M) - pad
            b_miny = min(b_bounds[1], civici_bounds[1] - BUFFER_M) - pad
            b_maxx = max(b_bounds[2], civici_bounds[2] + BUFFER_M) + pad
            b_maxy = max(b_bounds[3], civici_bounds[3] + BUFFER_M) + pad
            width = int((b_maxx - b_minx) / RESOLUTION_M)
            height = int((b_maxy - b_miny) / RESOLUTION_M)
            transform = from_bounds(b_minx, b_miny, b_maxx, b_maxy, width, height)
            # Rasterize building footprints with their heights to create a DSM (Digital Surface Model) raster.
            shapes = ((mapping(g), h) for g, h in zip(buildings_tile.geometry, buildings_tile["height"]) if g is not None and h > 0)
            dsm_data = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0.0, dtype="float32")

            with rasterio.open(tile_dsm, "w", driver="GTiff", height=height, width=width, count=1,
                               dtype="float32", crs=CITY_METRIC_CRS, transform=transform) as dst:
                dst.write(dsm_data, 1)
            #tile_records is a DataFrame that contains the relevant information for each civico in the current tile, including its unique key, full address, building ID, floor count, and geometry. The x and y coordinates of each civico are also extracted from the geometry for use in the visibility calculations.
            tile_records = civici_tile[["civkey", "indirizzo_completo", "building_id", "floor_count"]].copy()
            tile_records["x"] = civici_tile.geometry.x
            tile_records["y"] = civici_tile.geometry.y
            tile_records["geometry"] = civici_tile.geometry.values
            tile_dicts = tile_records.to_dict(orient="records")
            
            batch_size = max(1, len(tile_dicts) // N_WORKERS)
            batches = [tile_dicts[i:i+batch_size] for i in range(0, len(tile_dicts), batch_size)]

            all_rows = []
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                futures = [
                    pool.submit(process_chunk_worker, batch, tile_dsm, tree_idx, tree_array, tree_ids, tree_heights)
                    for batch in batches
                ]
                for f in as_completed(futures):
                    all_rows.extend(f.result())

            if all_rows:
                chunk_gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs=CITY_METRIC_CRS)
                chunk_gdf.to_parquet(RESULTS_DIR / f"chunk_{chunk_id:04d}.parquet")

            if os.path.exists(tile_dsm):
                os.remove(tile_dsm)
            chunk_id += 1

    all_chunks = sorted(RESULTS_DIR.glob("*.parquet"))
    if not all_chunks:
        print("[ERROR] No chunks generated.")
        con.close()
        return

    result_gdf = gpd.GeoDataFrame(
        pd.concat([gpd.read_parquet(f) for f in all_chunks]),
        geometry="geometry", crs=CITY_METRIC_CRS,
    )

    #comliance: per civico, any floor compliant = True if any floor has >= 3 visible trees within 30m buffer
    civico_compliance = result_gdf.groupby("civkey")["compliant"].any().rename("any_floor_compliant").reset_index()
    result_gdf = result_gdf.merge(civico_compliance, on="civkey", how="left")
 
    out_wgs = result_gdf.to_crs(CRS_WGS)
    out_wgs["longitude"] = out_wgs.geometry.x
    out_wgs["latitude"] = out_wgs.geometry.y
 
    con.execute("CREATE SCHEMA IF NOT EXISTS rule_3")
    gdf_to_duckdb(con, result_gdf, f"rule_3.{CITY}_results")
    
    gpkg_path = paths["gpkg"].replace(".gpkg", "_rule3.gpkg")
    parquet_path = paths["parquet"].replace(".parquet", "_rule3.parquet")
    result_gdf.to_file(gpkg_path, driver="GPKG", layer="civici")
    out_wgs.to_parquet(parquet_path, index=False)
 
    # NEW: per-civico aggregate layer (one row per address, not one per
    # floor) — any_floor_compliant + best floor's tree count, much easier
    # to symbolize compliant/non-compliant in QGIS than the exploded
    # per-floor layer.
    civico_agg = result_gdf.groupby("civkey").agg(
        indirizzo_completo=("indirizzo_completo", "first"),
        building_id=("building_id", "first"),
        any_floor_compliant=("any_floor_compliant", "first"),
        max_trees_visible_any_floor=("n_trees_visible", "max"),
        n_floors=("floor", "count"),
        geometry=("geometry", "first"),
    ).reset_index()
    civico_agg = gpd.GeoDataFrame(civico_agg, geometry="geometry", crs=CITY_METRIC_CRS)
    civico_agg.to_file(gpkg_path, driver="GPKG", layer="civici_by_address")
    print(f"  Per-civico layer saved -> layer='civici_by_address' in {gpkg_path}")
 
    # NEW: trees layer, so compliance can be visually cross-checked
    # against actual detected tree positions/heights in QGIS.
    trees_out = trees[["tree_id", "score", "height_m", "geometry"]].copy() \
        if "tree_id" in trees.columns else trees[["score", "height_m", "geometry"]].copy()
    trees_out = trees_out.to_crs(CITY_METRIC_CRS) if trees_out.crs != CITY_METRIC_CRS else trees_out
    trees_out.to_file(gpkg_path, driver="GPKG", layer="trees")
    print(f"  Trees layer saved -> layer='trees' in {gpkg_path} ({len(trees_out):,} trees)")
 
    total_addresses = int(result_gdf["civkey"].nunique())
    compliant_addresses_any_floor = int(result_gdf[result_gdf["any_floor_compliant"] == True]["civkey"].nunique())
    address_level_pct = (compliant_addresses_any_floor / total_addresses * 100) if total_addresses > 0 else 0.0
 
    # Address-level ("any floor compliant") is the PRIMARY reported city
    # compliance number — matches standard 3-30-300 framing ("does this
    # address have adequate tree access at all"). Floor-level is still
    # computed and reported alongside it as a secondary, more
    # conservative view (population/dwelling-level access), since it
    # answers a genuinely different question worth having on hand.
    total_floor_rows = len(result_gdf)
    compliant_floor_rows = int(result_gdf["compliant"].sum())
    floor_level_pct = (compliant_floor_rows / total_floor_rows * 100) if total_floor_rows > 0 else 0.0
 
    print(f"\nSummary for {CITY}:")
    print(f"  Address-level (PRIMARY — any floor compliant): "
          f"{compliant_addresses_any_floor:,}/{total_addresses:,} addresses ({address_level_pct:.2f}%)")
    print(f"  Floor-level (secondary — every floor counted individually): "
          f"{compliant_floor_rows:,}/{total_floor_rows:,} floor-occupancies ({floor_level_pct:.2f}%)")
 
    summary_data = pd.DataFrame([{
        "city": CITY,
        "total_addresses": total_addresses,
        "compliant_addresses_any_floor": compliant_addresses_any_floor,
        "address_level_compliance_pct": round(address_level_pct, 2),
        "total_floor_occupancies": total_floor_rows,
        "compliant_floor_occupancies": compliant_floor_rows,
        "floor_level_compliance_pct": round(floor_level_pct, 2),
    }])
    
    if CSV_PATH.exists():
        existing_summary = pd.read_csv(CSV_PATH)
        existing_summary = existing_summary[existing_summary["city"] != CITY]
        summary_data = pd.concat([existing_summary, summary_data], ignore_index=True)
        
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_data.to_csv(CSV_PATH, index=False)
    print(f"Logged summary statistics to {CSV_PATH}")
 
    print(f"\n[SUCCESS] Normalized multi-city code applied for {CITY}. Outputs written.")
    con.close()
 
if __name__ == "__main__":
    main()