# ── Post-processing after SAM2 ───────────────────────────────────────────
# Input:  `results` list produced by the SAM2 loop (cell 5)
#         Each entry: {'df_score': float, 'sam_score': float, 'geometry': Polygon}
# Output: gdf_clean  — filtered, deduplicated crown polygons ready for export
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
from pathlib import Path
SAM2_CFG      = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT     = "/workspace/3-30-300-index/data/sam2.1_hiera_large.pt"
TIF_PATH      = "/workspace/3-30-300-index/data/bologna_rgb_2023.tif"
RAW_PREDS_CSV = "/workspace/3-30-300-index/data/raw_predictions_v2.csv"
results   = "/workspace/3-30-300-index/data/tree_detections_sam2_v5.gpkg"

# ── Post-processing filters ────────────────────────────────────────────────
SCORE_THRESH     = 0.15
MIN_AREA_M2      = 1.5
MAX_AREA_M2      = 60.0
MAX_ASPECT_RATIO = 3.0

# ── 0. Build raw GeoDataFrame from SAM2 results ───────────────────────────
with rasterio.open(TIF_PATH) as src:
    crs = src.crs

gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=crs)
print(f"Raw SAM2 polygons:   {len(gdf):,}")

# ── 1. Fill holes & keep only the largest part of multi-polygons ──────────
# SAM2 sometimes produces crowns with internal holes (shadowed centre,
# occluded branch) or small disconnected fragments.  Filling holes prevents
# underestimates in canopy area; keeping only the largest ring removes noise.
def fill_holes(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)          # drop all interior rings
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda p: p.area)
        return Polygon(largest.exterior)
    return geom                                # LineString artefact — leave as-is

gdf["geometry"] = gdf.geometry.apply(fill_holes)
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
print(f"After hole-fill:     {len(gdf):,}")

# ── 2. Simplify polygon boundaries ───────────────────────────────────────
# rasterio.features.shapes produces staircase outlines at pixel resolution.
# tolerance=0.3 m smooths without visibly distorting crown shapes.
# A small buffer–unbuffer also rounds sharp corners (optional visual polish).
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.3, preserve_topology=True)
# gdf["geometry"] = gdf.geometry.buffer(0.2).buffer(-0.2)   # uncomment for smoother crowns

# ── 3. Area, compactness & aspect-ratio filters ───────────────────────────
# Now using real polygon area (not bbox estimate).
gdf["area_m2"]     = gdf.geometry.area
gdf["perimeter_m"] = gdf.geometry.length
# Polsby-Popper compactness: 1.0 = perfect circle, <0.05 = wire-thin artefact
gdf["compactness"] = (4 * np.pi * gdf["area_m2"]) / (gdf["perimeter_m"] ** 2).replace(0, np.nan)

# Minimum bounding rectangle for aspect ratio
def bbox_aspect(geom):
    if geom is None or geom.is_empty:
        return np.nan
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    if min(w, h) == 0:
        return np.nan
    return max(w, h) / min(w, h)

gdf["aspect_ratio"] = gdf.geometry.apply(bbox_aspect)

mask_area       = gdf["area_m2"].between(MIN_AREA_M2, MAX_AREA_M2)
mask_compact    = gdf["compactness"] > 0.05     # removes road/edge artefacts
mask_aspect     = gdf["aspect_ratio"] <= MAX_ASPECT_RATIO

gdf_filtered    = gdf[mask_area & mask_compact & mask_aspect].copy()
gdf_rejected    = gdf[~(mask_area & mask_compact & mask_aspect)].copy()
print(f"After shape filters: {len(gdf_filtered):,}  (rejected {len(gdf_rejected):,})")

# ── 4. Spatial-index IoU deduplication ───────────────────────────────────
# Two SAM2 prompts from overlapping DeepForest boxes can produce near-
# identical polygons.  Greedy dedup: sort by sam_score DESC, then for each
# polygon discard any later polygon whose IoU with it exceeds iou_thresh.
# Uses an R-tree spatial index to skip non-overlapping pairs (O(n log n)).
IOU_DEDUP_THRESH = 0.50

def dedup_by_iou(gdf_in, iou_thresh):
    gdf_s  = gdf_in.sort_values("sam_score", ascending=False).reset_index(drop=True)
    sindex = gdf_s.sindex
    keep   = np.ones(len(gdf_s), dtype=bool)

    for i, row in gdf_s.iterrows():
        if not keep[i]:
            continue
        # Only check candidates whose bounding box intersects
        candidates = list(sindex.intersection(row.geometry.bounds))
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            geom_j = gdf_s.at[j, "geometry"]
            inter  = row.geometry.intersection(geom_j).area
            if inter == 0:
                continue
            union  = row.geometry.union(geom_j).area
            if union > 0 and inter / union > iou_thresh:
                keep[j] = False   # lower-score duplicate → drop

    result = gdf_s[keep].reset_index(drop=True)
    return result

gdf_dedup = dedup_by_iou(gdf_filtered, IOU_DEDUP_THRESH)
print(f"After IoU dedup:     {len(gdf_dedup):,}  (removed {len(gdf_filtered) - len(gdf_dedup):,} duplicates)")

# ── 5. Crown-type labelling ───────────────────────────────────────────────
# Classify detections by size, matching the cat_id=1/2 split from training.
# Thresholds can be tuned from the area histogram in cell 7.
gdf_dedup["crown_type"] = np.where(
    gdf_dedup["area_m2"] < 25, "individual", "canopy_patch"
)
print(f"  individual (<25 m²): {(gdf_dedup['crown_type'] == 'individual').sum():,}")
print(f"  canopy patch:        {(gdf_dedup['crown_type'] == 'canopy_patch').sum():,}")

# ── 6. Final clean GeoDataFrame ───────────────────────────────────────────
gdf_clean = gdf_dedup[[
    "df_score", "sam_score", "area_m2", "compactness", "aspect_ratio",
    "crown_type", "geometry"
]].copy()

# ── 7. Export to GeoPackage (two layers) ─────────────────────────────────
GPKG_OUT = "/workspace/3-30-300-index/data/bologna_trees_sam2.gpkg"
gdf_clean.to_file(GPKG_OUT, driver="GPKG", layer="tree_crowns_clean")
gdf_rejected.to_file(GPKG_OUT, driver="GPKG", layer="tree_crowns_rejected")

print(f"\nSaved → {GPKG_OUT}")
print(f"  layer=tree_crowns_clean    {len(gdf_clean):,} features")
print(f"  layer=tree_crowns_rejected {len(gdf_rejected):,} features")
print(f"\nArea stats (clean):")
print(gdf_clean["area_m2"].describe().round(2).to_string())
