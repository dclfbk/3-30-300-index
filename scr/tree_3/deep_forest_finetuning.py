
#running deepforest on TCD data to adjust it to european- otalian forest
import os, json, random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from pathlib import Path
from PIL import Image
from shapely.geometry import box
#from sklearn.model_selection import StratifiedShuffleSplit
import sys
import torch
print(torch.__version__, torch.__file__)
from deepforest import main as df_main
from datasets import load_dataset

import time
from rasterio.windows import Window
from rasterio.transform import xy as rio_xy
from shapely.geometry import shape
from shapely.affinity import translate
from rasterio.features import shapes as rio_shapes

from config.config import get_city_cfg, connect_db, gdf_to_duckdb


import os, json, random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from pathlib import Path
from PIL import Image
from shapely.geometry import box
#from sklearn.model_selection import StratifiedShuffleSplit
import sys
import torch
print(torch.__version__, torch.__file__)
from deepforest import main as df_main
from datasets import load_dataset

import time
from rasterio.windows import Window
from rasterio.transform import xy as rio_xy
from shapely.geometry import shape
from shapely.affinity import translate
from rasterio.features import shapes as rio_shapes

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from rasterio.drivers import StratifiedShuffleSplit


RAW_BOXES_CSV = "/workspace/3-30-300-index/data/raw_predictions_v2.csv"
TIF_PATH = "/workspace/3-30-300-index/data/bologna_rgb_2023.tif"
WORK_DIR        = "workspace/3-30-300-index"
TILES_DIR       = str(Path(WORK_DIR) / "tiles")
MODEL_SAVE      = str(Path(WORK_DIR) / "deepforest_tcd_eu_v2.pt")
TRAIN_CSV       = str(Path(WORK_DIR) / "train_v2.csv")
VAL_CSV         = str(Path(WORK_DIR) / "val_v2.csv")
RAW_PREDS_CSV   = str(Path(WORK_DIR) / "raw_predictions_v2.csv")
OUTPUT_CLEAN    = str(Path(WORK_DIR) / "tree_detections_clean_v2.gpkg")

# config cell — rename to match:
MAX_ASPECT = 3.0   # was MAX_ASPECT_RATIO
# ── Dataset ────────────────────────────────────────────────────────────────
MAX_TILES    = 2000
IMG_SIZE     = 2048

# ── Training ───────────────────────────────────────────────────────────────
TRAIN_EPOCHS = 15
TRAIN_LR     = 1e-4
TRAIN_BATCH  = 4



print("Config OK")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")




ds = load_dataset("restor/tcd", split="train", streaming=True)
european = (
    s for s in ds
    if 35 <= s["lat"] <= 72 and -25 <= s["lon"] <= 45
)

rows_canopy = []   # cat_id == 1
rows_tree   = []   # cat_id == 2  (individual trees — what we want to improve)

for i, sample in enumerate(european):
    if i >= MAX_TILES:
        break

    img_path = TILES_DIR / f"tile_{i:04d}.png"
    if not img_path.exists():
        sample["image"].save(img_path)

    raw = sample.get("coco_annotations")
    if not raw:
        continue
    try:
        annots = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        continue
    if isinstance(annots, dict):
        annots = annots.get("annotations", [])

    for ann in annots:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w < 5 or h < 5:
            continue
        cat = ann.get("category_id")
        if cat not in [1, 2]:
            continue

        entry = {
            "image_path": str(img_path),
            "xmin": float(x),
            "ymin": float(y),
            "xmax": float(x + w),
            "ymax": float(y + h),
            "label": "Tree",
            "source_cat": cat,
        }
        if cat == 1:
            rows_canopy.append(entry)
        else:
            rows_tree.append(entry)

print(f"Raw counts — canopy (cat1): {len(rows_canopy):,}  |  individual trees (cat2): {len(rows_tree):,}")

# ── Upsample cat_id=2 to match cat_id=1 count ─────────────────────────────
if len(rows_tree) > 0 and len(rows_canopy) > len(rows_tree):
    ratio         = len(rows_canopy) / len(rows_tree)
    n_full        = int(ratio)
    n_partial     = round((ratio % 1) * len(rows_tree))
    rows_tree_up  = rows_tree * n_full + random.sample(rows_tree, n_partial)
    print(f"Upsampled cat2: {len(rows_tree):,} → {len(rows_tree_up):,} (×{ratio:.2f})")
else:
    rows_tree_up  = rows_tree
    print("No upsampling needed (cat2 >= cat1 or cat2 is empty)")

rows = rows_canopy + rows_tree_up
df   = pd.DataFrame(rows)

assert "image_path" in df.columns and len(df) > 0, "Empty — check coco_annotations structure"

# ── Clip & drop tiny boxes ─────────────────────────────────────────────────
df[["xmin", "ymin"]] = df[["xmin", "ymin"]].clip(lower=0)
df["xmax"] = df["xmax"].clip(upper=IMG_SIZE)
df["ymax"] = df["ymax"].clip(upper=IMG_SIZE)
df = df[((df["xmax"] - df["xmin"]) >= 5) & ((df["ymax"] - df["ymin"]) >= 5)].copy()

print(f"Total annotations after upsampling: {len(df):,}")


# Annotation count per tile as a density proxy
tile_counts = df.groupby("image_path").size().reset_index(name="ann_count")
tile_counts["density_bin"] = pd.qcut(
    tile_counts["ann_count"], q=4, labels=False, duplicates="drop"
)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(tile_counts, tile_counts["density_bin"]))

train_tiles = tile_counts.iloc[train_idx]["image_path"].values
val_tiles   = tile_counts.iloc[val_idx]["image_path"].values

# DeepForest expects image_path to be just the filename (relative to root_dir)
def make_relative(df_in):
    df_out = df_in.copy()
    df_out["image_path"] = df_out["image_path"].apply(lambda p: Path(p).name)
    return df_out[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]]

train_df = make_relative(df[df["image_path"].isin(train_tiles)])
val_df   = make_relative(df[df["image_path"].isin(val_tiles)])

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV,     index=False)

print(f"{len(df):,} boxes from {len(tile_counts)} tiles")
print(f"Train: {len(train_df):,} boxes ({len(train_tiles)} tiles)")
print(f"Val:   {len(val_df):,} boxes ({len(val_tiles)} tiles)")
print("\nDensity distribution in val:")
print(tile_counts.iloc[val_idx]["density_bin"].value_counts().sort_index())