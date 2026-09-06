"""
finetune_deepforest_tcd_individual.py — Fine-tune DeepForest on the
restor/tcd (OAM-TCD) dataset, restricted to individual-tree
(category_id=2) annotations only, at a GSD matched to your actual
target imagery.

── WHY category_id=2 ONLY (not both categories, not a 50/50 split) ────────
OAM-TCD's own creators note the individual-vs-group boundary in their
annotations reflects genuinely inconsistent human judgement calls —
annotators had freedom to choose whether to individually label a tree
or group it as "canopy" — not a clean, objective split. This project's
earlier fine-tuning attempt (deepforest_tcd_eu.pt / _v2) mixed both
categories under one "Tree" label and produced large bounding-box
artifacts. DeepForest/RetinaNet has no built-in mechanism for treating
group-labeled regions as "ignore" zones the way the broader Sparsely-
Annotated-Object-Detection literature recommends (see OAM-TCD's own
"sparse-annotation loss" and the follow-up CrownViM paper's partial-
supervision loss — reimplementing that properly is out of scope here).
Instead this script uses a coarser, tile-level proxy: any tile where
category_id=1 (canopy/group) boxes cover more than
GROUP_COVERAGE_SKIP_FRAC of the tile area is dropped entirely from
training, so real-but-unlabeled canopy in those tiles never gets
implicitly taught as background.

── WHY GSD-matched resampling, and why RANDOM per-tile GSD beats a single fixed one ──
restor/tcd images are fixed at 10 cm/px (0.10 m/px) — confirmed on the
dataset's HuggingFace page, which explicitly warns models trained at
its nominal resolution may under-perform on imagery at a significantly
different resolution. Your Italian cities' actual measured GSDs vary
(genova ~0.143, torino_rgb ~0.3079, others closer to ~0.20) — there is
no single "correct" target GSD for a multi-city project.

An earlier version of this script resampled every training tile to one
FIXED target GSD. That's better than not resampling at all, but it
still leaves cities far from that fixed value with a residual train/
inference scale mismatch — and "fixing" that by resampling the real
target imagery instead just moves the problem (downsampling throws
away real captured detail; upsampling a coarser city up to the target
fabricates pixels via interpolation rather than adding real
information).

The better-supported approach, confirmed against the remote-sensing
object detection literature: multi-sensor/multi-GSD robustness is
standardly built by training with RANDOM rescale augmentation spanning
the real range of GSDs the model will see at inference, rather than
normalizing every image to one value (see e.g. cropped-block training
with rescale augmentation used specifically to build cross-sensor
robustness in remote sensing detectors). This script now does that by
default: each kept tile is resampled to a GSD drawn uniformly from
[--gsd-min, --gsd-max] (defaults span this project's currently observed
range, update them if your cities' measured GSDs differ), with box
coordinates rescaled identically. Your real target imagery is never
touched — every city keeps its full native resolution at inference.

Pass a single --target-gsd instead of --gsd-min/--gsd-max only if
you're deliberately fine-tuning a city-specific checkpoint for one
known city.

Usage:
    python finetune_deepforest_tcd_individual.py
        (default: multi-scale training, random GSD per tile in [0.14, 0.31])
    python finetune_deepforest_tcd_individual.py --gsd-min 0.10 --gsd-max 0.40 --max-tiles 3000
    python finetune_deepforest_tcd_individual.py --target-gsd 0.3079   # single city, fixed GSD instead
    python finetune_deepforest_tcd_individual.py --epochs 25 --lr 5e-5
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

from deepforest import main as df_main

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

ROOT_PATH = "/workspace/3-30-300-index"
sys.path.insert(0, ROOT_PATH)

SOURCE_GSD_M = 0.10   # restor/tcd native resolution — fixed, confirmed on HF dataset card
SOURCE_IMG_PX = 2048  # restor/tcd native tile size

# Europe bounding box used to filter the streamed global dataset
EU_LAT_RANGE = (35, 72)
EU_LON_RANGE = (-25, 45)

MIN_BOX_PX_SRC = 5     # in SOURCE (10cm) pixel space, before resampling
MAX_BOX_PX_SRC = 300   # ~30m at 10cm/px — generous upper bound for a single
                        # crown; category_id=2 boxes shouldn't normally
                        # exceed this, this just guards against annotation
                        # errors slipping through

GROUP_COVERAGE_SKIP_FRAC = 0.15   # drop a tile entirely if category_id=1
                                   # (group/canopy) boxes cover more than
                                   # this fraction of its area — see
                                   # module docstring


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tiles", type=int, default=2000,
                     help="Max tiles to pull from the streamed restor/tcd dataset "
                          "(before European filtering, so actual kept count is lower)")
    ap.add_argument("--target-gsd", type=float, default=None,
                     help="Resample every tile to this SINGLE fixed GSD (m/px). "
                          "Mutually exclusive with --gsd-min/--gsd-max. Only use "
                          "this if you're fine-tuning for one specific city whose "
                          "measured GSD you already know — for a multi-city "
                          "project, --gsd-min/--gsd-max (the default) is the "
                          "better choice; see module docstring.")
    ap.add_argument("--gsd-min", type=float, default=0.14,
                     help="Lower bound of the GSD range to train across (m/px). "
                          "Default spans genova's measured ~0.143 to "
                          "torino_rgb's measured ~0.3079 — update these to match "
                          "the actual range of measured GSDs across YOUR cities "
                          "before a real run, this default is just this "
                          "project's current observed range.")
    ap.add_argument("--gsd-max", type=float, default=0.31)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--group-coverage-skip-frac", type=float, default=GROUP_COVERAGE_SKIP_FRAC)
    ap.add_argument("--out-dir", default=str(Path(ROOT_PATH) / "finetuning_individual_v3"))
    ap.add_argument("--model-name", default="deepforest_tcd_eu_individual_multiscale_v4.pt",
                     help="Checkpoint filename. Kept distinct from prior "
                          "deepforest_tcd_eu.pt / _v2.pt / _v3.pt so there's no "
                          "ambiguity about which checkpoint is which.")
    return ap.parse_args()


def load_and_prepare_tiles(args, tiles_dir: Path):
    """Stream restor/tcd, filter to Europe, resample each kept tile to a
    per-tile GSD (either a single fixed value, or randomly drawn from
    [gsd_min, gsd_max] — see module docstring for why a range is the
    better default), keep only category_id=2 (individual tree) boxes as
    positive training instances, and skip any tile where category_id=1
    (group/canopy) coverage exceeds GROUP_COVERAGE_SKIP_FRAC."""
    use_fixed_gsd = args.target_gsd is not None
    if use_fixed_gsd:
        print(f"[prep] source GSD={SOURCE_GSD_M}m/px -> FIXED target GSD={args.target_gsd}m/px")
    else:
        print(f"[prep] source GSD={SOURCE_GSD_M}m/px -> RANDOM target GSD per tile, "
              f"range=[{args.gsd_min}, {args.gsd_max}]m/px "
              f"(multi-scale training — see module docstring for why this beats "
              f"a single fixed target)")

    ds = load_dataset("restor/tcd", split="train", streaming=True)

    rows = []
    n_seen = 0
    n_kept_eu = 0
    n_skipped_group_heavy = 0
    n_skipped_no_individual = 0
    gsd_used_log = []   # track what actually got used, for the summary print

    for sample in ds:
        if n_seen >= args.max_tiles:
            break
        n_seen += 1

        if not (EU_LAT_RANGE[0] <= sample["lat"] <= EU_LAT_RANGE[1]
                and EU_LON_RANGE[0] <= sample["lon"] <= EU_LON_RANGE[1]):
            continue
        n_kept_eu += 1

        raw = sample.get("coco_annotations")
        if not raw:
            continue
        try:
            annots = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if isinstance(annots, dict):
            annots = annots.get("annotations", [])

        group_area_px2 = 0.0
        individual_boxes = []   # in SOURCE pixel space

        for ann in annots:
            if ann.get("iscrowd", 0):
                continue
            cat = ann.get("category_id")
            x, y, w, h = ann["bbox"]

            if cat == 1:  # group/canopy — used only to measure coverage
                group_area_px2 += w * h
                continue
            if cat != 2:  # individual tree — the only positive class we train on
                continue
            if w < MIN_BOX_PX_SRC or h < MIN_BOX_PX_SRC:
                continue
            if w > MAX_BOX_PX_SRC or h > MAX_BOX_PX_SRC:
                continue
            individual_boxes.append((x, y, x + w, y + h))

        group_frac = group_area_px2 / (SOURCE_IMG_PX * SOURCE_IMG_PX)
        if group_frac > args.group_coverage_skip_frac:
            n_skipped_group_heavy += 1
            continue
        if not individual_boxes:
            n_skipped_no_individual += 1
            continue

        # ── Per-tile target GSD: fixed, or randomly drawn from the range ──
        this_target_gsd = args.target_gsd if use_fixed_gsd else random.uniform(args.gsd_min, args.gsd_max)
        gsd_used_log.append(this_target_gsd)
        scale = SOURCE_GSD_M / this_target_gsd
        out_img_px = max(1, round(SOURCE_IMG_PX * scale))

        # ── Resample image to this tile's target GSD ────────────────────
        tile_name = f"tile_{n_kept_eu:04d}.png"
        out_path = tiles_dir / tile_name
        if not out_path.exists():
            img = sample["image"].convert("RGB").resize(
                (out_img_px, out_img_px), Image.LANCZOS
            )
            img.save(out_path)

        # ── Rescale + clip boxes to the resampled image ────────────────
        for xmin, ymin, xmax, ymax in individual_boxes:
            rxmin = max(0.0, xmin * scale)
            rymin = max(0.0, ymin * scale)
            rxmax = min(float(out_img_px), xmax * scale)
            rymax = min(float(out_img_px), ymax * scale)
            if (rxmax - rxmin) < 2 or (rymax - rymin) < 2:
                continue  # degenerate after clipping/downsampling
            rows.append({
                "image_path": tile_name,
                "xmin": rxmin, "ymin": rymin, "xmax": rxmax, "ymax": rymax,
                "label": "Tree",
            })

    print(f"[prep] streamed {n_seen} tiles, {n_kept_eu} in Europe bbox")
    print(f"[prep] skipped {n_skipped_group_heavy} tiles (>{args.group_coverage_skip_frac:.0%} "
          f"group/canopy coverage)")
    print(f"[prep] skipped {n_skipped_no_individual} tiles (no valid individual-tree boxes)")
    if gsd_used_log:
        print(f"[prep] target GSD actually used across kept tiles: "
              f"min={min(gsd_used_log):.4f} max={max(gsd_used_log):.4f} "
              f"mean={sum(gsd_used_log)/len(gsd_used_log):.4f}")

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("[prep] No usable annotations after filtering — check dataset access / filters")

    print(f"[prep] {len(df):,} individual-tree boxes from "
          f"{df['image_path'].nunique()} tiles")
    return df


def stratified_split(df, test_size=0.15, random_state=42):
    """Split by tile, stratified on per-tile box-count density quartile,
    so sparse and dense scenes are both represented in train and val."""
    tile_counts = df.groupby("image_path").size().reset_index(name="ann_count")
    try:
        tile_counts["density_bin"] = pd.qcut(
            tile_counts["ann_count"], q=4, labels=False, duplicates="drop"
        )
    except ValueError:
        # too few unique values for 4 bins — fall back to fewer bins
        tile_counts["density_bin"] = pd.qcut(
            tile_counts["ann_count"], q=2, labels=False, duplicates="drop"
        )

    # StratifiedShuffleSplit needs at least 2 members per class; collapse
    # any singleton bins into the nearest bin rather than erroring out
    bin_counts = tile_counts["density_bin"].value_counts()
    singleton_bins = bin_counts[bin_counts < 2].index
    if len(singleton_bins) > 0:
        most_common_bin = bin_counts.idxmax()
        tile_counts.loc[tile_counts["density_bin"].isin(singleton_bins), "density_bin"] = most_common_bin

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(tile_counts, tile_counts["density_bin"]))

    train_tiles = set(tile_counts.iloc[train_idx]["image_path"])
    val_tiles = set(tile_counts.iloc[val_idx]["image_path"])

    train_df = df[df["image_path"].isin(train_tiles)].reset_index(drop=True)
    val_df = df[df["image_path"].isin(val_tiles)].reset_index(drop=True)

    print(f"[split] train: {len(train_df):,} boxes ({len(train_tiles)} tiles)")
    print(f"[split] val:   {len(val_df):,} boxes ({len(val_tiles)} tiles)")
    print("[split] val density-bin distribution:")
    print(tile_counts.iloc[val_idx]["density_bin"].value_counts().sort_index().to_string())

    return train_df, val_df


def main():
    args = parse_args()
    random.seed(42)

    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    model_save_path = out_dir / args.model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_and_prepare_tiles(args, tiles_dir)
    train_df, val_df = stratified_split(df)

    train_df[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]].to_csv(train_csv, index=False)
    val_df[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]].to_csv(val_csv, index=False)
    print(f"[split] wrote {train_csv} and {val_csv}")

    print("\n[train] Loading pretrained release weights (fine-tuning, not training from scratch) ...")
    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")

    model.config["train"]["csv_file"] = str(train_csv)
    model.config["train"]["root_dir"] = str(tiles_dir)
    model.config["validation"]["csv_file"] = str(val_csv)
    model.config["validation"]["root_dir"] = str(tiles_dir)
    model.config["train"]["epochs"] = args.epochs
    model.config["train"]["lr"] = args.lr
    model.config["train"]["batch_size"] = args.batch_size
    model.config["train"]["augment"] = True

    print(f"[train] epochs={args.epochs} lr={args.lr} batch_size={args.batch_size} "
          f"device={device}")

    model.create_trainer(fast_dev_run=False)
    model.trainer.fit(model)
    model.save_model(str(model_save_path))
    print(f"\n[train] Model saved: {model_save_path}")

    print("\n[sanity check] Note: score_thresh in model.config is NOT reliably "
          "honored by predict_tile in DeepForest 2.1.0 (see project notes). "
          "Always filter predictions explicitly by score after predict_tile, "
          "e.g.:\n"
          "    preds = model.predict_tile(...)\n"
          "    preds = preds[preds['score'] >= YOUR_THRESHOLD]\n"
          "rather than relying on model.config['score_thresh'] alone.")


if __name__ == "__main__":
    main()