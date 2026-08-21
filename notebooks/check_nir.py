"""
check_cir_tile.py — Inspect a single Firenze CIR tile: verify band order,
compute NDVI, and preview the mask before committing to a full run.

Usage:
    python check_cir_tile.py /path/to/tile_XXXX.tif
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index/")
from scr.canopy_30.canopy_extract_nir import detect_nir_red, ndvi_mask, CITY_NDVI_THRESH

tile_path = sys.argv[1]

with rasterio.open(tile_path) as src:
    print(f"bands: {src.count}  dtype: {src.dtypes}  shape: {src.shape}")
    data = src.read(list(range(1, min(src.count, 4) + 1)), out_dtype=np.float32)

for i in range(data.shape[0]):
    b = data[i]
    print(f"  band {i}: min={b.min():.1f} max={b.max():.1f} mean={b.mean():.1f}")

# Force is_cir=True to match your Firenze config
nir, red = detect_nir_red(data, is_cir=True)
print(f"\nAssumed NIR mean: {nir.mean():.1f}  |  Assumed Red mean: {red.mean():.1f}")
if nir.mean() < red.mean():
    print("⚠️  NIR mean < Red mean over the whole tile — bands may be swapped, "
          "or this tile is mostly non-vegetated (built-up/water). Check a "
          "known-vegetated tile before concluding anything.")

thresh = CITY_NDVI_THRESH.get("firenze", 0.30)
mask = ndvi_mask(nir, red, thresh)
print(f"\nNDVI thresh used: {thresh}")
print(f"Mask coverage: {mask.mean()*100:.1f}% of tile")

# Quick visual: RGB-ish composite (using NIR/R/G as false color) + mask overlay
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
composite = np.stack([data[0], data[1], data[2] if data.shape[0] > 2 else data[1]], axis=-1)
composite = np.clip(composite / np.percentile(composite, 98), 0, 1)
axes[0].imshow(composite)
axes[0].set_title("False-color (NIR/R/G)")
axes[0].axis("off")

axes[1].imshow(composite)
axes[1].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
axes[1].set_title(f"NDVI mask overlay (thresh={thresh})")
axes[1].axis("off")

plt.tight_layout()
out_path = Path(tile_path).stem + "_ndvi_check.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved preview: {out_path}")