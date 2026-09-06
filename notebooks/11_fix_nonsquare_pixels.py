"""
fix_nonsquare_pixels.py — corrects non-square-pixel orthophoto tiles by
resampling to true square pixels at the FINER of the tile's measured x/y
resolutions (preserves maximum real captured detail rather than
downsampling detail away).

Confirmed affected: trento (ratio ~1.52), torino_rgb (ratio ~1.36) — both
consistent with the known latitude-dependent degree/meter conversion bug
in download_orthophotos.py. Writes corrected tiles to a NEW sibling
directory ({city}_ortho_tiles_squarepx/) rather than overwriting
originals, so this is safe to run and inspect before committing to it.

Usage:
    python fix_nonsquare_pixels.py --city trento
    python fix_nonsquare_pixels.py --city torino_rgb --sample-check-only
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

DATA_ROOT = Path("/workspace/3-30-300-index/data")


def find_tile_dir(city: str) -> Path:
    base_city = city.replace("_rgb", "")
    for candidate in [
        DATA_ROOT / city / f"{city}_ortho_tiles",
        DATA_ROOT / base_city / f"{city}_ortho_tiles",
        DATA_ROOT / base_city / f"{base_city}_ortho_tiles",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No tile directory found for city='{city}'")


def measure_gsd(tile_dir: Path, n_sample: int = 15) -> tuple[float, float]:
    """Sample tiles to measure typical x/y resolution. Returns (median_x, median_y)."""
    import random
    random.seed(1)
    tiles = list(tile_dir.glob("*.tif"))
    sample = random.sample(tiles, min(n_sample, len(tiles)))
    xs, ys = [], []
    for t in sample:
        with rasterio.open(t) as src:
            xs.append(abs(src.transform.a))
            ys.append(abs(src.transform.e))
    return float(np.median(xs)), float(np.median(ys))


def resample_tile_to_square(src_path: Path, dst_path: Path, target_gsd: float):
    with rasterio.open(src_path) as src:
        bounds = src.bounds
        new_width = max(1, round((bounds.right - bounds.left) / target_gsd))
        new_height = max(1, round((bounds.top - bounds.bottom) / target_gsd))
        new_transform = from_bounds(*bounds, new_width, new_height)

        new_data = np.zeros((src.count, new_height, new_width), dtype=src.dtypes[0])
        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=new_data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )

        profile = src.profile.copy()
        profile.update(width=new_width, height=new_height, transform=new_transform)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(new_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--sample-check-only", action="store_true",
                     help="Just measure and report the x/y ratio, don't resample anything.")
    args = ap.parse_args()

    tile_dir = find_tile_dir(args.city)
    gsd_x, gsd_y = measure_gsd(tile_dir)
    ratio = gsd_y / gsd_x
    print(f"[{args.city}] Measured GSD: x={gsd_x:.4f}  y={gsd_y:.4f}  ratio={ratio:.3f}")

    if abs(ratio - 1.0) < 0.02:
        print(f"[{args.city}] Pixels are already square (within 2%) — no correction needed.")
        return

    target_gsd = min(gsd_x, gsd_y)  # finer axis — preserves max real detail
    print(f"[{args.city}] Non-square confirmed. Target square GSD: {target_gsd:.4f} "
          f"(the finer of the two measured axes)")

    if args.sample_check_only:
        return

    out_dir = tile_dir.parent / f"{tile_dir.name}_squarepx"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(tile_dir.glob("*.tif"))
    print(f"[{args.city}] Resampling {len(tiles):,} tiles -> {out_dir}")

    t0 = time.time()
    for i, t in enumerate(tiles):
        out_path = out_dir / t.name
        if out_path.exists():
            continue
        try:
            resample_tile_to_square(t, out_path, target_gsd)
        except Exception as e:
            print(f"  [WARN] {t.name}: {str(e)[:100]}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(tiles) - i - 1) / rate / 60
            print(f"  {i+1}/{len(tiles)}  ({elapsed:.0f}s elapsed, ~{eta_min:.1f} min remaining)")

    print(f"[{args.city}] Done in {(time.time()-t0)/60:.1f} min. "
          f"Corrected tiles in: {out_dir}")
    print(f"Next: point your inference script's tile_dir at this new folder "
          f"(or swap it in for the original once you've spot-checked a few "
          f"tiles in QGIS), and update EFFECTIVE_GSD['{args.city}'] = {target_gsd:.4f}")


if __name__ == "__main__":
    main()