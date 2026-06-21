"""
download_orthophotos.py  —  Batch WMS orthophoto tile downloader (patched for Emilia-Romagna)
Usage:
    python download_orthophotos.py --city bologna_rgb
    python download_orthophotos.py --city bologna_rgb --retry-failed
"""
import argparse
import io
import math
import os
import sys
import time
from glob import glob

import numpy as np
import rasterio
from PIL import Image
from owslib.wms import WebMapService
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

ROOT_PATH = "/Users/terezasaskova/Library/Mobile Documents/com~apple~CloudDocs/3-30-300-index/"
sys.path.insert(0, ROOT_PATH)
os.chdir(ROOT_PATH)

CITY_WMS = {
    "bologna": {
        "url": "https://servizigis.regione.emilia-romagna.it/wms/rer2023_24_nir",
        "layer": "RER2023_24_NIR",
        "wms_version": "1.3.0",
        "nir": True,
        "year": 2023,
        "max_px": 2048,
        "fmt": "image/tiff",
        "notes": "AGEA 2023 NIR — full Emilia-Romagna coverage",
    },
    "bolzano": {
        "url": "https://geoservices.buergernetz.bz.it/mapproxy/ows",
        "layer": "p_bz-Orthoimagery:Aerial-2023-CIR",
        "wms_version": "1.3.0",
        "nir": True,
        "year": 2023,
        "max_px": 2000,
        "notes": "Bolzano 2023 CIR/NIR 20cm",
    },
    "genova": {
        "url": "https://geoservizi.regione.liguria.it/geoserver/M2555/ows",
        "layer": "L10401",
        "wms_version": "1.3.0",
        "nir": False,
        "year": 2022,
        "max_px": 4096,
        "notes": "Liguria GeoServer — verify layer with --list-layers",
    },
    "milano": {
        "url": "https://www.cartografia.servizirl.it/arcgis5/services/BaseMap/Ortofoto2021/ImageServer/WMSServer",
        "layer": "0",
        "wms_version": "1.1.1",
        "nir": False,
        "year": 2021,
        "max_px": 4096,
        "notes": "Lombardia 2021 RGB",
    },
    "perugia": {
        "url": "https://siat.regione.umbria.it/arcgis/services/public/ORTOFOTO_2020_WGS84_UTM33N/MapServer/WMSServer",
        "layer": "0", "wms_version": "1.1.1", "nir": False, "year": 2020, "max_px": 4096,
        "notes": "Umbria 2020 RGB 20cm AGEA — no NIR available",
    },
    "roma": {
        "url": "https://geoportale.comune.roma.it/geoserver/ows",
        "layer": "REGLAZ:2020_AGEA_25833_COG",
        "wms_version": "1.3.0",
        "nir": False,
        "year": 2020,
        "max_px": 4096,
        "notes": "Roma/Lazio AGEA 2020 RGB orthophoto via Comune di Roma GeoServer",
    },
    "torino": {
        "url": "https://geomap.reteunitaria.piemonte.it/mapproxy/service",
        "layer": "coto_ortofoto_ir_2023",
        "wms_version": "1.3.0",
        "nir": True,
        "year": 2023,
        "max_px": 1024,
        "fmt": "image/png",
        "notes": "Torino 2023 NIR orthophoto — Comune di Torino via CSI Piemonte",
    },
    "trento": {
        "url": "https://siat.provincia.tn.it/geoserver/stem/ecw-rgb-2015/wms",
        "layer": "ecw-rgb-2015",
        "wms_version": "1.1.1",
        "nir": False,
        "year": 2015,
        "max_px": 4096,
        "notes": "Trento 2015 RGB — verify layer with --list-layers",
    },
    "venezia": {
        "url": "https://idt2-geoserver.regione.veneto.it/geoserver/ows",
        "layer": "rv:ortofoto_agea_2024",
        "wms_version": "1.3.0",
        "nir": False,
        "year": 2024,
        "max_px": 4096,
        "notes": "Veneto 2024 RGB — verify layer with --list-layers",
    },
    "verona": {
        "url": "https://idt2-geoserver.regione.veneto.it/geoserver/ows",
        "layer": "rv:ortofoto_agea_2024",
        "wms_version": "1.3.0",
        "nir": False,
        "year": 2024,
        "max_px": 4096,
        "notes": "Veneto 2024 RGB — verify layer with --list-layers",
    },
    "bologna_rgb": {
        "base_city": "bologna",
        "url": "https://servizigis.regione.emilia-romagna.it/wms/agea2023_rgb",
        "layer": "Agea2023_RGB",
        "wms_version": "1.1.1",
        "native_crs": "EPSG:3003",
        "nir": False,
        "year": 2023,
        "max_px": 2048,
        "fmt": "image/png",
        "notes": "AGEA 2023 RGB — Emilia-Romagna coverage, native EPSG:3003",
    },
    "firenze": {
        "url": "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms?map=owsofc_rt",
        "layer": "rt_ofc.5k25.32bit.nir",
        "wms_version": "1.1.1",
        "nir": True,
        "year": 2023,
        "max_px": 4096,
        "fmt": "image/tiff",
        "notes": "Toscana 2023 CIR (NIR/R/G) — Regione Toscana Geoscopio OFC", # <-- Make sure this line is exactly here
    },
    "firenze_rgb": {
        "base_city": "firenze",
        "url": "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms?map=owsofc_rt",
        "layer": "rt_ofc.5k23.32bit",
        "wms_version": "1.1.1",
        "nir": False,
        "year": 2023,
        "max_px": 4096,
        "fmt": "image/tiff",
        "notes": "Toscana 2023 RGB 20cm — same endpoint as CIR, different layer (32bit colore)",
    },
    "torino_rgb": {
        "base_city": "torino",
        "url": "https://geomap.reteunitaria.piemonte.it/mapproxy/service",
        "layer": "coto_ortofoto_2023",
        "wms_version": "1.3.0",
        "nir": False,
        "year": 2023,
        "max_px": 1024,
        "fmt": "image/png",
        "notes": "Torino 2023 RGB — CSI Piemonte, same server as NIR variant",
    },
    "bolzano_rgb": {
        "base_city": "bolzano",   
        "url": "https://geoservices.buergernetz.bz.it/mapproxy/ows",
        "layer": "p_bz-Orthoimagery:Aerial-2023-RGB",
        "wms_version": "1.3.0",
        "nir": False,
        "year": 2023,
        "max_px": 2000,
        "notes": "Bolzano 2023 RGB 20cm — mapproxy, RGB twin of CIR layer",
    },
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--city",         required=True, help="City name or 'all'")
p.add_argument("--gsd",          type=float, default=0.20)
p.add_argument("--tile-px",      type=int,   default=2048)
p.add_argument("--overlap-px",   type=int,   default=64)
p.add_argument("--save-crs",     default="EPSG:32632")
p.add_argument("--layer",        default=None)
p.add_argument("--fmt",          default="image/jpeg")
p.add_argument("--out-dir",      default=None)
p.add_argument("--boundary",     default=None)
p.add_argument("--list-layers",  action="store_true")
p.add_argument("--retry-failed", action="store_true")
p.add_argument("--max-retries",  type=int, default=3)
p.add_argument("--blank-std-thresh", type=float, default=1.0,
               help="If array std() is below this, treat response as blank/error and retry")
args = p.parse_args()

DEG_PER_M = 1 / 111_320

def get_boundary_data(city, boundary_path):
    import geopandas as gpd
    if boundary_path:
        gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    else:
        from config.config import connect_db, load_table
        con = connect_db(read_only=True)
        cfg = CITY_WMS.get(city, {})
        
        lookup_city = cfg.get("base_city", city)
        lookup_city = lookup_city.replace("_rgb", "").replace("_nir", "").replace("_cir", "")
        
        gdf = load_table(con, "raw", f"{lookup_city}_boundary").to_crs("EPSG:4326")
        con.close()
    
    boundary_geom = unary_union(gdf.geometry.values)
    return boundary_geom, boundary_geom.bounds

def list_layers(cfg):
    print(f"Connecting to: {cfg['url']}")
    wms = WebMapService(cfg["url"], version=cfg["wms_version"], timeout=60)
    print(f"\nAvailable layers ({len(wms.contents)}):")
    for name, layer in wms.contents.items():
        title = getattr(layer, "title", "")
        print(f"  {name!r:60s}  # {title}")
    sys.exit(0)

def download_city(city, cfg):
    layer_name = args.layer or cfg["layer"]
    gsd        = args.gsd
    tile_px    = args.tile_px
    overlap_px = args.overlap_px
    req_px     = min(tile_px + 2 * overlap_px, cfg["max_px"])

    out_dir = args.out_dir or os.path.join("data", city, f"{city}_ortho_tiles")
    os.makedirs(out_dir, exist_ok=True)

    failed_log = os.path.join(out_dir, "failed_tiles.txt")

    boundary_geom, (xmin, ymin, xmax, ymax) = get_boundary_data(city, args.boundary)
    
    gsd_deg     = gsd * DEG_PER_M
    tile_deg    = tile_px * gsd_deg
    overlap_deg = overlap_px * gsd_deg

    cols = math.ceil((xmax - xmin) / tile_deg)
    rows = math.ceil((ymax - ymin) / tile_deg)

    existing_tifs = len(glob(os.path.join(out_dir, "tile_*.tif")))
    if existing_tifs >= (cols * rows) * 0.75 and not args.retry_failed:
        print(f"[{city}] Skipping: {existing_tifs} tiles already exist on disk out of max {cols * rows} grid cells.")
        return

    print(f"\n[{city}] Extent: ({xmin:.4f}, {ymin:.4f}) -> ({xmax:.4f}, {ymax:.4f})")
    print(f"  WMS   : {cfg['url']}")
    print(f"  Year  : {cfg['year']}  |  NIR: {cfg['nir']}  |  {cfg['notes']}")
    print(f"  Grid  : {cols} cols x {rows} rows = {cols * rows} bounding box tiles")

    print("  Connecting ...")
    try:
        wms = WebMapService(cfg["url"], version=cfg["wms_version"], timeout=120)
    except Exception as e:
        print(f"  ERROR connecting: {e}")
        return

    layers = list(wms.contents.keys())
    if layer_name is None:
        layer_name = layers[0]
        print(f"  Auto-selected layer: {layer_name!r}")
    if layer_name not in wms.contents:
        print(f"  ERROR: layer {layer_name!r} not found.")
        return

    transformer = Transformer.from_crs("EPSG:4326", args.save_crs, always_xy=True)
    
    # Track projection requirements cleanly
    if "native_crs" in cfg:
        native_crs = cfg["native_crs"]
    elif "bologna" in city:
        native_crs = "EPSG:32632"
    else:
        native_crs = "EPSG:3857"
        
    t_to_native = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)

    retry_set = set()
    if args.retry_failed and os.path.exists(failed_log):
        with open(failed_log) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    retry_set.add((int(parts[0]), int(parts[1])))
        print(f"  Retrying {len(retry_set)} previously failed tiles")

    failed = []
    skipped_count = 0
    downloaded_count = 0
    
    with tqdm(total=rows * cols, desc=city) as pbar:
        for row in range(rows):
            for col in range(cols):
                pbar.update(1)

                if args.retry_failed and (row, col) not in retry_set:
                    continue

                tx0 = xmin + col * tile_deg
                ty0 = ymin + row * tile_deg
                tx1 = min(tx0 + tile_deg, xmax)
                ty1 = min(ty0 + tile_deg, ymax)

                tile_footprint = box(tx0, ty0, tx1, ty1)
                if not tile_footprint.intersects(boundary_geom):
                    skipped_count += 1
                    continue

                tile_path = os.path.join(out_dir, f"tile_{row:04d}_{col:04d}.tif")
                if os.path.exists(tile_path) and not args.retry_failed:
                    downloaded_count += 1
                    continue

                ex0 = max(tx0 - overlap_deg, xmin)
                ey0 = max(ty0 - overlap_deg, ymin)
                ex1 = min(tx1 + overlap_deg, xmax)
                ey1 = min(ty1 + overlap_deg, ymax)

                px_w = max(1, min(round((ex1 - ex0) / gsd_deg), req_px))
                px_h = max(1, min(round((ey1 - ey0) / gsd_deg), req_px))

                wx0, wy0 = t_to_native.transform(ex0, ey0)
                wx1, wy1 = t_to_native.transform(ex1, ey1)

                # CRITICAL PATCH: Handle WMS 1.3.0 Axis Ordering Swap for EPSG:32632
                if "bologna" in city and cfg["wms_version"] == "1.3.0":
                    wms_bbox = (wy0, wx0, wy1, wx1)
                else:
                    wms_bbox = (wx0, wy0, wx1, wy1)

                success = False
                for attempt in range(args.max_retries):
                    try:
                        tile_fmt = cfg.get("fmt", args.fmt)
                        resp = wms.getmap(
                            layers=[layer_name],
                            srs=native_crs,
                            bbox=wms_bbox,
                            size=(px_w, px_h),
                            format=tile_fmt,
                            transparent=False,
                        )

                        raw = resp.read()
                        
                        # --- DIAGNOSTIC CHECK: Is the server returning text/XML instead of an image? ---
                        content_type = resp.info().get("Content-Type", "").lower()
                        if "xml" in content_type or "text" in content_type:
                            print(f"\n[SERVER ERROR DETECTED] Tile ({row},{col}) returned text instead of an image:")
                            try:
                                print(raw.decode("utf-8")[:1000]) # Print first 1000 characters of the error
                            except:
                                print(raw[:500])
                            sys.exit("\nStopping script to fix the server exception above.")
                        # -----------------------------------------------------------------------------

                        if tile_fmt in ("image/tiff", "image/geotiff"):

                            with rasterio.MemoryFile(raw) as memfile:
                                with memfile.open() as src:
                                    n_bands = src.count
                                    arr = src.read()
                                    arr = np.moveaxis(arr, 0, -1)
                                    if arr.dtype != np.uint8:
                                        arr = ((arr / arr.max()) * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
                        else:
                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                            arr = np.array(img)
                            n_bands = 3

                        if arr.std() < args.blank_std_thresh:
                            raise ValueError(f"blank/uniform response (std={arr.std():.4f})")

                        mx0, my0 = transformer.transform(ex0, ey0)
                        mx1, my1 = transformer.transform(ex1, ey1)
                        tmx0, tmy0 = transformer.transform(tx0, ty0)
                        tmx1, tmy1 = transformer.transform(tx1, ty1)

                        with rasterio.open(
                            tile_path, "w",
                            driver="GTiff",
                            height=px_h, width=px_w,
                            count=n_bands,
                            dtype=arr.dtype,
                            crs=CRS.from_string(args.save_crs),
                            transform=from_bounds(mx0, my0, mx1, my1, px_w, px_h),
                            compress="lzw",
                            tiled=True,
                            blockxsize=256, blockysize=256,
                        ) as dst:
                            for i in range(n_bands):
                                dst.write(arr[:, :, i], i + 1)
                            dst.update_tags(
                                city=city,
                                year=cfg["year"],
                                nir=cfg["nir"],
                                layer=layer_name,
                                overlap_px=overlap_px,
                                true_xmin=tmx0, true_ymin=tmy0,
                                true_xmax=tmx1, true_ymax=tmy1,
                            )

                        success = True
                        downloaded_count += 1
                        time.sleep(0.5)
                        break

                    except Exception as e:
                        if attempt < args.max_retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            tqdm.write(f"  FAILED ({row},{col}) after {args.max_retries} attempts: {e}")
                            failed.append((row, col))

    print(f"\nDone! Downloaded/Verified: {downloaded_count} | Skipped outside boundary: {skipped_count} | Failed: {len(failed)}")
    if failed:
        with open(failed_log, "w") as f:
            for r, c in failed:
                f.write(f"{r},{c}\n")

if args.list_layers:
    city = args.city if args.city != "all" else list(CITY_WMS.keys())[0]
    if city not in CITY_WMS:
        sys.exit(f"Unknown city: {city}")
    list_layers(CITY_WMS[city])

cities_to_run = list(CITY_WMS.keys()) if args.city == "all" else [args.city]

for city in cities_to_run:
    if city not in CITY_WMS:
        print(f"No WMS configured for '{city}'.")
        continue
    download_city(city, CITY_WMS[city])