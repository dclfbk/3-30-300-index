# 3-30-300 Urban Greenery Index — Italian Cities

Project evaluates compliance with the **3-30-300 rule** for urban greenery across Italian cities at the civic-address level:

1. **Rule 3** — every resident should see at least 3 large, mature trees from home.
2. **Rule 30%** — at least 30% tree canopy cover in the surrounding neighbourhood.
3. **Rule 300m** — a public park within 300m walking distance.

Compliance is computed per civic address (not per grid cell or population estimate), using orthophoto imagery, OpenStreetMap, municipal open data, and computer-vision tree detection, rather than relying on coarser pan-European land-use products.

---

## Repository Structure

3-30-300-index/
├── config/
│ ├── config.py #set up, osm tags etc.
│ └── config.yaml # Per-city configuration
├── data/
│ └── {city}/ # One folder per city
├── notebooks/
│ ├── 02_download_orthophotos.py # WMS orthophoto tile downloader (check updates about wms, if necessary)
│ └── 10_csv2gpkg.py # CSV → GPKG (for testing deepforest results)
├── scr/
│ ├── tree_3/ # Rule 3 pipeline
│ │ ├── 01_deepforest.py # Zero-shot DeepForest
│ │ ├── 02_sam2.py # SAM2 crown polygon refinement
│ │ └── 03_tree_compliance.py #  gdal_viewshed
│ └── canopy_30/ # Rule 30% pipeline
│ │ ├── canopy_extract_rgb.py # ExG-based canopy ext
│ │ ├── canopy_extract_nir.py # NDVI-based canopy ext
│ │ └── 02_canopy_compliance.py # H3 hex-grid canopy coverage compliance
│ └── park_300/
│ ├── 01_green_spaces_extraction.py # creating one dataset
│ ├── 02_green_spaces_compliance.py #computing closest network path
│ └── 03_green_spaces_visualization.py 
│
├── 3_30_300_master.duckdb # Shared spatial database (all cities, all rules)


## Setup

### Environment

```bash
# General project dependencies
pip install -r requirements.txt --break-system-packages

# SAM2 (editable install — required, not optional; SAM2 isn't on PyPI as
# a standalone package, this is how `import sam2` resolves at all)
cd {ROOT_PATH}/sam2
pip install -e . --break-system-packages
cd {ROOT_PATH}

# Verify SAM2 resolves correctly before running any script that imports it
python3 -c "import sam2; print(sam2.__file__)"

# DuckDB spatial extension (one-time, environment-wide)
python3 -c "import duckdb; con = duckdb.connect(); con.execute('INSTALL spatial; LOAD spatial')"
```

### GPU

This project was primarily run on GPU resources with a persistent network volume at `/workspace/3-30-300-index`. For a successful execution of tree detection GPU resources are needed. Rule 30 and 300 is mainly CPU dependent.

### Database

All scripts should connect to the single shared DuckDB file via the retry-wrapped connector, not the bare one — concurrent city-processing runs regularly contend for the write lock (see info below):

```python
from config.config import connect_db_with_retry
con = connect_db_with_retry()  # opens /workspace/3-30-300-index/3_30_300_master.duckdb, retries on lock conflicts
```

`connect_db()` (no retry) still exists in `config.py` for read-only/one-off/interactive use where a lock conflict isn't worth retrying — but any script doing a real write pass against the shared DB should use `connect_db_with_retry()`.

`DB_PATH` in `config.py` is derived relative to the repo root (`ROOT_PATH / "3_30_300_master.duckdb"`), so it resolves correctly on any machine as long as the `.duckdb` file sits at the project root.

### Shared helpers in `config.py`

Rather than re-implementing per-script, these now live centrally and should be imported, not copy-pasted:

| Name | Purpose |
|---|---|
| `connect_db`, `connect_db_with_retry` | DB connection, with/without polite lock-retry |
| `gdf_to_duckdb`, `load_table`, `gpkg_to_duckdb` | GeoDataFrame ↔ DuckDB I/O |
| `get_city_cfg`, `get_output_paths`, `get_produced_paths` | Per-city config.yaml lookups and standard output path layout |
| `find_tile_dir`, `load_boundary` | Locate a city's ortho-tile directory / load its boundary geometry, with base-city fallback for RGB/NIR/CIR variants |
| `strip_source_suffix` | Strip `_rgb`/`_nir`/`_cir` from a city key to get the base city |
| `fetch_osm_tag`, `build_osm_exclusion`, `build_osm_inclusion` | Per-tag-cached OSM fetch and exclusion/inclusion polygon assembly, scoped (test/full) and polygon-based |
| `OSM_EXCLUDE_TAGS`, `OSM_INCLUDE_TAGS`, `WATER_TAG_KEYS`, `OSM_GREEN_TAGS`, `PRIVATE_ACCESS` | Shared OSM tag dictionaries |
| `OSM_BUFFER_M`, `WATER_BUFFER_M`, `TEST_OSM_BUFFER_DEG` | Buffer distances used by the OSM exclusion/inclusion helpers |
| `CRS_WGS`, `CRS_METRIC`, `DATA_DIR`, `ROOT_PATH`, `DB_PATH` | Path/CRS constants |

---

## Data Layout

Each city has a folder under `data/{city}/`:
data/{city}/
├── {city}ortho_tiles/ # Downloaded orthophoto tiles (GeoTIFF)
├── canopy_polygons.gpkg # Rule 30% extraction output
├── tree_detections_zeroshot{city}.gpkg # Rule 3 SAM2 crown output

Cities with both RGB and NIR/CIR orthophoto sources use a `{city}_rgb` variant folder (e.g. `torino_rgb`, `bologna_rgb`) for the RGB-specific pipeline outputs, since NIR/CIR imagery is unsuitable for RGB-trained models (DeepForest) and ExG-based canopy extraction — vegetation appears in false colour and channel order does not match what these pipelines assume. Physical-city data shared across variants (building footprints, civic addresses, administrative boundary) is **not** duplicated per variant; both `torino` and `torino_rgb` resolve to the same underlying `raw.torino_*` tables via base-city fallback (`strip_source_suffix` + `find_tile_dir`/`load_boundary`'s fallback logic) in each script.

DuckDB tables follow this convention:

| Schema | Table | Contents |
|---|---|---|
| `raw` | `{city}_boundary`, `{city}_buildings`, `{city}_buildings_sn`, `{city}_water` | Source data (buildings, civic addresses, admin boundary) |
| `preprocessing` | `{city}_canopy`, `tree_polygons_{city}`, `hex_grid_{city}` | Pipeline intermediate outputs |
| `rule_3` | `{city}_results` | Per-floor tree-visibility compliance |
| `processed` | `canopy_compliance_{city}`, `canopy_compliance_{city}_by_civico` | Rule 30% compliance |

---

## Running the Pipeline for a New City

### 1. Download orthophotos

```bash
python notebooks/02_download_orthophotos.py --city {city}
python notebooks/02_download_orthophotos.py --city {city} --list-layers   # find the right WMS layer first
```

Add the city's WMS endpoint to `CITY_WMS` in the download script before running. **GSD is always measured directly from the downloaded tiles' geotransform**, never trusted from provider-stated nominal resolution — some providers have been found to misstate resolution by 3× or more (see Appendix on Genova's GSD correction in the thesis). If a city's measured x/y GSD differ by more than ~2%, its tiles have a non-square-pixel artefact (see `fix_nonsquare_pixels.py` below) and must be corrected before detection.

### 2. Rule 3 — Tree Detection

```bash
python scr/tree_3/01_deepforest.py --city {city} --n-tiles 15   # quick test first
python scr/tree_3/01_deepforest.py --city {city}                # full run, or --city all for every configured city

python scr/tree_3/02_sam2.py --city {city} --test    # quick test first
python scr/tree_3/02_sam2.py --city {city}

python scr/tree_3/03_tree_compliance.py --city {city}
```

Each city typically needs its own calibrated confidence/IoU thresholds (`batch_zeroshot_inference.py`'s `EFFECTIVE_GSD`, `IOU_THRESHOLD`, `INFER_SCORE_THRESH`) — tune by visual inspection in QGIS before a full run. This is also a main reason the whole pipeline is not chained into one script — each step needs a visual check, rather than fully automatic processing.

### 3. Rule 30% — Canopy Coverage

```bash
python scr/canopy_30/canopy_extract_rgb.py --city {city} --n-tiles 15 --skip-osm   # quick test first
python scr/canopy_30/canopy_extract_rgb.py --city {city}                           # full run

# for cities with NIR/CIR imagery instead of RGB:
TEST_MODE=1 python scr/canopy_30/canopy_extract_nir.py --city {city}_cir --boundary-city {city}
python scr/canopy_30/canopy_extract_nir.py --city {city}_nir

python scr/canopy_30/02_canopy_compliance.py --city {city}
```

Per-city spectral thresholds (ExG, hue/saturation range, brightness gate, water guard, morphological closing) live in `config.yaml` under each city's `canopy_extraction:` block — **not** hardcoded in the script. Add a new city's block there rather than editing the script directly.

**NDVI thresholds are only partially migrated to config.yaml.** `canopy_extract_nir.py` checks a city's `ndvi_extraction:` block in `config.yaml` first, but falls back to hardcoded `CITY_NDVI_THRESH_FALLBACK`/`CITY_MIN_AREA_M2_FALLBACK` dicts in the script for any city that doesn't have one yet. Prefer adding an `ndvi_extraction:` block for new cities rather than extending the fallback dicts.

### 4. Rule 300m — Park Access

```bash
python scr/park_300/01_green_spaces_extraction.py --city {city}   # merges OSM + DBSN + municipal park layers

```

---

## Configuration

`config.yaml` holds one block per city:

```yaml
{city}:
  query: "City, Italy"                 # Nominatim query for boundary fetch
  building_id_col: "NUMEROCOMPLETO"
  avg_tree_height_m: 12.2               # from ETH Global Canopy Height dataset
  data_sources:
    boundary: "data/{city}/{city}_boundary.gpkg"
    buildings_sn: "data/{city}/{city}_buildings_streetnum.gpkg"
    wms: "https://..."                  # orthophoto WMS endpoint
  canopy_extraction:
    exg_thresh: 0.15
    hue_range: [0.20, 0.28]
    sat_min: 0.28
    water_guard_gap: 0.03               # only for cities with turbid/algae-tinted water
    closing_enabled: true               # only where dense canopy fragments/shrinks
  ndvi_extraction:                      # only for _nir/_cir cities; falls back to a
    ndvi_thresh: 0.30                   # hardcoded dict in canopy_extract_nir.py if omitted
    min_area_m2: 1.5
```

Civic-address schemas vary genuinely between cities (different providers, different column-naming conventions). `03_tree_compliance.py` tries several candidate column names per role (via, number, suffix) case-insensitively before falling back to explicit per-city overrides (`civico_via_col`, `address_num_col`, etc.) in `config.yaml`. 

---

##  Problems that might occur: 

**WMS connection** Use --list-layers command to check whihc layers are currently available, it can happend that orthophotos which has been used for original processing are no longer available

**Orthophoto coverage** Use /notebooks/07_check_coverage.py to check if tiles of given city are actually correcponding with OSM downloaded municipality boundary - it can happend that orthophotos are not covering whole boundary.

**Threshold adjustment and poor results** Pipelines for both 30% rule and 3 trees and not connected with sh script on puprose since each steps need an individual calibration. This is why a command --n-tiles <number> need to be used before running full city.

**FN/FP zeroshot/sam2 processing** This is known limitation of the pipeline, where thresholds need to be adjusted individually, which cities with worse quality of orthophotos or other posibile influences tend to have lower quality outputs. 

**Concurrent DB writers.** Multiple scripts running simultaneously against the shared DuckDB file can hit `Conflicting lock is held` errors. `connect_db_with_retry()` (used in `01_zeroshot_inference.py`, `02_sam2.py`, `02_canopy_compliance.py`, `03_tree_compliance.py`, and the park-extraction script) retries politely rather than crashing and losing in-memory results. `02_sam2.py` additionally writes its per-tile checkpoint to GPKG/CSV before ever touching the shared DB, and does its one shared-DB commit at the very end of a multi-hour run via `connect_db_with_retry`, so a failed final DB commit never loses computed detections.

**DB corruption.** If `connect_db()` raises `IO Error: Corrupt database file`, **back up the file immediately** before further reads (`cp 3_30_300_master.duckdb 3_30_300_master.duckdb.BACKUP`).

---