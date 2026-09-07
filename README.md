# 3-30-300 Urban Greenery Index — Italian Cities

Evaluates compliance with the **3-30-300 rule** for urban greenery across Italian cities, at the civic-address level:

1. **Rule 3**: 3+ mature trees visible from home.
2. **Rule 30%**: 30%+ tree canopy cover in the surrounding neighbourhood.
3. **Rule 300m**: a public park within 300m walking distance.

Computed per civic address (not per grid cell or population estimate), using orthophoto imagery, OpenStreetMap, municipal open data, and computer-vision tree detection.

---

## Repository Structure

```
3-30-300-index/
├── config/
│   ├── config.py                 # Shared helpers, constants, OSM tags
│   ├── config.yaml                # Per-city configuration
│   ├── gh_cities.yml               # City → regional OSM extract mapping (for GraphHopper)
│   ├── config_template.yml         # GraphHopper server config template
│   └── run_gh.sh                   # Starts local GraphHopper routing server for a city
├── data/
│   └── {city}/                    # One folder per city
├── notebooks/                     # Boundary/orthophoto/streetnum/height downloaders,
│                                   # coverage checks, pixel-fix utilities (01–11)
├── scr/
│   ├── tree_3/                    # Rule 3 pipeline
│   │   ├── 01_deepforest.py          # Zero-shot DeepForest
│   │   ├── 02_sam2_inference.py      # SAM2 crown polygon refinement
│   │   └── 03_tree_compliance.py     # gdal_viewshed-based compliance
│   ├── canopy_30/                 # Rule 30% pipeline
│   │   ├── 01_a_canopy_extraction_rgb.py   # ExG-based canopy extraction
│   │   ├── 01_b_canopy_extraction_nir.py   # NDVI-based canopy extraction
│   │   └── 02_canopy_compliance.py         # H3 hex-grid canopy coverage
│   └── park_300/                  # Rule 300m pipeline
│       ├── 01_green_spaces_dataset_0ha.py  # Merges OSM + DBSN + municipal layers
│       ├── 01_green_spaces_dataset_1ha.py  # (0ha/1ha = min. park-size threshold — confirm which is primary)
│       ├── 02_green_spaces_compliance.ipynb  # GraphHopper walking-distance routing
│       └── 03_green_spaces_vizualization_*.py
└── 3_30_300_master.duckdb        # Shared spatial database (all cities, all rules)
```

---

## Setup

Before running setup.sh, set ROOT_PATH to wherever you've cloned this repo:

```bash
export ROOT_PATH=/path/to/your/3-30-300-index
./setup.sh
```

Run `setup.sh` (below) for the one-time environment install. Summary of what it covers:

- Python dependencies (`requirements.txt`)
- SAM2 (editable install — required, not on PyPI as a standalone package)
- DuckDB spatial extension
- GDAL CLI tools (`gdal_viewshed`, used by Rule 3 compliance — **not** provided by `pip install gdal`/rasterio)
- Java 17 + GraphHopper jar (used by Rule 300m routing)

**Per-session, before running Rule 3:**
```bash
python3 -c "import sam2; print(sam2.__file__)"   # confirm SAM2 resolves
which gdal_viewshed                               # confirm GDAL CLI is present
```

**Per-session, before running Rule 300m:**
```bash
./config/run_gh.sh {city}     # downloads regional OSM extract if missing, starts GraphHopper server
# wait for "Server - Started" log line, then in another terminal:
curl -s http://localhost:{GH_PORT}/health
```
Extracts in `config/gh_cities.yml` are **regional**, not per-city — several cities share one extract (e.g. Bologna, Bolzano, Trento, Trieste, Venezia, Verona all use `nord-est-latest.osm.pbf`).

### GPU

Tree detection (Rule 3) requires GPU. Rule 30% and Rule 300m are CPU-only.

### Database

Always use the retry-wrapped connector for scripts doing real write passes — concurrent city runs regularly contend for the write lock:
```python
from config.config import connect_db_with_retry
con = connect_db_with_retry()
```
`connect_db()` (no retry) is fine for read-only/interactive use.

If `connect_db()` raises `IO Error: Corrupt database file`, **back up immediately**: `cp 3_30_300_master.duckdb 3_30_300_master.duckdb.BACKUP`.

### Key shared helpers (`config.py`)

| Name | Purpose |
|---|---|
| `connect_db`, `connect_db_with_retry` | DB connection, with/without lock-retry |
| `gdf_to_duckdb`, `load_table`, `gpkg_to_duckdb` | GeoDataFrame ↔ DuckDB I/O |
| `get_city_cfg`, `get_output_paths` | Per-city config.yaml lookups, output path layout |
| `find_tile_dir`, `load_boundary`, `strip_source_suffix` | Ortho-tile/boundary lookup with RGB/NIR/CIR base-city fallback |
| `fetch_osm_tag`, `build_osm_exclusion`, `build_osm_inclusion` | Cached OSM fetch + exclusion/inclusion polygons |
| `GH_PORT`, `WALKING_DISTANCE`, `EUCLID_CUTOFF` | GraphHopper port + Rule 300m thresholds |
| `CRS_WGS`, `CRS_METRIC`, `DATA_DIR`, `ROOT_PATH`, `DB_PATH` | Path/CRS constants |

---

## Data Layout

```
data/{city}/
├── {city}_ortho_tiles/                          # Downloaded orthophoto tiles (GeoTIFF)
├── canopy_polygons.gpkg                         # Rule 30% output
└── tree_detections_zeroshot_{city}.gpkg         # Rule 3 SAM2 crown output
```

Cities with both RGB and NIR/CIR sources use a `{city}_rgb` variant folder (NIR/CIR is unsuitable for RGB-trained DeepForest and ExG canopy extraction). Physical-city data (buildings, civic addresses, boundary) is **not** duplicated per variant — both resolve to the same `raw.{city}_*` tables via base-city fallback.

| Schema | Table | Contents |
|---|---|---|
| `raw` | `{city}_boundary`, `{city}_buildings`, `{city}_buildings_sn`, `{city}_water` | Source data |
| `preprocessing` | `{city}_canopy`, `tree_polygons_{city}`, `hex_grid_{city}` | Pipeline intermediates |
| `rule_3` | `{city}_results` | Per-floor tree-visibility compliance |
| `processed` | `canopy_compliance_{city}`, `canopy_compliance_{city}_by_civico` | Rule 30% compliance |
| `rule_300m` *(confirm schema name — notebook currently writes to `rule_300m_2`)* | `{city}_routing`, `{city}_results` | Rule 300m routing + compliance |

---

## Running the Pipeline for a New City

```bash
# 1. Orthophotos
python notebooks/02_download_orthophotos.py --city {city} --list-layers   # find the right WMS layer
python notebooks/02_download_orthophotos.py --city {city}

# 2. Rule 3 — Tree Detection
python scr/tree_3/01_deepforest.py --city {city} --n-tiles 15   # test first
python scr/tree_3/01_deepforest.py --city {city}
python scr/tree_3/02_sam2_inference.py --city {city} --test
python scr/tree_3/02_sam2_inference.py --city {city}
python scr/tree_3/03_tree_compliance.py --city {city}

# 3. Rule 30% — Canopy Coverage
python scr/canopy_30/01_a_canopy_extraction_rgb.py --city {city} --n-tiles 15 --skip-osm
python scr/canopy_30/01_a_canopy_extraction_rgb.py --city {city}
# for NIR/CIR cities instead:
TEST_MODE=1 python scr/canopy_30/01_b_canopy_extraction_nir.py --city {city}_cir --boundary-city {city}
python scr/canopy_30/01_b_canopy_extraction_nir.py --city {city}_nir
python scr/canopy_30/02_canopy_compliance.py --city {city}

# 4. Rule 300m — Park Access
./config/run_gh.sh {city}    # start GraphHopper first, separate terminal
python scr/park_300/01_green_spaces_dataset_0ha.py --city {city}
# then run 02_green_spaces_compliance.ipynb
```

**GSD is always measured directly from tile geotransform**, never trusted from provider metadata (some misstate by 3×+). If x/y GSD differ by >2%, run `notebooks/11_fix_nonsquare_pixels.py` before detection.

Each city needs its own calibrated thresholds (`EFFECTIVE_GSD`, `IOU_THRESHOLD`, `INFER_SCORE_THRESH` for Rule 3; ExG/hue/sat/brightness for Rule 30%) — tune via `--n-tiles` test runs and QGIS inspection before a full run. This is why steps aren't chained into one script.

Canopy thresholds live in `config.yaml` under each city's `canopy_extraction:`/`ndvi_extraction:` block — add new cities there, not as script edits. NDVI has a hardcoded fallback dict in `01_b_canopy_extraction_nir.py` for cities without a config block yet.

Civic-address schemas vary per city/provider. `03_tree_compliance.py` tries several candidate column names case-insensitively before falling back to `config.yaml` overrides (`civico_via_col`, `address_num_col`, etc.).

---

## Problems That Might Occur

**WMS connection.** Use `--list-layers` — original processing layers can go offline.

**Orthophoto coverage.** Use `notebooks/07_check_coverage.py` — tiles sometimes don't cover the full OSM boundary.

**FN/FP in zeroshot/SAM2.** Known limitation — thresholds need per-city tuning; lower-quality orthophotos produce lower-quality outputs.

**Concurrent DB writers.** `Conflicting lock is held` errors happen under simultaneous city runs. Scripts using `connect_db_with_retry()` retry politely. `02_sam2_inference.py` checkpoints to GPKG/CSV before its final DB commit, so a failed commit never loses computed detections.

**GraphHopper graph rebuild.** `run_gh.sh` wipes `graph-cache` on every invocation — expect a rebuild delay each time you (re)start it, even for a previously-run city/region.