import os
import sys
import argparse
import time
import torch
import pandas as pd
from pathlib import Path

# Ensure we can import local config modules
sys.path.insert(0, "/workspace/3-30-300-index")
os.chdir("/workspace/3-30-300-index")

# ────────────────────────────────────────────────────────────────
# Imports from your project structure
# ────────────────────────────────────────────────────────────────
try:
    from deepforest import main as df_main
    from config.config import get_city_cfg, connect_db, gdf_to_duckdb
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure 'config' module exists in /workspace/3-30-300-index/config/")
    sys.exit(1)

# ────────────────────────────────────────────────────────────────
# Configuration Constants
# ────────────────────────────────────────────────────────────────
WORK_DIR = "/workspace/3-30-300-index"
MODEL_SAVE      = str(Path(WORK_DIR) / "deepforest_tcd_eu_v2.pt")

# Inference Params 
PATCH_SIZE          = 400
PATCH_OVERLAP       = 0.2
IOU_THRESHOLD       = 0.2
INFER_SCORE_THRESH  = 0.15

# Target Cities - Edit this list to include all your cities
ALL_CITIES = ["perugia", "bologna", "firenze", "genova", "torino", "venezia", "verona", "trento", "bolzano", "bari", "roma", "milano"]

# ────────────────────────────────────────────────────────────────
# Core Logic
# ────────────────────────────────────────────────────────────────

def load_model(device):
    """Loads the base DeepForest model and applies fine-tuned weights."""
    print("\n[INIT] Loading DeepForest model...")
    
    m = df_main.deepforest()
    
    # 1. Load Base Model
    print("   -> Loading base 'weecology/deepforest-tree' model...")
    m.load_model(model_name="weecology/deepforest-tree", revision="main")
    
    # 2. Load Fine-Tuned Weights
    if MODEL_SAVE and Path(MODEL_SAVE).exists():
        print(f"   -> Applying fine-tuned weights from {Path(MODEL_SAVE).name}...")
        try:
            state = torch.load(MODEL_SAVE, map_location=device, weights_only=False)
            # Handle different save formats (dict vs wrapper)
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            m.model.load_state_dict(state_dict)
            print("   -> Weights loaded successfully.")
        except Exception as e:
            print(f"   -> WARNING: Failed to load fine-tuned weights: {e}. Using base model only.")
    else:
        print(f"   -> WARNING: Fine-tuned model not found at {MODEL_SAVE}. Using base model.")

    # 3. Configure for Inference
    m.config["score_thresh"] = INFER_SCORE_THRESH
    m.config["batch_size"] = 2
    m.config["workers"] = 0
    
    # Create trainer (Required for predict_tile in recent versions)
    m.create_trainer()
    print("[INIT] Model ready.\n")
    return m

def process_single_city(city, tile_dir, model):
    """Runs inference on a single city's tiles."""
    print(f"\n>>> Processing City: {city.upper()}")
    print(f"   Tile Directory: {tile_dir}")
    
    if not tile_dir.exists():
        print(f"   [SKIP] Directory not found.")
        return []

    tif_files = sorted(tile_dir.glob("*.tif"))
    if not tif_files:
        print(f"   [SKIP] No .tif files found.")
        return []
    
    print(f"   Found {len(tif_files)} tiles.")
    city_predictions = []

    for i, tif_path in enumerate(tif_files):
        # Skip small/broken files
        if tif_path.stat().st_size < 1_000_000: 
            continue

        try:
            print(f"   [{i+1}/{len(tif_files)}] Running: {tif_path.name} ...", end=" ", flush=True)
            
            preds = model.predict_tile(
                path=str(tif_path),
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                iou_threshold=IOU_THRESHOLD,
                dataloader_strategy="window",
            )
            
            if not preds.empty:
                # Tag with source info
                preds["source_tile"] = tif_path.stem
                preds["city"] = city
                
                # Ensure geometry column exists if needed for DB later
                if hasattr(preds, 'geometry') and preds.geometry.isna().all():
                    # If predictions returned without geometry (sometimes happens with older versions), 
                    # you might need to reconstruct or skip DB upload. 
                    # Assuming DeepForest returns GeoDataFrame here.
                    pass

                city_predictions.append(preds)
                print(f"OK ({len(preds)})")
            else:
                print("OK (0)")
                
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}...")
            # Log error but continue to next tile
            continue
            
    return city_predictions

def main():
    parser = argparse.ArgumentParser(description="Batch Tree Detection across Cities")
    parser.add_argument("--city", type=str, default="all", 
                        help="Specific city name (e.g., 'torino') or 'all' to run all configured cities.")
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # 1. Load Model Once
    model = load_model(device)

    # 2. Select Cities
    cities_to_run = ALL_CITIES
    if args.city != "all":
        target = args.city.lower()
        if target in [c.lower() for c in cities_to_run]:
            cities_to_run = [target]
        else:
            print(f"Error: '{args.city}' not in configured list. Options: {cities_to_run}")
            sys.exit(1)

    start_time = time.time()
    master_df_list = []

    # 3. Batch Loop
    for city in cities_to_run:
        # Construct expected tile directory path
        # Adjust this logic if your directories are named differently (e.g., _nir_tiles_local)
        potential_dirs = [
            Path("/workspace/3-30-300-index/data") / f"{city}_nir_tiles_local",
            Path("/workspace/3-30-300-index/data") / f"{city}_tiles",
            Path("/workspace/3-30-300-index/data") / city,
        ]
        
        # Find first valid directory
        tile_dir = next((d for d in potential_dirs if d.exists()), None)
        
        if not tile_dir:
            print(f"\n[WARN] No tiles found for {city}. Skipping.")
            continue

        preds = process_single_city(city, tile_dir, model)
        if preds:
            city_df = pd.concat(preds, ignore_index=True)
        
            # Save per-city CSV immediately
            city_csv = Path(WORK_DIR) / f"{city}_tree_detections.csv"
            city_df.to_csv(city_csv, index=False)
            print(f"[SAVED] {city}: {len(city_df):,} detections -> {city_csv.name}")
            
            master_df_list.append(city_df)

    # 4. Finalize & Save
    if master_df_list:
        final_df = pd.concat(master_df_list, ignore_index=True)
        
        # Save CSV
        output_csv = Path(WORK_DIR) / "all_cities_tree_detections.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Total detections: {len(final_df):,}")
        print(f"[SUCCESS] Saved to: {output_csv}")
        
        # Optional: Upload to DuckDB if you have the table structure ready
        # Note: DeepForest output might need conversion to GeoDataFrame before gdf_to_duckdb
        try:
            con = connect_db()
            # Check if it's a GeoDataFrame first
            if hasattr(final_df, 'geometry'):
                from shapely.wkb import loads
                # If geometry is WKB bytes, convert; otherwise assume it's already Shapely
                if final_df['geometry'].dtype == 'object' and len(final_df) > 0:
                     # Simple check, adapt if your specific DF format differs
                     pass
                
                table_name = f"inference.detectors_{time.strftime('%Y%m%d_%H%M%S')}"
                gdf_to_duckdb(con, final_df, table_name)
                print(f"[SUCCESS] Uploaded to DB: {table_name}")
            else:
                print("[INFO] Dataframe has no geometry column. Skipping DB upload.")
            con.close()
        except Exception as db_err:
            print(f"[WARN] DB upload failed: {db_err}")
            
        print(f"Total Runtime: {(time.time() - start_time)/60:.1f} minutes")
    else:
        print("\n[INFO] No detections found in any city.")

if __name__ == "__main__":
    main()