"""
config.py -- shared paths for the tree-detection pipeline (scripts 00-05).

Every script imports its paths from here, so there is ONE place to change where the
data and the outputs live. PROJECT_DIR is the folder containing this file, so the repo
works wherever it is cloned: put the data in the folders named below, or point these
paths at your own copies. This module is cheap (no heavy imports, no data reads).
"""
from pathlib import Path

# Project root = the folder containing this file, so the repo works wherever it is cloned.
PROJECT_DIR = Path(__file__).resolve().parent

# --- Google Earth Engine (00) ---
EE_PROJECT = "howe-et-al"    # Google Earth Engine project ID: replace with your own

# --- LiDAR products (inputs; see the Data section of the README) ---
LIDAR_DIR     = PROJECT_DIR / "Rainier_lidar_2019"
LAS_CLEAN     = LIDAR_DIR / "point_cloud_clean"      # cleaned .laz tiles; 02 uses only their names, to rebuild the 1 km tiles
TILES_GEOJSON = LIDAR_DIR / "lidar_tiles.geojson"    # footprints of the LiDAR tiles; 00 downloads NAIP over this area

CHM_DIR       = LIDAR_DIR / "Khosravipour_et_al_CHM"
CHM_PATH      = CHM_DIR / "Rainier_lidar_2019_Khosravipour_CHM.tiff"        # 1 m pit-free canopy height model

RUMPLE_PATH   = LIDAR_DIR / "Rumple" / "pit free" / "rumple_pitfree.tiff"    # 30 m canopy surface complexity

# --- NAIP imagery (written by 00) ---
NAIP_DIR      = PROJECT_DIR / "Rainier_NAIP_2019"
NAIP_PATH     = NAIP_DIR / "Mount_Rainier_NAIP_2019_RGB.tif"

# --- outputs, one folder per step ---
CROWNS_DIR    = PROJECT_DIR / "watershed_outputs"             # 01: reference crowns and treetops
CROWNS_PATH   = CROWNS_DIR / "watershed_crowns.shp"
TOPS_PATH     = CROWNS_DIR / "watershed_treetops.shp"

TRAIN_DIR     = PROJECT_DIR / "deepforest_training_data_cv"   # 02: NAIP patches and the tables of each fold
TILES_PATH    = TRAIN_DIR / "tile_folds.shp"                  #     the tiles and the fold each one is tested in

MODEL_DIR     = PROJECT_DIR / "deepforest_models"             # 03: fine-tuned model of each fold
LOG_DIR       = PROJECT_DIR / "lightning_logs"                #     loss and validation metrics by epoch

PRED_DIR      = PROJECT_DIR / "deepforest_predictions_cv"     # 04: predictions, scored trees and accuracy tables

ANALYSIS_DIR  = PROJECT_DIR / "rumple_analysis_cv"            # 05: tables and figures of the rumple analysis
