import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box
from deepforest import preprocess

import config

# inputs (every path is set in config.py)
naip_path = str(config.NAIP_PATH)
rumple_path = str(config.RUMPLE_PATH)
chm_path = str(config.CHM_PATH)
laz_dir = str(config.LAS_CLEAN)
crowns_path = str(config.CROWNS_PATH)
tops_path = str(config.TOPS_PATH)

# outputs (all inside this folder). Spatial cross-validation: patches for every tile in one folder, plus a
# train and a validation table for each fold.
out_dir = str(config.TRAIN_DIR)
patch_dir = os.path.join(out_dir, "patches")
os.makedirs(patch_dir, exist_ok=True)

# settings
min_height = 5.0      # m, drop crowns shorter than this (barely visible in NAIP)
min_crowns = 200      # skip LiDAR tiles with fewer crowns than this (slivers at the survey edge)
patch_size = 200      # px = 120 m at 0.6 m; RetinaNet enlarges each patch to 800 px internally
patch_overlap = 0.1
n_folds = 5           # spatial cross-validation: every tile is a test tile in exactly one fold

# Two harmless warnings DeepForest's split_raster prints for every tile: it misreads the
# band count of (bands, rows, cols) arrays (it still uses the 3 RGB bands correctly), and
# an internal geopandas indexing note. Hide just those two.
warnings.filterwarnings("ignore", message="Input image had non-3 band shape")
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed")


# ############################
# 1. Crowns -> training boxes
# ############################
crowns = gpd.read_file(crowns_path)
tops = gpd.read_file(tops_path)
crowns = crowns.merge(tops[["crown_id", "height"]], on="crown_id")

boxes = crowns[crowns.height >= min_height].copy()
boxes["geometry"] = boxes.envelope              # DeepForest trains on rectangles
print(f"{len(boxes):,} of {len(crowns):,} crowns are >= {min_height} m")


# ############################
# 2. LiDAR tiles (1 km squares from the .laz file names)
# ############################
# A USGS tile name like 10TFS06115192 encodes its south-west corner in km:
# easting 0611 -> 611000 m, northing 5192 -> 5192000 m.
laz_files = glob.glob(os.path.join(laz_dir, "*.laz"))
names = [os.path.basename(f).replace(".laz", "").split("_")[-1] for f in laz_files]
tiles = gpd.GeoDataFrame(
    {"tile": names},
    geometry=[box(int(n[5:9]) * 1000, int(n[9:13]) * 1000,
                  int(n[5:9]) * 1000 + 1000, int(n[9:13]) * 1000 + 1000) for n in names],
    crs=boxes.crs)

# trim tiles to the NAIP extent (edge tiles stick out past it)
with rasterio.open(naip_path) as src:
    naip_extent = box(*src.bounds)
tiles["geometry"] = tiles.geometry.intersection(naip_extent)

# each crown belongs to the tile that contains its centre
centres = boxes[["crown_id"]].copy()
centres = gpd.GeoDataFrame(centres, geometry=boxes.centroid, crs=boxes.crs)
centres = gpd.sjoin(centres, tiles[["tile", "geometry"]], predicate="within")
boxes = boxes.merge(centres[["crown_id", "tile"]], on="crown_id")
tiles["n_crowns"] = tiles.tile.map(boxes.tile.value_counts()).fillna(0).astype(int)

# mean rumple per tile
with rasterio.open(rumple_path) as src:
    mean_rumple = []
    for g in tiles.geometry:
        w = from_bounds(*g.bounds, transform=src.transform).round_offsets().round_lengths()
        r = src.read(1, window=w, masked=True)
        mean_rumple.append(float(r.mean()) if r.count() > 0 else np.nan)
tiles["rumple"] = mean_rumple


# ############################
# 3. Cross-validation folds by tile
# ############################
# Sort tiles by mean rumple and deal them out like cards, so every fold spans the rumple range.
#  - test tiles of fold k (20%): every 5th tile, starting at the k-th
#  - validation tiles of fold k (10%, only used to pick the best epoch): every 10th tile, starting at
#    the ((k + 2) % 5)-th, i.e. half of another fold's tiles
#  - training tiles: the rest (70%)
# Fold 2 is then exactly the single split used for runs 1-4 (test = every 5th tile from the 2nd,
# validation = every 10th from the 4th), so 03 can reuse run 4's model for it.
tiles = tiles[tiles.n_crowns >= min_crowns].sort_values("rumple").reset_index(drop=True)
rank = np.arange(len(tiles))
tiles["fold"] = rank % n_folds                       # the fold in which this tile is a test tile
for k in range(n_folds):
    tiles[f"fold{k}"] = np.where(rank % n_folds == k, "test",
                                 np.where(rank % (2 * n_folds) == (k + 2) % n_folds, "val", "train"))
print(f"{len(tiles)} tiles, {n_folds} folds:")
print(pd.DataFrame({f"fold{k}": tiles[f"fold{k}"].value_counts() for k in range(n_folds)}).to_string())

tiles.to_file(os.path.join(out_dir, "tile_folds.shp"))


# ############################
# 4. Cut NAIP into patches with box labels (every tile, once)
# ############################
annotations = []
with rasterio.open(naip_path) as src, rasterio.open(chm_path) as chm_src:
    res = src.res[0]
    for t in tiles.itertuples():
        # NAIP for this tile, on whole pixels
        w = from_bounds(*t.geometry.bounds, transform=src.transform).round_offsets().round_lengths()
        if w.width < patch_size or w.height < patch_size:
            print(f"skipping {t.tile}: smaller than one patch")
            continue
        img = src.read([1, 2, 3], window=w)                 # (bands, rows, cols)
        tile_transform = src.window_transform(w)

        # Blank NAIP wherever the CHM has no data. There are no watershed crowns there, so any
        # trees in those areas would be shown to the model as background (11-14% of run 3's
        # predictions fell in such areas, on real vegetation).
        chm = chm_src.read(1, window=from_bounds(*window_bounds(w, src.transform), transform=chm_src.transform),
                           out_shape=(w.height, w.width), resampling=Resampling.nearest)
        no_lidar = ~(np.isfinite(chm) & (chm > -100))
        img[:, no_lidar] = 0

        # crown boxes -> pixel coordinates of this tile
        b = boxes[boxes.tile == t.tile].geometry.bounds
        left, top = tile_transform.c, tile_transform.f
        ann = pd.DataFrame({
            "image_path": f"{t.tile}.tif",
            "xmin": (b.minx - left) / res,
            "ymin": (top - b.maxy) / res,
            "xmax": (b.maxx - left) / res,
            "ymax": (top - b.miny) / res,
            "label": "Tree",
        })

        patches = preprocess.split_raster(
            annotations_file=ann,
            numpy_image=img,
            image_name=f"{t.tile}.tif",
            root_dir=out_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            save_dir=patch_dir,
        )
        annotations.append(patches.assign(tile=t.tile))
        print(f"{t.tile} (test in fold {t.fold}): {patches.image_path.nunique()} patches, {len(patches):,} boxes "
              f"({no_lidar.mean():.0%} of the tile blanked, no LiDAR)")
annotations = pd.concat(annotations, ignore_index=True)


# ############################
# 5. Train and validation tables for each fold
# ############################
# image_path is relative to the patches folder (03 uses it as root_dir for both tables)
for k in range(n_folds):
    for role in ["train", "val"]:
        df = annotations[annotations.tile.isin(tiles.tile[tiles[f"fold{k}"] == role])]
        df = df[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]]
        df.to_csv(os.path.join(out_dir, f"fold{k}_{role}.csv"), index=False)
        print(f"fold {k} {role:5s}: {df.image_path.nunique():,} patches, {len(df):,} boxes "
              f"-> {os.path.join(out_dir, f'fold{k}_{role}.csv')}")
