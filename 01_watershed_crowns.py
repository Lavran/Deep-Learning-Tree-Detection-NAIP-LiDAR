import os
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.features import shapes
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
import geopandas as gpd
from shapely.geometry import shape

import config

# paths (set in config.py)
chm_path = str(config.CHM_PATH)            # input: 1 m pit-free CHM mosaic
crowns_path = str(config.CROWNS_PATH)      # outputs: the crowns and their treetops, read by 02, 04 and 05
tops_path = str(config.TOPS_PATH)
os.makedirs(config.CROWNS_DIR, exist_ok=True)


def treetops_vwf(chm, res, hmin=2.0, a=1.2, b=0.15):
    """Variable-window treetop detection (Popescu & Wynne 2004 style).

    Search radius in metres = a + b * height, so crown-spacing scales with
    tree height. a, b are forest-type-specific allometric coefficients.
    Returns treetop pixel positions as an (N, 2) array of (row, col).
    """
    # 1. candidate local maxima (fine 3x3), above the height threshold
    mx = ndi.maximum_filter(chm, size=3, mode="constant", cval=0)
    cand = (chm == mx) & (chm >= hmin)
    rows, cols = np.where(cand)
    h = chm[rows, cols]

    # 2. tallest first
    order = np.argsort(-h)
    rows, cols, h = rows[order], cols[order], h[order]

    # 3. greedy suppression: each accepted top clears shorter maxima
    #    within its own height-dependent radius (converted to pixels)
    radii_px = (a + b * h) / res
    xy_px = np.c_[cols.astype(float), rows.astype(float)]
    tree = cKDTree(xy_px)
    taken, keep = np.zeros(len(h), bool), []
    for k in range(len(h)):
        if taken[k]:
            continue
        keep.append(k)
        taken[tree.query_ball_point(xy_px[k], radii_px[k])] = True
    keep = np.array(keep)

    # return pixel positions (segment_trees turns them into map coordinates)
    r, c = rows[keep], cols[keep]
    return np.c_[r, c]

def segment_trees(chm_path, hmin=2.0, smooth_sigma=1.0):
    with rasterio.open(chm_path) as src:
        chm = src.read(1).astype("float32")
        transform, nodata, crs = src.transform, src.nodata, src.crs
    res = abs(transform.a)                           # CHM cell size (m)

    chm[(chm == nodata) | ~np.isfinite(chm) | (chm < 0)] = 0.0   # clean nodata / NaN / negatives
    chm_s = ndi.gaussian_filter(chm, sigma=smooth_sigma)  # suppress spurious peaks
    canopy = chm_s > hmin                            # only segment vegetation

    # tree tops = variable-window local maxima
    coords = treetops_vwf(chm_s, res, hmin=hmin)
    markers = np.zeros(chm_s.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    # crowns = watershed on the inverted CHM, confined to canopy
    labels = watershed(-chm_s, markers, mask=canopy)

    rows, cols = coords[:, 0], coords[:, 1]
    xs, ys = xy(transform, rows, cols)               # treetop map coordinates
    heights = chm[rows, cols]
    return labels, np.array(xs), np.array(ys), heights, transform, crs

labels, xs, ys, heights, transform, crs = segment_trees(chm_path, hmin=2.0, smooth_sigma=1.0)

def crowns_to_gdf(labels, transform, crs):
    geoms, ids = [], []
    for geom, val in shapes(labels.astype("int32"), mask=labels > 0, transform=transform):
        geoms.append(shape(geom)); ids.append(int(val))
    gdf = gpd.GeoDataFrame({"crown_id": ids}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve("crown_id").reset_index()   # one polygon per crown
    gdf["area_m2"] = gdf.area
    return gdf

crowns = crowns_to_gdf(labels, transform, crs)

# treetops as points (crown_id i ↔ coords[i-1], so heights line up)
tops = gpd.GeoDataFrame({"crown_id": range(1, len(xs)+1), "height": heights},
                        geometry=gpd.points_from_xy(xs, ys), crs=crs)

# write the crowns and the treetops as two shapefiles
crowns.to_file(crowns_path)
tops.to_file(tops_path)
print(f"{len(crowns):,} crowns and {len(tops):,} treetops -> {config.CROWNS_DIR}")
