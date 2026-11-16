# ============================================================
# 0. SETUP
# ============================================================
import os
# NAIP Imagary
import ee
import geemap
import geopandas as gpd

import config

# ############################
# 1. Earth Engine project (set in config.py, along with every path)
# ############################
ee.Initialize(project=config.EE_PROJECT)

# ############################
# 2. Define the AOI
# ############################
# The 45 LiDAR tiles, which is the extent of the existing Mount_Rainier_NAIP_2019_RGB.tif
# (was Rainier_boundary/mora_boundary.shp, the whole park). Keep only the geometry: the
# tile table has nested columns Earth Engine doesn't need.
rainier = gpd.read_file(str(config.TILES_GEOJSON))[["geometry"]].dissolve()

# Earth Engine expects longitude/latitude
rainier = rainier.to_crs("EPSG:4326")

# Convert GeoDataFrame to an Earth Engine FeatureCollection
rainier_ee = geemap.gdf_to_ee(rainier)

# Extract one Earth Engine geometry for the AOI
aoi = rainier_ee.geometry()


naip = (
    ee.ImageCollection("USDA/NAIP/DOQQ")
    .filterBounds(aoi)
)

# Find the NAIP years available over the LiDAR tiles
years = (
    naip
    .aggregate_array("system:time_start")
    .map(lambda date: ee.Date(date).get("year"))
    .distinct()
    .sort()
    .getInfo()
)

print("Available NAIP years:", years)

# ############################
# LiDAR download (already done). download_lidar() is a separate helper that isn't part of
# this repository and isn't imported here, so this block raised a NameError. Left
# commented out; the README says where the LiDAR comes from.
# ############################
# tiles = download_lidar(
#     aoi=rainier,
#     year=2019,
#     output_directory="Rainier_lidar_2019",
#     download=True
# )


# ############################
# 3. (Optional) Export 2019 RGBN NAIP to Google Drive
# ############################
# This is the export that made Mount_Rainier_NAIP_2019_RGBN.tif. Commented out so
# running the script doesn't start another large Drive export; uncomment if needed.
#
# naip_2019 = (
#     ee.ImageCollection("USDA/NAIP/DOQQ")
#     .filterBounds(aoi)
#     .filterDate("2019-01-01", "2020-01-01")
# )
#
# # Combine the NAIP tiles and clip them to Mount Rainier
# naip_mosaic = (
#     naip_2019
#     .mosaic()
#     .clip(aoi)
#     .select(["R", "G", "B", "N"])
# )
#
# task = ee.batch.Export.image.toDrive(
#     image=naip_mosaic,
#     description="Mount_Rainier_NAIP_2019",
#     folder="Rainier_NAIP",
#     fileNamePrefix="Mount_Rainier_NAIP_2019",
#     region=aoi,
#     scale=0.6,
#     crs="EPSG:26910",
#     maxPixels=1e13,
#     fileDimensions=8192,
#     skipEmptyTiles=True,
#     fileFormat="GeoTIFF",
#     formatOptions={
#         "cloudOptimized": True
#     }
# )
#
# task.start()
#
# print(task.status())

# ############################
# 4. Download 2019 RGB NAIP imagery
# ############################
output_dir = str(config.NAIP_DIR)        # the folder 02 and 04 read the mosaic from
os.makedirs(output_dir, exist_ok=True)

output_file = str(config.NAIP_PATH)

naip_2019 = (
    ee.ImageCollection("USDA/NAIP/DOQQ")
    .filterBounds(aoi)
    .filterDate("2019-01-01", "2020-01-01")
)

print("2019 source images:", naip_2019.size().getInfo())

naip_rgb = (
    naip_2019
    .mosaic()
    .clip(aoi)
    .select(["R", "G", "B"])
    .toUint8()
)

# Download on the LiDAR grid (was crs="EPSG:26910", scale=0.6): EPSG:6339 is the LiDAR
# CRS, the transform starts at the top-left corner of the 1 m pit-free CHM
# (Rainier_lidar_2019_Khosravipour_CHM.tiff) with 0.6 m pixels, and the shape covers the
# CHM's 7,470 m x 16,710 m extent. 50 NAIP pixels = one 30 m rumple cell.
geemap.download_ee_image(
    image=naip_rgb,
    filename=output_file,
    region=aoi,
    crs="EPSG:6339",
    crs_transform=[0.6, 0, 611550, 0, -0.6, 5196690],
    shape=(27850, 12450),   # (rows, columns)
    dtype="uint8",
    resampling="near",
    num_threads=8,
    max_tile_size=32,
    max_tile_dim=10000,
    overwrite=True
)

print(f"Saved to: {output_file}")
