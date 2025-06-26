

import os
import rasterio
from rasterio import features
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from tqdm import tqdm

TILE_SIZE = 256  # pixels
STRIDE = 128     # 50% overlap
IMAGE_DIR = 'data/raw/AOI_2_Vegas_Train/MUL-PanSharpen'
LABEL_DIR = 'data/raw/AOI_2_Vegas_Train/geojson/buildings'
OUT_IMAGE_DIR = 'data/tiles/images'
OUT_LABEL_DIR = 'data/tiles/labels'

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

def generate_tiles(image_path, label_path):
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load image
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        transform = src.transform
        
        # Load geojson labels and transform to image CRS
        gdf = gpd.read_file(label_path)
        gdf = gdf.to_crs(src.crs)

        for top in range(0, height, STRIDE):
            for left in range(0, width, STRIDE):
                window = Window(left, top, TILE_SIZE, TILE_SIZE)
                transform_window = rasterio.windows.transform(window, transform)
                bounds = rasterio.windows.bounds(window, transform)
                tile_geom = box(*bounds)

                # Check if tile intersects any buildings
                intersects = gdf.intersects(tile_geom).any()
                if not intersects:
                    continue

                tile_image = src.read(window=window)
                if tile_image.shape[1] != TILE_SIZE or tile_image.shape[2] != TILE_SIZE:
                    continue  # skip incomplete edge tiles

                # Save image tile
                tile_name = f"{image_id}_{top}_{left}"
                out_img_path = os.path.join(OUT_IMAGE_DIR, f"{tile_name}.npy")
                np.save(out_img_path, tile_image)

                # Save binary label mask (1 = building, 0 = background)
                mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

                # Before rasterizing, clean up geometries
                valid_shapes = [
                    (geom, 1)
                    for geom in gdf.intersection(tile_geom).buffer(0)
                    if geom.is_valid and not geom.is_empty
                ]

                if valid_shapes:
                    out_label_path = os.path.join(OUT_LABEL_DIR, f"{tile_name}.npy")
                    burned = features.rasterize(
                        valid_shapes,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        transform=transform_window,
                        fill=0,
                        dtype=mask.dtype
                    )
                    np.save(out_label_path, burned)

def main():
    image_files = sorted(os.listdir(IMAGE_DIR))
    for img_file in tqdm(image_files, desc="Generating tiles"):
        if not img_file.endswith('.tif'):
            continue
        image_path = os.path.join(IMAGE_DIR, img_file)
        label_name = 'buildings_' + img_file.replace('MUL-PanSharpen_', '').replace('.tif', '.geojson')
        label_path = os.path.join(LABEL_DIR, label_name)
        if not os.path.exists(label_path):
            continue
        generate_tiles(image_path, label_path)

if __name__ == "__main__":
    main()