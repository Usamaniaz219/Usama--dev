import cv2
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt
import os

TILE_SIZE = 128
OVERLAP = 5
eps = 20
min_samples = 200
image_path = "/home/usama/usama_dev_test/CUTS/data/zoning_map_resized_dataset/map_images/demo92_tile_11_18.jpg"

def apply_dbscan_on_tiles(tile, eps, min_samples):
    pixels = tile.reshape((-1, 3))

    # Apply DBSCAN clustering
    dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(pixels)
    labels = dbscan.labels_

    # Reshape labels into tile shape
    labels = labels.reshape(tile.shape[:2])
    
    return labels

def apply_dbscan_on_image(image_path):
    # Load high-resolution image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create a directory to save masks
 
    mask_dir = "DBSCAN_results"
    os.makedirs(mask_dir, exist_ok=True)
    
    # Processed mask for visualization
    processed_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize as black background
    
    index = 0
    for y in range(0, height - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
        for x in range(0, width - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
            tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
           
            labels = apply_dbscan_on_tiles(tile, eps, min_samples)

            # Create a binary mask for each label
            for label in np.unique(labels):
                if label == -1:  # Noise points
                    continue
                mask = np.uint8(labels == label) * 255
                cv2.imwrite(f'{mask_dir}/tile_{index}_Label_{label}.jpg', tile)
                
                # Save mask as a file
                cv2.imwrite(f'{mask_dir}/Mask_{index}_Label_{label}.jpg', mask)
                
                # Set foreground points to white (255) in the processed mask
                processed_mask[y:y+TILE_SIZE, x:x+TILE_SIZE][mask > 0] = 255
            
            index += 1
    cv2.imwrite(f'{mask_dir}/Masked_{index}_Label_{label}.jpg', mask)