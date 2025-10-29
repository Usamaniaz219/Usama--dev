import cv2
import numpy as np
import os

def detect_edges_with_postprocessing(image, sigma=0.33, use_clahe=True):
    # Step 1: Resize image to normalize resolution
    # image_resized = cv2.resize(image, resize_dim)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) if enabled
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(100, 100))
        gray = clahe.apply(gray)

    # Step 4: Apply Gaussian Blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    lower = 10
    upper = 200

    edges = cv2.Canny(blurred, lower, upper)

    # Step 7: Post-Processing: Morphological operations (Dilation and Closing)
    Define a kernel size based on the image resolution
    kernel = np.ones((3, 3), np.uint8)

    # Apply Dilation to thicken the edges
    # dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply Closing (Dilation followed by Erosion) to close gaps between edges
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges

# image = cv2.imread('map_image.jpg')
folder_path = "/media/usama/SSD/Usama_dev_ssd/Edge_detection_of_map_images/TEED/data1/"
output_dir = "/media/usama/SSD/Usama_dev_ssd/Edge_detection_of_map_images/canny_edge_detection/canny_edge_detection_results_with_preprocessing_and_post_processing/"
folder_name = os.path.basename(os.path.dirname(folder_path))
all_ori_images = os.listdir(folder_path)
ori_image__renamed = [ori_image.replace(".jpg", "").replace(".png", "") for ori_image in all_ori_images]

for renamed_ori_image in ori_image__renamed:
    print("renamed mask",renamed_ori_image)
    image_path = f"{folder_path}/{renamed_ori_image}.jpg"
    image = cv2.imread(image_path)

    edges_with_postprocessing = detect_edges_with_postprocessing(image)
    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(folder_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{renamed_ori_image}_edge.png")
    cv2.imwrite(output_file_path, edges_with_postprocessing)


