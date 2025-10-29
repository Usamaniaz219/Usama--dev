
import os
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon
from shapely.validation import make_valid
from sklearn.cluster import MeanShift
import logging



logging.basicConfig(filename='im_process3.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def MeanShift_Zoning_Segmenter(image, output_subdir):
    pixels = image.reshape((-1, 3))
    clustering = MeanShift(bandwidth=8, bin_seeding=True).fit(pixels)
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    mask_paths = []
    for label in unique_labels:
        label_mask = (labels == label).reshape(image.shape[:2]).astype(np.uint8)
        area_of_interest = cv2.bitwise_and(image, image, mask=label_mask * 255)
        
        # Save mask for each cluster
        mask_name = f"{os.path.splitext(os.path.basename(output_subdir))[0]}_{label}.jpg"
        output_directory_path = os.path.join(output_subdir, mask_name)
        cv2.imwrite(output_directory_path, area_of_interest)

        # Save the mask path for further denoising
        mask_paths.append(output_directory_path)

    return mask_paths


def retrieve_poly(args):
    ori, contours_filled = args
    cnt_ori_2d = np.squeeze(ori)
    if cnt_ori_2d.shape[0] < 4:
        return None

    polygon_ori = Polygon(cnt_ori_2d)
    valid_polygon_ori = make_valid(polygon_ori)

    for cnt_fill in contours_filled:
        cnt_fill_2d = np.squeeze(cnt_fill)
        if cnt_fill_2d.shape[0] < 4:
            continue

        polygon_fill = Polygon(cnt_fill_2d)
        polygon_fill = make_valid(polygon_fill)
        
        if valid_polygon_ori.intersects(polygon_fill):
            return ori
    return None



def process_denoise_zone_mask_image(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing image: {image_name}")

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Error reading image: {image_path}")
        return None

    # Thresholding to get the original contours
    _, thresh_original = cv2.threshold(original, 20, 255, cv2.THRESH_BINARY)
    contours_original, _ = cv2.findContours(thresh_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Applying median blur and thresholding to get the filled contours
    median = cv2.medianBlur(original, 3)
    _, thresh_median = cv2.threshold(median, 25, 255, cv2.THRESH_BINARY)
    contours_filled, _ = cv2.findContours(thresh_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a blank mask
    mask = np.zeros(original.shape, dtype=np.uint8)

    # Prepare arguments for parallel processing
    args = [(ori, contours_filled) for ori in contours_original if ori.shape[0] > 3]

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(retrieve_poly, args))

    # Drawing valid contours on the mask
    for result in results:
        if result is not None:
            cv2.drawContours(mask, [result], -1, 255, cv2.FILLED)

    mask = cv2.bitwise_and(mask, thresh_original)

    # Save the final denoised mask
    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{image_name}_denoised_mask.jpg")
    cv2.imwrite(output_file_path, mask)

    print(f"Image processed: {image_name}")
    return mask



def process_ori_image(image_path, output_dir):
    image = cv2.imread(image_path)
    start_time = time.time()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)  
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_subdir = os.path.join(output_dir, image_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Apply the MeanShift segmentation and get the mask paths
    mask_paths = MeanShift_Zoning_Segmenter(image, output_subdir)

    # Process each segmentation mask with the denoising function
    for mask_path in mask_paths:
        process_denoise_zone_mask_image(mask_path, output_subdir)

    logging.info(f"Processed image '{image_path}' - Resolution: {image.shape[1]}x{image.shape[0]}, Processing Time: {time.time() - start_time:.4f} seconds")


def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist
    for filename in sorted(os.listdir(input_dir)):  # Iterate over all files in the input directory
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            image_path = os.path.join(input_dir, filename)
            process_ori_image(image_path, output_dir)


input_directory = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/Test_data/ca_colma_city/input/ca_colma_ori_image/"
output_directory = "clustering_results"
process_images(input_directory, output_directory) # Call the process_images function
