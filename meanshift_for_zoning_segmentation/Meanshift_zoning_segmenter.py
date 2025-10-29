import os
import cv2
import numpy as np
import time
import logging
from sklearn.cluster import MeanShift
logging.basicConfig(filename='im_process2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set up logging configuration 
def MeanShift_Zoning_Segmenter(image, output_subdir):
    pixels = image.reshape((-1, 3))
    # clustering = MeanShift(bandwidth=10, n_jobs=-1, bin_seeding=True, min_bin_freq=1, cluster_all=False).fit(pixels)
    clustering = MeanShift(bandwidth=8, bin_seeding=True).fit(pixels)
    # clustering = MeanShift(bandwidth=15, n_jobs=-1).fit(pixels)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    for label in unique_labels:  # Extract areas of interest based on unique labels using logical AND operation
        label_mask = (labels == label).reshape(image.shape[:2]).astype(np.uint8)
        area_of_interest = cv2.bitwise_and(image, image, mask=label_mask * 255)
        mask_name = f"{os.path.splitext(os.path.basename(output_subdir))[0]}_{label}.jpg"   # Output image naming includes the original image name and clustering label
        output_directory_path = os.path.join(output_subdir, mask_name)
        cv2.imwrite(output_directory_path, area_of_interest)

def process_image(image_path, output_dir):
    try:
       
        image = cv2.imread(image_path)   # Read the image
        start_time = time.time()       # Measure processing time
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)    # Resize and convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS) 
        # Resized image using bicubic interpolation
        # image = cv2.resize(image, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print("image name", image_name)
        output_subdir = os.path.join(output_dir, image_name)
        os.makedirs(output_subdir, exist_ok=True)
        MeanShift_Zoning_Segmenter(image, output_subdir) # Apply the MeanShift_Zoning_Segmenter function
        logging.info(f"Processed image '{image_path}' - Resolution: {image.shape[1]}x{image.shape[0]}, Processing Time: {time.time() - start_time:.4f} seconds")  # Log image resolution and processing time

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {str(e)}")

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist
    for filename in sorted(os.listdir(input_dir)):  # Iterate over all files in the input directory
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)

input_directory = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/multi_color_space_pipeline/final_image/"
output_directory = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/multi_color_space_pipeline/final_image/"
process_images(input_directory, output_directory) # Call the process_images function





