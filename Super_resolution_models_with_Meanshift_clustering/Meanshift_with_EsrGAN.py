# !pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
import torch
from PIL import Image
import os
import time
import logging
from RealESRGAN import RealESRGAN
import cv2
import numpy as np
from sklearn.cluster import MeanShift

logging.basicConfig(filename='im_process1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Set up logging configuration 

# RealESRGAN setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# MeanShift_Zoning_Segmenter function
def MeanShift_Zoning_Segmenter(image, output_subdir):
    pixels = image.reshape((-1, 3))
    clustering = MeanShift(bandwidth=15, n_jobs=-1, bin_seeding=True, min_bin_freq=1, cluster_all=False).fit(pixels)
    # clustering = MeanShift(bandwidth=15, n_jobs=-1).fit(pixels)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    for label in unique_labels:  # Extract areas of interest based on unique labels using logical AND operation
        label_mask = (labels == label).reshape(image.shape[:2]).astype(np.uint8)
        area_of_interest = cv2.bitwise_and(image, image, mask=label_mask * 255)
        mask_name = f"{os.path.splitext(os.path.basename(output_subdir))[0]}_{label}.jpg"   # Output image naming includes the original image name and clustering label
        output_directory_path = os.path.join(output_subdir, mask_name)
        cv2.imwrite(output_directory_path, area_of_interest)
    

# Process image function
def process_image(image_path, output_dir):
    # try:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print("image name", image_name)
    output_subdir = os.path.join(output_dir, image_name, "15")
    os.makedirs(output_subdir, exist_ok=True)

    # Load LR image
    path_to_image = image_path
    lr_image = Image.open(path_to_image).convert('RGB')

    # Generate SR image using RealESRGAN
    sr_image = model.predict(lr_image)
    sr_image = np.array(sr_image)

    # Apply the MeanShift_Zoning_Segmenter function
    MeanShift_Zoning_Segmenter(sr_image, output_subdir)

    logging.info(f"Processed image '{image_path}' - Resolution: {sr_image.shape[1]}x{sr_image.shape[0]}, Processing Time: {time.time() - start_time:.4f} seconds")  # Log image resolution and processing time

    # except Exception as e:
    #     logging.error(f"Error processing image '{image_path}': {str(e)}")

# Process images function
def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    for filename in sorted(os.listdir(input_dir)):  # Iterate over all files in the input directory
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)

# Input and output directories
input_directory = "/workspace/data/"
output_directory = "/workspace/results_1"

# Call the process_images function
process_images(input_directory, output_directory)