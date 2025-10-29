import os
import cv2
import numpy as np

folder_path = "/media/usama/SSD/Usama_dev_ssd/data_12_Sep/"
output_dir = "/media/usama/SSD/Usama_dev_ssd/Edge_detection_of_map_images/canny_edge_detection/adaptive_threshold_results/"
folder_name = os.path.basename(os.path.dirname(folder_path))
print("folder name",folder_name)

all_masks = os.listdir(folder_path)
masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]
for renamed_mask in masks_renamed:
    mask_path = f"{folder_path}/{renamed_mask}.jpg"
    mask_image = cv2.imread(mask_path)
    img_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
    # img_gray = cv2.medianBlur(img_gray,3)
    thresh1 = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2) 
    # kernel = np.ones((5, 5), np.uint8)
    # thresh_image_with_morph_closing = cv2.morphologyEx(thresh1,cv2.MORPH_CLOSE,kernel)

    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(folder_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
    cv2.imwrite(output_file_path,thresh1)