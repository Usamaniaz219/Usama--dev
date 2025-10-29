import os
import cv2
import numpy as np

folder_path = "/media/usama/SSD/Usama_dev_ssd/data_demo_157/"
output_dir = "/media/usama/SSD/Usama_dev_ssd/Edge_detection_of_map_images/canny_edge_detection/contour_results_after_morph_gradients_19_sep_2024/"
folder_name = os.path.basename(os.path.dirname(folder_path))
print("folder name",folder_name)

all_masks = os.listdir(folder_path)
masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]
for renamed_mask in masks_renamed:
    mask_path = f"{folder_path}/{renamed_mask}.jpg"
    mask_image = cv2.imread(mask_path)
    img_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
    # img_gray = cv2.medianBlur(img_gray,3)
    kernel = np.ones((3, 3), np.uint8)
    # # thresh_image_with_morph_gradient = cv2.morphologyEx(img_gray,cv2.MORPH_GRADIENT,kernel)
    # tophat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

# Apply Gradient
    gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
    # _,thresh = cv2.threshold(thresh_image_with_morph_gradient,25,255,cv2.THRESH_BINARY)
    # thresh1 = cv2.adaptiveThreshold(thresh_image_with_morph_gradient, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2) 
    # kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # thresh_image_with_morph_closing = cv2.morphologyEx(thresh1,cv2.MORPH_CLOSE,kernel)
    # resultant = cv2.bitwise_or(tophat,gradient)


    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(folder_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
    cv2.imwrite(output_file_path,gradient)