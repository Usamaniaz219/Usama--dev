import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

def process_images(mask_final_dir, ori_mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for mask_filename in os.listdir(mask_final_dir):
        mask_final_path = os.path.join(mask_final_dir, mask_filename)
        ori_mask_path = os.path.join(ori_mask_dir, mask_filename)
        
        if not os.path.exists(ori_mask_path):
            print(f"Skipping {mask_filename}: Corresponding original mask not found.")
            continue
        
        mask_final = cv2.imread(mask_final_path, cv2.IMREAD_GRAYSCALE)
        ori_mask = cv2.imread(ori_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_final is None or ori_mask is None:
            print(f"Skipping {mask_filename}: Error reading image.")
            continue
        
        _, mask_final = cv2.threshold(mask_final, 50, 255, cv2.THRESH_BINARY)
        _, ori_mask = cv2.threshold(ori_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Generate skeleton
        org_mask_bool = ori_mask > 0
        org_skeleton = skeletonize(org_mask_bool).astype(np.uint8) * 255 
        # _, org_skeleton = cv2.threshold(org_skeleton, 240, 255, cv2.THRESH_BINARY)
        
        # Dilate the skeleton
        kernel = np.ones((3,3), np.uint8)
        org_skeleton = cv2.dilate(org_skeleton, kernel, iterations=1)
        
        # Save skeleton image
        skeleton_output_path = os.path.join(output_dir, f"skeleton_{mask_filename}")
        cv2.imwrite(skeleton_output_path, org_skeleton)
        
        # Compute intersection
        intersected_region = cv2.bitwise_and(ori_mask, mask_final)
        
        # Remove filled areas
        removed_filled_areas = cv2.subtract(intersected_region, org_skeleton)
        org_skeleton_inverted = cv2.bitwise_not(org_skeleton)
        # removed_filled_areas_inverted = cv2.bitwise_not(removed_filled_areas)

        removed_filled_areas = cv2.bitwise_and(removed_filled_areas,org_skeleton_inverted)
        removed_filled_areas = cv2.erode(removed_filled_areas,kernel,iterations=2)
        
        

        output_path = os.path.join(output_dir, f"removed_filled_{mask_filename}")
        cv2.imwrite(output_path, removed_filled_areas)
        print(f"Processed {mask_filename}")

mask_final_directory = "/media/usama/SSD/Line_removal_using_skeletonization/mixture_outputs_7_feb_11/"
ori_mask_directory = "/media/usama/SSD/Line_removal_using_skeletonization/mixture_images_full_7_feb_2025_data_for_testing/"
output_directory = "processed_outputs_7_feb_2025_mixture_11"

process_images(mask_final_directory, ori_mask_directory, output_directory)













































# # import cv2
# # import numpy as np
# # from skimage.morphology import skeletonize

# # mask_final_path = "Filled_regions_test_data_3_feb_2025/Filled_regions_outputs_data_4_feb_2025/mixture_outputs_with_child_cont_filled_4_feb_11/ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_87.jpg"
# mask_final_path = "Filled_regions_test_data_3_feb_2025/Filled_regions_outputs_data_4_feb_2025/roads_outputs_with_child_cont_filled_4_feb_11/binary_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_61.jpg"
# ori_mask_path = "Filled_regions_test_data_3_feb_2025/roads_images_previous/ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_61.jpg"

# # ori_mask_path = "Filled_regions_test_data_3_feb_2025/mixture_images_previous/ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_87.jpg"

# mask_final = cv2.imread(mask_final_path,cv2.IMREAD_GRAYSCALE)
# _,mask_final = cv2.threshold(mask_final, 50, 255, cv2.THRESH_BINARY)
# ori_mask = cv2.imread(ori_mask_path, cv2.IMREAD_GRAYSCALE)
# _,ori_mask = cv2.threshold(ori_mask, 50, 255,cv2.IMREAD_GRAYSCALE)
# ori_mask = cv2.GaussianBlur(ori_mask,(5,5),0)

# org_mask_bool = ori_mask > 0
# org_skeleton = skeletonize(org_mask_bool).astype(np.uint8) * 255 
# _,org_skeleton = cv2.threshold(org_skeleton,128,255,cv2.THRESH_BINARY)

# kernel = np.ones((3,3),np.uint8)
# org_skeleton = cv2.dilate(org_skeleton,kernel,iterations=4)
# cv2.imwrite("org_skelton.jpg",org_skeleton)

# Intersected_region = cv2.bitwise_and(ori_mask,mask_final)

# # removed_filled_areas = cv2.bitwise_xor(Intersected_region,org_skeleton)

# removed_filled_areas = cv2.subtract(Intersected_region,org_skeleton)

# # removed_filled_areas = cv2.bitwise_and(mask_final,removed_filled_areas)



# cv2.imwrite("removed_filled_Areas1.jpg",removed_filled_areas)




