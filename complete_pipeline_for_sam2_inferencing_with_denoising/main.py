import os

import cv2

from denoising_masks_using_cluster_o_and_edges_3_march_2025 import remove_noise_using_cluster_0
from mean_shift_noise_removal import remove_noise_2,denoise_images_by_subtraction

# from testing.scripts.extract_and_select import select_polygons
from merge import merge_tiles
from sam2_testing_and_Automatic_prompt_selection_25_march_2025_by_utilizing_both_cpu_and_gpu import \
    get_sam_results
# from testing.scripts.select_mean_shift_masks import select_masks
from tiling_each_mask import process_mask_directory

base_location = '/media/usama/SSD/Data_for_complete_sam2_scripts/Data_for_complete_sam2_scripts_15_may_2025_11/'

for subdir in os.listdir(base_location):
    image_dir_path = os.path.join(base_location, subdir)
    image_path = image_dir_path + '/' + subdir +'.jpg'
    input_image = cv2.imread(image_path)
    # input_image = cv2.GaussianBlur(input_image,(9,9),0)
    
    points_path = os.path.join(image_dir_path,'legend_coords.txt')
    mean_shift_path = os.path.join(image_dir_path,'original_meanshift_masks/')

    #STEP 1
    #Select only zone based masks
    # mean_shifts = select_masks(points_path, mean_shift_path)
    # Currently selecting all mean shifts for now
    # mean_shifts = []
    #
    # for image_filename in os.listdir(mean_shift_path):
    #     if image_filename.endswith(('.jpg', '.png', '.jpeg')):
    #         mean_shifts.append(image_filename)
    # print(mean_shifts)

    #STEP 2
    #Noise removal and tiling

    denoised_dir = os.path.join(image_dir_path,'denoised_masks_1/')
    os.makedirs(denoised_dir, exist_ok=True)

    remove_noise_using_cluster_0(image_path, mean_shift_path, denoised_dir)
    denoise_images_by_subtraction(denoised_dir)

    denoised_dir_2 = os.path.join(image_dir_path, 'denoised_masks_2/')
    os.makedirs(denoised_dir_2, exist_ok=True)



    
    mean_shifts = remove_noise_2(denoised_dir,denoised_dir_2)

    image_tiles_path= image_dir_path + '/image_tiles/'
    os.makedirs(image_tiles_path, exist_ok=True)
    process_mask_directory(image_path, image_tiles_path)

    mean_tiles_dir = image_dir_path + '/mean_tiles_dir/'
    os.makedirs(mean_tiles_dir, exist_ok=True)

    no_noise_mean_tiles_dir = image_dir_path + '/noise_removed_mean_tiles_dir/'
    os.makedirs(no_noise_mean_tiles_dir, exist_ok=True)


    for mean_mask in mean_shifts:
        mean_tiles_path = mean_tiles_dir + mean_mask.split('.')[0]
        os.makedirs(mean_tiles_path, exist_ok=True)
        process_mask_directory(mean_shift_path + mean_mask, mean_tiles_path)

        no_noise_mean_tiles_path = no_noise_mean_tiles_dir + mean_mask.split('.')[0]
        os.makedirs(no_noise_mean_tiles_path, exist_ok=True)
        process_mask_directory(denoised_dir_2 + mean_mask, no_noise_mean_tiles_path)

    #STEP 3
    #SAM
    for zones in os.listdir(no_noise_mean_tiles_dir):
        get_sam_results(zones,image_dir_path,image_tiles_path, mean_tiles_dir,no_noise_mean_tiles_dir)

    #STEP 4
    #Image Regeneration

    sam_path = image_dir_path + '/sam_results/'
    merge_path = image_dir_path + '/sam_merged/'
    os.makedirs(merge_path, exist_ok=True)
    for sam_subdir in os.listdir(sam_path):
        merged = merge_tiles(input_image, os.path.join(sam_path, sam_subdir))
        cv2.imwrite(merge_path + sam_subdir + '.jpg', merged)

    #STEP 5
    #Extract Polygon
    # result_path = image_dir_path + '/final_result/'
    # os.makedirs(result_path, exist_ok=True)
    # for merge_subdir in os.listdir(merge_path):
    #     result = select_polygons(os.path.join(merge_path, merge_subdir))
    #     cv2.imwrite(result_path + merge_subdir, result)
    #





