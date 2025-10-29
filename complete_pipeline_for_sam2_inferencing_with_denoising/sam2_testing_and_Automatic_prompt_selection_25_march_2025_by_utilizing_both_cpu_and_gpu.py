# Inference Code #
#######################

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import concurrent.futures
import threading

from get_representative_pt import automatic_foreground_prompt_selector_from_directory, \
    automatic_foreground_prompt_selector_from_image, process_single_image_using_rep_point_logic, is_foreground_pixel

# torch.cuda.memory._record_memory_history(max_entries=100000)

# models = [("/home/asfand/Work/sam2/checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml", "/home/asfand/Work/sam2/testing/models/fine_tuned_sam2_13_feb_2025_with_8_accumulation_steps_90.torch","large"),
# ("/home/asfand/Work/sam2/checkpoints/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml", "/home/asfand/Work/sam2/testing/models/fine_tuned_sam2_tiny_23_march_2025_8_accumulations_step_55.torch","tiny")
# ]

models = [("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_large.pt", "sam2_hiera_l.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_13_feb_2025_with_8_accumulation_steps_90.torch","large"),
("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/fine_tuned_sam2_tiny_23_march_2025_8_accumulations_step_55.torch","tiny")
]

max_iterations = 40
lower_bound = 0.99
upper_bound = 1.0
semaphore = threading.BoundedSemaphore(4)

# os.makedirs(output_images_dir, exist_ok=True)

def load_image_and_points(image_dir, dilated_mask_dir,org_mask_dir):
    """
    Load images and corresponding mask points for processing.

    Args:
        image_dir (str): Path to the directory containing images.
        dilated_mask_dir (str): Path to the directory containing dilated mask images.
        org_mask_dir (str): Path to the directory containing mask images.

    Yields:
        tuple: Processed image, resized mask, selected points, and image filename.
    """
    # Get points from the mask using the automatic selector
    points_dict = automatic_foreground_prompt_selector_from_directory(dilated_mask_dir,org_mask_dir)

    mask_files = [f for f in os.listdir(dilated_mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in mask_files:
        print("Processing image:", image_filename)

        # Construct corresponding mask file path
        dilated_mask_file_path = os.path.join(dilated_mask_dir, image_filename)
        
        org_mask_path = os.path.join(org_mask_dir,image_filename)
        if not os.path.exists(dilated_mask_dir) or not os.path.exists(org_mask_path):
            print(f"Mask for {image_filename} not found. Skipping.")
            continue

        # Read image and mask
        temp_list = image_filename.split('_tile_')[0].split('_')[0:-1]
        temp_name ="_".join(temp_list)
        image_name = temp_name + '_tile_' + image_filename.split('_tile_')[1]
        img = cv2.imread(os.path.join(image_dir, image_name))[..., ::-1]  # Convert BGR to RGB
        gt_dilated_mask = cv2.imread(dilated_mask_file_path, cv2.IMREAD_GRAYSCALE)
        gt_org_mask = cv2.imread(org_mask_path, cv2.IMREAD_GRAYSCALE)

        # Get points for the current image
        points = points_dict.get(image_filename, [])
        points_after = np.array(points).reshape(-1, 1, 2)
        # r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
        # img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        # gt_dilated_mask = cv2.resize(gt_dilated_mask, (int(gt_dilated_mask.shape[1] * r), int(gt_dilated_mask.shape[0] * r)))
        # gt_org_mask = cv2.resize(gt_org_mask, (int(gt_org_mask.shape[1] * r), int(gt_org_mask.shape[0] * r)))

        # Yield the processed image, mask, points, and filename
        yield img, gt_dilated_mask,gt_org_mask, points_after, image_filename
    

#models = [
 #   ("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_tiny_23_march_2025_8_accumulations_step_55.torch","tiny"),
  #  ("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_large.pt", "sam2_hiera_l.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_13_feb_2025_with_8_accumulation_steps_90.torch","large")
#]

def load_model(model_cfg,sam2_checkpoint,fine_tuned_sam2_checkpoint):
    sam2_model_cuda = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    sam2_model_cpu = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    # Build net and load weights
    predictor_cuda = SAM2ImagePredictor(sam2_model_cuda)
    predictor_cpu = SAM2ImagePredictor(sam2_model_cpu)
    predictor_cuda.model.load_state_dict(torch.load(fine_tuned_sam2_checkpoint))
    predictor_cpu.model.load_state_dict(torch.load(fine_tuned_sam2_checkpoint, map_location="cpu"))
    return predictor_cuda,predictor_cpu
    # return  predictor_cpu


def calculate_iou(pred_mask, ground_truth_mask):
   
    intersection = np.sum(pred_mask & ground_truth_mask)
    union = np.sum(pred_mask | ground_truth_mask)
    return intersection / union if union > 0 else 0


def testing_loop(input_points, predictor):
        
    point_label = np.ones([input_points.shape[0], 1])
      
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=point_label)
        
    if scores.ndim == 1:
        np_masks = np.array(masks)
        np_scores = scores  # If it's 1D, use it directly
    else:
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0] 

    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)

    mask_11 = sorted_masks
    mask_bool = mask_11.astype(bool)

    for i in range(mask_bool.shape[0]):
        mask = mask_bool[i]
        seg_map[mask]=1

    seg_map = seg_map.astype(np.uint8)
    _,seg_map = cv2.threshold(seg_map,0,255,cv2.THRESH_BINARY)
    
    # print(idx)
    return seg_map


def calculate_best_iou(gt_dilated_mask, gt_org_mask, no_of_prompt, best_iou, best_seg_map, best_prompt, predictor):
    # with semaphore:
    for iteration in range(max_iterations):
        # print(f"{no_of_prompt} point selection Iteration {iteration + 1}:")

        # Generate new points for refinement
        selected_points, all_selected_points_1 = automatic_foreground_prompt_selector_from_image(gt_dilated_mask,gt_org_mask, no_of_prompt)
        if len(all_selected_points_1) > 0:
            input_points_31 =  np.array(all_selected_points_1).reshape(-1, 1, 2)
            # Generate new segmentation map and compute IoU
            seg_map_ = testing_loop(input_points_31, predictor)
            iou = calculate_iou(seg_map_, gt_dilated_mask)
            # iou = calculate_iou(seg_map_, gt_org_mask)
            # print("Updated IoU:", iou)

            # Update the best segmentation map and IoU if it improves
            if iou > best_iou:
                best_iou = iou
                best_seg_map = seg_map_
                best_prompt = input_points_31  # Update best prompt
                # print("Best IoU updated:", best_iou)

            # Check if IoU is within the acceptable range
            if lower_bound <= iou < upper_bound:
                print("IoU is now within the acceptable range. Ending process.")
                break
    return best_iou, best_seg_map, best_prompt


def get_sam_results(zone_name, base_path,image_dir, org_mask_dir, dilated_mask_dir):
    # Ensure the output directory exists
    output_txt_files_dir = os.path.join(base_path, 'sam_prompts/' + zone_name)
    output_masks_dir = os.path.join(base_path, 'sam_results/' + zone_name)
    # output_images_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Evaluation_results_17_feb_2025/image_files_21_feb_2025_30_itterations_test_set_25_feb_2025_latest_with_one_two_three_and_four_points"

    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_txt_files_dir, exist_ok=True)

    org_mask_zone_dir = os.path.join(org_mask_dir,zone_name)
    dilated_mask_zone_dir = os.path.join(dilated_mask_dir,zone_name)

    data_generator_41  = load_image_and_points(image_dir,dilated_mask_zone_dir,org_mask_zone_dir)

    for idx, (img1, gt_dilated_mask,gt_org_mask, input_points_21, img_name) in enumerate(data_generator_41):
        # edged = cv2.Canny(gt_dilated_mask, 50, 150)
        gt_dilated_mask_thresholded = cv2.threshold(gt_dilated_mask,128,255,cv2.THRESH_BINARY)[1]

        best_iou, best_seg_map = 0, None
        best_prompt = input_points_21

        for sam2_checkpoint, model_cfg, finetuned_sam2_checkpoint,model_name in models:
            print(f"{model_name} is loaded!")
            predictor_cuda,predictor_cpu = load_model(model_cfg, sam2_checkpoint,finetuned_sam2_checkpoint)
            # predictor_cpu = load_model(model_cfg, sam2_checkpoint,finetuned_sam2_checkpoint)
            contours, _ = cv2.findContours(gt_dilated_mask_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            no_of_polygons = len(contours)

            total_no_of_prompts = (4 * no_of_polygons) +( 3 * no_of_polygons) + (2 * no_of_polygons) + no_of_polygons
            if total_no_of_prompts > 100:
                print(f"no of prompts is greater than 100 so computation is run on cpu.total no of prompts is {total_no_of_prompts}")
                predictor = predictor_cpu
            else:
                predictor = predictor_cuda
            with torch.no_grad():
                predictor.set_image(img1.copy())
                print("################################")
                # Initial segmentation and IoU
                seg_map = testing_loop(input_points_21,predictor)
                iou = calculate_iou(seg_map, gt_dilated_mask)

                _,gt_org_mask = cv2.threshold(gt_org_mask,10,255,cv2.THRESH_BINARY)

                _,gt_dilated_mask = cv2.threshold(gt_dilated_mask,128,255,cv2.THRESH_BINARY)

                # Perform refinement if IoU is below the lower bound
                if iou < lower_bound:
                    prompt_list = [1,2,3,4]
                    futures = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    # Start the load operations and mark each future with its URL
                        for promp in prompt_list:
                            futures.append(executor.submit(calculate_best_iou, gt_dilated_mask, gt_org_mask, promp, best_iou, best_seg_map, best_prompt,predictor))
                        for future in concurrent.futures.as_completed(futures):
                            current_iou, current_seg_map, current_prompt = future.result()
                            # print(f"current iou {model_name}",current_iou)
                            if current_iou > best_iou:
                                best_iou = current_iou
                                best_seg_map = current_seg_map
                                best_prompt = current_prompt

                print("###########################################################################")
                # print("before")
                print(f"Best Iou {model_name} :", best_iou)
                print("###########################################################################")

                if best_iou < lower_bound:
                # Process the image using representative point logic
                    rep_points = process_single_image_using_rep_point_logic(gt_dilated_mask)

                    print("Rep points:", rep_points)
                    # Initialize transformed group
                    transformed_group = []

                    for rep_group in rep_points:
                        for coord in rep_group:
                            # Ensure the coordinate is checked for being on the foreground
                            x, y = coord
                            if is_foreground_pixel(x, y, gt_org_mask):
                                # Append as a single-item nested list
                                transformed_group.append([[int(x), int(y)]])
                            else:
                                print(f"Point ({x}, {y}) is not on the foreground.")

                    # Convert the list to a numpy array
                    if len(transformed_group)>0:
                        rep_array = np.array(transformed_group, dtype=np.int32)
                        # print("Points after rep point logic and foreground check:", rep_array)

                        # Process the transformed points
                        seg_map_ = testing_loop(rep_array, predictor)
                        print("Ground truth mask shape:", gt_dilated_mask.shape)
                        iou = calculate_iou(seg_map_, gt_dilated_mask)
                        print("Updated IoU after representative point:", iou)

                        if iou > best_iou:
                            best_iou = iou
                            best_seg_map = seg_map_
                            best_prompt = rep_array  # Update best prompt
                            # print(f"Best IoU updated of {model_name}:", best_iou)

        txt_filename = os.path.join(output_txt_files_dir, f"{os.path.splitext(img_name)[0]}.txt")

        # # Save the best prompt and IoU to the text file
        with open(txt_filename, "w") as file:
            for point in best_prompt:
                for pt in point:
                    # if len(point) == 2:
                    file.write(f"{pt[0]},{pt[1]}\n")

        cv2.imwrite(f"{output_masks_dir}/{img_name}",best_seg_map)


if __name__ == '__main__':
    base_dir = "/home/asfand/Work/sam2/testing/data/complete_test/demo121/"
    image_directory = "/home/asfand/Work/sam2/testing/data/complete_test/demo121/image_tiles"
    org_mask_directory = "/home/asfand/Work/sam2/testing/data/complete_test/demo121/mean_tiles_dir/"
    dilated_mask_directory = "/home/asfand/Work/sam2/testing/data/complete_test/demo121/noise_removed_mean_tiles_dir/"
    cur_name = 'demo121_1'
    get_sam_results(cur_name,base_dir,image_directory, org_mask_directory, dilated_mask_directory)