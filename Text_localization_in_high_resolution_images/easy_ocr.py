
import cv2
import numpy as np
import math
import os
import easyocr


def load_image(image_path):
    image = cv2.imread(image_path)
    
    return image

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols


def extract_overlapping_tiles(image, tile_width, tile_height, overlap):
    tiles = []
    
    # Iterate over the image based on tile_width and tile_height with overlaps
    for start_y in range(0, image.shape[0], tile_height - overlap):
        for start_x in range(0, image.shape[1], tile_width - overlap):
            
            # Calculate the end coordinates with overlap
            end_x = min(start_x + tile_width, image.shape[1])
            end_y = min(start_y + tile_height, image.shape[0])

            # Extract the tile and append it to the tiles list
            tile = image[start_y:end_y, start_x:end_x]
            tiles.append(tile)
    
    return tiles



def detect_text_in_tile(image, tile_width, tile_height, overlap, reader):
    bounding_boxes = []
    output_image = np.copy(image)

    # Calculate number of tiles
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)
    
    for r in range(num_rows):
        for c in range(num_cols):
            start_x = c * tile_width
            start_y = r * tile_height

            # Extract overlapping tiles
            tiles = extract_overlapping_tiles(image, tile_width, tile_height, overlap)
            print(f"Number of tiles: {len(tiles)}")

            for i, tile in enumerate(tiles):
                # Save tile images (optional)
                cv2.imwrite(f"/home/usama/Tiles_results_30_sep_2024/tile_{i}.jpg", tile)

                # Perform OCR on the tile
                result = reader.readtext(tile)

                if len(result) > 0:
                    for (bbox, text, _) in result:
                        try:
                            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
                        except ValueError:
                            continue

                        # Adjust the coordinates to the original image
                        x1 += start_x
                        y1 += start_y
                        x2 += start_x
                        y2 += start_y
                        x3 += start_x
                        y3 += start_y
                        x4 += start_x
                        y4 += start_y

                        # Map the bounding box back to the full image
                        mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        mapped_bbox = np.array(mapped_bbox, dtype=np.int32).reshape((-1, 1, 2))
                        bounding_boxes.append(mapped_bbox)

                        # Draw bounding boxes on the output image
                        output_image = cv2.polylines(output_image, [mapped_bbox], isClosed=True, color=(0, 255, 0), thickness=2)

                        # Write the recognized text at the top-left corner of the bounding box
                        text_x, text_y = int(x1), int(y1) - 10  # Slightly above the box
                        cv2.putText(output_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    return bounding_boxes, output_image



def process_images(directory_path,output_dir, tile_width=2450, tile_height=2450,max_dimension_threshold=2500,overlap=1000):
    bounding_boxes_of_image = []
   
    os.makedirs(output_dir, exist_ok=True)
    all_masks = os.listdir(directory_path)
    masks_renamed = [mask.replace(".jpg", "").replace(".png", "") for mask in all_masks]

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):  # Add other extensions if needed
            image_path = os.path.join(directory_path, filename)
            ori_image_name = os.path.splitext(os.path.basename(image_path))[0]
            print("Original image name:", ori_image_name)
            image = load_image(image_path)
            output_image1 = np.copy(image)
            if image is not None:
               
                reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory
                height, width = image.shape[:2]
                largest_dimension = max(height, width)

                if largest_dimension <= max_dimension_threshold:
                    result = reader.readtext(image,width_ths=0.65)

                    if len(result) > 0:
                        for bbox, text, _ in result:
                            bbox_arr = np.array(bbox, dtype=np.int32)
                            bbox_arr_reshaped = bbox_arr.reshape((-1, 1, 2))
                            
                            # Draw the bounding box
                            output_image1 = cv2.polylines(output_image1, [bbox_arr_reshaped], isClosed=True, color=(0, 255, 0), thickness=2)
                            
                            # Calculate the position to put the text (at the top-left corner of the bounding box)
                            x, y = bbox[0]
                            x = int(x)
                            y = int(y)
                            # print("bounding box coord:",bbox[0])
                            # print("y:",y)
                            cv2.putText(output_image1, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 2, cv2.LINE_AA)

                        # Save the output image
                    output_file_path = os.path.join(output_dir, f"{ori_image_name}_detected_bbox_result_text_for_default_parameter_settings.jpg")
                    cv2.imwrite(output_file_path, output_image1)


                else:

                    bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height,overlap, reader)
                    output_file_path = os.path.join(output_dir, f"{ori_image_name}_detected_bbox_result_text_for_default_parameter_settings.jpg")
                    print("bounding Boxes length",len(bounding_boxes))
                    cv2.imwrite(output_file_path, output_image)
                
            else:
                print(f"Failed to read {image_path}")

# image_directory = '/home/usama/Home_data/Tiles_results_30_sep_2024/ca_dublin_latest_res_30_22/'
# output_dir = "/home/usama/Tiles_results_30_sep_2024/ca_dublin_latest_res_30_11/"
# # tile_width = 2000
# # tile_height = 2000
# # overlap = 400

# process_images(image_directory, output_dir)









