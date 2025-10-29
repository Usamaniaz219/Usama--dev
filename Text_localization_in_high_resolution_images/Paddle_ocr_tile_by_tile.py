import numpy as np
import easyocr
import cv2
import math
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

bounding_boxes = []

def load_image(image_path):
    image = cv2.imread(image_path)
    
    return cv2.medianBlur(image,3)

# def resize_image_to_multiple(image, tile_width, tile_height):
#     # Resize the image to be a multiple of tile_width and tile_height
#     new_width = math.ceil(image.shape[1] / tile_width) * tile_width
#     new_height = math.ceil(image.shape[0] / tile_height) * tile_height
#     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#     return resized_image

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    # Calculate the number of rows and columns
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols

def extract_tile(image, start_x, start_y, tile_width, tile_height):
    # Extract the tile from the image
    end_x = min(start_x + tile_width, image.shape[1])
    end_y = min(start_y + tile_height, image.shape[0])
    return image[start_y:end_y, start_x:end_x]

def detect_text_in_tile(image, tile_width, tile_height, ocr):
    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)
    # allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Iterate over each row
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)
    for r in range(num_rows):
        # Iterate over each column
        for c in range(num_cols):
            # Calculate the starting coordinates of the tile
            start_x = c * tile_width
            start_y = r * tile_height

            # Extract the tile from the image
            tile = extract_tile(image, start_x, start_y, tile_width, tile_height)
            # result = reader.readtext(tile, text_threshold=0.5, min_size=10, low_text=0.48, mag_ratio=1.2, contrast_ths=0.1)
            # result = reader.readtext(tile, text_threshold=0.5)
            # result = reader.readtext(tile)
            result = ocr.ocr(tile, cls=True)

            # Check if any bounding boxes were returned
            if len(result) > 0:
                for idx in range(len(result)):
                    res = result[idx]
                    if res is not None:
                        boxes_tile = [line[0] for line in res]
                    else:
                        boxes_tile = None
                # Extract the bounding box coordinates and text from the result
                # bounding_boxes_tile = [bbox for bbox, _, _ in result]
                if np.any(boxes_tile):
                # Map the bounding box coordinates back to the original image coordinates
                    for bbox in boxes_tile:
                        try:
                            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
                        except ValueError:
                            continue

                        # Adjust bounding box coordinates to fit the original image
                        x1 += start_x
                        y1 += start_y
                        x2 += start_x
                        y2 += start_y
                        x3 += start_x
                        y3 += start_y
                        x4 += start_x
                        y4 += start_y

                        mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        mapped_bbox = np.array(mapped_bbox, dtype=np.int32)
                        mapped_bbox = mapped_bbox.reshape((-1, 1, 2))
                        bounding_boxes.append(mapped_bbox)
                        output_image = cv2.polylines(output_image, [mapped_bbox], isClosed=True, color=(0, 0, 255), thickness=2)
                    

    return bounding_boxes, output_image

def process_images(directory_path, tile_width, tile_height, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):  # Add other extensions if needed
            image_path = os.path.join(directory_path, filename)
            ori_image_name = os.path.splitext(os.path.basename(image_path))[0]
            print("Original image name:", ori_image_name)
            image = load_image(image_path)
            if image is not None:
                # Resize image to be a multiple of tile_width and tile_height
                # resized_image = resize_image_to_multiple(image, tile_width, tile_height)
                # reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, ocr)
                output_file_path = os.path.join(output_dir, f"{ori_image_name}_detected_bbox_result_text_for_default_parameter_settings.jpg")
                cv2.imwrite('Page_0.png', output_image)

                cv2.imwrite(output_file_path, output_image)
            else:
                print(f"Failed to read {image_path}")

# Directory containing the images
image_directory = '/media/usama/SSD/Usama_dev_ssd/ocr_based_text_detection/drive/'
output_dir = "/media/usama/SSD/Usama_dev_ssd/ocr_based_text_detection/paddle_ocr_detection_and_recognition_results_8_oct_2024/"
tile_width = 1024
tile_height = 1024

process_images(image_directory, tile_width, tile_height, output_dir)




