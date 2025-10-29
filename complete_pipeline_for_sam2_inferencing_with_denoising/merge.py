import math
import os
from typing import final

import cv2
import numpy as np

def merge_tiles(image, mask_path):
    tile_size = 800
    h, w = image.shape[:2]

    no_of_cols = math.ceil(w/tile_size)
    no_of_rows = math.ceil(h/tile_size)

    max_tiles = no_of_cols * no_of_rows

    base_name = os.path.basename(mask_path)
    base_img = np.zeros((tile_size, tile_size), dtype = np.uint8)

    img_list =[]
    if os.path.isfile(mask_path + '/' + base_name + '_tile_0.jpg'):
        zero_img = cv2.imread(mask_path + '/' + base_name + '_tile_0.jpg', cv2.IMREAD_GRAYSCALE)
        zero_img = zero_img[112:912, 112:912]
        img_list.append(zero_img)
    else:
        img_list.append(base_img)
    cur_row = 0

    for index in range(1,max_tiles):
        cur_path = mask_path + '/' + base_name + '_tile_' + str(index) + '.jpg'
        cur_img = base_img.copy()
        if os.path.isfile(cur_path):
            cur_img = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
            cur_img = cur_img[112:912, 112:912]
        if index%no_of_cols == 0:
            cur_row +=1
            img_list.append(cur_img)
        else:
            img_list[cur_row] = np.hstack((img_list[cur_row],cur_img))
            print(img_list[cur_row].shape)
            # cv2.imshow('image', img_list[cur_row])
            # cv2.waitKey(0)

    final_img = img_list[0]
    for index_2 in range(1,len(img_list)):
        final_img = np.vstack((final_img,img_list[index_2]))

    final_img = final_img[:h,:w]
    return final_img

if __name__ == "__main__":
    image = cv2.imread("/media/usama/SSD/Data_for_complete_sam2_scripts/Data_for_complete_sam2_scripts_15_may_2025/demo115/demo115.jpg")
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image',700,700)
    # mask_path = '/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/sam_results/wa_darrington_10'
    merge_path = '/media/usama/SSD/Data_for_complete_sam2_scripts/Data_for_complete_sam2_scripts_15_may_2025/demo115/sam_merged/'
    image_dir_path = '/media/usama/SSD/Data_for_complete_sam2_scripts/Data_for_complete_sam2_scripts_15_may_2025/demo115/'
    sam_path = image_dir_path + '/sam_results/'
    for sam_subdir in os.listdir(sam_path):
        merged = merge_tiles(image, os.path.join(sam_path, sam_subdir))
        cv2.imwrite(merge_path + f'{sam_subdir}.jpg', merged)
    #     cv2.imshow('image', merged)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
