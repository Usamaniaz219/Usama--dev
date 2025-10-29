import math
import os
import cv2
import numpy as np

def click_event(event, x, y, flags, params): 
    font = cv2.FONT_HERSHEY_SIMPLEX 

	# checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
		# displaying the coordinates on the image window 
        cv2.putText(Img, str(x) + ',' + str(y), (x,y), font, 1, (0, 255, 0), 2) 
        cv2.putText(Img, '.', (x,y), font, 1, (0, 255, 255), 2)
        cv2.putText(mask, str(x) + ',' + str(y), (x,y), font, 1, (0, 255, 0), 2) 
        cv2.putText(mask, '.', (x,y), font, 1, (0, 255, 255), 2)
        pt_list.append([x,y])
        with open(txt_filename, "a")  as file1:# append mode
            file1.write(str(x) + ', ' + str(y) +"\n")
            file1.close()
        cv2.imshow('image', Img) 
        cv2.imshow('mask', mask)

    # checking for middle mouse clicks 
    if event==cv2.EVENT_MBUTTONDOWN: 
        target = (x, y)

        to_del = min(pt_list, key=lambda point: math.hypot(target[1]-point[1], target[0]-point[0]))
        # cv2.putText(Img, str(to_del[0]) + ',' + str(to_del[1]), (to_del[0],to_del[1]), font, 1, (255, 255, 0), 2)
        cv2.putText(Img, 'X', (to_del[0],to_del[1]), font, 1, (0, 0, 255), 2) 
        cv2.putText(mask, 'X', (to_del[0],to_del[1]), font, 1, (0, 0, 255), 2) 
        pt_list.remove(to_del)
        cv2.imshow('image', Img) 
        cv2.imshow('mask', mask)
        with open( txt_filename, "r+" ) as f:
            lines = f.readlines()   
            print("lines",lines)        # Get a list of all lines
            f.seek(0)                       # Reset the file to the beginning
            # idx = lines.index(str(to_del[0]) + " ," + str(to_del[1]) + "\n") # Don't forget the '\n'
            idx = lines.index(str(to_del[0]) + ", " + str(to_del[1]) + "\n") # Don't forget the '\n'
            lines.pop( idx )                # Remove the corresponding index
            f.truncate()                    # Stop processing now because len(file_lines) > len( lines ) 
            f.writelines( lines )           # write back

if __name__=="__main__": 

    image_dir = '/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/image_1/'
    mask_dir = '/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/mask_1/'
    txt_dir = '/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/sam2_outputs_with_no_parrallel_processing/txt_files_24_march_2025_1/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_filename in image_files:
        print(image_filename)
        Img = cv2.imread(os.path.join(image_dir, image_filename), 1)
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        mask = cv2.imread(os.path.join(mask_dir, image_filename), 1)
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)))
        cv2.imwrite("resized_image.jpg",mask)
        txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt
        font = cv2.FONT_HERSHEY_SIMPLEX 
        pt_list =[]
        with open(txt_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Only process non-empty lines
                    point= list(map(int, line.split(',')))
                    pt_list.append(point)
                    cv2.putText(Img, str(point[0]) + ',' + str(point[1]), (point[0],point[1]), font, 1, (255, 0, 0), 2)
                    cv2.putText(Img, '.', (point[0],point[1]), font, 1, (0, 0, 255), 2)
                    cv2.putText(mask, str(point[0]) + ',' + str(point[1]), (point[0],point[1]), font, 1, (255, 0, 0), 2)
                    cv2.putText(mask, '.', (point[0],point[1]), font, 1, (0, 0, 255), 2)
        cv2.imshow('image', Img)
        cv2.imshow('mask', mask)     
        cv2.setMouseCallback('image', click_event) 
        
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
