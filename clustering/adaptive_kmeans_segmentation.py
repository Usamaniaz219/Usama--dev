import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def process_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    i = -1
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all files in the input directory
    for filename in sorted(os.listdir(input_dir)):
        
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            # image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_name = os.path.splitext(filename)[0]
            print(image_name)
            output_subdir = os.path.join(output_dir, image_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # Apply the KMeans_Zoning_Segmenter function
            def KMeans_Zoning_Segmenter(image,k_values):
                # image = cv2.resize(image, (image.shape[1] // 5, image.shape[0] // ))
                pixels = image.reshape((-1, 3))
                pixels = pixels.astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = k_values[i]
                print(k)
                _,labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
                unique_labels = np.unique(labels)
    
                # Extract areas of interest based on unique labels using logical AND operation
                interest_areas = []
                # gmm_extracted_images = []  # To store images extracted from GMM on interest areas
                for label in unique_labels:
                    label_mask = np.zeros_like(labels, dtype=np.uint8)
                    label_mask[labels == label] = 1  # Set pixels belonging to the current label to 1
                    # Reshape label_mask to the shape of the original image
                    label_mask = label_mask.reshape(image.shape[:2])

                    # Perform logical AND operation to extract areas of interest
                    area_of_interest = cv2.bitwise_and(image, image, mask=label_mask.astype(np.uint8) * 255)
                    interest_areas.append(area_of_interest)
                    mask_name = f"KMean_AOI_image_{label}.jpg"
                    output_directory_path  = os.path.join(output_subdir,mask_name)
                    cv2.imwrite(output_directory_path, area_of_interest)
                    # Apply GMM on the area of interest
                    area_pixels = area_of_interest.reshape((-1, 3))
                    # gmm_area = GaussianMixture(n_components=8, covariance_type='tied', random_state=10, init_params='k-means++')
                    kmean_area = KMeans(n_clusters=7,random_state=10)
                    # _,area_labels,centers = cv2.kmeans(pixels, 7, None, criteria, 10,None, centers)
                    # gmm_area.fit(area_pixels)
                    kmean_area.fit(area_pixels)
                    # area_labels = gmm_area.predict(area_pixels)
                    area_labels = kmean_area.predict(area_pixels)
                    unique_labelss = np.unique(area_labels)
                    for labelss in unique_labelss:
                        label_maskss = np.zeros_like(area_labels, dtype=np.uint8)
                        label_maskss[area_labels == labelss] = 1  # S
                        label_maskss = label_maskss.reshape(image.shape[:2])
                        extracted_mask = cv2.bitwise_and(area_of_interest, area_of_interest, mask=label_maskss.astype(np.uint8) * 255)
                        # cv2.imwrite(f"GMM_output_masks/68__/GMM_extracted_image_{label}_{labelss}.jpg", extracted_mask)
                        extracted_mask = cv2.cvtColor(extracted_mask,cv2.COLOR_RGB2BGR)
                        mask_name_2 = f"KMean_extracted_image_{label}_{labelss}.jpg"
                        output_directory_path_2  = os.path.join(output_subdir,mask_name_2)
                        cv2.imwrite(output_directory_path_2, extracted_mask)

            i= i+1
            print(i)
                        # cv2.imwrite(f"GMM_output_masks/Kmeans_output/79/KMean_extracted_image_{label}_{labelss}.jpg", extracted_mask)
                    
        k_values = [9,17,5,8,12,10,8,27,18,13,15,12,10,11,14,7,32,14,16,14,38,12,43,6,20,37,15,13,8,17,25,20,19,20,26,6,8,43,22,14,8] 
        print(len(k_values))      
        unique_labels= KMeans_Zoning_Segmenter(image,k_values)

           

# Specify the input and output directories
input_directory = "GMM_output_masks/data/"
output_directory = "GMM_output_masks/output_images1"

# Call the process_images function
process_images(input_directory, output_directory)
