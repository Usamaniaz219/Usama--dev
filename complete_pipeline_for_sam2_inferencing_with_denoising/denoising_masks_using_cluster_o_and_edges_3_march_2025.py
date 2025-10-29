

import cv2
import numpy as np
import os
import rasterio.features
import shapely



def find_mask_with_suffix(directory, suffix):
    """
    Find the first file ending with the given suffix in the directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            return os.path.join(directory, filename)
    return None

def apply_canny_edge_detector(image_path, output_path):
    """
    Step 1: Perform adaptive thresholding to detect edges.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image,(5,5),0)
    edges = cv2.Canny(image,100,200)

    # Apply adaptive threshold for edge detection
    # edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY_INV, 21, 7)

    cv2.imwrite(output_path, edges)
    return edges

def combine_mask_with_edges(mask_path, edges):
    """
    Step 2: Combine zoning mask with detected edges.
    
    """
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask
    _, mask_image = cv2.threshold(mask_image, 50, 255, cv2.THRESH_BINARY)

    # Combine mask and edges using bitwise OR
    combined_mask = cv2.bitwise_or(mask_image, edges)

    # Morphological closing to remove small gaps
    kernel = np.ones((3, 3), np.uint8)
    # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # combined_mask = cv2.dilate(combined_mask,kernel,iterations=1)

    return combined_mask

def subtract_and_denoise(target_mask_path, combined_mask):
    """
    Step 3: Subtract edges from target cluster mask to denoise.
    """
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask
    _, target_mask = cv2.threshold(target_mask, 50, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    # target_mask = cv2.dilate(target_mask,kernel,iterations=1)
    # target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('Edges', target_mask)

  #     cv2.imshow('Combined Mask', combined_mask)
#   cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Subtract edges from the mask
    denoised_mask = cv2.subtract(target_mask, combined_mask)

    # Apply median blur for smoothing

    denoised_mask = cv2.medianBlur(denoised_mask, 5)
    # kernel = np.ones((3, 3), np.uint8)
    # denoised_mask = cv2.erode(denoised_mask,kernel,iterations=1)


    return denoised_mask

def remove_noise_using_cluster_0(image_path, mask_directory, denoised_output_dir):
    """
    Main processing loop: find _0.jpg mask, then process all other masks.
    """
    # Find the base _0 mask (this mask remains unchanged and used for all others)
    base_mask_path = find_mask_with_suffix(mask_directory, "_0.jpg")
    if base_mask_path is None:
        print("No _0.jpg mask found in directory.")
        return

    print(f"Using base mask: {base_mask_path}")

    # Detect edges once from the main image
    edge_output = os.path.join(mask_directory, "fl_edges.jpg")
    edges = apply_canny_edge_detector(image_path, edge_output)

    # Combine base mask with edges (this combined mask will be used for all images)
    combined_mask = combine_mask_with_edges(base_mask_path, edges)

    # Process each mask file (skip _0.jpg itself)
    for filename in os.listdir(mask_directory):
        if filename.endswith(".jpg") and not filename.endswith("_0.jpg"):
            target_mask_path = os.path.join(mask_directory, filename)

            # Apply denoising process
            denoised_mask = subtract_and_denoise(target_mask_path, combined_mask)

            ###########################################################################################
            mean_img_path = os.path.join(mask_directory,filename)
            mean_img = cv2.imread(mean_img_path)
            h, w = mean_img.shape[:2]
            mean_gray = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)
            _,mean_thresh = cv2.threshold(mean_gray,20,255,0)

            mean_shape = rasterio.features.shapes(mean_thresh.astype('uint8'))
            mean_polygons = [
                shapely.geometry.shape(shape[0])
                for shape in mean_shape if shape[1] == 255
            ]
            
            noise_shapes = rasterio.features.shapes(denoised_mask)

            # Select only cells with 1's
            noise_polygons = [
                shapely.geometry.shape(shape[0])
                for shape in noise_shapes if shape[1] == 255
            ]
            
            
            result_poly = []
            for mean_poly in mean_polygons:
                for noise_poly in noise_polygons:
                    if mean_poly.intersects(noise_poly):
                        result_poly.append(mean_poly)
                        continue
            denoised_mask_with_org_poly_reterival = rasterio.features.rasterize(result_poly, out_shape=(h, w))

            denoised_mask_with_org_poly_reterival = denoised_mask_with_org_poly_reterival.astype('float64')
            denoised_mask_with_org_poly_reterival *= (255/denoised_mask_with_org_poly_reterival.max())




            ###########################################################################################


            # kernel = np.ones((3,3),np.uint8)
            # Save denoised mask with the same name in "denoised_masks" folder
            denoised_output_path = os.path.join(denoised_output_dir, filename)


            # denoised_mask = cv2.erode(denoised_mask,kernel,iterations=1)
            # cv2.imwrite(denoised_output_path, denoised_mask)
            cv2.imwrite(denoised_output_path, denoised_mask_with_org_poly_reterival.astype('uint8'))
            #  cv2.imwrite(denoised_output_path, denoised_mask_with_org_poly_reterival.astype('uint8'))
            
            print(f"Denoised mask saved: {denoised_output_path}")

if __name__ == "__main__":

    img_path = "/home/asfand/Work/sam2/testing/data/complete_test/demo122/demo122.jpg"
    # Base directory where masks are stored
    mask_dir = "/home/asfand/Work/sam2/testing/data/complete_test/demo122/original_meanshift_masks"

    denoised_dir = "/home/asfand/Work/sam2/testing/data/complete_test/demo122/denoised_masks"
    os.makedirs(denoised_dir, exist_ok=True)

    remove_noise_using_cluster_0(img_path, mask_dir, denoised_dir)