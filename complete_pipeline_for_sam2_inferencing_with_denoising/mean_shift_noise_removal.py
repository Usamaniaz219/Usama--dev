import cv2
import os
import time
import numpy as np

def remove_noise_2(mask_directory,output_dir):
    filenames = []
    for filename in os.listdir(mask_directory):
        if filename.endswith(".jpg") and not filename.endswith("_0.jpg"):

            mask_img = cv2.imread(os.path.join(mask_directory, filename), cv2.IMREAD_GRAYSCALE)
            # cv2.imshow('mask', mask)
            dilation= cv2.dilate(mask_img, np.ones((5, 5), np.uint8), iterations=2)
            # cv2.imshow('dilate', dilation)

            erosion= cv2.erode(dilation, np.ones((5, 5), np.uint8), iterations=5)
            # cv2.imshow('erosion', erosion)

            dilation2= cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=3)
            # cv2.imshow('dilate2', dilation2)
            _, binary_mask = cv2.threshold(dilation2, 20, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(binary_mask) == 0:
                continue
            cv2.imwrite(os.path.join(output_dir, filename), binary_mask)
            filenames.append(filename)
    return filenames
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 700, 700)
    # cv2.imshow('image', dilation2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

###############################################
def load_images_from_directory(directory):
    images = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            path = os.path.join(directory, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.astype(np.float32))
                filenames.append(filename)
    return images, filenames

def denoise_images_by_subtraction(directory):
    images, filenames = load_images_from_directory(directory)

    if not images:
        print("No images found in the directory.")
        return

    num_images = len(images)
    height, width = images[0].shape[:2]
    kernel = np.ones((3,3),np.uint8)

    # Ensure all images are same size
    for img in images:
        if img.shape != images[0].shape:
            raise ValueError("All images must be of the same dimensions and channels.")

    for i in range(num_images):
        # Clone the original image
        result = np.copy(images[i])
        

        # Subtract all other images
        for j in range(num_images):
            if i != j:
                image_j = images[j]
                
                image_j_closed = cv2.morphologyEx(image_j,cv2.MORPH_CLOSE,kernel,iterations=2)
                
                result -= image_j_closed

        # Normalize result to valid image range
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Save result over original file
        save_path = os.path.join(directory, filenames[i])
        cv2.imwrite(save_path, result)
        print(f"Saved denoised image: {save_path}")


#################################################################

if __name__ == "__main__":
    mask_dir = cv2.imread('/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/mean_tiles_dir/wa_darrington_3/denoised_masks_1')
    out_dir = cv2.imread('/home/asfand/Work/sam2/testing/data/complete_test/wa_darrington/mean_tiles_dir/wa_darrington_3/denoised_masks_2')
    image_path = "/media/usama/SSD/Data_for_complete_sam2_scripts/test_folder/denoised_masks_1/"
    denoise_images_by_subtraction(image_path)




    remove_noise_2(mask_dir, out_dir)