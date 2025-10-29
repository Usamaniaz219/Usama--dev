import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def find_optimal_gmm_k(pixels, max_k=10):
    """
    Determine optimal number of clusters using Bayesian Information Criterion (BIC).
    """
    lowest_bic = np.inf
    best_k = 2
    bics = []

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(pixels)
        bics.append(gmm.bic(pixels))
        if bics[-1] < lowest_bic:
            lowest_bic = bics[-1]
            best_k = k
    print(f" Optimal GMM clusters = {best_k} (BIC method)")
    return best_k


def GMM_Zoning_Segmenter(image, output_dir, n_colors=None):
    """
    Apply Gaussian Mixture Model (GMM) clustering to segment an image.
    Automatically determines the optimal number of clusters if not provided.
    """
    pixels = image.reshape((-1, 3))

    # Automatically find optimal cluster count
    if n_colors is None:
        n_colors = find_optimal_gmm_k(pixels, max_k=10)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_colors, random_state=42)
    gmm.fit(pixels)
    labels = gmm.predict(pixels).reshape(image.shape[:2])
    unique_labels = np.unique(labels)

    print(f"Segmenting into {len(unique_labels)} zones...")

    for label in unique_labels:
        mask = (labels == label).astype(np.uint8) * 255
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        mask_path = os.path.join(output_dir, f"gmm_segment_{label}.png")
        cv2.imwrite(mask_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

    return labels


def process_images(input_dir, output_dir):
    """
    Process all images in a folder with GMM segmentation.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_name = os.path.splitext(filename)[0]
        output_subdir = os.path.join(output_dir, image_name)
        os.makedirs(output_subdir, exist_ok=True)

        print(f"\n🖼 Processing: {filename}")
        GMM_Zoning_Segmenter(image, output_subdir)

