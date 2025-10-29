import cv2
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

image = cv2.imread("/home/usama/usama_dev_test/CUTS/data/zoning_map_resized_dataset/map_images/demo92_tile_4162.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image.reshape((-1, 3))
hierarchical = AgglomerativeClustering(n_clusters=4)  
hierarchical.fit(pixels)
labels = hierarchical.labels_
linkage_matrix= hierarchical.children_
print("linkage",dir(hierarchical.children_))
# print("labels",labels)
# print("linkage matrix",linkage_matrix)

# Reshape labels into image shape
labels = labels.reshape(image.shape[:2])

unique_labels = np.unique(labels)

for label in unique_labels:
    label_mask = np.zeros_like(labels)
    label_mask[labels == label] = 255  # Setting the pixels of the specific label to white (255)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(label_mask, cmap='gray')  # Displaying the binary mask
    plt.title(f"Binary Mask for Label {label}")
    plt.axis("off")
    plt.show()