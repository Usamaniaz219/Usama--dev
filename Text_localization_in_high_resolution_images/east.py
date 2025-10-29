import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression


def tile_image_opencv(input_image_path, tile_size):
    
    input_image = cv2.imread(input_image_path)
    input_height, input_width = input_image.shape[:2]
    
    num_tiles_x = input_width // tile_size[0]
    num_tiles_y = input_height // tile_size[1]
    
    tiled_images = []
    
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = x * tile_size[0]
            upper = y * tile_size[1]
            right = left + tile_size[0]
            lower = upper + tile_size[1]
         
            tile_image = input_image[upper:lower, left:right]
            
            tiled_images.append(tile_image)
    
    return tiled_images
    

def detect_text(image):
    orig = image.copy()
    layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
   
    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
  
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
  
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        
        for x in range(0, numCols):
 
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
 
    for (startX, startY, endX, endY) in boxes:
    
        startX = int(startX )
        startY = int(startY)
        endX = int(endX)
        endY = int(endY )
       
        cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 0, 0), 2)

    return orig



def concat_tiles_opencv(tiled_images, tile_size, input_height, input_width):

    output_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)

    num_tiles_x = input_width // tile_size[0]
    num_tiles_y = input_height // tile_size[1]

    for i, tile_image in enumerate(tiled_images):

        x = (i % num_tiles_x) * tile_size[0]
        y = (i // num_tiles_x) * tile_size[1]

        output_image[y:y+tile_size[1], x:x+tile_size[0], :] = tile_image

    return output_image


image_path = '/home/usama/usama-dev/image_impainting_models/RN_impainting/examples/images/downsampled_image0.png'
image = cv2.imread(image_path)
tile_size=(320,320)

tiled_images = tile_image_opencv(image_path, tile_size)

results = []
for tiled_image in tiled_images:
    
    img=tiled_image 
   
    result = detect_text(img)
    results.append(result)
    cv2.imshow("img_out_tile",result)
    cv2.waitKey(0)


image_height, image_width = image.shape[:2]

output_image = concat_tiles_opencv(results, tile_size, image_height, image_width)

cv2.imwrite('output_image.jpg', output_image)















