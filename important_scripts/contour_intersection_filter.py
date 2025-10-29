import time
import cv2
import numpy
import shapely
from shapely.validation import make_valid
import multiprocessing


def retrieve_poly(ori):
    # (ori, eroded_cnt) = args
    for cnt_fill in contours_filled:
        if cnt_fill.size <= 4 or ori.size <= 4:
            continue
        cnt_ori_2d = numpy.squeeze(ori)
        polygon_ori = shapely.geometry.Polygon(cnt_ori_2d)

        cnt_fill_2d = numpy.squeeze(cnt_fill)
        polygon_fill = shapely.geometry.Polygon(cnt_fill_2d)

        if make_valid(polygon_ori).intersects(make_valid(polygon_fill)):
            return ori

start_time = time.time()

ori_path = '/home/usama/Asfand_data_811/ca_emeryville_SwinIR/'
# thresh_path = 'threshold_3/ct_avon_thresh_3_1/'
result_path = 'cnt_intersection_based_boundary_retaining_results/cnt_test_code_results/'

# Erosion and then median filter 3
name = "ca_emeryville_SwinIR_8"
# original = cv2.imread(ori_path + name + '.jpg')
original = cv2.imread(ori_path + name + '.jpg')
gry_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
_, thresh_original = cv2.threshold(gry_original, 20, 255, 0)
contours_original, _ = cv2.findContours(thresh_original, 1, cv2.CHAIN_APPROX_NONE)


median = cv2.medianBlur(original, 3)
_,median = cv2.threshold(median, 25, 255, cv2.THRESH_BINARY )
contours,_ = cv2.findContours(median, 1, cv2.CHAIN_APPROX_SIMPLE)
extent_threshold = 0.1
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    bounding_box_area = w * h
    
    # Calculate extent
    extent = float(area) / bounding_box_area
    # print("aspect ratio",aspect_ratio)
    if cv2.contourArea(cnt) > 10:
        # cv2.drawContours(filtered_image1, [cnt], 0, 255, -1)
        if extent > extent_threshold:  # Assu
            cv2.drawContours(median, [cnt], 0, 255, -1)
        
    else: 
        pass


gry_filled = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
ret, thresh_filled = cv2.threshold(gry_filled, 20, 255, 0)
contours_filled, _ = cv2.findContours(thresh_filled, 1, cv2.CHAIN_APPROX_NONE)

mask = numpy.zeros(original.shape, numpy.uint8)
intersect_cnt = []
# pool = multiprocessing.Pool(processes=7)
with multiprocessing.Pool(6) as pool:
    intersect_cnt.append(pool.map(retrieve_poly, contours_original))

res = [i for i in intersect_cnt[0] if i is not None]
cv2.drawContours(mask, res, -1, (255, 255, 255), cv2.FILLED)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow("median", cv2.resize(median, (600, 750)))
# cv2.imshow("median", cv2.resize(median, (600, 750)))
cv2.imshow("mask", cv2.resize(mask, (600, 750)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(result_path + name + '.jpg', mask)

