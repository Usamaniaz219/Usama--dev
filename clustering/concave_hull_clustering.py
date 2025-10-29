import cv2
import numpy
import concave_hull

from concave_hull import concave_hull
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def get_clusters(X, y):
    s = numpy.argsort(y)
    return numpy.split(X[s], numpy.unique(y[s], return_index=True)[1][1:])


# name = 'demo158_1.jpg'
# img = cv2.imread('clean_masks/' + name)
img = cv2.imread('Median_erode__contours_low__and_high_resolution_test/median_contours_reterival_test_results/ca_la_habra_15.jpg')

distance = 20/1

img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gry, 100, 255, 0)

# plt.imshow(img_gry)
# plt.show()
contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_NONE)
#
mask = numpy.zeros(img.shape, numpy.uint8)
cnt_arr = []
for cnt in [contours]:
    for i in cnt:
        # if cv2.contourArea(i) < (distance/2):
        #     continue
        for j in i:
            cnt_arr.append([j[0][0],j[0][1]])

result_cnt = DBSCAN(eps=distance, min_samples=10).fit(cnt_arr)
new_arr = []
new_ind = []
for i,j in zip(result_cnt.labels_,cnt_arr):
    if i == -1:
        continue
    new_arr.append(j)
    new_ind.append(i)
clst = get_clusters(numpy.array(new_arr),numpy.array(new_ind))
for res_cnt in clst:

    approx = concave_hull(res_cnt, concavity=0.3, length_threshold=distance)
    if approx is not None:
        cv2.drawContours(img, [approx], 0, (255, 0, 0), 5)

cv2.imwrite('ca_escondido_cnt.jpg',img)
# plt.axis('off')
# plt.imshow(img)
# plt.show()
# _, thresh_mask = cv2.threshold(img_gry, 127, 255, 0)
# cv2.imwrite('bw_masks/mask_' + name,thresh)
# cv2.imshow("0.1", cv2.resize(mask, (600, 750)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
