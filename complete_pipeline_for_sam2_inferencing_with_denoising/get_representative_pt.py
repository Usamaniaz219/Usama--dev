import os

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
import random
import cv2


def get_quadrant_representative_points(polygon):
    """Get representative points from the quadrants of a polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    quadrants = [
        Polygon([(min_x, min_y), (center_x, min_y), (center_x, center_y), (min_x, center_y)]),
        Polygon([(center_x, min_y), (max_x, min_y), (max_x, center_y), (center_x, center_y)]),
        Polygon([(min_x, center_y), (center_x, center_y), (center_x, max_y), (min_x, max_y)]),
        Polygon([(center_x, center_y), (max_x, center_y), (max_x, max_y), (center_x, max_y)])
    ]

    temp_points = []  # Temporary list to hold quadrant representative points

    for quadrant in quadrants:
        if quadrant.intersects(polygon):
            intersection = quadrant.intersection(polygon)
            if not intersection.is_empty:
                rep_point = intersection.representative_point()
                temp_points.append((rep_point.x, rep_point.y))

    return temp_points


def is_foreground_pixel(x, y, mask):
    """Check if a point lies on the foreground pixel of the annotation mask."""
    rows, cols = mask.shape
    if 0 <= int(y) < rows and 0 <= int(x) < cols:
        return mask[int(y), int(x)]==255
    return False


def get_representative_points_within_contours(contours, contours_1,mask):
    """Get representative points within each part of the polygon or a reduced number if there's intersection with contours_1."""
    representative_points = []

    for contour_1 in contours_1:
        try:
            shapely_polygon = Polygon([(point[0][0], point[0][1]) for point in contour_1])
            shapely_polygon = make_valid(shapely_polygon)  # Ensure the polygon is valid
            count = 0
            tmp_pts = []

            for contour in contours:
                # shapely_polygon_1 = Polygon([(point[0][0], point[0][1]) for point in contour])
                coordinates = []
                for cont_point in contour:
                    x = cont_point[0][0]
                    y = cont_point[0][1]
                    coordinates.append((x, y))
                tmp_pts_1 =[]
                if len(coordinates)>3:
                # Create the polygon using the list of coordinates
                    shapely_polygon_1 = Polygon(coordinates)
                    shapely_polygon_1 = make_valid(shapely_polygon_1)  # Ensure the polygon is valid

                    if shapely_polygon.intersects(shapely_polygon_1):
                        count += 1

                        if shapely_polygon_1.area <= 200:
                            rep_point = shapely_polygon_1.representative_point()
                            representative_points.append(([(rep_point.x, rep_point.y)]))
                        else:
                            pts = get_quadrant_representative_points(shapely_polygon_1)
                            for pt in pts:
                                if is_foreground_pixel(pt[0],pt[1],mask):
                                    tmp_pts_1.append(pt)
                            tmp_pts.append(tmp_pts_1)
            if tmp_pts:
                if count > 1:
                    # print("length of tmp_pts",len(tmp_pts))
                    if len(tmp_pts) >= 2:
                        representative_points.append(list(random.sample(tmp_pts[0], 2)))
                        representative_points.append(list(random.sample(tmp_pts[1], 2)))
                    elif tmp_pts:
                        representative_points.append(list(tmp_pts[0]))
                elif count==1 and tmp_pts:
                    representative_points.append(list(tmp_pts[0]))
                else:
                    rep_point = shapely_polygon.representative_point()
                    representative_points.append([(rep_point.x, rep_point.y)])

        except ValueError as e:
            print(f"Error creating polygon: {e}")
            continue

    return representative_points


def process_single_image_using_rep_point_logic(mask):
    """
    Process a single image and its corresponding mask to extract representative points.

    Parameters:
        mask (Img): mask image.

    Returns:
        rep_points (list): Representative points extracted from the contours of the mask.
    """

    if mask is None:
        # print(f"Error: Could not read mask from path {mask_path}")
        return []

    # Threshold the mask to create a binary image
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours_1, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Erode the mask and find contours again
    eroded_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
    _, binary_mask_eroded = cv2.threshold(eroded_mask, 127, 255, cv2.THRESH_BINARY)
    contours_2, _ = cv2.findContours(binary_mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the final mask based on contour count
    final_mask = eroded_mask if len(contours_2) >= len(contours_1) else mask
    _, binary_mask_final = cv2.threshold(final_mask, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get representative points with intersection logic
    rep_points = get_representative_points_within_contours(contours, contours_1, mask)

    return rep_points


def automatic_foreground_prompt_selector_from_image(dilated_mask, org_mask, no_of_prompts):
    # Binarize the mask
    _, dilated_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("length of dilated masks contours", len(dilated_contours))
    org_contours, _ = cv2.findContours(org_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("length of original masks contours", len(org_contours))

    # If more dilated contours than original, filter by largest area
    if len(dilated_contours) > len(org_contours):
        contours_with_areas = [(cv2.contourArea(c), c) for c in dilated_contours]
        contours_with_areas.sort(reverse=True, key=lambda x: x[0])  # Sort by area (descending)
        dilated_contours_1 = [c for _, c in contours_with_areas[:len(org_contours)]]
        print("points are selected on the dilated masks contours")
    elif len(dilated_contours) + 1 == len(org_contours) or len(dilated_contours) + 2 == len(org_contours) or len(
            dilated_contours) == len(org_contours):
        dilated_contours_1 = org_contours
        print("points are selected on the org masks contours!")
    else:
        dilated_contours_1 = dilated_contours

    # print("length of dilated contours",len(dilated_contours))
    all_selected_points = []
    selected_points = []
    for contour in dilated_contours_1:
        #         # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter points inside the bounding rectangle but within the contour
        contour_points = [
            (px, py)
            for px in range(x + 2, x + w - 2)  # Exclude the boundary by skipping edges
            for py in range(y + 2, y + h - 2)
            if cv2.pointPolygonTest(contour, (px, py), False) > 0  # Check if inside the contour
        ]
        # Filter only foreground points
        foreground_points = [pt for pt in contour_points if is_foreground_pixel(pt[0], pt[1], org_mask)]

        # Randomly select up to 10 points if enough points exist
        if len(foreground_points) > no_of_prompts:
            if cv2.contourArea(contour) < 200:
                selected_indices = np.random.choice(len(foreground_points), 1, replace=False)
            else:
                selected_indices = np.random.choice(len(foreground_points), no_of_prompts, replace=False)
            selected_points = [foreground_points[i] for i in selected_indices]
        else:
            selected_points = foreground_points

        all_selected_points.extend(selected_points)

    return selected_points, all_selected_points


def automatic_foreground_prompt_selector_from_directory(dilated_mask_dir, org_mask_dir):
    """
    Select points automatically from mask images by processing contours.

    Args:

        dilated_mask_dir (str): Path to the directory containing dilated mask images.
        org_mask_dir (str): Path to the directory containing mask images.

    Returns:
        dict: A dictionary with filenames as keys and selected points as values.
    """
    selected_points_dict = {}

    # Iterate through all mask images
    for filename in os.listdir(dilated_mask_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            print("image filename", filename)
            # Read the mask image
            dilated_mask_path = os.path.join(dilated_mask_dir, filename)
            org_mask_path = os.path.join(org_mask_dir, filename)
            dilated_mask = cv2.imread(dilated_mask_path, cv2.IMREAD_GRAYSCALE)
            _, dilated_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)
            org_mask = cv2.imread(org_mask_path, cv2.IMREAD_GRAYSCALE)

            # r = np.min([1024 / dilated_mask.shape[1], 1024 / dilated_mask.shape[0]])  # Scaling factor
            # dilated_mask = cv2.resize(dilated_mask, (int(dilated_mask.shape[1] * r), int(dilated_mask.shape[0] * r)))
            # org_mask = cv2.resize(org_mask, (int(org_mask.shape[1] * r), int(org_mask.shape[0] * r)))
            # # org_mask = cv2.medianBlur(org_mask,5)
            _, org_mask = cv2.threshold(org_mask, 10, 255, cv2.THRESH_BINARY)
            # cv2.imshow("org_mask.jpg",org_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            selected_points, all_selected_points = automatic_foreground_prompt_selector_from_image(dilated_mask,
                                                                                                   org_mask, 4)
            # print("all selected points",all_selected_points)

            # Store the points in the dictionary
            selected_points_dict[filename] = all_selected_points
            # all_selected_points = all_selected_points.clear()

    return selected_points_dict