import cv2
import numpy as np
import math
from src.utils.configuration import Configuration
from src.utils.metautils import adjust_coordinate_rectangle

def haversine(p1, p2):
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(math.radians, [p1[0], p1[1], p2[0], p2[1]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_countour(mask):
    largest_contour = None
    mask_img = mask['segmentation'].astype(np.uint8) * 255
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def compute_roundness(mask) -> float:
    roundness = 0
    largest_contour = get_countour(mask)
    if largest_contour is not None:
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            roundness = (4 * np.pi * area) / (perimeter ** 2)
    return roundness

def compute_eccentricity(mask) -> float:
    eccentricity = 0
    largest_contour = get_countour(mask)
    if largest_contour is not None:
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
        return eccentricity

def compute_percentage(mask) -> float:
    total_area = mask['segmentation'].size
    mask_area = mask['area']
    retval = 100 * float(mask_area) / float(total_area)
    return retval

def compute_pixels(mask) -> float:
    return mask['area']

def compute_stability(mask) -> float:
    return mask['stability_score']

def compute_iou(mask) -> float:
    return mask['predicted_iou']

def compute_meters(mask) -> float:
    configuration = Configuration()
    lat_min, lat_max, long_min, long_max = adjust_coordinate_rectangle(configuration.get('areaofinterest_earth'))
    height = haversine(lat_min, long_min, lat_max, long_min)
    width = haversine(lat_min, long_min, lat_min, long_max)
    total_meters = width * height
    total_pixels = mask['segmentation'].size
    mask_pixels = mask['area']
    mask_meters = (total_meters * mask_pixels) / total_pixels;
    return mask_meters

