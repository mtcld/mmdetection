import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import os

carpart_color_list = {'sli_side_turn_light': [[0.25, 0.5, 0.5]], \
                      'mirror': [[0.65, 0.5, 0.5], [0.5, 0.5, 0.5]], \
                      'fbu_front_bumper': [[0.15, 1, 1]], \
                      'grille': [[0.8, 1, 1]], \
                      'trunk': [[0.9, 0.5, 0.5]], \
                      'fender': [[0.3, 1, 1], [0.3, 0.5, 1]], \
                      'hli_head_light': [[0.5, 1, 1], [0.5, 0.5, 1]], \
                      'rbu_rear_bumper': [[0.15, 0.5, 0.5], [0.8, 0.3, 1]], \
                      'door': [[0.1, 0.8, 0.5], [0.2, 0.6, 0.5], [0.45, 1, 1], [0.45, 0.5, 0.5]], \
                      'hood': [[0, 1, 1]], \
                      'tyre': [[0.7, 1, 1], [0.7, 0.5, 0.5], [0.5, 0.45, 1], [0.3, 0.5, 0.5]], \
                      'alloy_wheel': [[0.9, 1, 1], [0.35, 1, 1], [0.6, 1, 1], [0.75, 1, 1]], \
                      'rocker_panel': [[0.75, 0.5, 0.5], [0.95, 1, 1]], \
                      'qpa_quarter_panel': [[0.4, 0.6, 0.5], [0.4, 1, 1]], \
                      'tli_tail_light': [[0.35, 0.5, 0.5], [0.6, 0.5, 0.8]]
                      }

carpart_side_color_list = {'sli_side_turn_light': [[0.25, 0.5, 0.5]], \
                      'mirror+l': [[0.65, 0.5, 0.5]],'mirror+r': [[0.5, 0.5, 0.5]], \
                      'fbu_front_bumper': [[0.15, 1, 1]], \
                      'grille': [[0.8, 1, 1]], \
                      'trunk': [[0.9, 0.5, 0.5]], \
                      'fender+l': [[0.3, 1, 1]],'fender+r' :[[0.3, 0.5, 1]], \
                      'hli_head_light+l': [[0.5, 1, 1]], 'hli_head_light+r':[[0.5, 0.5, 1]], \
                      'rbu_rear_bumper': [[0.15, 0.5, 0.5], [0.8, 0.3, 1]], \
                      'door+lf': [[0.1, 0.8, 0.5]],'door+lb':[[0.2, 0.6, 0.5]],'door+rb':[[0.45, 1, 1]],'door+rf': [[0.45, 0.5, 0.5]], \
                      'hood': [[0, 1, 1]], \
                      'tyre+lf': [[0.7, 1, 1]],'tyre+rf': [[0.7, 0.5, 0.5]], 'tyre+rb':[[0.5, 0.45, 1]],'tyre+lb':[[0.3, 0.5, 0.5]], \
                      'alloy_wheel+lf': [[0.9, 1, 1]],'alloy_wheel+lb':[[0.35, 1, 1]],'alloy_wheel+rb':[[0.6, 1, 1]],'alloy_wheel+rf': [[0.75, 1, 1]], \
                      'rocker_panel+r': [[0.75, 0.5, 0.5]], 'rocker_panel+l': [[0.95, 1, 1]], \
                      'qpa_quarter_panel+l': [[0.4, 0.6, 0.5]],'qpa_quarter_panel+r':[[0.4, 1, 1]], \
                      'tli_tail_light+r': [[0.35, 0.5, 0.5]],'tli_tail_light+l':[[0.6, 0.5, 0.8]]
                      }

are_max_list={'sli_side_turn_light': 925.0, 'tyre+rb': 27696.0, 'fbu_front_bumper': 50722.5, 'door+rb': 48327.0, 'rocker_panel+l': 10850.0, 'alloy_wheel+lb': 16438.5, 'alloy_wheel+rf': 15363.5, 'hli_head_light+l': 4554.5, 'qpa_quarter_panel+r': 26272.5, 'fender+l': 21781.0, 'trunk': 34980.0, 'mirror+l': 2295.0, 'alloy_wheel+rb': 17404.0, 'fender+r': 25704.5, 'qpa_quarter_panel+l': 29738.0, 'hli_head_light+r': 4831.0, 'tli_tail_light+r': 8239.0, 'alloy_wheel+lf': 16243.5, 'hood': 40387.0, 'grille': 5442.5, 'mirror+r': 2351.0, 'rocker_panel+r': 10857.5, 'tyre+lb': 29466.0, 'door+rf': 56199.5, 'tli_tail_light+l': 8919.5, 'tyre+lf': 27138.0, 'rbu_rear_bumper': 62255.5, 'tyre+rf': 26081.5, 'door+lb': 49916.0, 'door+lf': 53241.0}

def get_area_max_contour(contours):
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    if (len(areas) == 0):
        return 0, 0
    return max(areas), np.argmax(areas)


def segment_carpart(img, min_hsv, max_hsv):
    nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_nemo, min_hsv, max_hsv)
    return mask


def check_carpart_in_image(img, value, carpart_name,check_side=False):
    min_hsv = (max(0, int(value[0] * 180.0 - 2)), max(0, int(value[1] * 255.0 - 10)), max(0, int(value[2] * 255.0 - 5)))
    max_hsv = (
    min(180, int(value[0] * 180.0 + 2)), min(255, int(value[1] * 255.0 + 10)), min(255, int(value[2] * 255.0 + 5)))
    mask = segment_carpart(img, min_hsv, max_hsv)

    if (carpart_name == 'car'):
        mask = cv2.bitwise_not(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour, index = get_area_max_contour(contours)
    if(check_side):
        if(max_contour/int(are_max_list[carpart_name]) >0.5):
            print(carpart_name,max_contour,are_max_list[carpart_name])
            return True, contours[index]
        else:
            return False, []

    if(max_contour> 50):
        return True, contours[index]

    return False, []


def get_car_bbox(img):
    _, car_contour = check_carpart_in_image(img, (0, 0, 0.24), "car")
    left_bbox, top_bbox, bbox_width, bbox_height = cv2.boundingRect(car_contour)
    return left_bbox, top_bbox, bbox_width, bbox_height


def get_center_contours(carpart_name, cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def get_center_carpart_list(image,check_side):
    carpart_center_list = []
    x, y, r_w, r_h = get_car_bbox(image)

    if(check_side):
        carpart_list=carpart_side_color_list
    else :
        carpart_list =carpart_color_list
    for carpart_name, colors in carpart_list.items():
        for color in colors:
            is_exit_carpart_name, contours = check_carpart_in_image(image, color, carpart_name,check_side)
            if (is_exit_carpart_name):
                cx, cy = get_center_contours(carpart_name, contours)
                info = [carpart_name, (cx - x) / r_w, (cy - y) / r_h]
                carpart_center_list.append(info)
    return carpart_center_list


def get_carpart_center_list_in_all_2d_image(files,check_side=False):
    carpart_center_lists = []
    for path in files:
        image = cv2.imread(path)
        carpart_center_list = get_center_carpart_list(image,check_side)
        carpart_center_lists.append(carpart_center_list)
    return carpart_center_lists


def get_2d_image_path_lists(image_folder):

    files = []
    for r, d, f in os.walk(image_folder):
        for file in f:
            if ('.jpg' in file) or ('.png' in file):  # todo : not all the image extension
                files.append(os.path.join(r, file))
    return files

