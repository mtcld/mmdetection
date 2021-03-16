import cv2
import matplotlib.pyplot as plt
import numpy as np

test_image_path = "capture_image_40290.png"

carpart_color_list = {'sli_side_turn_light': [[0.25, 0.5, 0.5]], \
                      'mirror': [[0.65, 0.5, 0.5], [0.5, 0.5, 0.5]], \
                      'fbu_front_bumper': [[0.15, 1, 1]], \
                      'grille': [[0.8, 1, 1]], \
                      'tail_gate': [[0.9, 0.5, 0.5]], \
                      'fender': [[0.3, 1, 1], [0.3, 0.5, 1]], \
                      'hli_head_light': [[0.5, 1, 1], [0.5, 0.5, 1]], \
                      'rbu_rear_bumper': [[0.8, 0.3, 1]], \
                      'door': [[0.1, 0.8, 0.5], [0.2, 0.6, 0.5], [0.45, 1, 1], [0.45, 0.5, 0.5]], \
                      'hood': [[0, 1, 1]], \
                      'tyre': [[0.7, 1, 1], [0.7, 0.5, 0.5], [0.15, 0.5, 0.5], [0.3, 0.5, 0.5]], \
                      'alloy_wheel': [[0.9, 1, 1], [0.35, 1, 1], [0.6, 1, 1], [0.75, 1, 1]], \
                      'rocker_panel': [[0.75, 0.5, 0.5], [0.95, 1, 1]], \
                      'qpa_quarter_panel': [[0.4, 0.6, 0.5], [0.4, 1, 1]], \
                      'tli_tail_light': [[0.35, 0.5, 0.5], [0.6, 0.5, 0.8]]
                      }


def get_are_max_contour(contours):
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    if (len(areas) == 0):
        return 0, 0
    return max(areas), np.argmax(areas)


def check_carpart_in_image(img, value, key):
    min_red = (max(0, int(value[0] * 180.0 - 2)), max(0, int(value[1] * 255.0 - 10)), max(0, int(value[2] * 255.0 - 5)))
    max_red = (min(180, int(value[0] * 180.0 + 2)), min(255, int(value[1] * 255.0 + 10)), min(255, int(value[2] * 255.0 + 5)))

    nemo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_nemo, min_red, max_red)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour, index = get_are_max_contour(contours)
    if (max_contour > 50):
        cnt = contours[index]
        cv2.drawContours(img, [cnt], 0, (0, 0, 0), 3)
        cv2.imshow(key, img)
        cv2.waitKey()
        return True
    return False


image = cv2.imread(test_image_path)

for key, values in carpart_color_list.items():
    image1 = image.copy()
    for value in values:
        if (check_carpart_in_image(image1, value, key)):
            print(key)
