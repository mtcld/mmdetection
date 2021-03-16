import os

os.chdir('/workspace/share/maskrcnn-benchmark')
from  three_d_car import color_segment
from three_d_car.car_detection import CarDetect
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="/workspace/share/dataset/KRUG/unzipped/01/44010111822074/img2176806.jpg",
                    type=str)
args = parser.parse_args()

_3D_image_folder = "/workspace/share/maskrcnn-benchmark/3d/3d_images"
config_file = "demo/carparts_test.yaml"
test_imgs_folder = "/workspace/share/dataset/KRUG/unzipped"
car = CarDetect()
from carpart_detection import CarpartDetection

all_image_check = color_segment.get_2d_image_path_lists(test_imgs_folder)
# process with projection image
_2d_image_path_lists = color_segment.get_2d_image_path_lists(_3D_image_folder)
_2d_carpart_center_lists = color_segment.get_carpart_center_list_in_all_2d_image(_2d_image_path_lists)


def get_save_file_path(path_file):
    sub_folder_list = file_path.split("/")[-6:-2]
    path_save = ""
    for sub_folder in sub_folder_list:
        path_save = path_save + sub_folder + "/"
    return path_save


for i, file_path in enumerate(all_image_check):

    image, predictions_list = car.car_prediction(file_path)
    lagest_car_area, lagest_car_mask, lagest_car_bbox = car.process_after_car_prediction(image, predictions_list)

    # from carpart_detection import CarpartDetection
    carpartdetection = CarpartDetection(lagest_car_area, lagest_car_mask, lagest_car_bbox)

    # process with real image
    image, real_carpart_center_list, real_bbox_list = carpartdetection.capart_prediction(file_path)
    is_full_view = carpartdetection.check_view_full_car(real_bbox_list)
    path_save = get_save_file_path(file_path)

    if (is_full_view == True):
        path_save = path_save + "full_car/"

    elif (carpartdetection.check_half_view_car(real_carpart_center_list)):
        path_save = path_save + "half_car/"

    else:
        path_save = path_save + "quater_car/"
    print(path_save)
    if (os.path.isdir(path_save) == False):
        os.makedirs(path_save)

    cv2.imwrite(path_save + str(i) + ".jpg", image)
