import os

os.chdir('/workspace/share/maskrcnn-benchmark')
import three_d_car.color_segment
from three_d_car.car_detection import CarDetect
from three_d_car.carpart_detection import CarpartDetection
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="/workspace/share/dataset/KRUG/unzipped/01/44010111822074/img2176806.jpg",
                    type=str)
args = parser.parse_args()

_3D_image_folder = "./image_3d"
config_file = "demo/carparts_test.yaml"

car = CarDetect()
image, predictions_list = car.car_prediction(args.file_path)
_, lagest_car_mask, lagest_car_bbox = car.process_after_car_prediction(image, predictions_list)

carpartdetection = CarpartDetection(_, lagest_car_mask, lagest_car_bbox)

# process with real image
image, real_carpart_center_list, real_bbox_list = carpartdetection.capart_prediction(args.file_path)

# process with projection image
_2d_image_path_lists = color_segment.get_2d_image_path_lists(_3D_image_folder)
_2d_carpart_center_lists = color_segment.get_carpart_center_list_in_all_2d_image(_2d_image_path_lists)

print("is_full_view", carpartdetection.check_view_full_car(real_bbox_list))

print(carpartdetection.estimate_car_view(_2d_carpart_center_lists, real_carpart_center_list, _2d_image_path_lists))
