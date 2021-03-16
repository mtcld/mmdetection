import cv2
from three_d_car import color_segment
from three_d_car.car_detection import CarDetect
from three_d_car.carpart_detection import CarpartDetection
from three_d_car.side_util import get_side_carpart,overlay_class_names

class Car_View():
    def __init__(self,_3D_image_folder="/workspace/share/batch_mxnet/maskrcnn-benchmark/image_3d"):
        self.car_detection = CarDetect()
        self.carpart_detection = CarpartDetection()
        self.carpart_categories = {'fli_fog_light': 1, 'sli_side_turn_light': 1, 'mirror+r': 0,'mirror+l': 0, 'fbu_front_bumper': 1,
                                   'bpn_bed_panel': -1, 'grille': 1, 'tail_gate': -1, 'mpa_mid_panel': 0, 'fender+l': 1,'fender+r': 1,
                                   'hli_head_light+l': 1,'hli_head_light+r': 1, 'car': 0, 'rbu_rear_bumper': -1,
                                   'door+lf': 0,'door+lb': 0,'door+rf': 0,'door+rb': 0,
                                   'lli_low_bumper_tail_light': -1,
                                   'hood': 1, 'hpa_header_panel': 1, 'trunk': -1, 'tyre+lf': 0,
                                   'tyre+lb': 0,'tyre+lf': 0,'tyre+rb': 0 ,'tyre+rf': 0 ,
                                   'alloy_wheel+lb': 0,'alloy_wheel+lf': 0,'alloy_wheel+rb': 0,'alloy_wheel+rf': 0,
                                   'hsl_high_mount_stop_light': -1, 'rocker_panel+l': 0,'rocker_panel+r':0,  'qpa_quarter_panel+r': -1, 'qpa_quarter_panel+l':1,
                                   'rpa_rear_panel': -1, 'rdo_rear_door': -1, 'tli_tail_light+r': -1,'tli_tail_light+l': -1,
                                   'fbe_fog_light_bezel': 1}
        self.process_3d_images(_3D_image_folder)

    def get_car_detection(self):
        return self.car_detection

    def set_car_detection(self, detection):
        self.car_detection = detection

    def get_carpart_detection(self):
        return self.carpart_detection

    def set_carpart_detection(self, detection):
        self.carpart_detection = detection

    def get_part_view(self, part):
        return self.carpart_categories[part]

    def get_view_from_carparts(self, carparts):
        view_count = 0
        for i in carparts:
            view_count += self.carpart_categories[i]
        print(view_count)
        return int(view_count / abs(view_count)) if view_count != 0 else 0

    def process_3d_images(self,_3D_image_folder):
        self._2d_image_path_lists = color_segment.get_2d_image_path_lists(_3D_image_folder)
        self._2d_carpart_center_lists = color_segment.get_carpart_center_list_in_all_2d_image(self._2d_image_path_lists)

    def detect_maincar_carpart(self,json_contents):
        image, predictions_list = json_contents['car'].result()
        lagest_car_area, self.lagest_car_mask, lagest_car_bbox = self.car_detection.process_after_car_prediction(image, predictions_list)

        # process with real image
        carpart_predictions_list = json_contents['carpart'].result()[1]
        real_carpart_center_list, real_bbox_list,_ ,masks_list_lagest_car= self.carpart_detection.process_after_carpart_prediction(
            carpart_predictions_list, self.lagest_car_mask, lagest_car_bbox)

        return real_carpart_center_list,real_bbox_list,lagest_car_area,masks_list_lagest_car

    def get_view_carpart_list(self,real_carpart_center_list):
        # get carpart list in projection image
        self.index, label = self.carpart_detection.estimate_car_view(self._2d_carpart_center_lists, real_carpart_center_list,
                                                                self._2d_image_path_lists)

        view_angle = (label.split(".png")[0]).split("_")[-1]
        print("view_angle",view_angle)
        m_real_carpart_center_list = get_side_carpart(real_carpart_center_list, view_angle)
        real_carpart_list = [info[0] for info in m_real_carpart_center_list]
        return real_carpart_list

    def filter_carpart_by_view(self,main_view,proposed_view=None):
        # check full view
        if (main_view !=0):
            return proposed_view, None
        image=cv2.imread(self._2d_image_path_lists[self.index])
        _2d_center_list = color_segment.get_center_carpart_list(image,check_side=True)

        view_carpart_list = [info[0] for info in _2d_center_list]
        ignored_neutral_view_parts = list(
            filter(lambda x: self.carpart_categories[x] != 0 and x != "sli_side_turn_light", view_carpart_list))
        view = self.get_view_from_carparts(ignored_neutral_view_parts)

        if proposed_view and proposed_view != view:
            return proposed_view, None

        proposed_carpart_list = list(filter(lambda x: self.carpart_categories[x] == view, ignored_neutral_view_parts))
        return view, proposed_carpart_list

    def check_view_car(self,real_bbox_list,lagest_car_area,image_w,image_h):
        is_full_view = self.carpart_detection.check_view_full_car(real_bbox_list, image_w,image_h)
        if(is_full_view):
            return 0
        is_half_view= self.carpart_detection.check_half_view_car(len(real_bbox_list),lagest_car_area, image_w,image_h)
        if(is_half_view):
            return 1
        return 2

