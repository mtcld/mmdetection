import numpy as np
# from demo.predictor_car import COCODemo
from numpy import linalg as LA
import cv2
from pipeline.async_service import predict


class CarpartDetection():
    def __init__(self):
        # cfg.merge_from_file("/workspace/share/maskrcnn-benchmark/carpart_model/carparts_test.yaml")
        # manual override some options
        # cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

        carpart_view = [0, 1, 1, 0, 1, -1, 1, -1, 0, 1, 1, 0, -1, 0, -1, 1, 1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 1]
        categories = [{'id': 0, 'name': 'fli_fog_light', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 1, 'name': 'sli_side_turn_light', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 2, 'name': 'mirror', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 3, 'name': 'fbu_front_bumper', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 4, 'name': 'bpn_bed_panel', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 5, 'name': 'grille', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 6, 'name': 'tail_gate', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 7, 'name': 'mpa_mid_panel', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 8, 'name': 'fender', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 9, 'name': 'hli_head_light', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 10, 'name': 'car', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 11, 'name': 'rbu_rear_bumper', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 12, 'name': 'door', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 13, 'name': 'lli_low_bumper_tail_light', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 14, 'name': 'hood', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 15, 'name': 'hpa_header_panel', 'supercategory': 'Carparts', 'view': 1},
                      {'id': 16, 'name': 'trunk', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 17, 'name': 'tyre', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 18, 'name': 'alloy_wheel', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 19, 'name': 'hsl_high_mount_stop_light', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 20, 'name': 'rocker_panel', 'supercategory': 'Carparts', 'view': 0},
                      {'id': 21, 'name': 'qpa_quarter_panel', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 22, 'name': 'rpa_rear_panel', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 23, 'name': 'rdo_rear_door', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 24, 'name': 'tli_tail_light', 'supercategory': 'Carparts', 'view': -1},
                      {'id': 25, 'name': 'fbe_fog_light_bezel', 'supercategory': 'Carparts', 'view': 1}]

        self.cat = ['back_ground']
        for cate in categories:
            self.cat.append(cate['name'])


    def compare_key(self, real_carpart_center_list, _2d_carpart_center_list):
        score = 0
        for real_list_info in real_carpart_center_list:
            real_label = real_list_info[0]
            real_position = np.asarray(real_list_info[1:])
            key_is_exit = False
            for projection_info in _2d_carpart_center_list:
                if (real_label == projection_info[0]):
                    projection_position = np.asarray(projection_info[1:])
                    score = score + np.exp(-LA.norm(real_position - projection_position) * 5)
                    key_is_exit = True
                else :
                    score =score - 0.005
            if (key_is_exit == False):
                score = score - 0.5
        return score

    def check_label_in_largest_car(self, car_part_masks,max_car_mask):
        #car_part_masks = self.convert_mask(car_part_masks)
        #car_mask = self.convert_mask(lagest_car_mask)
        #print(car_mask.shape)
        
        car_mask_size = cv2.countNonZero(max_car_mask)
        if (car_mask_size == 0):
            return True


        overlaping_mask = cv2.bitwise_and(max_car_mask, car_part_masks)

        car_part_area = self.get_area_mask(car_part_masks)
        overlaping_area = self.get_area_mask(overlaping_mask)

        if (car_part_area == 0):
            return False
        if (overlaping_area / car_part_area > 0.2):
            return True
        return False
    
    '''
    def convert_mask(self, mask):
        mask = np.asarray(mask)
        thresh = mask[0, :, :, None]
        thresh = np.uint8(thresh)
        return thresh
    '''
    def get_area_mask(self, thresh_mask):
        contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            return -100
        area = cv2.contourArea(contours[0])
        return area

    def get_center_contour(self, mask):

        #thresh = self.convert_mask(mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            return -1, -1

        max_area= cv2.contourArea(contours[0])
        max_contour= contours[0]
        for contour in contours:
            area=cv2.contourArea(contour)
            if(max_area< area):
                max_are=area
                max_contour= contour

        if(cv2.contourArea(max_contour) <10):
            return -1,-1

        M = cv2.moments(max_contour)

        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    def process_after_forward(self, cat_result, bbox_list ,masks,scores,lagest_car_mask,lagest_car_bbox):
        capart_list_infos = []
        bbox_list_of_lagest_car = []
        scores_list_of_lagest_car =[]
        masks_list_of_lagest_car=[]
        for i, cat_id in enumerate(cat_result):
            if (self.check_label_in_largest_car(masks[i],lagest_car_mask)):
                cx, cy = self.get_center_contour(masks[i])
                if (cx < 0):
                    continue
                info = [cat_id, (cx - lagest_car_bbox[0]) / lagest_car_bbox[2], # self.cat[cat_id]
                        (cy - lagest_car_bbox[1]) / lagest_car_bbox[3]]
                capart_list_infos.append(info)
                bbox_list_of_lagest_car.append(bbox_list[i])
                scores_list_of_lagest_car.append(scores[i])
                masks_list_of_lagest_car.append(masks[i])

        return capart_list_infos, bbox_list_of_lagest_car,scores_list_of_lagest_car,masks_list_of_lagest_car

    def preprocess_predict_result(self, predictions_list):

        cat_result = predictions_list["labels"]
        masks = predictions_list["mask"]
        scores = predictions_list["scores"]

        bbox_car_list = []
        for mask in masks:
            #thresh = self.convert_mask(mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            area = 0
            x = 0
            y = 0
            rw = 0
            rh = 0
            for c in contours:
                area_c = cv2.contourArea(c)
                x, y, rw, rh = cv2.boundingRect(c)
            if area_c > area:
                bbox_car = [x, y, rw, rh]
            bbox_car_list.append(bbox_car)
        return cat_result, bbox_car_list, masks,scores

    def capart_prediction(self,image):
        json_contents = predict(image, ['carpart'])
        return json_contents['carpart'].result()[1]

    def estimate_car_view(self, _2d_carpart_center_lists, real_carpart_center_list, files):
        score_list = []
        for i, _2d_carpart_center_list in enumerate(_2d_carpart_center_lists):
            score = self.compare_key(real_carpart_center_list, _2d_carpart_center_list)
            score_list.append(score)
        index_max = [i for i, value in enumerate(score_list) if (value == max(score_list))]
        try:
            index_max[0], files[index_max[0]]
        except:
            print("")
        return index_max[0],files[index_max[0]]

    def check_view_full_car(self, bbox_list,width,height):
        number_box_out_of_image = 0
        for i, bbox in enumerate(bbox_list):
            left_top, right_bottom = bbox[:2], bbox[2:]
            if (left_top[0] / width < 0.03 or left_top[1] / height < 0.03 or right_bottom[
                0] / width > 0.97 or
                    right_bottom[1] / height > 0.97):
                number_box_out_of_image = number_box_out_of_image + 1
        if (number_box_out_of_image > 2 or len(bbox_list) < 7):
            return False
        return True

    def check_half_view_car(self, number_carpart,lagest_car_area,width,height):
        area_ratio = lagest_car_area / (width * height)
        if (area_ratio < 0.8 and number_carpart > 6):
            return True
        return False

    def process_after_carpart_prediction(self,predictions_list,lagest_car_mask,lagest_car_bbox):
        cat_result, bbox_list, masks,scores = self.preprocess_predict_result(predictions_list)
        capart_list_infos, bbox_list_of_lagest_car,scores_list_of_lagest_car,masks_list_of_lagest_car= self.process_after_forward(cat_result, bbox_list, masks,scores,lagest_car_mask,lagest_car_bbox)
        return capart_list_infos, bbox_list_of_lagest_car,scores_list_of_lagest_car,masks_list_of_lagest_car


