import numpy as np
# from maskrcnn_benchmark.config import cfg
# from demo.predictor_car import COCODemo
from numpy import linalg as LA
import cv2
import torch
from pipeline.async_service import predict

class CarDetect():
    def __init__(self, confidence_threshold=0.6, min_image_size=800, device="cuda"):
        self.confidence_threshold = confidence_threshold
        self.min_image_size = min_image_size
        self.device = device
        self.initialize()

    def initialize(self, context=None):
        pass

    def car_prediction(self, file_path):
        # image = cv2.imread(file_path)
        # predictions_list = self.coco_demo.run_on_opencv_image(image)
        # return image, predictions_list[1]
        json_contents = predict(file_path, ['car'])
        return json_contents['car'].result()

    def find_contour(self, thresh):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            return []
        return contours[0]
    
    '''
    def convert_mask(self, mask):
        mask = np.asarray(mask)
        thresh = mask[0, :, :, None]
        thresh = np.uint8(thresh)
        return thresh
    '''
    def find_max_mask_car(self, masks, labels, w, h, image):
        car_masks = [mask for i, mask in enumerate(masks) if (labels[i] == 'car' or labels[i] == 'truck')]
        if (len(car_masks)==0):
            return image.shape[1]*image.shape[0], np.ones_like(image), [0, 0, image.shape[1], image.shape[0]]

        area_list = []
        bbox_car_list = []
        for mask in car_masks:
            #thresh = self.convert_mask(mask)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, rw, rh = cv2.boundingRect(contours[0])
            bbox_car = [x, y, rw, rh]
            area = cv2.contourArea(contours[0])
            area_list.append(area)
            bbox_car_list.append(bbox_car)
        
        return max(area_list), car_masks[np.argmax(area_list)], bbox_car_list[np.argmax(area_list)]

    def process_after_car_prediction(self, image, predictions_list):
        # masks = predictions_list.get_field("mask").numpy()
        # labels = predictions_list.get_field("labels").tolist()
        # labels = [self.coco_demo.CATEGORIES[i] for i in labels]
        labels = predictions_list["labels"]
        masks = predictions_list["mask"]
        scores = predictions_list["scores"]
        h, w = image.shape[:2]
        lagest_car_area, lagest_car_mask, car_bbox = self.find_max_mask_car(masks, labels, w, h,
                                                                            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return lagest_car_area, lagest_car_mask, car_bbox
