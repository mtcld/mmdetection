# # Importing libraries
import mmcv
import torch
import numpy as np
import configparser
import random, string
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config = configparser.ConfigParser()
config.read('./model/config.ini')

damage_list=['scratch','crack','dent']

class Model():

    def __init__(self, config_file, checkpoint_file, category_name):
        self.config_file= config_file
        self.checkpoint_file = checkpoint_file
        self.category_name = category_name
        self.category_model = category_name
        self.initialize_model()
        self.score_threshold = 0.3
        self.bbox_color = (72, 101, 241)
        self.text_color = (72, 101, 241)

    def initialize_model(self):
        # Building the model from a config file and a checkpoint file
        model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')
        self.category_model = model

    def inference(self, img):

        model = self.category_model
        # # Infering on a single image
        result = inference_detector(model, img)

        pred_image = model.show_result(
        img,
        result,
        score_thr=self.score_threshold,
        bbox_color=self.bbox_color,
        text_color=self.text_color)

        height = pred_image.shape[0]
        width = pred_image.shape[1]

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        class_name = model.CLASSES
        scores = bboxes[:, -1]

        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        sample_labels = []
        sample_filtered_score = []
        for i in range(len(scores)):
            if scores[i] > self.score_threshold:
                sample_labels.append(labels[i])
                sample_filtered_score.append(scores[i])
                
        sample_labels_with_class_names = []
        for i in range(len(sample_labels)):
            sample_labels_with_class_names.append(class_name[sample_labels[i]])

        filtered_segms = []
        total_number_of_labels = 0
        for i in range(len(scores)):
            if scores[i] > self.score_threshold:
                filtered_segms.append(segms[i])
                total_number_of_labels += 1

        filtered_segms_ndarray = np.array(filtered_segms)
        filtered_segms_ndarray = filtered_segms_ndarray.astype('uint8')
        sample_segms_ndarray = filtered_segms_ndarray.reshape(total_number_of_labels, 1, height, width)
        sample_segms_ndarray= [mask[0, :, :, None] for mask in sample_segms_ndarray]

        return [pred_image, {
                                "labels": sample_labels_with_class_names, 
                                "scores": sample_filtered_score, 
                                "mask": sample_segms_ndarray
                                }]

def get_all_model():
    car_model=Model(config['CAR']['config_file'], config['CAR']['checkpoint_file'], config['CAR']['category_name'] )
    carpart_model=Model(config['CARPART']['config_file'], config['CARPART']['checkpoint_file'], config['CARPART']['category_name'] )
    dent_model=Model(config['DENT']['config_file'], config['DENT']['checkpoint_file'], config['DENT']['category_name'] )
    crack_model=Model(config['CRACK']['config_file'], config['CRACK']['checkpoint_file'], config['CRACK']['category_name'] )
    scratch_model=Model(config['SCRATCH']['config_file'], config['SCRATCH']['checkpoint_file'], config['SCRATCH']['category_name'] )

    return {'car':car_model,'carpart':carpart_model,'scratch':scratch_model,'dent':dent_model,'crack':crack_model}

def get_damage_list():
    return damage_list