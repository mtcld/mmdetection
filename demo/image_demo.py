from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from matplotlib import pyplot as plt
import cv2


config_file = '../configs/detectors/dent_detector_updated_segm.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../data/disk1/dent_updated_mmdet_model/epoch_13.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'repair-car-dent-800x800.jpg'
result = inference_detector(model, img)

# show the results
out=show_result_pyplot(model, img, result)