from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from matplotlib import pyplot as plt
import cv2


config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)
print(result)
# show the results
#out=show_result_pyplot(model, img, result)