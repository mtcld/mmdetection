import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from matplotlib import pyplot as plt
import cv2
import json
from tqdm import tqdm

config_file = 'configs/detectors/dent_detector_updated_segm.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'data/crack_latest_mmdet_model2/epoch_14.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

test_json='/mmdetection/data/crack_latest/annotations/crack_test_new.json'
img_dir='/mmdetection/data/crack_latest/images/'

with open(test_json) as f:
    data = json.load(f)

for i in tqdm(range(len(data['images']))):
    file_name = data['images'][i]['file_name']
    img = cv2.imread(img_dir + file_name)
    result = inference_detector(model, img)
    out = show_result_pyplot(model, img, result)
    mask_pred=out[2].astype(np.uint8)
    print(img.shape)
    print(mask_pred.shape)
    print(np.unique(mask_pred))
    mask=np.zeros(mask_pred.shape[1:],np.uint8)
    for p in mask_pred:
        mask=cv2.bitwise_or(mask,p)
    cv2.imwrite(str(i)+'.png',mask)
