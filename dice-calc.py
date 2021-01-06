import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from matplotlib import pyplot as plt
import cv2
import json
from tqdm import tqdm
import pydensecrf.densecrf as dcrf

def crf(original_image, mask_img):
    if (len(mask_img.shape) < 3):
        mask_img = gray2rgb(mask_img)
    annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
    colors, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = 2
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseBilateral(sxy=25,srgb=4,rgbim=original_image, compat=10, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    return MAP.reshape((original_image.shape[0], original_image.shape[1]))

config_file = 'configs/detectors/dent_detector_updated_segm.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'data/crack_latest_mmdet_model2/epoch_14.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

test_json='/mmdetection/data/crack_latest/annotations/crack_test_new.json'
img_dir='/mmdetection/data/crack_latest/images/'

with open(test_json) as f:
    data = json.load(f)

    
iou=0
l=0
for i in range(len(data['images'])):
    h=data['images'][i]['height']
    w=data['images'][i]['width']
    mask_act=np.zeros((h,w),dtype='uint8')
    for j in range(len(data['annotations'])):
        if data['annotations'][j]['image_id']==data['images'][i]['id']:
            p1=data['annotations'][j]['segmentation'][0]
            p1=[int(i) for i in p1]
            p2=[]
            for p in range(int(len(p1)/2)):
                p2.append([p1[2*p],p1[2*p+1]])
            fill_pts = np.array([p2], np.int32)
            cv2.fillPoly(mask_act, fill_pts, 255)
    if np.unique(mask_act,return_counts=True)[1][1]/(w*h)>0.00:
        l=l+1
        file_name = data['images'][i]['file_name']
        img = cv2.imread(img_dir + file_name)
        result = inference_detector(model, img)
        out = show_result_pyplot(model, img, result)
        mask_pred=255*out[2].astype(np.uint8)
        mask_pred_sum=np.zeros(img.shape[:2],dtype='uint8')
    
        for m in mask_pred:
            mask_pred_sum=cv2.bitwise_or(mask_pred_sum,m)
        
        #cv2.imwrite('pred'+str(i)+'.jpg',255*mask_act)
        intersection = np.logical_and(mask_act, mask_pred_sum)
        union = np.logical_or(mask_act, mask_pred_sum)
        iou_score = np.sum(intersection) / np.sum(union)
        print('iou_score')
        print(iou_score)
        iou=iou+iou_score
    
print('l_'+str(l))
print('final_iou')
print(iou/l)
    
    
# for i in tqdm(range(len(data['images']))):
#     file_name = data['images'][i]['file_name']
#     img = cv2.imread(img_dir + file_name)
#     result = inference_detector(model, img)
#     out = show_result_pyplot(model, img, result)
#     mask_pred=255*out[2].astype(np.uint8)
#     print(img.shape)
#     print(mask_pred.shape)
#     print(np.unique(mask_pred))
#     mask=np.zeros(mask_pred.shape[1:],np.uint8)
#     for p in mask_pred:
#         mask=cv2.bitwise_or(mask,p)
#     #cv2.imwrite('mask/'+str(i)+'.png',mask)
