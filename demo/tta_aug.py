from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib
from matplotlib import pyplot as plt
import cv2
import json
import numpy as np
matplotlib.rcParams['figure.figsize'] = (20, 10)



def draw_poly_org(p1,image1,color):
    image=image1.copy()
    mask=np.zeros(image.shape[:2],np.uint8)
    p1=[int(i) for i in p1]
    p2=[]
    for p in range(int(len(p1)/2)):
        p2.append([p1[2*p],p1[2*p+1]])
    fill_pts = np.array([p2], np.int32)
    cv2.fillPoly(mask, fill_pts, 255)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     
    cv2.drawContours(image, contours, -1, color, 2)    
    return image,contours



config_file = '../configs/detectors/dent_detector_updated_segm.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../data/crack_latest_mmdet_model2/epoch_14.pth'



# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')



test_json='/mmdetection/data/crack_latest/annotations/crack_test_new.json'
img_dir='/mmdetection/data/crack_latest/images/'

with open(test_json) as f:
    data = json.load(f)
    
    
    
#7,20
index=7


plt.figure()
file_name = data['images'][index]['file_name']
img=cv2.imread(img_dir+file_name)    
image_new_org=img.copy()
for j in range(len(data['annotations'])):
    if data['annotations'][j]['image_id']==data['images'][index]['id']:
        p1=data['annotations'][j]['segmentation'][0]
        image_new_org,_=draw_poly_org(p1,image_new_org,(0,255,0))
plt.imshow(cv2.cvtColor(image_new_org, cv2.COLOR_BGR2RGB))
plt.show()


plt.figure()
file_name = data['images'][index]['file_name']
img = cv2.imread(img_dir + file_name)
result = inference_detector(model, img)
# show the results
out=show_result_pyplot(model, img, result,score_thr=0.4)
plt.imshow(cv2.cvtColor(out[0], cv2.COLOR_BGR2RGB))
plt.show()


file_name = data['images'][index]['file_name']
img = cv2.imread(img_dir + file_name)
h,w=img.shape[:2]
img1=img[0:int(h/2),0:int(w/2)]
img2=img[int(h/2):h,0:int(w/2)]
img3=img[0:int(h/2),int(w/2):w]
img4=img[int(h/2):h,int(w/2):w]

for im in [img1,img2,img3,img4]:
    result = inference_detector(model, im)
    # show the results
    out=show_result_pyplot(model, im, result,score_thr=0.4)
    plt.imshow(cv2.cvtColor(out[0], cv2.COLOR_BGR2RGB))
    plt.show()
    
    









