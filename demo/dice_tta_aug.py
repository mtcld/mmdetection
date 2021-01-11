from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib
from matplotlib import pyplot as plt
import cv2
import json
import numpy as np
import pandas as pd
import sys
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
carparts_json='/mmdetection/data/crack_test_cp_0.json'
img_dir='/mmdetection/data/crack_latest/images/'
csv_path='/mmdetection/data/crack_cp.csv'

carparts_df=pd.read_csv(csv_path)



with open(test_json) as f:
    data = json.load(f)
    
    
with open(carparts_json) as f:
    carparts_data = json.load(f)
    
carparts_dict={}
for i in carparts_data['predict']:
    carparts_dict[i['image_name']]=i['carparts']
    
with open('carparts.json', 'w') as outfile:
        json.dump(carparts_dict,outfile,indent=4,ensure_ascii = False)

    
iou=0
l=0
#for i in range(len(data['images'])):
for i in range(len(data['images'])):
    h=data['images'][i]['height']
    w=data['images'][i]['width']
    mask_car=np.zeros((h,w),dtype='uint8')
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
        
        carparts_df_sub_num=carparts_df[carparts_df['image_name']==file_name][['number']].iloc[0]
        print(carparts_df_sub_num)
        
        if int(carparts_df_sub_num)>3:
            carparts_segm= carparts_dict[file_name]

            for cp in carparts_segm.keys():
                poly_points=carparts_segm[cp]
                for pp in poly_points:
                    p1=[]
                    for i2 in range(int(0.5*len(pp))):
                        p1.append([pp[2*i2],pp[2*i2 +1]])
                    p1=np.array([p1],np.int32)
                    cv2.fillPoly(mask_car, p1, 255 )
        else:
            mask_car=255*np.ones((h,w),dtype='uint8')
        
        cv2.imwrite('mask/'+file_name,mask_car)
        
        img = cv2.imread(img_dir + file_name)
        pred_img_comb=np.zeros(img.shape,np.uint8)
        mask_pred_sum=np.zeros(img.shape[:2],dtype='uint8')
        img1=img[0:int(h/2),0:int(w/2)]
        img2=img[int(h/2):h,0:int(w/2)]
        img3=img[0:int(h/2),int(w/2):w]
        img4=img[int(h/2):h,int(w/2):w]
        
        resultf = inference_detector(model, img)
        outf=show_result_pyplot(model, img, resultf,score_thr=0.4)
        pred_img=outf[0]
        
        mode=0
        for im in [img1,img2,img3,img4]:
            mode=mode+1
            result = inference_detector(model, im)

            out=show_result_pyplot(model, im, result,score_thr=0.4)
            
            if mode==1:
                pred_img_comb[0:int(h/2),0:int(w/2)]=out[0]
            if mode==2:
                pred_img_comb[int(h/2):h,0:int(w/2)]=out[0]
            if mode==3:
                pred_img_comb[0:int(h/2),int(w/2):w]=out[0]
            if mode==4:
                pred_img_comb[int(h/2):h,int(w/2):w]=out[0]            
            
            
            if len(out[2])==0:
                continue
            mask_pred=255*out[2].astype(np.uint8)
            
            shape=mask_pred[0].shape
            mask_pred_sub=np.zeros(shape,np.uint8)
                             
            for m in mask_pred:
                mask_pred_sub=cv2.bitwise_or(mask_pred_sub,m)
                
            if mode==1:
                mask_pred_sum[0:int(h/2),0:int(w/2)]=mask_pred_sub
                pred_img_comb[0:int(h/2),0:int(w/2)]=out[0]
            if mode==2:
                mask_pred_sum[int(h/2):h,0:int(w/2)]=mask_pred_sub
                pred_img_comb[int(h/2):h,0:int(w/2)]=out[0]
            if mode==3:
                mask_pred_sum[0:int(h/2),int(w/2):w]=mask_pred_sub
                pred_img_comb[0:int(h/2),int(w/2):w]=out[0]
            if mode==4:
                mask_pred_sum[int(h/2):h,int(w/2):w]=mask_pred_sub
                pred_img_comb[int(h/2):h,int(w/2):w]=out[0]
            
        intersection = np.logical_and(mask_act, mask_pred_sum)
        union = np.logical_or(mask_act, mask_pred_sum)
        iou_score = np.sum(intersection) / np.sum(union)
        print('iou_score')
        print(iou_score)
        iou=iou+iou_score
    
    vis = np.concatenate((pred_img, pred_img_comb), axis=1)
    cv2.imwrite('predicted_images/'+file_name, vis)

    
print('l_'+str(l))
print('final_iou')
print(iou/l)
        
