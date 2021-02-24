import json
import cv2
import numpy as np
import pathlib
from tqdm import tqdm

damage_name='total-missing'
image_dir='data/'+damage_name+'/images/'
mask_dir='data/'+damage_name+'_mask'
pathlib.Path(mask_dir).mkdir(parents=True, exist_ok=True) 
modes=['test','train','valid']

for mode in modes:    
    p1='data/'+damage_name+'/annotations/missing_'+mode+'.json'
    
    with open(p1) as f:
        data=json.load(f)

    for i in tqdm(range(len(data['images']))):
        dict_act={}
        dict_pred={}
        fn=data['images'][i]['file_name']
        cv2.imread(image_dir+fn)
        fn=fn[0:fn.rfind('.')]+'.png'
        h=data['images'][i]['height']
        w=data['images'][i]['width']
        mask=np.zeros((h,w),dtype='uint8')
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==data['images'][i]['id']:
                p1=data['annotations'][j]['segmentation'][0]

                p1=[int(i) for i in p1]
                p2=[]
                for p in range(int(len(p1)/2)):
                    p2.append([p1[2*p],p1[2*p+1]])
                fill_pts = np.array([p2], np.int32)
                cv2.fillPoly(mask, fill_pts, 255)
        cv2.imwrite(mask_dir+'/'+fn,mask)
