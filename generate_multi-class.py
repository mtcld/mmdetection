import json
import cv2
import numpy as np
import pathlib
from tqdm import tqdm
import random
import sys
damage_name = 'carpart'
image_dir = 'data/' + damage_name + '/images/'
mask_dir = 'data/' + damage_name + '_mask'
#mask_dir='carparts_mask'
pathlib.Path(mask_dir).mkdir(parents=True, exist_ok=True)
modes = ['train','valid','test']

#palette=[]

#random.seed(0)
#for i in range(18):
#    r = random.randint(100,255)
#    g = random.randint(100,255)
#    b = random.randint(100,255)
#    if [b,g,r] not in palette:
#        palette.append([b,g,r])
#print(palette)

for mode in modes:
    p1 = 'data/' + damage_name + '/annotations/carpart_' + mode + '.json'
    #p1='test.json'
    with open(p1) as f:
        data = json.load(f)

    for i in tqdm(range(len(data['images']))):
        dict_act = {}
        dict_pred = {}
        fn = data['images'][i]['file_name']
        fn = fn[0:fn.rfind('.')] + '.png'
        #print(fn)
        h = data['images'][i]['height']
        w = data['images'][i]['width']
        mask = np.zeros((h, w), dtype='uint8')
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id'] == data['images'][i]['id']:
                categ_id=data['annotations'][j]['category_id']
                #print(categ_id)
                p1 = data['annotations'][j]['segmentation'][0]
                p1 = [int(i) for i in p1]
                p2 = []
                for p in range(int(len(p1) / 2)):
                    p2.append([p1[2 * p], p1[2 * p + 1]])
                #print(p2)
                fill_pts = np.array([p2], np.int32)
                cv2.fillPoly(mask, fill_pts, int(categ_id)+1)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(mask_dir + '/' + fn, mask)

classes=["sli_side_turn_light","tyre","alloy_wheel","hli_head_light","hood",
         "fwi_windshield","flp_front_license_plate","door","mirror","handle",
         "qpa_quarter_panel","fender","grille","fbu_front_bumper","rocker_panel",
         "rbu_rear_bumper","pillar","roof","blp_back_license_plate","window",
         "rwi_rear_windshield","tail_gate","tli_tail_light","fbe_fog_light_bezel",
         "fli_fog_light","fuel_tank_door","lli_low_bumper_tail_light"]
