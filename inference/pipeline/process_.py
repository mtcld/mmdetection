import sys
import os
work_dir =os.getcwd()
# print("work_dir ******************",work_dir )
sys.path.append(work_dir)

import asyncio
import json
import pickle
import cv2
import numpy as np
import requests
import random
import time

from demo.view.test_full_view import Car_View
car_view = Car_View(os.path.join(work_dir,"image_3d/"))

from model import model_config
model_list = model_config.get_all_model()
damage_chars = model_config.get_damage_list()

def get_mask_from_demo(predictions, res):
    res_mask = res["mask"]
    pred_mask = np.asarray(res_mask)  # it works in pytorch tensor
    pred_mask = np.array(pred_mask)
    mask_act = np.zeros(predictions.shape[:2], dtype='uint8')
    return pred_mask, mask_act

async def API_call(img_path, model_name):
    '''
    This block is the only code who talks with the model_config[currently DetectoRS]
    
    Input requirement:
    img_path: The path of the image which will be detected 
    model_name: The name of the model which will be detecting the image

    Output requirement:
    [pred_image, {
            "labels": sample_labels_with_class_names, 
            "scores": sample_filtered_score, 
            "mask": sample_segms_ndarray
        }]
    '''
    sample_model = model_list[model_name]
    sample_json = sample_model.inference(img_path)
    return sample_json


def resize_img(cv2_img, max_side):
    ratio= cv2_img.shape[1]*1.0/cv2_img.shape[0]
    w= int(max_side*min(ratio,1))
    h= int(max_side*min(1.0/ratio,1))
    return cv2.resize(cv2_img, (w, h), interpolation=cv2.INTER_AREA)

async def main(img_path, loop):
    #damage_chars = ["scratch", "crack", "dent", "loose", "totaled"]  # 'totaled'
    json_contents = [loop.create_task(API_call(img_path, cat)) for cat in damage_chars]
    await asyncio.wait(json_contents)
    results = {cat: json_content for cat, json_content in zip(damage_chars, json_contents)}
    return results

async def predict_car_asyc(img_path, loop):
    categories = ["car", "carpart"]  # 'totaled'
    json_contents = [loop.create_task(API_call(img_path, cat)) for cat in categories]
    await asyncio.wait(json_contents)
    results = {cat: json_content for cat, json_content in zip(categories, json_contents)}
    return results

async def main_async(img_path, loop, damage_list):
    categories = damage_list # ["car", "carpart", "scratch", "crack", "dent", "loose", "totaled"]  # 'totaled'
    json_contents = [loop.create_task(API_call(img_path, cat)) for cat in categories]
    await asyncio.wait(json_contents)
    results = {cat: json_content for cat, json_content in zip(categories, json_contents)}
    return results

def final_damage_mask(json_content):
    damage_img = json_content.result()[0]
    damage_res = json_content.result()[1]
    pred_mask1, mask_act1 = get_mask_from_demo(damage_img, damage_res)
    return [damage_res, pred_mask1]

def get_predicted_car_masks(predict_car_asyc, img_path):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # loop = asyncio.new_event_loop()
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
    
        car_json_contents = loop.run_until_complete(predict_car_asyc(img_path, loop))
    finally:
        loop.close()

    if car_json_contents:
        _, car_res = car_json_contents['car'].result()[0], car_json_contents['car'].result()[1]
        car_masks = [mask for i, mask in enumerate(car_res["mask"]) if car_res["labels"][i] == 'car'
                     or car_res["labels"][i] == 'truck']
        num_of_predicted_carparts = len(car_json_contents['carpart'].result()[1]["labels"])
        return car_masks, num_of_predicted_carparts,car_json_contents
    else:
        return [], 0,car_json_contents


def rotate(img, degree=180):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = degree
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def is_eligible_damage_on_part(carpart, damage):
    # print(carpart, damage)
    if "grille" in carpart  and damage in ["scratch", "dent"]:
        return False
    if "light" in carpart and "bezel" not in carpart and "dent" in damage:
        return False
    return True


def add_proposed_totaled(m_car_view, final_output, segmented_carparts):
    totaled_damage_parts = []
    for part, damages in final_output.items():
        damages_label = set(map(lambda x: x[0], part))
        if "totaled" in damages_label:
            totaled_damage_parts.append(part)

    if totaled_damage_parts:
        return final_output
    

    view, parts_of_view = car_view.filter_carpart_by_view(m_car_view)
    if parts_of_view:
        proposed_totaled_missing_part=[part_of_view for part_of_view in parts_of_view if (part_of_view not in segmented_carparts)]
        #proposed_totaled_missing_part = set(parts_of_view) - set(segmented_carparts)
        for part in proposed_totaled_missing_part:
            if part not in final_output:
                final_output[part] = []
            final_output[part].append(('totaled', random.uniform(0.81, 0.85)))
    return final_output


def detect_car_and_carpart(img_path, save_img_folder):
    seg_img_folder = os.path.join(save_img_folder, img_path.rsplit(".", 1)[0].split("/")[-1])
    if not os.path.exists(seg_img_folder):
        os.makedirs(seg_img_folder, exist_ok=True)

    img = cv2.imread(img_path)
    # logfile.info("Image size h: %d - w: %d", img.shape[0], img.shape[1])
    if np.max(img.shape[0:2]) > 800:
        max_side=800
        img = resize_img(img,max_side)
        cv2.imwrite(img_path, img)
    

    ori_car_masks, num_of_predicted_ori_carparts,car_json_contents = get_predicted_car_masks(predict_car_asyc, img_path)
    
    if car_json_contents:
        for category, v in car_json_contents.items():
            each_result = v.result()[0]
            seg_img_path = os.path.join(seg_img_folder, category + ".jpg")
            cv2.imwrite(seg_img_path, each_result)
    
    real_carpart_center_list,real_bbox_list,lagest_car_area,masks_list_lagest_car=car_view.detect_maincar_carpart(car_json_contents)

    car_mask = car_view.lagest_car_mask * 255
    # get main car image

    img = cv2.bitwise_and(img, img, mask=car_mask)
    maincar_path=os.path.join(seg_img_folder, img_path.split("/")[-1])
    cv2.imwrite(maincar_path, img)

    # check  view car
    h,w=img.shape[0:2]
    m_car_view=car_view.check_view_car(real_bbox_list,lagest_car_area,w,h)
    carpart_image = car_json_contents['carpart'].result()[0]

    return {"carpart_image":carpart_image,\
            "main_car":maincar_path,\
            "m_car_view":m_car_view,\
            "info_car":{"seg_img_folder":seg_img_folder,"max_mask_car":car_view.lagest_car_mask,\
                        "real_carpart_center_list":real_carpart_center_list,"pred_mask_carpart":masks_list_lagest_car}}


def detect_damage(img_path,info_car,m_car_view,damage_list=None):
    real_carpart_center_list=info_car["real_carpart_center_list"]
    pred_mask_carpart = info_car["pred_mask_carpart"]
    max_mask_car = info_car["max_mask_car"]
    seg_img_folder=info_car["seg_img_folder"]
   
    has_max_mask_car=False
    if  len(np.unique(max_mask_car, return_counts=True)[1]) > 1:
        has_max_mask_car=True

    img = cv2.imread(img_path)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        json_contents = loop.run_until_complete(main(img_path, loop))
    finally:
        loop.close()
    
    
    if json_contents:
        for category, v in json_contents.items():
            each_result = v.result()[0]
            seg_img_path = os.path.join(seg_img_folder, category + ".jpg")
            cv2.imwrite(seg_img_path, each_result)

        final_output = {}
        damage_dict = {}

        for damage_char in damage_chars:
            damage_dict[damage_char] = final_damage_mask(json_contents[damage_char])

        filted_carparts=car_view.get_view_carpart_list(real_carpart_center_list)
        #print("filted_carparts", filted_carparts)
       
        for m in range(len(pred_mask_carpart)):

            cat=filted_carparts[m]
            #pred_part = np.squeeze(pred_mask_carpart[m])
            pred_part =pred_mask_carpart[m]

            part_damage_prop = None
            if  has_max_mask_car:
                part_damage_prop = cv2.bitwise_and(max_mask_car, pred_part)

            for dam in damage_chars:
                max_score = 0

                if not is_eligible_damage_on_part(cat, dam):
                    continue
                damage_res, pred_damage_mask1n = damage_dict[dam][0], damage_dict[dam][1]
                
                for kn in range(len(pred_damage_mask1n)):
                    pred_part1n = pred_damage_mask1n[kn]
                    #pred_part1n = np.squeeze(pred_part1n)
                    pred_part1n =pred_part1n
                
                    if part_damage_prop is None or part_damage_prop is not None and np.sum(part_damage_prop) == 0:
                        part_damage_prop = pred_part
                    
                    part_damage_prop = cv2.bitwise_and(part_damage_prop, part_damage_prop, mask=np.uint8(pred_part1n))            
                    if len(np.unique(part_damage_prop, return_counts=True)[1]) > 1:
                        a = np.unique(part_damage_prop, return_counts=True)[1][1]
                        b = np.unique(pred_part, return_counts=True)[1][1]
                        damage_pixel = a / b
                    else:
                        damage_pixel = 0

                    if damage_pixel > 0:
                        max_score = max(max_score, damage_res["scores"][kn])

                damage_score = (dam, max_score)
                #damage_score=dam
                if max_score > 0:
                    if cat not in final_output.keys():
                        final_output[cat] = []
                        final_output[cat].append(damage_score)
                    else:
                        final_output[cat].append(damage_score)
        rm_part_list=[carpart for carpart, value in final_output.items() if("tyre" in carpart or  "alloy_wheel" in carpart or 'windshield' in carpart or  'window' in carpart)]
        for rm_part in rm_part_list:
            if rm_part in final_output:
                del final_output[rm_part]

        return final_output,filted_carparts
