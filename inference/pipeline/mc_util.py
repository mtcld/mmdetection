import json
import os
from pipeline.process_vin_claim import get_all_child_folder_name, damage_info_from_forder
from pipeline.process_ import detect_damage,detect_car_and_carpart
import requests
import cv2
import base64

def post_process(vin, results):
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "../", "external_api/case_vin.json")) as f:
        caseId_vin = json.load(f)

    child_folders = get_all_child_folder_name(results)
    out_put_list = {}
    out_dict = {}
    for child_folder in child_folders:
        out_dict = damage_info_from_forder(child_folder, results, caseId_vin, vin)
        out_put_list = {**out_put_list, **out_dict}

    return out_dict

def predict_car_and_carpart(image_path, output_folders):
    output = detect_car_and_carpart(image_path, output_folders)
    decoded_image = encode_image_2_base64(output["carpart_image"])
    return output,decoded_image

def predict_damage(image_path, carpart_output):
    id = image_path.split("/")[-2]
    final_output ,carpart_list= detect_damage(image_path, carpart_output["info_car"],carpart_output["m_car_view"])
    data = {id: {"img_path": image_path, "damage": final_output}}
    return data,carpart_list

def encode_image_2_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def get_all_file(folder_path):
    files = []
    for r, d, f in os.walk(folder_path):
        for file in f:
            if ('.jpg' in file.lower()) or ('.jpeg' in file.lower() or ('.png' in file.lower())):
                files.append(os.path.join(r, file))
    return files

def matching(img_path1,img_path2):
    url = "http://modelserver:8080/predictions/" + "mapping"
    files = {'data1': open(img_path1, 'rb'),'data2':open(img_path2, 'rb')}
    r = requests.post(url, files=files)
    if r.ok:
       j = r.json()
       print(j)

