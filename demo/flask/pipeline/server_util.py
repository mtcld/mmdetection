import os
import cv2
import json
import boto3
import base64
import shutil
import logging
import numpy as np
# from app import get_car_from_vin, get_quote
from botocore.exceptions import ClientError

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def endcode_image_2_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        pass

def zip_folder_and_remove(folder_path, output_file):
    try:
        shutil.make_archive(folder_path, 'zip', folder_path)
        os.rename(folder_path + ".zip", output_file)
        shutil.rmtree(folder_path, ignore_errors=True)
    except:
        pass
    return output_file

def get_all_file(folder_path):
    files = []
    for r, d, f in os.walk(folder_path):
        for file in f:
            if ('.jpg' in file.lower()) or ('.jpeg' in file.lower() or ('.png' in file.lower())):
                files.append(os.path.join(r, file))
    return files

def upload_file_to_S3(file_name, bucket, object_name):
    '''Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    '''

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.Session(region_name='eu-central-1').client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name,ExtraArgs={'ACL': 'public-read'})
    except ClientError as e:
        logging.error(e)
        return None
    return "https://" + bucket + ".s3.amazonaws.com/" + object_name

def download_folder_from_S3(folder_path,bucket,save_folder):
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)
    prefix = folder_path
    for s3_object in my_bucket.objects.filter(Prefix=prefix):
        path, filename = os.path.split(s3_object.key)
        my_bucket.download_file(s3_object.key, save_folder + filename)

def trans_json_result(vin, request_id, results):
    final_result = {"make": "N/A", "model": "N/A", "pic_url": "/pic/1571068279556.JPEG", "quote": "\u20ac3076",
                    "vin": "undefined", "year": "N/A"}
    quote = 0
    print("vin", vin)

    final_result["items"] = []
    for result in results[request_id]:
        result = result['parts']

        for damage in result:
            tmp = {}
            carpart_side = damage['part'].split("+")
            quote += get_quote(carpart_side[0], damage['treatment'])
            if (len(carpart_side) > 1):
                tmp["carpart"] = carpart_side[0]
                if (len(carpart_side[1]) > 1):
                    tmp["Side 1"] = "Front" if carpart_side[1][1] == 'f' else "Rear"
                    tmp["Side 2"] = "Left" if carpart_side[1][0] == 'l' else "Right"
                else:
                    tmp["Side 1"] = "N/A"
                    tmp["Side 2"] = "Left" if carpart_side[1][0] == 'l' else "Right"
            else:
                tmp["carpart"] = damage['part']
                tmp["Side 1"] = "N/A"
                tmp["Side 2"] = "N/A"

            tmp["confidence"] = [round(confi['confidence'], 2) for confi in damage['damage']]
            tmp["damage"] = []
            tmp["damage"] =[type['type'].replace('totaled','severe') for type in damage['damage']]
            tmp["treatment"] = damage['treatment']
            final_result["items"].append(tmp)

    # if (vin == "undefined"):
    #     #     final_result["quote"] = "\u20ac0" + str(quote)
    fixed_car_lookup = get_car_from_vin(vin)

    final_result["year"] = fixed_car_lookup["year"]
    final_result["model"] = fixed_car_lookup["model"]
    final_result["make"] = fixed_car_lookup["make"]
    if final_result["make"] == "N/A":
        final_result["quote"] = "N/A"
    else:
        final_result["quote"] = "\u20ac" + str(quote)

    print("results", results)
    print("final_result", final_result)
    return final_result