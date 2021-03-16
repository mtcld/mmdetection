
import os
import time
import io
from io import BytesIO
from pipeline.mc_util import *

work_dir="/workspace/images/test"

def get_result_pipline(uuid,vin):
    files = get_all_file(os.path.join(work_dir,"images/"+ uuid))
    output_dir=os.path.join(work_dir,"images/"+ uuid + "/output")
    carpart_output_list=[]

    for i, file in enumerate(files):
        carpart_output, labeled_image = predict_car_and_carpart(file, output_dir)
        carpart_output_list.append(carpart_output)

    damage_output = []
    for i, file in enumerate(files):
        data,_ = predict_damage(file,carpart_output_list[i])
        damage_output.append(data)
    
    print('giving the dict to postprocess')
    print(damage_output)
    out_dict = post_process(vin, damage_output)
    print('returning out_dict')
    print(out_dict)
    return [{"out_dict":out_dict}]

def preprocess(data):
    uuid= data[0]["uuid"]
    vin= data[0]["vin"]
    return uuid.decode(),vin.decode()

def handle(data, context):
    if data is None:
        return None
    uuid,vin=preprocess(data)

    return get_result_pipline(uuid,vin)


get_result_pipline('1abc', 'None')