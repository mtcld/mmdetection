import os
import json
import mmcv
import logging
import configparser
import random, string
from flask import Flask, request
from utils import get_type_confidence
from base64 import encodebytes, decodebytes
from mmdet.apis import init_detector, inference_detector

app = Flask(__name__)
bbox_color = (72, 101, 241)
text_color = (72, 101, 241)
config = configparser.ConfigParser()
config.read('config.ini')
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename=config['LOGGER']['filename'], level=logging.DEBUG)

@app.route('/infer', methods = ['POST', 'GET'])
def infer():
    '''
    This API will infer the image coming from the json of POST method

    Input JSON:
    {
        "data": "image encoded"
    }

    Output JSON:
    {
    "car": [
        {
            "type": "car",
            "confidence": 0
        }
    ],
    "part": [
        {
            "type": "tyre",
            "confidence": 0
        },
        {
            "type": "alloy_wheel",
            "confidence": 0
        }        
    ],
    "dent": [
        {
            "type": "dent",
            "confidence": 0
        },
        {
            "type": "dent",
            "confidence": 0
        }
    ],
    "crack": [
        {
            "type": "crack",
            "confidence": 0
        }
    ],
    "scratch": [
        {
            "type": "scratch",
            "confidence": 0
        }
    ]
    }

    inference_detector takes model and original image and returns results in the form of an array which is used later to show results

    model.show_result takes the original image, results, bbox, text color and output file
    '''
    img_original_data = request.get_json()
    img_data = bytes(img_original_data['data'], 'utf-8')
    image_64_decode = decodebytes(img_data) 
    base_image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=int(20)))
    image_result = open(config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension'], 'wb')
    image_result.write(image_64_decode)
    base_folder_path = config['IMAGE']['inferencepath'] + base_image_name + '/'

    type_confidence_dict = {}

    logging.info('Inferring Car ...')
    car_model.show_result(config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension'], inference_detector(car_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), bbox_color=bbox_color, text_color=text_color, out_file= base_folder_path + 'car' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['car'] = get_type_confidence(inference_detector(car_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), car_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Car is Inferred')

    logging.info('Inferring Carpart ...')
    carpart_model.show_result(config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension'], inference_detector(carpart_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), bbox_color=bbox_color, text_color=text_color, out_file=base_folder_path + 'carpart' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['part'] = get_type_confidence(inference_detector(carpart_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), carpart_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Carpart is Inferred')

    logging.info('Inferring Dent ...')
    dent_model.show_result(config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension'], inference_detector(dent_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), bbox_color=bbox_color, text_color=text_color, out_file=base_folder_path + 'dent' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['dent'] = get_type_confidence(inference_detector(dent_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), dent_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Dent is Inferred')

    logging.info('Inferring Crack ...')
    crack_model.show_result(config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension'], inference_detector(crack_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), bbox_color=bbox_color, text_color=text_color, out_file=base_folder_path + 'crack' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['crack'] = get_type_confidence(inference_detector(crack_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), crack_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Crack is Inferred')

    logging.info('Inferring Scratch ...')
    scratch_model.show_result(config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension'], inference_detector(scratch_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), bbox_color=bbox_color, text_color=text_color, out_file=base_folder_path + 'scratch' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['scratch'] = get_type_confidence(inference_detector(scratch_model, config['IMAGE']['cachepath'] + base_image_name + config['IMAGE']['extension']), scratch_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Scratch is Inferred')

    logging.info('Image was Successfully Inferred')
    type_confidence_json = json.dumps(eval(str(type_confidence_dict)), indent = 4)
    return type_confidence_json

@app.route('/predictions', methods = ['POST', 'GET'])
def predictions():
    '''
    This API will mimic what was done for maskrcnn_benchmark models

    Inputs: 
    {
        'uuid': roomstr, 
        'vin': vin
    }

    Output String:
    {
    "car": [
        {
            "type": "car",
            "confidence": 0
        }
    ],
    "part": [
        {
            "type": "tyre",
            "confidence": 0
        },
        {
            "type": "alloy_wheel",
            "confidence": 0
        }        
    ],
    "dent": [
        {
            "type": "dent",
            "confidence": 0
        },
        {
            "type": "dent",
            "confidence": 0
        }
    ],
    "crack": [
        {
            "type": "crack",
            "confidence": 0
        }
    ],
    "scratch": [
        {
            "type": "scratch",
            "confidence": 0
        }
    ]
    }

    '''
    img_original_data = request.get_json()
    image_predict_path = config['IMAGE']['cachepath'] + img_original_data['uuid'] +'/'
    image_full_path = config['IMAGE']['cachepath'] + img_original_data['uuid'] +'/' + os.listdir(config['IMAGE']['cachepath'] + uuid)[0]

    type_confidence_dict = {}

    logging.info('Inferring Car ...')
    car_model.show_result(image_full_path, inference_detector(car_model, image_full_path), bbox_color=bbox_color, text_color=text_color, out_file= image_predict_path + 'car' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['car'] = get_type_confidence(inference_detector(car_model, image_full_path), car_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Car is Inferred')

    logging.info('Inferring Carpart ...')
    carpart_model.show_result(image_full_path, inference_detector(carpart_model, image_full_path), bbox_color=bbox_color, text_color=text_color, out_file=image_predict_path + 'carpart' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['part'] = get_type_confidence(inference_detector(carpart_model, image_full_path), carpart_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Carpart is Inferred')

    logging.info('Inferring Dent ...')
    dent_model.show_result(image_full_path, inference_detector(dent_model, image_full_path), bbox_color=bbox_color, text_color=text_color, out_file=image_predict_path + 'dent' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['dent'] = get_type_confidence(inference_detector(dent_model, image_full_path), dent_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Dent is Inferred')

    logging.info('Inferring Crack ...')
    crack_model.show_result(image_full_path, inference_detector(crack_model, image_full_path), bbox_color=bbox_color, text_color=text_color, out_file=image_predict_path + 'crack' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['crack'] = get_type_confidence(inference_detector(crack_model, image_full_path), crack_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Crack is Inferred')

    logging.info('Inferring Scratch ...')
    scratch_model.show_result(image_full_path, inference_detector(scratch_model, image_full_path), bbox_color=bbox_color, text_color=text_color, out_file=image_predict_path + 'scratch' + config['IMAGE']['extension'], score_thr=float(config['IMAGE']['threshold']))
    type_confidence_dict['scratch'] = get_type_confidence(inference_detector(scratch_model, image_full_path), scratch_model.CLASSES, float(config['IMAGE']['threshold']))
    logging.info('Scratch is Inferred')

    logging.info('Image was Successfully Inferred')
    type_confidence_json = json.dumps(eval(str(type_confidence_dict)), indent = 4)
    return type_confidence_json

if __name__ == '__main__':
    '''
    As soon as the API is setup, the models are being initialized

    init_detector takes configuration file, model file and the device where it should be infered
    '''
    logging.info('Building Car Model ...')
    car_model = init_detector(config['CAR']['car_config_file'], config['CAR']['car_checkpoint_file'], device=config['BUILD']['device'])
    logging.info('Car Model Built')

    logging.info('Building Carpart Model ...')
    carpart_model = init_detector(config['CARPART']['carpart_config_file'], config['CARPART']['carpart_checkpoint_file'], device=config['BUILD']['device'])
    logging.info('Carpart Model Built')

    logging.info('Building Dent Model ...')
    dent_model = init_detector(config['DENT']['dent_config_file'], config['DENT']['dent_checkpoint_file'], device=config['BUILD']['device'])
    logging.info('Dent Model Built')

    logging.info('Building Crack Model ...')
    crack_model = init_detector(config['CRACK']['crack_config_file'], config['CRACK']['crack_checkpoint_file'], device=config['BUILD']['device'])
    logging.info('Crack Model Built')

    logging.info('Building Scratch Model ...')
    scratch_model = init_detector(config['SCRATCH']['scratch_config_file'], config['SCRATCH']['scratch_checkpoint_file'], device=config['BUILD']['device'])
    logging.info('Scratch Model Built')

    app.run(host='0.0.0.0', port=3333)
