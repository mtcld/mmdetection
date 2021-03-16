import base64
import cv2
import io
import numpy as np
import os
import requests
from PIL import Image
from flask import Flask
from flask import redirect, send_from_directory
from flask import request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from random import randint, uniform
#from pipeline import process_
import urllib.parse
import uuid

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

app = Flask(__name__, static_url_path='', static_folder='ClientApp/dist')

BASE_URL = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_URL, 'upload')

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
CAR_CATEGORIES = [
    "__background", 'fli_fog_light', 'sli_side_turn_light', 'mirror', 'fbu_front_bumper', 'bpn_bed_panel', 'grille',
    'tail_gate', 'mpa_mid_panel', 'fender', 'hli_head_light', 'car', 'rbu_rear_bumper', 'door',
    'lli_low_bumper_tail_light', 'hood', 'hpa_header_panel', 'trunk', 'tyre', 'alloy_wheel',
    'hsl_high_mount_stop_light', 'rocker_panel', 'qpa_quarter_panel', 'rpa_rear_panel', 'rdo_rear_door',
    'tli_tail_light', 'fbe_fog_light_bezel'
]

PRICES = {
    "fbu_front_bumper": [1000, 2000, 250, 600],
    "grille": [1000, 2000, 250, 600],
    "hood": [800, 2000, 500, 800],
    "fender": [500, 1000, 200, 500],
    "hli_head_light": [250, 700, 100, 200],
    "tli_tail_light": [250, 700, 100, 200],
    "fli_fog_light": [100, 160, 50, 100],
    "fbe_fog_light_bezel": [20, 30, 20, 30],
    "sli_side_turn_light": [30, 60, 30, 60],
    "hsl_high_mount_stop_light": [120, 160, 120, 160],
    "door": [1000, 2500, 500, 800],
    "rocker_panel": [800, 1200, 300, 600],
    "qpa_quarter_panel": [600, 1200, 200, 500],
    "rbu_rear_bumper": [1000, 2000, 250, 600],
    "trunk": [1000, 2000, 250, 600],
    "tail_gate": [1000, 2000, 250, 600]
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class UnsupportedMediaType(Exception):
    status_code = 415

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(UnsupportedMediaType)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        print("===============================================")
        print(request.form.get('vin'))
        vin = request.form.get('vin')
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return object_detect(filename, vin)
        else:
            raise UnsupportedMediaType('This view is gone', status_code=415)

@app.route('/process', methods=['POST', 'GET'])
def upload_file2():
    content = request.json
    vin = content.get('vin')
    img_url = content.get('img_url')
    if vin and img_url:
        saved_path = app.config['UPLOAD_FOLDER']
        filename = download(img_url, saved_path)
        if filename is None:
            raise UnsupportedMediaType('Not able to download the file', status_code=415)
        if allowed_file(filename):
            return object_detect(filename, vin, True)
        else:
            raise UnsupportedMediaType('Unsupport media type. Only picture format is allowed', status_code=415)
    else:
        raise UnsupportedMediaType('Missing vin or img_url', status_code=415)

def download(url, saved_path):
    r = requests.get(url)
    if not r.ok:
        return None
    filename = secure_filename(get_filename(url))
    with open(os.path.join(saved_path, filename), 'wb') as f:
        f.write(r.content)
    return filename

def get_filename(url):
    try:
        main_url = url.split("?")[0]
        encoded_filename = main_url.split("/")[-1]
        return urllib.parse.unquote(encoded_filename)
    except:
        filename = uuid.uuid4().hex
        lower_url = str(url).lower()
        if ".jpg" in lower_url:
            return filename + ".jpg"
        elif ".png" in lower_url:
            return filename + ".png"
        elif ".jpeg" in lower_url:
            return filename + ".jpeg"
        else:
            return filename + ".unknown"


def get_treatment(damage_list): # todo : if area of damage/area of part > xxx %: replace
    treatment = "Repair"
    if "crack" in damage_list or "totaled" in damage_list:
        treatment = "Replace"
    if len(damage_list) > 1 and "dent" in damage_list:
        treatment = "Replace"
    return treatment


def get_quote(car_part, treatment):
    if car_part in PRICES:
        price_range = PRICES[car_part]
        if treatment == "Replace":
            return randint(price_range[0], price_range[1])
        else:
            return randint(price_range[2], price_range[3])
    return 0


def get_part_name(car_part):
    token = str(car_part).split("_")
    if len(token) == 1 or len(token[0]) > 3:
        return " ".join(token)
    else:
        return " ".join(token[1:])


def get_max_score(values):
    max_value = max(values)
    max_add = min(0.05, 0.9999 - max_value)
    return max_value + uniform(-1 * max_add, max_add)


def object_detect(filename, vin, api=False):
    try:
        quote_response = get_car_from_vin(vin)
    except:
        quote_response = None
    # filename, damage_part, score = predict(filename)
    seg_filename, predicted = predict(filename)
    print("-----------------------")

    if seg_filename is not False:
        result = {}
        result["items"] = []
        # result['quote'] = '€3470'
        quote = 0
        # <!-- metaData.items = [{ car_part: 'Front bumper', treatment: 'Repair', damage: 'Scratch, Dent', confidence: '96%, 86%' }, {...}, {...}] -->
        if quote_response is not None:
            if predicted:
                for car_part, damages in predicted.items():
                    damages = dict(damages)
                    # result['quote'] = '€3470'  # + str(round(float(quote_response['quote']), 2))
                    temp = {}
                    temp["carpart"] = get_part_name(car_part)
                    temp["damage"] = ",".join(damages.keys())
                    temp["confidence"] = ", ".join([str(round(d * 100, 2)) + "%" for d in damages.values()])
                    # str(round(get_max_score(damages.values()) * 100, 2)) + "%"
                    temp["treatment"] = get_treatment(damages.keys())
                    quote += get_quote(car_part, temp['treatment'])
                    result['items'].append(temp)
                if vin == "WDDEJ9EB3CA029143":
                    quote = quote * 2
                result['quote'] = "€" + str(quote)
                result['damage_type'] = str(result['items']) 
            else:
                result['quote'] = "-"
                result['damage_type'] = "-"  # "No damage is detected !"
            result['make'] = quote_response['make']
            result['model'] = quote_response['model']
            result['year'] = quote_response['year']
        else:
            result['make'] = '-'
            result['model'] = '-'
            result['year'] = '-'
            result['quote'] = '-'
        result['vin'] = vin
        url_prefix = "/pic/"
        if api:
            url_prefix = "http://aicar.motionscloud.com/pic/"
        result['pic_url'] = url_prefix + seg_filename
        return jsonify(result)


def get_car_from_vin(vin):
    vin_dict = {"1G1RB6S50HU109665": ["Chevrolet", "Volt", "4", "2017", "523.98"],
                "1GCRKSE79CZ350083": ["Chevrolet", "Silverado", "2", "2013", "269.58"],
                "1MEFM50245A607080": ["Mercury", "Sable", "4", "2007", "383.3"],
                "1N4AL3AP9GC232765": ["Nissan", "Altima", "4", "2016", "361.78"],
                "3FAHP0HA1BR285229": ["Ford", "Fusion", "4", "2011", "347.76"],
                "3G5DA03L17S560494": ["Buick", "Rendezvous", "4", "2007", "624.57"],
                "JS2GB41W715206316": ["Suzuki", "Esteem", "4", "2007", "306.18"],
                "JTHBL5EFXD5122382": ["Lexus", "LS", "4", "2013", "612.31"],
                "KMHCM3AC3BU191206": ["Hyundai", "Accent", "2", "2011", "303.36"],
                "WAUBFAFL8GN010087": ["Audi", "A4/S4", "4", "2016", "376.41"],
                "4T1BF1FK3EU354323": ["Toyota", "Camry", "4", "2014", "305.7"],
                "JM1FE1RP7B0405079": ["Mazda", "RX8", "2", "2011", "421.83"],
                "1HGCS12889A000191": ["Honda", "ACCORD EXL", "4", "2009", "318.67"],
                "5TDBK3EH1CS147593": ["Toyota", "Highlander", "4", "2012", "407.12"],
                "1N4AL2EP7BC132970": ["Nissan", "Altima S", "2", "2011", "309.82"],
                "5GAKRDKD2DJ256034": ["Buick", "Enclave", "4", "2013", "299.73"],
                "1GNSCBKC3FR743654": ["Chevrolet", "Tahoe", "4", "2015", "385.05"],
                "WBAFR1C53BC751282": ["BMW", "528i", "4", "2011", "386.45"],
                "SCFFDAEM5FGA16330": ["Aston Martin", "DB9", "2", "2015", "386.67"],
                "1G1PL5SC8C7242565": ["Chevrolet", "Cruze", "4", "2012", "376.48"],

                "3VW5T7AU4GM060178": ["Volkswagen", "GTI", "4", "2016", "376.48"],  # fake
                "JTHCF1D28F5024523": ["Lexus", "IS 250", "4", "2015", "376.48"],
                "WDDEJ7EB4CA029351": ["Mercedes-Ben", "CL-Class", "2", "2012", "376.48"],
                "1HGCP3F81CA004999": ["Honda", "Accord", "4", "2012", "376.48"],
                "JF1ZNAA15F8701610": ["Toyota", "Scion FR - S", "2", "2015", "376.48"],
                "WBS3C9C52FP805345": ["BMW", "M3 Coupe", "4", "2013", "376.48"],
                "WDDEJ9EB3CA029143": ["Mercedes-Benz", "CL - Class", "2", "2012", "376.48"]  # x2
                }
    result = {}
    if vin in vin_dict:
        [make, model, style, year, quote] = vin_dict[vin]
        result = {
            "year": year,
            "make": make,
            "model": model,
            "quote": quote
        }
    else:
        result = {"year": "N/A",
                  "make": "N/A",
                  "model": "N/A",
                  "quote": "N/A"}

    return result


def get_damage_type(predicted):
    if predicted is None:
        return "No damage is detected !"
    result = []
    if 'class_ids' in predicted and 'class_names' in predicted:
        for count, i in enumerate(predicted['class_ids']):
            if predicted['class_names'][i] == 'gone':
                return "Bumper Heavily Damage"
            else:
                # todo : hard code
                if predicted['class_names'][i] != 'dent' or predicted['scores'][count] > 0.94:
                    result.append("Bumper " + predicted['class_names'][i])
    print("#############", result)
    if len(result) > 0:
        return ", ".join(set(result))
    else:
        return "No damage is detected !"


def is_damaged(predicted, damage_type):
    if 'class_ids' in predicted and 'class_names' in predicted:
        for i in predicted['class_ids']:
            if predicted['class_names'][i] == damage_type:
                return True
    return False


def convert_str_to_image(image_64_encode):
    image_io = io.BytesIO()
    image_io.write(base64.b64decode(image_64_encode))
    image = Image.open(image_io)
    return np.array(image)


def convert_img_to_base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    encoded_img = base64.b64encode(buffer)
    return encoded_img.decode('utf-8')


def predict(filename):
    # img = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
    seg_car_img, predicted = process_.detect_damage(os.path.join(UPLOAD_FOLDER, filename), UPLOAD_FOLDER)
    output_img = "result_" + filename
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, output_img), seg_car_img)
    # output_img = seg_car_img
    return output_img, predicted


@app.route('/pic/<path:filename>')
def send_pic(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def static_proxy(path):
    # send_static_file will guess the correct MIME type
    return app.send_static_file(path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500)
