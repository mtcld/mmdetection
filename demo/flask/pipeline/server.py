import sys
work_dir = "/workspace/mmdetection/inference/"
sys.path.append(work_dir)

import os
import cv2
import time
import requests
import configparser
from gevent import monkey
from celery import Celery
from pipeline.server_util import *
from celery.utils.log import get_task_logger
from flask_socketio import SocketIO, emit, join_room, rooms
from flask import Flask, jsonify, request, after_this_request

config = configparser.ConfigParser()
config.read('config.ini')

app = Flask(__name__)
monkey.patch_all(thread=True, select=True)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
socketio = SocketIO(app, message_queue='redis://localhost:6379/0', cors_allowed_origins='*', path='/socket.io/')
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
logger = get_task_logger(__name__)
bucket = "mc-core"

work_dir = "/workspace/"

@socketio.on('Finish_uploading')
def Finish_uploading(data):
    roomstr = str(request.sid)
    image_save_dir = "images/cache/" + roomstr + "/"
    output_dir = work_dir + image_save_dir + "output"
    vin = "None"
    if ('vin' in data):
        vin = data['vin']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    join_room(roomstr)
    process_image_folder.apply_async(args=[True, vin, image_save_dir, output_dir, roomstr], task_id=request.sid)


@socketio.on('Upload')
def Upload(data):
    image = data_uri_to_cv2_img(data['data'])
    image_name = data['fileName']
    image_save_dir = work_dir + "/images/cache" + str(request.sid) + "/"
    make_dir(image_save_dir)
    cv2.imwrite(image_save_dir + image_name, image)
    emit('Done', {})


@socketio.on('Start')
def Start(message):
    print("Start", message)
    startingRange = 0
    emit('MoreData', {'startingRange': startingRange, 'percent': 0})


@celery.task(bind=True)
def process_image_folder(self, mode_socket, vin, image_save_dir, output_dir, roomstr, callback=None, bucket='mc-core'):
    if mode_socket:
        socketio = SocketIO(message_queue='redis://')
    self.update_state(state='PROGRESS',
                      meta={
                          'status': 'upload_image',
                          'progress': 0.05
                      })

    files = get_all_file(work_dir + image_save_dir)
    url = "http://127.0.0.1:3333/predictions/"
    data = {'uuid': roomstr, 'vin': vin}
    r = requests.post(url, data=data)
    print('r')
    print(r)
    out_dict = {}
    if r.ok:
        out_dict = r.json()
        for i, file in enumerate(files):
            if mode_socket:
                image_path = os.path.join(output_dir, (file.split("/")[-1]).split('.')[0] + '/carpart.jpg')

                labeled_image = cv2.imread(image_path)
                labeled_image = endcode_image_2_base64(labeled_image)
                socketio.emit('labeled_image', str(labeled_image.decode('ascii')), room=roomstr)

    self.update_state(state='PROGRESS',
                      meta={
                          'status': 'predict_image',
                          'progress': 1

                      })

    final_result = out_dict['out_dict'][roomstr]
    final_result["requestID"] = roomstr

    if mode_socket:
        socketio.emit('result', final_result, room=roomstr)
    else:
        files = get_all_file(output_dir)
        for f in files:
            filename = f.split("/images/")[-1]
            if "carpart" in filename:
                s3_url = upload_file_to_S3(f, bucket, filename)

        requests.post(callback,
                      json={'payload': final_result, 'uuid': roomstr})

    # upload all results to S3
    output_zip = os.path.join(output_dir.replace('output', ''), roomstr + ".zip")
    filename = roomstr + ".zip"
    zip_folder_and_remove(output_dir, output_zip)
    is_fully_uploaded = upload_file_to_S3(output_zip, bucket, "demo/" + filename)
    if is_fully_uploaded:
        shutil.rmtree(output_zip, ignore_errors=True)

@socketio.on('status')
def taskstatus(data):
    task = process_image_folder.AsyncResult(task_id=request.sid)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info.get('status'),
            'progress': task.info.get('progress')
        }

    else:
        response = {
            'state': task.state,
            'status': str(task.info)  # this is the exception raised
        }

    emit('resuls_status', response)


@app.route('/app/', methods=['GET', 'POST'])
def receive_message():
    content = request.json

    uuid = content['uuid']
    vin = content['vin']
    callback = content['callback']

    bucket = content['bucket']
    print('bucket', content['bucket'])

    image_save_dir = 'images/' + uuid + "/"
    shutil.rmtree(image_save_dir, ignore_errors=True)
    make_dir(image_save_dir)

    download_folder_from_S3(uuid + '/input', bucket, image_save_dir)

    output_dir = work_dir + image_save_dir + "output"
    make_dir(output_dir)

    process_image_folder.apply_async(args=[False, vin, image_save_dir, output_dir, uuid, callback, bucket],
                                     task_id=uuid)
    return jsonify({"result": "ok"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', keyfile='ssl_key/key.pem', certfile='ssl_key/cert.pem')