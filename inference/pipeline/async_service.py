import asyncio
import requests
import pickle
import json
import cv2


async def API_call(img_path, model):
    # with open(pickle_file, 'rb') as handle:
    #     res = pickle.load(handle)
    # res = json.loads(res)
    # seg_img = str(pickle_file).replace("pickle", "jpg")
    # return [cv2.imread(seg_img), res]
    url = "http://modelserver:8080/predictions/" + model
    requests.post(url=url, data=img_path)
    await asyncio.sleep(0.0001)
    files = {'data': open(img_path, 'rb')}
    r = requests.post(url, files=files)  # , data=values)
    if r.ok:
        j = r.json()
    pickle_file = j['filepath']
    with open(pickle_file, 'rb') as handle:
        res = pickle.load(handle)
    res = json.loads(res)
    seg_img = str(pickle_file).replace("pickle", "jpg")
    return [cv2.imread(seg_img), res]

async def main_async(img_path, loop, damage_list):
    categories = damage_list # ["car", "carpart", "scratch", "crack", "dent", "loose", "totaled"]  # 'totaled'
    json_contents = [loop.create_task(API_call(img_path, cat)) for cat in categories]
    await asyncio.wait(json_contents)
    results = {cat: json_content for cat, json_content in zip(categories, json_contents)}
    return results

def predict(img_path, label_list):
    json_contents = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        loop.set_debug(1)
        json_contents = loop.run_until_complete(main_async(img_path, loop, label_list))
    finally:
        loop.close()
    return json_contents