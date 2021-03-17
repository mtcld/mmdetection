# Inference Code 

<hr />

## Contents:

- [Inference Code](#inference-code)
  - [Contents:](#contents)
  - [Motivation:](#motivation)
  - [Prerequisites:](#prerequisites)
  - [Structure:](#structure)
  - [Explanation:](#Explanation)
  - [Authors:](#authors)

<hr/>

## Motivation:
<p>This code will help us infer the image and output the result. The code can be run in two ways. One way is
to create an API using mxnet-model-server and the another way is to just run the AI_api.py. The latter can 
be used for debugging purposes</p>

<hr />

## Prerequisites:

Create the docker from the given docker folder [for up to date docker code go to mtcld/ai-infrastructure]. Strictly follow the Dockerfile for requirements and essential configurations that need to be configured,
installed and copied.

<hr />

## Structure:

This is a sample of the Infrastructure Code that we will be using
```
mmdetection
|-------inference/
|       |-------configs/
|               |-------config.properties
|       |-------demo/
|               |-------view/
|                       |-------__init__.py
|                       |-------test_full_view.py
|       |-------external_api/
|               |-------case_vin.json
|               |-------vin.py
|       |-------image_3d/
|               |-------[All the images of 3d model]
|       |-------logs/
|               |-------access_log.log
|               |-------mms_log.log
|               |-------mms_metrics.log
|               |-------model_log.log
|               |-------model_metrics.log
|       |-------model/
|               |-------config.ini
|               |-------model_config.py
|       |-------pipeline/
|               |-------async_services.py
|               |-------mc_utils.py
|               |-------process_.py
|               |-------process_vin_claim.py
|       |-------three_d_car/
|               |-------capture_image_40290.png
|               |-------car_detection.py
|               |-------car3D.blend
|               |-------car3D.blend1
|               |-------carpart_detection.py
|               |-------color_segment_test.py
|               |-------color_segment.py
|               |-------projection3d.py
|               |-------side_util.py
|               |-------test_check_half_car.py
|               |-------test_full_view.py
|       |-------AI_api.py
|       |-------app.py
|       |-------README.md (This file that you are reading)
|       |-------test_api.py
|-------modelconfiguration/
|       |-------otf_message_handler.py
|       |-------start.sh
|-------demo/
|       |-------inference_demo.ipynb
|       |-------MotionsCloud-01.jpg
|       |-------MotionsCloud-06.jpg
```

<hr />

## Explanation:
- Only the code that should be changed incase the code has to be reused are these ones: <br/>
  1. pipeline/process_.py
  2. model/

- Test using *AI_api.py* <br />

- The code to change the steps are written here '*../demo/inference_demo.ipynb*' <br/>

- For debugging purposes, run *AI_api.py* and for demo purposes run *AI_api.py* through mxnet-model-server <br />

- For Inputs and Output of the API:
  - Input: UUID and VIN number <br/>
  - Output: [carpart: 'door', damage: [0.6, 0.5], confidence: ['dent', 'scratch'], treatment: "Replace", Side 1: N/A, Side 2: N/A] <br />

- For Input and Output for detectors, see '*../demo/inference_demo.ipynb*'


## Authors:
1. [**Sulabh Shrestha**](https://github.com/codexponent)
<hr />