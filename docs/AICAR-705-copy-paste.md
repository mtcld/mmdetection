# Machine 1 
- get raw dataturk and scalabel json files from [here](https://drive.google.com/drive/folders/1jb13jq7RZihhCsoxL5ua9RCGxwjKPCu2)
- convert into coco json format (train,valid,test)
```
# access docker container
docker start mms2
docker exec -it mms2 /bin/bash
# go to convert code
cd share/ai-damage-detection/damage_detection/augmentation

# checkout to this branch
git checkout feat/AICAR-705-copy-paste-augment
```
- follow this [tutorial](https://github.com/mtcld/ai-damage-detection/tree/feat/AICAR-705-copy-paste-augment/damage_detection/augmentation#run-convertpy-to-convert-dataturk-and-scalabel-to-coco-format) to convert 

- Augment train sets (copy paste, random crop, rotation)
```
python augment_data.py /path/to/images/folder /path/to/train/json /path/to/new/train/json /path/to/car/part/data -r -cp -cr

# /path/to/car/part/data = ../../../maskrcnn-benchmark/predict_cp
```
# Machine 4 (using detectoRS backbone : ResNet50 HTC)
- Access docker container
```
docker start mmd
docker exec -it mmd /bin/bash
```
- Generate binary mask for segmentation brach of HTC 
```
python generate_binary_mask.py
```
- change `data` in `configs/detectors/crack_detector_latest_segm.py` with path to json,binary mask, images
- start training
```
./tools/dist_train.sh configs/detectors/crack_detector_latest_segm.py 2
``` 
- evaluate 
```
# evaluate ap bbox and segm 
python tools/test.py work_dirs/crack_cp_2/crack_detector_latest_segm.py work_dirs/crack_cp_2/epoch_9.pth --eval bbox segm
# evaluate damage level result in crack_confusion_matrix.json
python generating-cf-results-per-image.py
```
