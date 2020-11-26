# MMdetection  training



### 1.Preparing MMdetection docker
```
a) Use this docker file to build mmdetection docker- https://github.com/gaurav67890/mmdetection/blob/master/docker/Dockerfile
b) docker build -t mmdetection docker/
c) docker run --gpus all --shm-size=8g -p 8080:8080 -p 8081:8081 -it -v /datasets/:/mmdetection/data mmdetection

```
### 2. Installation of mmcv within the above docker
```
a) git clone https://github.com/gaurav67890/mmcv.git
b) MMCV_WITH_OPS=1 pip install -e .
```

### 3. Config file
```
a) Config file is where you will add the training,testing and validation images and annoation path and the classes.
b) Also you will update all the paramters within the config file.
c) One example of mask-rcnn config file with updated path of a dataset is here https://github.com/gaurav67890/mmdetection/blob/master/configs/mask_rcnn/dent_maskrcnn.py

```

### 4. Training in mmdetection
```
a) /tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
b) Example command for training /tools/dist_train.sh configs/mask-rcnn/dent_maskrcnn.py 2
c) All the log files and weights for the above command will be stored inside ./work_dirs/dent_maskrcnn/ if the work_dir is not mentioned inside the config file
else it will be save inside the work_dir mentioned inside the config file
```


### 5. Tensorboard
```
a) Tensorboard command for the above training
tensorboard --logdir=work_dirs/dent_maskrcnn/ --host 0.0.0.0 --port 8080

```

### 6. Testing/ Inference
```
a) ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--cfg-options]
b) Example testing command: ./tools/dist_test.sh configs/mask-rcnn/dent_maskrcnn.py  work_dirs/dent_maskrcnn/epoch_10.pth 2 -out result.pkl --eval bbox segm
```

