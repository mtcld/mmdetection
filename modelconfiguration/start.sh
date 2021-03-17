#!/bin/bash

cd /workspace/mmdetection/inference
model-archiver --model-name AI_api --model-path ./ --handler AI_api:handle --export-path ./
mxnet-model-server --start --mms-config configs/config.properties --model-store ./

# # Caller
# curl -X POST "localhost:8081/models?url=AI_api.mar&batch_size=1&max_batch_delay=10&initial_workers=1"

while :
do
  sleep 1
done

exec "/bin/bash";