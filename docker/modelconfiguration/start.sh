#!/bin/bash


cd /workspace/mmdetection/inference
model-archiver --model-name AI_api --model-path ./ --handler AI_api:handle --export-path ./

# tmux new -s modelcaller -d
# tmux send-keys './wait-for-it.sh -h localhost -p 8081 -t 0 -- curl -X POST "localhost:8081/models?url=AI_api.mar&batch_size=1&max_batch_delay=10&initial_workers=3"' C-m

mxnet-model-server --start --mms-config configs/config.properties --model-store ./

# tmux new -s modelcaller -d
# tmux send-keys './wait-for-it.sh -h localhost -p 8081 -t 0 -- curl -X POST "localhost:8081/models?url=AI_api.mar&batch_size=1&max_batch_delay=10&initial_workers=3"' C-m

# # Caller
# curl -X POST "localhost:8081/models?url=AI_api.mar&batch_size=1&max_batch_delay=10&initial_workers=1"

while :
do
  sleep 1
done

exec "/bin/bash";