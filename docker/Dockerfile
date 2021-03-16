ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# # Custom Dockerfile
RUN apt update && apt upgrade -y
RUN apt install wget -y
RUN apt install curl -y
RUN apt install tmux -y
RUN pip install flask

RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

# # Celery stuffs
#RUN pip install celery==5.0.5
#RUN pip install gevent==20.9.0
#RUN pip install flask_socketio==4.3.2
#RUN pip install flask_cors==3.0.10
#RUN pip install boto3==1.17.12
#RUN pip install redis==3.5.3
#RUN apt-get install redis-server -y

RUN apt-get install default-jdk -y
RUN pip install flask_cors
RUN pip install mxnet-model-server


COPY modelconfiguration/otf_message_handler.py /opt/conda/lib/python3.7/site-packages/mms/protocol

# # New Build
COPY modelconfiguration/start.sh /
# COPY modelconfiguration/wait-for-it.sh /workspace/motionscloud/batch_mxnet/maskrcnn-benchmark

WORKDIR /
RUN chmod +x start.sh
ENTRYPOINT ["/bin/bash", "./start.sh"]

