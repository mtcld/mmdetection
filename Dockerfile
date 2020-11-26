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

#
COPY credentials.json /etc/
ENV GOOGLE_APPLICATION_CREDENTIALS="/etc/credentials.json"
RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py

RUN pip install tensorboard future


# Install MMCV
#RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN git clone https://github.com/gaurav67890/mmcv.git
RUN cd mmcv && MMCV_WITH_OPS=1 pip install -e . && cd ..

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/gaurav67890/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

ENTRYPOINT ['./tools/dist_train.sh,configs/mask-rcnn/dent_maskrcnn.py,4']
