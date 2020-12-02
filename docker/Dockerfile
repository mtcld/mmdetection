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


RUN apt-get update && apt-get install -y curl zip wget
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin


COPY credentials.json /etc/
ENV GOOGLE_APPLICATION_CREDENTIALS="/etc/credentials.json"
RUN gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py

RUN pip install tensorboard future
RUN pip install pandas google-cloud-storage cloudml-hypertune

# Install MMCV
#RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN git clone https://github.com/gaurav67890/mmcv.git
RUN cd mmcv && MMCV_WITH_OPS=1 pip install -e . && cd ..

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/gaurav67890/mmdetection.git -b feat/AICAR-663-gcp-tuning-mmdetection /mmdetection
COPY scratch_latest /mmdetection/scratch_latest
COPY scratch_latest_mask /mmdetection/scratch_latest_mask
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

#ENTRYPOINT ["python","tools/trainer.py","configs/detectors/scratch_detector_latest_segm.py"]
