FROM nvcr.io/nvidia/pytorch:18.11-py3
WORKDIR /workspace
COPY pretrained_models pretrained_models
RUN cd pretrained_models && wget -c -i download.list
COPY . .
RUN pip install opencv-python