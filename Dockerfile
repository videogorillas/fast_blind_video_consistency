FROM nvcr.io/nvidia/pytorch:18.11-py3
WORKDIR /workspace
COPY . .
RUN cd pretrained_models && wget -c -i download.list
RUN pip install opencv-python