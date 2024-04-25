#!/bin/bash

docker run \
  --gpus all \
  -it \
  --rm \
  -v $HOME/shelf-ai/ws:/home/shelf-ai \
  nvcr.io/nvidia/tensorrt:24.03-py3

#  "cd /home/shelf-ai && trtexec --onnx=retail-yolo.onnx --saveEngine=retail-yolo.engine"
