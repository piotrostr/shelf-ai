#!/bin/bash

IMAGE_TAG=piotrostr/shelf-ai-retail-yolo:0.1.0

docker build \
  -t $IMAGE_TAG \
  -f Dockerfile \
  .

# optional
# docker push $IMAGE_TAG

docker run \
  --gpus all \
  --rm \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  -it $IMAGE_TAG

