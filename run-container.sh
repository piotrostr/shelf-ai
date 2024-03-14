#!/bin/bash

docker build \
  -t piotrostr/shelf-ai-retail-yolo:0.1.0 \
  -f Dockerfile \
  .

# optional
# docker push piotrostr/shelf-ai-retail-yolo:0.1.0

docker run \
  --gpus all \
  --rm \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  -it piotrostr/shelf-ai-retail-yolo:0.1.0

