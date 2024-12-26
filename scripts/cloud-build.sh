#!/bin/bash

CONTAINER_URI=europe-central2-docker.pkg.dev/vertex-ai-playground-402513/shelf-ai/retail-yolo:0.2.0

gcloud builds submit \
  --region=europe-central2 \
  --tag $CONTAINER_URI \
  .

