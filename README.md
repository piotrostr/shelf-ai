# Shelf AI

this repository covers code required to deploy a Retail YOLO model using the
ultralytics framework under Torchserve onto Vertex AI to an autoscalable
endpoint

https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai
https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements

https://github.com/vmc-7645/YOLOv8-retail - under GPL-3.0 License

## Contents

- `mar.sh`
  generates a model archive file for torchserve

- `handler.py`

- `test-ts.sh`

- `retail-yolo.pt`
  YoloV8 finetuned on the Retail SKKU dataset

## Requirements
