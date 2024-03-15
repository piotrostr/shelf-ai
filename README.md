# Shelf AI

THIS IS NOT AN OFFICIAL GOOGLE PRODUCT

This repository covers code required to deploy a Retail YOLO model using the
ultralytics framework under Torchserve onto Vertex AI to an autoscalable
endpoint

[Deploying Torchserve on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)

[Torchserve inference API](https://pytorch.org/serve/inference_api.html)

[Vertex AI - custom container requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements)

[YOLOv8 Retail](https://github.com/vmc-7645/YOLOv8-retail) - under GPL-3.0 License

## Contents

* Local utilities
  * `mar.sh`
    generates a model archive file for torchserve
  * `run-container.sh`
    runs a local deployment of the custom container
  * `test-ts.sh`
    sends requests to local deployment of torchserve

* Inference
  * `infer.py`
  * `infer.sh`

* `handler.py`

* Artifacts
  * `retail-yolo.pt`
    YoloV8 finetuned on the Retail SKKU dataset
  * ``

## Requirements
