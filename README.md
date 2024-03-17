# Shelf AI

THIS IS NOT AN OFFICIAL GOOGLE PRODUCT

This repository covers code required to deploy a Retail YOLO model using the
ultralytics framework under Torchserve onto Vertex AI to an autoscalable
endpoint

[Deploying Torchserve on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)

[Torchserve inference API](https://pytorch.org/serve/inference_api.html)

[Vertex AI - custom container requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements)

[YOLOv8 Retail](https://github.com/vmc-7645/YOLOv8-retail) - under GPL-3.0 License

## Requirements

### Deployment

- Python >3.11 with `google-cloud-aiplatform` installed
- `gcloud` tool installed with project configured and the Vertex AI and Cloud
  Build APIs enabled

### Inference

- The requirements listed in the `requirements.txt`
- Stuff like `curl`, `jq`, `base64` for sending requests using Bash scripts

### Development

- Stuff listed in Deployment and Inference
- `docker`, ideally with NVIDIA capabilities (`nvidia-docker2`)

## Contents

### Local utilities

Those are used in the local development phase, the final artifact is the custom
Torchserve container that is ready to be deployed on Vertex AI

- `mar.sh` generates a model archive file for torchserve
- `run-container.sh` runs a local deployment of the custom container
- `test-ts.sh` sends requests to local deployment of torchserve
- `Dockerfile` contains the manifest of the Torchserve container that runs the
  YOLOv8 model trained on the SKU110k dataset
- `handler.py` is the custom handler that overrides the default Torchserve `BaseHandler`

### Deployment utilities

- `cloud-build.sh` Bash script that submits a Cloud Build and returns the custom
  container image to be deployed onto Vertex AI
- `deploy.py` Python script that deploys the custom conatiner onto Vertex AI
  1. It uploads the model to the Vertex AI Model Registry
  2. It deploys a new Vertex AI Endpoint
  3. It deploys the uploaded model to the deployed Endpoint with an accelerator
    (NVIDIA T4 was used in this example)

### Inference scripts

Running the scripts require a deployed endpoint with the model at serving

Before running the scripts, ammend the `ENDPOINT_ID` and `PROJECT_ID` configuration variables

- `infer.py` is a Python script for running inference on images or videos using
  the deployed Vertex AI Endpoint and plotting the results. It uses OpenCV for
  running the videos and annotating the images; It requires updating the
  `image_paths` and optionally the `VIDEO_FOOTAGE` variable
- `infer.sh` is a Bash script for sending sample requests (single image, batch)
  to the deployed Vertex AI Endpoint

### Artifacts

- `retail-yolo.pt` is the YoloV8 finetuned on the Retail SKKU dataset
- `res.json` is a sample response from the deployed Vertex AI Endpoint
- `sample_request.json` is a sample request for running single image inference
- `sample_request_batch.json` is a sample request for running batch inference

## Flow

Be sure to:

- Ensure Python 3.11 is installed, install the required packages
- Ensure the Google Cloud APIs are enabled

### Development Flow

- `pytest` to check if the custom handler is all good, the `test_handler.py`
script checks the intermediate output in between `preprocess`, `inference` and
`postprocess` Torchserve overriden methods of the custom handler. It has flags
that can be ammended to save intermediate output into `.pkl` format so that it
can be inspected and debugged during development.
- `run-container.sh` to build the custom container and `test-ts.sh` to check if
requests go through; the `test-ts.sh` script checks for file upload requests,
single image requests and batch image requests

### Deployment Flow

Ensure the config params in `cloud-build.sh` and `deploy.py` scripts are correct, then run

```sh
./cloud-build.sh && python deploy.py
```

This takes roughly 10-15 minutes, results in an endpoint ready to serve traffic

Deployment can then be tested using `infer.sh` and `infer.py` scripts, see the
Vertex AI console to grab the `ENDPOINT_ID` and `PROJECT_ID` that are required
for inference

The Python `infer.py` script usage is

```sh
python infer.py --visualize --use_mask
```

for running on images in the `image_paths` list and

```sh
python infer.py --visualize --track
```

for running on Video footage

## Considerations

The Vertex AI API has a gRPC API that runs over HTTP/2 and supports streaming
binary objects

In order to achieve +99% detection rate, I would highly recommend fine-tuning
the YOLO model on a specialized dataset

It also makes sense to introduce additional classes: empty spots and improperly
placed product classes alongside the 'retail' product class

The model achieves good results but finetuning is not a complex task, with even
~100 data points in the fine-tuning dataset a major improvement for a specific
store should be achievable

There are also some cool features like Benchmarking or Batch Prediction
available through the Vertex AI console that can be useful 