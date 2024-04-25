# WebSocket

The model here is exported to TensorRT and served via WebSocket rather than
Torchserve on Vertex AI and REST/gRPC

Exporting to TensorRT cuts down the inference time from ~50ms to ~10ms on Tesla
T4 GPU

This likely can be optimized further using NVIDIA V100 that has more Tensor
cores or a model that is slightly smaller, however the bottleneck is not the
inference, but the frames streaming

The Python client yields around 10 FPS, averaging 70-100ms per frame

Ping to the serving instance is 30-40ms (PL to europe-west4, NL)

## TensoRT

Setting it up is tricky, but a docker container simplifies the process a lot

Prerequisite: Install NVIDIA docker runtime on a machine that has CUDA drivers
and a GPU with Tensor cores, the GPU is used for optimizing the model,
quantizing, profiling etc all using a single `trtexec` call, but it has to run
on the same GPU as the inference later on (compute capability 7.5 for Tesla T4)

[CUDA Drivers Installation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
[NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Also have the ONNX model ready

With the prerequisites installed, the container can be run with the following command

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  -v $HOME/shelf-ai/ws:/home/shelf-ai \ # volume containing the ONNX model
  nvcr.io/nvidia/tensorrt:24.03-py3 -- bash
```

Then, to export the model run

```bash
cd /home/shelf-ai && \
  trtexec \
    --onnx=retail-yolo.onnx \
    --saveEngine=retail-yolo.engine \
    --best \
    --verbose
```

Ultralytics supports TensorRT engines out-of-the-box, once the model is
exported as an engine, it can be loaded and used with a sugared high-level API

```python
from ultralytics import YOLO

model = YOLO('retail-yolo.engine')
preds = model('sample_image.jpg')
```

## Deployment

See Dockerfile, the repository has two files that are used for inference,
high-level interface `Model` in `model.py` and `server.py` containing a simple
server with a single enpoint

There is also a `client.py` tool that can be used to test the server with
either video or sending single images

Install requirements with

```bash
pip install -r requirements.txt
```

And run locally with `./run.sh`

To run with Docker use `./run-docker.sh`, the server is exposed on port 3000,
container can be skimmed before prod, there is hanging Torch dependencies, as
mentioned in the root `README.md`, this is not a production architecture,
Triton Inference Server is required for running in prod

Docker container comes with the model included, but the `retail-yolo.engine`
has to match the compute capability of the GPU to be deployed at, so might
require re-exporting (takes around 10 minutes)

