"""
handler.py module overrides the BaseHandler class to define the custom handler
for the YOLO model. It follows the schema that can be found here:
https://pytorch.org/serve/custom_service.html
"""

import io
import os
import json

from typing import Any

import torch

from ultralytics import YOLO
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image


class YOLOHandler(BaseHandler):
    """
    Torchserve handler class for YOLO model using Ultralytics model
    """

    manifest: Any

    def initialize(self, context):
        # Initialize model and other artifacts
        if context:
            self.manifest = context.manifest
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            self.device = torch.device(
                "cuda:" + str(properties.get("gpu_id"))
                if torch.cuda.is_available()
                else "cpu"
            )
            serlialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serlialized_file)
            if not os.path.isfile(model_pt_path):
                raise RuntimeError(
                    "Missing the serialized file of the model. Path:",
                    model_pt_path,
                )
            self.model = YOLO(model_pt_path)
        else:  # running locally
            self.model = YOLO("./retail-yolo.pt")
            print("cuda: ", torch.cuda.is_available())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.model.device)
            print(self.model.device)

        self.initialized = True

    def preprocess(self, data):
        # Preprocess input data
        images = []

        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, (bytes, bytearray)):
                # if the image is a bytes object.
                image = Image.open(io.BytesIO(image))
            else:
                # if the input is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        # model inference
        results = self.model(data, *args, **kwargs)
        return results

    def postprocess(self, data):
        return [json.loads(result.tojson()) for result in data]
