"""
handler.py module overrides the BaseHandler class to define the custom handler
for the YOLO model. It follows the schema that can be found here:
https://pytorch.org/serve/custom_service.html
"""

import io
import os
import json
import base64

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
        """
        preprocess gets the Vertex AI formatted request and returns a list of
        PIL.Image objects

        this is suitable for the YOLO model used in this example via ultralytics

        expected payload is
        [
            {
                "body": {  # or "data" for uplod image reqs
                    "instances": [
                        {"data": "base64-encoded-image-utf-8" },
                        ...
                    ]
                }
            },
            ...
        ]
        provided that request sent is

        {
          "instances": [ { "data": "base64-encoded-image-utf-8" }, ... ]
        }
        """
        images = []

        for req in data:
            body = req.get("data") or req.get("body")
            # support binary files upload
            if isinstance(body, (bytes, bytearray)):
                image = Image.open(io.BytesIO(body))
                images.append(image)
                continue
            payload = body.get("instances")
            for item in payload:
                image = Image.open(io.BytesIO(base64.b64decode(item["data"])))
                images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        # model inference
        results = self.model(data, *args, **kwargs)
        return results

    def postprocess(self, data):
        # this only works for single 'instances' input
        return [{"predictions": [json.loads(result.tojson()) for result in data]}]


def is_nested(input_list: list):
    """
    Check if a list is nested
    """
    # Iterate through each item in the list
    for item in input_list:
        # Check if the current item is also a list
        if isinstance(item, list):
            # If any item is a list, return True (indicating it's a nested list)
            return True
    # If no items are lists, return False (indicating it's not a nested list)
    return False
