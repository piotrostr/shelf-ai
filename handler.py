from typing import Any
import json
import torch

from ultralytics import YOLO
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io


class YOLOHandler(BaseHandler):

    manifest: Any

    def initialize(self, context):
        # Initialize model and other artifacts
        if context:
            self.manifest = context.manifest
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

    def inference(self, data):
        # model inference
        results = self.model(data)
        return results

    def postprocess(self, data):
        return [json.loads(result.tojson()) for result in data]
