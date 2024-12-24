import logging
from typing import Optional
import cv2
import vertexai
import numpy as np

from pydantic import BaseModel
from PIL import Image as PILImage

from vertexai.vision_models import Image, MultiModalEmbeddingModel


class EmbeddingResponse(BaseModel):
    text_embedding: list[float]
    image_embedding: list[float]


class Embeddings(BaseModel):
    data: Optional[list[float]]


class CLIPEmbedder:
    def __init__(self):
        import clip
        import torch

        self.model, self.preprocess = clip.load("ViT-B/32")
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.warn(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                logging.warn(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )

        else:
            logging.info("Using MPS (OSX accelerator)")
            self.device = torch.device("mps")
            self.model = self.model.to(self.device)

        if not self.device:
            self.device = torch.device("cpu")

    def embed(self, img: np.ndarray, text: Optional[str] = None) -> Embeddings:
        image = self.preprocess(PILImage.fromarray(img)).unsqueeze(0).to(self.device)  # type: ignore
        image_features = self.model.encode_image(image).cpu()
        return Embeddings(data=image_features.detach().numpy().tolist()[0])

    def embed_path(self, img_path: str) -> Embeddings:
        try:
            image = (
                self.preprocess(PILImage.open(img_path)).unsqueeze(0).to(self.device)
            )  # type: ignore
        except:
            logging.warning("Failed to open image", img_path)
            return Embeddings(data=None)

        image_features = self.model.encode_image(image).cpu()
        return Embeddings(data=image_features.detach().numpy().tolist()[0])


class Embedder:
    def __init__(
        self,
        project_id: str,
        location: str = "europe-west4",
    ):
        vertexai.init(project=project_id, location=location)

        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

        logging.info("Embedder loaded")

    # a potential improvement would be to use both the text detected in the image
    # as well as the image itself to get a more accurate embedding
    def embed(
        self, img: np.ndarray, text: Optional[str] = None, dimension: int = 1408
    ) -> Embeddings:
        ok, image_bytes = cv2.imencode(".jpg", img)
        if not ok:
            raise ValueError("Failed to encode image")

        image = Image(image_bytes=image_bytes.tobytes())

        embeddings = self.model.get_embeddings(image=image, dimension=dimension)

        return Embeddings(data=embeddings.image_embedding)

    # this works for gcs too, pretty cool
    def embed_path(self, img_path: str, dimension: int = 1408) -> Embeddings:
        image = Image.load_from_file(img_path)

        embeddings = self.model.get_embeddings(image=image, dimension=dimension)

        return Embeddings(data=embeddings.image_embedding)
