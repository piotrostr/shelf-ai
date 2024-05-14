import logging
import cv2
import vertexai
import numpy as np

from pydantic import BaseModel

from vertexai.vision_models import Image, MultiModalEmbeddingModel


class EmbeddingResponse(BaseModel):
    text_embedding: list[float]
    image_embedding: list[float]


class Embeddings(BaseModel):
    data: list[float] | None


class Embedder:
    def __init__(
        self,
        project_id: str,
        location: str = "europe-west4",
    ):
        vertexai.init(project=project_id, location=location)

        self.model = MultiModalEmbeddingModel.from_pretrained(
            "multimodalembedding")

        logging.info("Embedder loaded")

    # a potential improvement would be to use both the text detected in the image
    # as well as the image itself to get a more accurate embedding
    def embed(self, img: np.ndarray, text: str | None = None, dimension: int = 1408) -> Embeddings:
        ok, image_bytes = cv2.imencode(".jpg", img)
        if not ok:
            raise ValueError("Failed to encode image")

        image = Image(image_bytes=image_bytes.tobytes())

        embeddings = self.model.get_embeddings(
            image=image,
            dimension=dimension
        )

        return Embeddings(
            data=embeddings.image_embedding
        )

    # this works for gcs too, pretty cool
    def embed_path(self, img_path: str, dimension: int = 1408) -> Embeddings:
        image = Image.load_from_file(img_path)

        embeddings = self.model.get_embeddings(
            image=image,
            dimension=dimension
        )

        return Embeddings(
            data=embeddings.image_embedding
        )
