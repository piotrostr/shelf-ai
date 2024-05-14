import logging
import numpy as np

from pydantic import BaseModel

from embedder import Embedder
from embeddings_store import EmbeddingsStore


class Recognition(BaseModel):
    detection_id: str
    similarity: float
    product_id: str
    meta: dict[str, str]


class Box(BaseModel):
    y2: float
    x1: float
    x2: float
    y1: float


class Detection(BaseModel):
    id: str
    confidence: float
    box: Box


class RecognizeRequest(BaseModel):
    image: np.ndarray
    detections: list[Detection]


class RecognizeResponse(BaseModel):
    results: list[Recognition]


class Recognizer:
    def __init__(self, embedder: Embedder, embeddings_store: EmbeddingsStore):
        self.embedder = embedder
        self.embeddings_store = embeddings_store

    def recognize(self, request: RecognizeRequest) -> RecognizeResponse:
        response = RecognizeResponse(results=[])
        image = request.image
        detections = request.detections
        for detection in detections:
            box = detection.box
            product_image = image[box.y1:box.y2, box.x1:box.x2]
            product_embeddings = self.embedder.embed(product_image)
            search_results = self.embeddings_store.search(product_embeddings)
            if not search_results:
                logging.info(f"No results found for detection {detection.id}")
                continue
            result = search_results[0]
            response.results.append(Recognition(
                detection_id=detection.id,
                product_id=result.product_id,
                similarity=result.similarity,
                meta={
                    "product_name": result.product_name
                },
            ))
        return response
