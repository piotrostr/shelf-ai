import logging
import chromadb

from pydantic import BaseModel
from embedder import Embeddings


class EmbeddingsSearchResult(BaseModel):
    product_id: str
    product_name: str
    similarity: float


class EmbeddingsStore:
    def __init__(self, chroma_dump_path: str | None = None, use_clip=False):
        self.client = chromadb.PersistentClient(
            "./chroma_dump" if not chroma_dump_path else chroma_dump_path
        )
        self.collection = self.client.create_collection(
            "product-embeddings-store", get_or_create=True)
        logging.info("EmbeddingsStore loaded")

    def search(self, embeddings: Embeddings) -> list[EmbeddingsSearchResult]:
        if not embeddings.data:
            raise ValueError("Embeddings data is empty")
        res = self.collection.query(embeddings.data, n_results=6)
        resz = zip(res['ids'][0], res['metadatas'][0],  # type: ignore
                   res['distances'][0])  # type: ignore
        return [
            EmbeddingsSearchResult(
                product_id=product_id,
                product_name=product_id,
                similarity=1 - distance,
            )
            for product_id, _, distance in resz
        ]

    def ingest(self, product_id: str, embeddings: Embeddings, meta: dict[str, str]) -> bool:
        if not embeddings.data:
            return False
        self.collection.add(
            ids=[product_id],
            embeddings=[embeddings.data],
            metadatas=[meta],
        )
        return True

    def exists(self, product_id: str) -> bool:
        res = self.collection.get(product_id)
        return len(res['ids']) > 0
