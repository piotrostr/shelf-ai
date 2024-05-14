import logging
import chromadb

from pydantic import BaseModel
from embedder import Embeddings


class EmbeddingsSearchResult(BaseModel):
    product_id: str
    product_name: str
    similarity: float


class EmbeddingsStore:
    def __init__(self):
        self.client = chromadb.PersistentClient("./chroma_dump")
        self.collection = self.client.create_collection(
            "product-embeddings-store", get_or_create=True)
        logging.info("EmbeddingsStore loaded")

    def search(self, embeddings: Embeddings) -> list[EmbeddingsSearchResult]:
        if not embeddings.data:
            raise ValueError("Embeddings data is empty")
        res = self.collection.query(embeddings.data, n_results=1)
        resz = zip(res['ids'], res['metadatas'],  # type: ignore
                   res['distances'])  # type: ignore
        print(res)
        return [
            EmbeddingsSearchResult(
                product_id=product_id[0],
                product_name=meta[0]['product_name'],
                similarity=1 - distance[0],
            )
            for product_id, meta, distance in resz
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
