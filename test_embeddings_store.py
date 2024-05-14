import numpy as np
from embeddings_store import EmbeddingsStore
from embedder import Embeddings


def test_embeddings_store_init():
    store = EmbeddingsStore()
    assert store is not None


def test_embeddings_store_ingest():
    store = EmbeddingsStore()
    ok = store.ingest("test", Embeddings(data=np.zeros(1408).tolist()), {
        "product_name": "test"})
    assert ok


def test_embeddings_store_search():
    store = EmbeddingsStore()
    embeddings = Embeddings(data=np.zeros(1408).tolist())
    results = store.search(embeddings)
    assert results is not None
