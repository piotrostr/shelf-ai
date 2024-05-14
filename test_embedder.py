import cv2

from embedder import Embedder


PROJECT_ID = "vertex-ai-playground-402513"


def test_embedder_init():
    embedder = Embedder(PROJECT_ID)
    assert embedder is not None


def test_embedder_embed_image():
    embedder = Embedder(PROJECT_ID)
    sample_img = cv2.imread("./sample_image.jpg")
    embeddings = embedder.embed(sample_img, dimension=1408)
    assert embeddings.data
    assert len(embeddings.data) == 1408
