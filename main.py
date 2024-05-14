import cv2
from tqdm import tqdm
import logging
import os
import argparse

PROJECT_ID = "vertex-ai-playground-402513"


def get_image_paths() -> list[str]:
    image_paths = []
    # change the data path to gcs and works the same
    data_path = "./products"
    product_ids = os.listdir(data_path)
    for product in tqdm(product_ids):
        images = os.listdir(os.path.join(data_path, product))
        for image in images:
            image_paths.append(os.path.join(data_path, product, image))
    return image_paths


def search_example():
    from embedder import Embedder
    from embeddings_store import EmbeddingsStore

    img = cv2.imread("./data/salatka_example_2.jpeg")

    e = Embedder(PROJECT_ID)
    es = EmbeddingsStore()

    img_embeddings = e.embed(img, dimension=256)
    res = es.search(img_embeddings)
    print(res)


def batch_ingest():
    from embedder import Embedder
    from embeddings_store import EmbeddingsStore

    embedder = Embedder(PROJECT_ID)
    embeddings_store = EmbeddingsStore()
    image_paths = get_image_paths()
    logging.info(f"embedding and ingesting total of {len(image_paths)} images")
    oks = 0
    # spawn multiple threads to speed up the process
    # i am happy to wait 2 hours, weather is nice
    for image_path in tqdm(image_paths):
        if embeddings_store.exists(image_path):
            logging.info(f"Skipping {image_path}, already exists")
            continue
        embeddings = embedder.embed_path(image_path, dimension=256)
        ok = embeddings_store.ingest(
            image_path,
            embeddings,
            {"product_name": image_path},
        )
        oks += 1
        if not ok:
            logging.error(f"Failed to ingest {image_path}")
    logging.info(f"Batch ingest complete, got {oks} images ingested")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-ingest",
        action="store_true",
        help="Run batch ingest",
    )
    parser.add_argument(
        "--search-example",
        action="store_true",
        help="Run search example",
    )
    args = parser.parse_args()

    if args.batch_ingest:
        batch_ingest()
    if args.search_example:
        search_example()
