import cv2
from tqdm import tqdm
import logging
import os
import argparse

from embedder import Embedder
from ultralytics import YOLO

from embeddings_store import EmbeddingsStore
from recognizer import Recognition

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

    img = cv2.imread("./data/salatka_example_2.jpeg")

    e = Embedder(PROJECT_ID)
    es = EmbeddingsStore()

    img_embeddings = e.embed(img, dimension=256)
    res = es.search(img_embeddings)
    print(res)


def scene_frame_example():
    yolo = YOLO("./retail-yolo.pt")
    img = cv2.imread("./data/IMG_0504.jpg")

    e = Embedder(PROJECT_ID)
    es = EmbeddingsStore()

    res = yolo(img)
    boxes = res[0].boxes.xyxy
    import matplotlib.pyplot as plt
    # plt.imshow(res[0].plot())
    # plt.show()
    # plt.clf()
    for detection_id in range(len(boxes)):
        x1, y1, x2, y2 = boxes[detection_id]
        product = img[int(y1):int(y2), int(x1):int(x2)]
        embeddings = e.embed(product, dimension=256)
        results = es.search(embeddings)
        if not results:
            logging.warning("No results found for %s", detection_id)
            continue
        print(results)
        _, axs = plt.subplots(1, len(results) + 1)
        axs[0].imshow(cv2.cvtColor(product, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Product")
        i = 1
        for search_res in results:
            # hacky, product ID is also path
            predicted_product = cv2.imread(search_res.product_id)

            # show the predicted image vs product with mpl
            axs[i].imshow(cv2.cvtColor(predicted_product, cv2.COLOR_BGR2RGB))
            axs[i].set_title(f"Predicted: {search_res.similarity} sim")
            i += 1
        plt.show()


def batch_ingest():
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
    parser.add_argument(
        "--scene-frame-example",
        action="store_true",
        help="Run scene frame example",
    )
    args = parser.parse_args()

    if args.batch_ingest:
        batch_ingest()
    if args.search_example:
        search_example()
    if args.scene_frame_example:
        scene_frame_example()
