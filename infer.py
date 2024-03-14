import cv2
import time
import argparse

from cv2.typing import MatLike
from google.cloud import aiplatform


def visualize(img, preds):
    for detection in preds:
        x1 = int(detection['box']['x1'])
        y1 = int(detection['box']['y1'])
        x2 = int(detection['box']['x2'])
        y2 = int(detection['box']['y2'])
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=(0, 255, 0), thickness=2)
        cv2.putText(img, detection['name'], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('detections', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def compress(img: MatLike, quality: int = 30):
    _, buffer = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    print("Compressed by a factor of", img.size / buffer.size)
    return buffer


def preprocess(img: MatLike):
    if img.size > 1_500_000:
        img = compress(img)
    return {"data": base64.b64encode(img.tobytes()).decode()}


def resize(img: MatLike) -> MatLike:
    w, h = img.shape[:2]
    return cv2.resize(img, (int(h/2), int(w/2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    ENDPOINT_ID = "2553457425835360256"
    PROJECT_ID = "352528412502"

    aiplatform.init(
        project=PROJECT_ID,
        location="europe-west4"
    )

    endpoint = aiplatform.Endpoint(
        f"projects/{PROJECT_ID}/locations/europe-west4/endpoints/{ENDPOINT_ID}"
    )

    image_paths = [
        "./IMG_0501.jpg",
        "./IMG_0502.jpg",
        "./IMG_0503.jpg",
        "./IMG_0504.jpg",
    ]
    import base64

    start_time = time.time()
    end_time = time.time()
    print("Time taken to preprocess: {}".format(end_time - start_time))
    images = [cv2.imread(image_path) for image_path in image_paths]
    images_resized = [resize(img) for img in images]
    instances = [preprocess(img) for img in images_resized]

    for i in range(len(instances)):
        start_time = time.time()
        res = endpoint.predict(instances=[instances[i]])
        end_time = time.time()
        print("Time taken for inference (round-trip): {}".format(end_time - start_time))
        print("Image Shape:", images_resized[i].shape)

        if args.visualize:
            visualize(images_resized[i], res.predictions[0])
