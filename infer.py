import base64
import cv2
import time
import argparse

from cv2.typing import MatLike
from google.cloud import aiplatform


ENDPOINT_ID = "2553457425835360256"  # Replace with your endpoint ID
PROJECT_ID = "352528412502"  # Replace with your project ID
FOOTAGE_PATH = "./data/video-footage.mp4"  # set to 0 to use webcam


def track(args, endpoint):
    capture = cv2.VideoCapture(FOOTAGE_PATH)
    while True:
        ret, frame = capture.read()
        for _ in range(2):
            ret, frame = capture.read()
        if not ret:
            print("Failed to capture image")
            return
        _frame = frame.copy()
        # _frame = resize(_frame)
        instance = preprocess(_frame)
        start_time = time.time()
        res = endpoint.predict(instances=[instance])
        end_time = time.time()
        print("Time taken for inference (round-trip): {:.2f}ms".format(
            (end_time - start_time) * 1000))
        print("Image Shape:", frame.shape)

        if args.visualize:
            visualize(_frame, res.predictions[0])

        if cv2.waitKey(22) & 0xFF == ord('q'):
            break

    capture.release()


def visualize(img, preds, use_mask=False):
    bounding_boxes = []
    for detection in preds:
        x1 = int(detection['box']['x1'])
        y1 = int(detection['box']['y1'])
        x2 = int(detection['box']['x2'])
        y2 = int(detection['box']['y2'])
        bounding_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=(0, 255, 0), thickness=2)
        cv2.putText(img, detection['name'], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if use_mask:
        import numpy as np
        mask = np.zeros(img.shape[:2], dtype="uint8")
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        mask_inv = cv2.bitwise_not(mask)
        blurred = cv2.GaussianBlur(img, (21, 21), 0)  # Adjust blur strength
        result = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
        result = cv2.add(result, img, mask=mask)

        combined_image = np.hstack([img, result])
        cv2.imshow('detections', combined_image)
        return
    cv2.imshow('detections', img)


def compress(img: MatLike, quality: int = 40):
    _, buffer = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    print("Compressed by a factor of", img.size / buffer.size)
    return buffer


def preprocess(img: MatLike):
    """
    preprocess compresses the images if they are larger than 1.5MB 
    and returns a dictionary with the base64 encoded image

    Args:
        img (MatLike): cv2 image

    Returns:
        dict: dictionary in endpoint-suitable format 
    """
    if img.size > 1_500_000:
        img = compress(img)
    return {"data": base64.b64encode(img.tobytes()).decode()}


def resize(img: MatLike) -> MatLike:
    w, h = img.shape[:2]
    return cv2.resize(img, (int(h/2), int(w/2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--use_mask", action="store_true")
    args = parser.parse_args()

    aiplatform.init(
        project=PROJECT_ID,
        location="europe-west4"
    )

    endpoint = aiplatform.Endpoint(
        f"projects/{PROJECT_ID}/locations/europe-west4/endpoints/{ENDPOINT_ID}"
    )

    if args.track:
        track(args, endpoint)
        exit(0)

    image_paths = [
        "./data/IMG_0501.jpg",
        "./data/IMG_0502.jpg",
        "./data/IMG_0503.jpg",
        "./data/IMG_0504.jpg",
        "./data/IMG_0508.jpg",
        "./data/IMG_0509.jpg",
        "./data/IMG_0510.jpg",
        "./data/IMG_0511.jpg",
    ]

    start_time = time.time()
    images = [cv2.imread(image_path) for image_path in image_paths]
    # resize the images to half the size to reduce request size
    # the images in image_paths are 5-6MB each, request size is 1.5MB
    images_resized = [resize(img) for img in images]
    instances = [preprocess(img) for img in images_resized]
    end_time = time.time()
    print("Time taken to preprocess: {:.2f}ms".format(
        (end_time - start_time) * 1000))

    for i in range(len(instances)):
        start_time = time.time()
        res = endpoint.predict(instances=[instances[i]])
        end_time = time.time()
        print(
            "Time taken for inference (round-trip): {:.2f}ms".format((end_time - start_time) * 1000))
        print("Image Shape:", images_resized[i].shape)

        if args.visualize:
            visualize(images_resized[i],
                      res.predictions[0], use_mask=args.use_mask)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
