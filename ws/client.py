import json
import time
import asyncio
import cv2
import websockets
import argparse

from server import compress, encode

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

def get_frames(video_path: str):
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    while ret:
        frame = cv2.resize(frame, (640, 360))
        yield frame
        ret, frame = capture.read()

async def infer_with_video(url: str, video_path: str):
    async with websockets.connect(url) as ws:
        i = 0
        full_start = time.time()
        for frame in get_frames(video_path):
            buffer = encode(frame)
            payload = compress(buffer)
            start = time.time()
            await ws.send(payload)
            res = await ws.recv()
            # format as raw string, including \n for me to see
            res = res[1:-1]
            try:
                visualize(frame, json.loads(res, strict=False))
            except Exception as e:
                print(e)
            end = time.time()
            i += 1
            print(f"FPS: {i / (end - full_start)}")
            print(f"Time taken this frame: {1000 * (end - start)}ms")

            if cv2.waitKey(22) & 0xFF == ord('q'):
                break

async def infer_with_saved_payload(url: str, image_path: str):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    buffer = encode(img)
    payload = compress(buffer)
    async with websockets.connect(url) as ws:
        for _ in range(100):
            start = time.time()
            await ws.send(payload)
            _ = await ws.recv()
            end = time.time()
            print(f"Time taken (ms): {1000 * (end - start)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../sample_image.jpg")
    parser.add_argument("--video", type=str)
    parser.add_argument("--url", type=str, default="ws://localhost:3000/ws")
    args = parser.parse_args()

    if args.video:
        asyncio.run(infer_with_video(args.url, args.video))

