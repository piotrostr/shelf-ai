import json
import time
import asyncio
import cv2
import websockets
import argparse

from server import compress, encode

def get_frames(video_path: str):
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    while ret:
        frame = cv2.resize(frame, (640, 360))
        yield frame
        ret, frame = capture.read()

async def bench_video(url: str, video_path: str):
    async with websockets.connect(url) as ws:
        i = 0
        full_start = time.time()
        for frame in get_frames(video_path):
            buffer = encode(frame)
            payload = compress(buffer)
            start = time.time()
            await ws.send(payload)
            _ = await ws.recv()
            end = time.time()
            i += 1
            cv2.imshow("frame", frame)
            print(f"FPS: {i / (end - full_start)}")
            print(f"Time taken this frame: {1000 * (end - start)}ms")

            if cv2.waitKey(22) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../sample_image.jpg")
    parser.add_argument("--video", type=str)
    parser.add_argument("--url", type=str, default="ws://localhost:3000/ws")
    args = parser.parse_args()

    if args.video:
        asyncio.run(bench_video(args.url, args.video))

