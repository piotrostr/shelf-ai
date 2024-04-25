import zlib
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from model import Model

app = FastAPI()
model = Model("./retail-yolo.engine")

def encode(img: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Error encoding image")
    return buffer.tobytes()

def compress(payload: bytes) -> bytes:
    return zlib.compress(payload)

def decompress(payload: bytes) -> bytes:
    return zlib.decompress(payload)

def decode_image(buffer: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_bytes()
            buffer = decompress(payload)
            img = decode_image(buffer)
            preds = model.predict(img)
            await websocket.send_json(preds[0].tojson())
    except WebSocketDisconnect as e:
        print("Client disconnected", e)

