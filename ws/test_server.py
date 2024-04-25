import cv2
import pytest

from fastapi.testclient import TestClient
from server import app, compress, encode

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_websocket_endpoint():
    client = TestClient(app)
    img = cv2.imread("../sample_image.jpg")
    buffer = encode(img)
    payload = compress(buffer)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(payload)
        response = ws.receive_json()
        assert response is not None
