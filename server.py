import asyncio
import websockets
import cv2
import numpy as np
import json
from ultralytics import YOLO

model = YOLO("models/5s_seg.pt")

async def handler(websocket):
    print("🔌 Клиент подключился")
    try:
        async for message in websocket:
            np_arr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            results = model(frame, conf=0.7, verbose=False)[0]

            boxes = []
            for xyxy, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(conf),
                    "class": int(cls)
                })

            await websocket.send(json.dumps({"boxes": boxes}))
    except Exception as e:
        print("Ошибка:", e)

async def main():
    print("🚀 Сервер запущен на порту 80")
    async with websockets.serve(handler, "0.0.0.0", 80, max_size=10 * 1024 * 1024):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
