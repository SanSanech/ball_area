import asyncio
import websockets
import numpy as np
import cv2
import json
from ultralytics import YOLO
from count import ShotCounter, draw_boxes  # твои классы из кода

# Загружаем модель
model = YOLO("models/5s_seg.pt")

# Словарь подключений -> ShotCounter
connections = {}

async def handler(websocket):
    # Инициализируем счётчик для этого клиента
    counter = ShotCounter()
    connections[websocket] = counter

    try:
        async for message in websocket:
            # Преобразуем бинарное сообщение в изображение
            np_arr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Инференс
            results = model(frame, conf=0.7, verbose=False)[0]
            counter.process(results.boxes, model.names)

            # Возвращаем результат
            response = {
                "thrown": counter.count,   # например, можно передавать все попадания
                "scored": counter.count    # пока счёт совпадает, можно отдельно если нужно
            }
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connections.pop(websocket, None)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8000, max_size=5 * 1024 * 1024):
        print("WebSocket сервер запущен на ws://0.0.0.0:8000")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
