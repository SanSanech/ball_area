import cv2
import os
import torch
from tqdm import tqdm
from ultralytics import YOLO

class ShotCounter:
    def __init__(self):
        self.state = 'WAIT_ENTRY'
        self.count = 0

    def process(self, boxes, names):
        ball_box, ring_box = None, None
        for xyxy, cls in zip(boxes.xyxy, boxes.cls):
            label = names[int(cls)].lower()
            if label in ['ball', 'мяч']:
                ball_box = list(map(int, xyxy.tolist()))
            elif label in ['ring', 'кольцо']:
                ring_box = list(map(int, xyxy.tolist()))

        if not ring_box or not ball_box:
            return

        bx = (ball_box[0] + ball_box[2]) // 2
        by = (ball_box[1] + ball_box[3]) // 2
        x1, y1, x2, y2 = ring_box

        if self.state == 'WAIT_ENTRY' and y1 < by < y2 and x1 < bx < x2:
            self.state = 'INSIDE'
        elif self.state == 'INSIDE' and by > y2:
            self.count += 1
            self.state = 'WAIT_ENTRY'

def draw_boxes(frame, boxes, names):
    annotated = frame.copy()
    for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        label = names[int(cls)].lower()
        color = (0, 255, 0)
        if label in ['ball', 'мяч']:
            color = (0, 0, 255)
        elif label in ['ring', 'кольцо']:
            color = (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, txt, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated

def process_and_save_video(input_path: str, output_path: str, model_path="models/5s_seg.pt"):
    # Проверка доступности GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Используемое устройство: {device.upper()}")
    
    if not os.path.exists(input_path):
        print(f"❌ Файл не найден: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Загрузка модели на GPU
    model = YOLO(model_path).to(device)
    counter = ShotCounter()

    print(f"📼 Обработка видео: {input_path}")
    pbar = tqdm(total=total, desc="Обработка")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Перенос кадра на GPU (если нужно)
        if device.startswith('cuda'):
            frame_tensor = torch.from_numpy(frame).to(device)
            results = model(frame_tensor, conf=0.7, verbose=False, device=device)[0]
        else:
            results = model(frame, conf=0.7, verbose=False, device=device)[0]

        counter.process(results.boxes, model.names)
        annotated = draw_boxes(frame, results.boxes, model.names)

        cv2.putText(annotated, f"Score: {counter.count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"✅ Готово! Результат сохранён: {output_path}")
    print(f"🏀 Итоговое количество попаданий: {counter.count}")

def menu():
    print("🎯 Меню обработки видео")
    print("1 - Обработать видео и сохранить с боксами")
    print("q - Выйти")

    while True:
        choice = input("Выберите действие: ").strip().lower()

        if choice == '1':
            input_path = input("Путь к видео: ").strip()
            output_path = input("Путь для сохранения (output.mp4): ").strip() or "output.mp4"
            process_and_save_video(input_path, output_path)
        elif choice == 'q':
            print("👋 Выход.")
            break
        else:
            print("❗️Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    # Проверка доступности CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    menu()