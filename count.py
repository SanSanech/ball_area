import cv2
from ultralytics import YOLO
from tqdm import tqdm

class ShotCounter:
    """
    Класс для подсчёта забитых бросков мяча в кольцо.
    Логика основана на отслеживании момента входа мяча в зону кольца сверху
    и выхода снизу, с учётом возможного пропадания детекции внутри кольца.
    """
    def __init__(self):
        # Состояния FSM:
        # WAIT_ENTRY - ждём входа мяча в зону кольца
        # INSIDE     - мяч находится внутри (забрасывается)
        self.state = 'WAIT_ENTRY'
        self.count = 0  # счётчик заброшенных мячей

    def process(self, boxes, names):
        """
        Обрабатываем детекции текущего кадра.
        boxes: результаты .boxes от YOLO
        names: список имён классов модели
        """
        # Инициализируем переменные для найденных боксов
        ball_box = None
        ring_box = None

        # Проходим по всем детекциям и определяем бокс мяча и кольца
        for xyxy, cls in zip(boxes.xyxy, boxes.cls):
            label = names[int(cls)].lower()
            if label in ['ball', 'мяч']:
                ball_box = list(map(int, xyxy.tolist()))
            elif label in ['ring', 'кольцо']:
                ring_box = list(map(int, xyxy.tolist()))

        # Если не видим ни кольцо, ни мяч, пропускаем кадр
        if not ring_box or not ball_box:
            return

        # Вычисляем центр мяча
        bx = (ball_box[0] + ball_box[2]) // 2
        by = (ball_box[1] + ball_box[3]) // 2
        x1, y1, x2, y2 = ring_box

        # Переход состояний FSM
        if self.state == 'WAIT_ENTRY' and y1 < by < y2 and x1 < bx < x2:
            self.state = 'INSIDE'
        elif self.state == 'INSIDE' and by > y2:
            self.count += 1
            self.state = 'WAIT_ENTRY'

# Функция рисования боксов

def draw_boxes(frame, boxes, names):
    annotated = frame.copy()
    for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        label = names[int(cls)].lower()
        color = (0,255,0)
        if label in ['ball','мяч']:
            color = (0,0,255)
        elif label in ['ring','кольцо']:
            color = (255,0,0)
        cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
        txt = f"{conf:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
        cv2.rectangle(annotated,(x1,y1-th-4),(x1+tw,y1), color,-1)
        cv2.putText(annotated, txt, (x1,y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
    return annotated

# Обработка фото

def process_image(path: str, model: YOLO):
    img = cv2.imread(path)
    if img is None:
        print(f"Не удалось загрузить изображение: {path}")
        return
    count_mode = input("Считать попадания? (1-да, 0-нет): ").strip() == '1'
    counter = ShotCounter() if count_mode else None
    results = model(img, verbose=False, conf=0.7)[0]
    if counter:
        counter.process(results.boxes, model.names)
        print(f"Попаданий: {counter.count}")
    annotated = draw_boxes(img, results.boxes, model.names)
    cv2.imshow("Image Inference", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Онлайн видео

def process_video(source, model: YOLO):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {source}")
        return
    count_mode = input("Считать попадания? (1-да, 0-нет): ").strip() == '1'
    counter = ShotCounter() if count_mode else None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False, conf=0.7)[0]
        if counter:
            counter.process(results.boxes, model.names)
        annotated = draw_boxes(frame, results.boxes, model.names)
        if counter:
            cv2.putText(annotated, f"Score: {counter.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Video Inference (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if counter:
        print(f"Итоговое количество попаданий: {counter.count}")
    cap.release()
    cv2.destroyAllWindows()

# Запись видео

def record_video(source, output_path: str, model: YOLO):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {source}")
        return
    count_mode = input("Считать попадания? (1-да, 0-нет): ").strip() == '1'
    counter = ShotCounter() if count_mode else None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc="Recording")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False, conf=0.7)[0]
            if counter:
                counter.process(results.boxes, model.names)
            annotated = draw_boxes(frame, results.boxes, model.names)
            if counter:
                cv2.putText(annotated, f"Score: {counter.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            out.write(annotated)
            pbar.update(1)
    except KeyboardInterrupt:
        print("\nЗапись прервана пользователем.")
    finally:
        pbar.close()
        cap.release()
        out.release()
        if counter:
            print(f"Итоговое количество попаданий: {counter.count}")
        print(f"Видео сохранено: {output_path}")

# Видео 360p

def process_video_360p(source, model: YOLO):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Не удалось открыть источник: {source}")
        return
    target_size = 640
    count_mode = input("Считать попадания? (1-да, 0-нет): ").strip() == '1'
    counter = ShotCounter() if count_mode else None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (640, 360))
        pad_vert = target_size - small.shape[0]
        pad_top = pad_vert // 2
        pad_bot = pad_vert - pad_top
        square = cv2.copyMakeBorder(small, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        results = model(square, verbose=False, conf=0.7)[0]
        if counter:
            counter.process(results.boxes, model.names)
        annotated_square = draw_boxes(square, results.boxes, model.names)
        annotated = annotated_square[pad_top:pad_top+360, :640]
        if counter:
            cv2.putText(annotated, f"Score: {counter.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Video 360p (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if counter:
        print(f"Итоговое количество попаданий: {counter.count}")
    cap.release()
    cv2.destroyAllWindows()

# Главное меню

def main():
    model_path = "models/5s_seg.pt"
    model = YOLO(model_path)
    print(f"Model: {model_path}")
    print("1 - Фото")
    print("2 - Видео")
    print("3 - Запись видео")
    print("q - Выйти")
    choice = input("Выберите режим 1-3: ").strip()
    if choice == '1':
        path = input("Путь к изображению: ").strip()
        process_image(path, model)
    elif choice == '2':
        src = input("Видео или 0 для камеры: ").strip()
        process_video(0 if src=='0' else src, model)
    elif choice == '3':
        src = input("Видео или 0 для камеры: ").strip()
        outp = input("Путь сохранять (output.mp4): ").strip()
        record_video(0 if src=='0' else src, outp, model)
    elif choice == '4':
        src = input("Видео или 0 для камеры: ").strip()
        process_video_360p(0 if src=='0' else src, model)
    elif choice == 'q':
        exit()
    else:
        print("Неверный выбор.")

if __name__ == "__main__":
    while True:
        main()
    



