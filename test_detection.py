from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO(
        r'C:\Users\Madina\PycharmProjects\pythonProject3\YOLOv8-SMOT\runs\detect\runs\paraplane\train\weights\best.pt')

    # Берём один кадр из видео
    cap = cv2.VideoCapture(r'C:\Users\Madina\Desktop\video\7s.mp4')
    ret, frame = cap.read()
    cap.release()

    # Запускаем с очень низким порогом
    results = model(frame, conf=0.05, verbose=True)

    # Показываем что нашли
    for r in results:
        print(f"Найдено объектов: {len(r.boxes)}")
        for box in r.boxes:
            print(f"  уверенность: {box.conf.item():.3f}, координаты: {box.xyxy}")

    # Сохраняем кадр с результатами
    res_frame = results[0].plot()
    cv2.imwrite('test_frame.jpg', res_frame)
    print("Кадр сохранён: test_frame.jpg")