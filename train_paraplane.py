from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')

    results = model.train(
        data='C:/Users/Madina/Desktop/505/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        patience=15,
        workers=0,
        project='runs/paraplane',
        name='train_manual',
        exist_ok=True
    )

    print("Готово! Веса: runs/paraplane/train_manual/weights/best.pt")