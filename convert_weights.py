from ultralytics import YOLO

if __name__ == '__main__':
    # Загружаем обученную модель
    model = YOLO(
        r'C:\Users\Madina\PycharmProjects\pythonProject3\YOLOv8-SMOT\runs\detect\runs\paraplane\train\weights\best.pt')

    # Экспортируем в формат torchscript (универсальный)
    model.export(format='torchscript')

    print("Готово! Файл best.torchscript создан рядом с best.pt")