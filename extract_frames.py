import cv2
import os

if __name__ == '__main__':
    video_path = r'C:\Users\Madina\Desktop\video\7s.mp4'
    output_folder = r'C:\Users\Madina\Desktop\video\7s_frames'

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Сохраняем кадры в формате 000001.jpg
        filename = os.path.join(output_folder, f'{frame_id:06d}.jpg')
        cv2.imwrite(filename, frame)
        frame_id += 1

    cap.release()
    print(f"Готово! Извлечено кадров: {frame_id - 1}")
    print(f"Сохранено в: {output_folder}")