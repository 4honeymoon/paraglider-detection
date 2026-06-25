YOLOv8-SMOT: Система мульти-объектного трекинга малых объектов

Данный репозиторий содержит официальную реализацию фреймворка YOLOv8-SMOT. Проект разработан на базе библиотек mmyolo и OCSORT.

Репозиторий предназначен для запуска готовых моделей и отслеживания траекторий объектов на видео или последовательностях кадров.
Проект протестирован в окружении с Python 3.10 и поддержкой CUDA 11.3.

# Клонирование репозитория
```bash
git clone https://github.com/4honeymoon/paraglider-detection.git YOLOv8-SMOT
cd YOLOv8-SMOT
```

# Подготовка окружения
Установите утилиту uv глобально через  pip:

```bash
pip install uv
```

# Создайте изолированное виртуальное окружение внутри папки проекта, зафиксировав версию Python 3.10:
```bash
uv venv --python 3.10
```

# Активируйте созданное окружение:
```bash
.venv\Scripts\Activate.ps1
```

# Установка PyTorch и базовых библиотек
```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
```

# Установка ядра MMYOLO
Для правильной работы фреймворка OpenMMLab установите базовые компоненты и сам пакет mmyolo в режиме редактирования через встроенный менеджер mim:
```bash
mim install -r mmyolo/requirements/mminstall.txt
mim install -v -e ./mmyolo
```

# Подготовка к запуску
Для проверки работы системы используется модель, обученная на специализированном датасете.
1. Создайте в корне проекта папку `checkpoints/`.
2. Скачайте файл весов модели по ссылке: https://drive.google.com/drive/folders/1La_TVb_uIM-RDHp3W9pA-LNcn5voFvnw?usp=sharing
   ( автоматическая разметка - v4, ручная - manual_best, веса модели, обученной на датасете из интернета - internet)
3. Поместите скачанный файл в созданную папку `checkpoints/`.
4. Конфигурация путей в коде
Перед запуском откройте файл `tools/predict.py` и проверьте переменные внутри функции `main()`. Укажите путь к архитектуре и вашему файлу весов:

config_file = "mmyolo/configs/yolov8/yolov8_l.py"  # или ваша архитектура (v11)
checkpoint_file = "checkpoints/best.pth"           # имя вашего файла весов

# Запуск 
Запустите скрипт анализа, передав путь к вашему исходному видеофайлу (или папке с последовательностью кадров) через аргумент --path.
```bash
python tools/predict.py --path path_to_video/input_video.mp4 --device gpu --save_result
```
После завершения обработки в корне проекта автоматически создастся папка results/. Внутри подпапки tracks/ сохранится текстовый файл (например, input_video.txt), содержащий координаты рамок и присвоенные ID объектов для каждого кадра.
# Визуализация
Чтобы превратить текстовые координаты в итоговое демонстрационное видео с графикой, запустите скрипт визуализации:
```bash
python tools/visualize_for_mot_ch.py -m results/predictions/yolov8l_custom_run/tracks/input_video.txt -i path_to_video/input_video.mp4 -o output_visualized.mp4 --mp4 --show-bbox
```
