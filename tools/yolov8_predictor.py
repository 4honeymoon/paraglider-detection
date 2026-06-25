import sys
import os

sys.path.append(r"C:\Users\Madina\PycharmProjects\pythonProject3\YOLOv8-SMOT")

import numpy as np
import os.path as osp
from loguru import logger
from ultralytics import YOLO
import mmcv


class YOLOV8Predictor:
    def __init__(self, checkpoint_file, config=None, config_file=None, device='cuda:0', fp16=False, tta=False):

        logger.info(f"Загружаем модель Ultralytics из: {checkpoint_file}")

        # Загружаем .pt файл напрямую через Ultralytics
        self.model = YOLO(r'C:\Users\Madina\PycharmProjects\pythonProject3\YOLOv8-SMOT\weights\internet_best.pt')

        # Устройство
        if 'cuda' in str(device):
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        logger.info("Модель успешно загружена!")

    def inference(self, img, timer):
        """
        Args:
            img: image path or numpy array
            timer: Timer object
        Returns:
            outputs: List[numpy.ndarray], [x1,y1,x2,y2,score]
            img_info: dict
        """
        img_info = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            raw_img = mmcv.imread(img)
        else:
            img_info["file_name"] = None
            raw_img = img

        # Запускаем детекцию
        results = self.model(raw_img, device=self.device, verbose=False, conf=0.2)
        logger.info(f"Детекций найдено: {len(results[0].boxes)}, conf порог: 0.2")

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            outputs = [None]
            logger.info("Параплан не найден на кадре")
        else:
            bboxes = boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
            scores = boxes.conf.cpu().numpy()  # уверенность

            outputs = [np.concatenate([
                bboxes,
                scores[:, None],
            ], axis=1)]

        return outputs, img_info