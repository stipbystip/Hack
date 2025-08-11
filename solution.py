import numpy as np
from typing import List, Union, Tuple
import cv2
from ultralytics import YOLO
import torch

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

import torchvision.ops


# Загружаем модель YOLOv11x


sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path="best_yolo_l_50_ep.pt",
    confidence_threshold=0.5,
    device="cuda:0",  # or 'cuda:0'
)


def infer_image_bbox_SAHI(image: np.ndarray) -> List[dict]:
    """Инференс изображения с использованием SAHI и тайлинга."""
    res_list = []

    image_h, image_w = image.shape[:2]

    slice_size = 1024
    prediction_result = get_sliced_prediction(
        image,
        detection_model=sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )

    for obj in prediction_result.object_prediction_list:
        bbox = obj.bbox.to_xywh()
        image_h, image_w = image.shape[:2]

        # Преобразование в нормализованные координаты
        xc = (bbox[0] + bbox[2] / 2) / image_w
        yc = (bbox[1] + bbox[3] / 2) / image_h
        w = bbox[2] / image_w
        h = bbox[3] / image_h

        formatted = {
            'xc': xc,
            'yc': yc,
            'w': w,
            'h': h,
            'label': obj.category.id,  # если всегда 0 — можно захардкодить
            'score': obj.score.value
        }
        res_list.append(formatted)

    return res_list

def infer_image_bbox(image: np.ndarray) -> List[dict]:
    """Функция для получения ограничивающих рамок объектов на изображении.

    Args:
        image (np.ndarray): Изображение, на котором будет производиться инференс.

    Returns:
        List[dict]: Список словарей с координатами ограничивающих рамок и оценками.
        Пример выходных данных:
        [
            {
                'xc': 0.5,
                'yc': 0.5,
                'w': 0.2,
                'h': 0.3,
                'label': 0,
                'score': 0.95
            },
            ...
        ]
    """

    res_list = []

    result = model.predict(source=image, imgsz=1024, device=0)
    
    result_numpy = []

    # Преобразуем результаты в numpy массивы
    for res in result:
        result_numpy.append(res.cpu().numpy())
    
    # Если есть результаты, обрабатываем их
    if len(result_numpy) > 0:
        for res in result_numpy:
            for box in res.boxes:
                xc = box.xywhn[0][0] 
                yc = box.xywhn[0][1]
                w = box.xywhn[0][2]
                h = box.xywhn[0][3]
                conf = box.conf[0].item()

                formatted = {
                    'xc': xc,
                    'yc': yc,
                    'w': w,
                    'h': h,
                    'label': 0,
                    'score': conf
                }
                res_list.append(formatted)

    return res_list

def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Список изображений или одно изображение.

    Returns:
        List[List[dict]]: Список списков словарей с результатами предикта 
        на найденных изображениях.
        Пример выходных данных:
        [
            [
                {
                    'xc': 0.5,
                    'yc': 0.5,
                    'w': 0.2,
                    'h': 0.3,
                    'label': 0,
                    'score': 0.95
                },
                ...
            ],
            ...
        ]
    """    
    results = []
    
    # Обрабатываем каждое изображение из полученного списка
    for image in images: 
        image_results = infer_image_bbox_SAHI(image)
        results.append(image_results)
    
    return results
