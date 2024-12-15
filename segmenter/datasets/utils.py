import cv2
import torch
import json
import numpy as np
import supervision as sv
from typing import Any, Tuple, Dict


def get_image_shape(image_path: str) -> Tuple[int, int]:
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return height, width


def create_mask_for_frame(annotations_file: str, frame_index: int, image_shape: Any):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    height, width = image_shape
    mask = np.zeros(image_shape, dtype=np.uint8)

    for annotation in annotations:
        for shape in annotation['shapes']:
            if shape['frame'] == frame_index:
                if shape['type'] == 'polygon':
                    label = shape['label']
                    points = np.array(shape['points'], dtype=np.float32).reshape((-1, 2))
                    polygon_mask = sv.polygon_to_mask(points, (width, height))
                    polygon_mask[polygon_mask == 1] = 1
                    mask = np.maximum(mask, polygon_mask)

    return mask


def get_bboxes_from_mask(mask, mode="xyxy"):
    contours = get_contours_from_mask(mask)
    bboxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if mode == "xywh":
            bboxes.append((x, y, width, height))
        else:
            bboxes.append((x, y, x + width, y + height))
    bboxes = torch.tensor(bboxes)
    return bboxes


def get_contours_from_mask(mask):
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours
