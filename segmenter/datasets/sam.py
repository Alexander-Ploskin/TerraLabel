from segmenter.datasets.utils import (
    get_bboxes_from_mask,
    create_mask_for_frame,
    get_image_shape
)

from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Callable
import os

import cv2


class SamDataset(Dataset):
    def __init__(self, images: List[Path], get_mask: Callable[[int], cv2.typing.MatLike]):
        self.images = images
        self.get_mask = get_mask

    @classmethod
    def from_cvat(cls, data_dir: str) -> Dataset:
        annotations_path = os.path.join(data_dir, 'annotations.json')
        data_dir = Path(data_dir)
        images = sorted(list(data_dir.glob("data/*.jpg")))
        get_mask = lambda frame_index: create_mask_for_frame(
            annotations_file=annotations_path,
            frame_index=frame_index,
            image_shape=get_image_shape(str(images[frame_index]))
        )
        
        return SamDataset(images, get_mask)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = self.get_mask(idx)
        
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = get_bboxes_from_mask(mask)

        return image, mask / 255, bboxes


def SAM_collate_fn(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]

    return images, masks, bboxes
