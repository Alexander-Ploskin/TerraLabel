from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, image_processor, img_pathes, mask_pathes):
        self.img_pathes = img_pathes
        self.mask_pathes = mask_pathes
        self.image_processor = image_processor

        assert len(self.img_pathes) == len(self.mask_pathes), "There must be as many images as there are segmentation maps"
        
        
    @classmethod
    def get_train_and_eval_datasets(cls, image_processor, img_dir, masks_dir, ratio=0.3, seed=52):
        img_pathes = [f'{img_dir}/{item}' for item in sorted(os.listdir(img_dir))]
        img_pathes = [item for item in img_pathes if item[-3:] == 'jpg']
        mask_pathes = [f'{masks_dir}/{item}' for item in sorted(os.listdir(masks_dir))]
        
        train_img_pathes, eval_img_pathes, train_mask_pathes, eval_mask_pathes = train_test_split(
            img_pathes,
            mask_pathes,
            test_size=ratio,
            random_state=seed
        )
        
        train_dataset = SemanticSegmentationDataset(image_processor, train_img_pathes, train_mask_pathes)
        eval_dataset = SemanticSegmentationDataset(image_processor, eval_img_pathes, eval_mask_pathes)
        
        return train_dataset, eval_dataset


    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, idx):
        mask = np.load(self.mask_pathes[idx])
        image = np.array( Image.open(self.img_pathes[idx])) 

        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
     