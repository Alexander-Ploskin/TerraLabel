import torch

from segment_anything import SamPredictor, sam_model_registry


class SAM:
    def __init__(self, model_type, checkpoint_path, device):
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(
            device
        )
        self.predictor = SamPredictor(self.model)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, images, bboxes):
        output_masks = [0] * len(images)
        for i in range(len(images)):
            if len(bboxes) == 0:
                output_masks[i] = torch.zeros_like(images[i])
            self.predictor.set_image(images[i].to(self.device))
            boxes = self.predictor.transform.apply_boxes_torch(
                bboxes[i], images[i].shape[:2]
            ).to(self.device)
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes,
                multimask_output=False,
            )
            mask = torch.any(masks, dim=0).squeeze()
            output_masks[i] = mask
        return output_masks

    def eval():
        pass
