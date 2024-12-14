import torch

from segmenter.metrics import calculate_metrics


class Evaluator:
    def __init__(self, model, loader, device, isSAM=False):
        self.model = model
        self.loader = loader
        self.device = device
        self.isSAM = isSAM

    def evaluate(self):
        self.model.eval()
        dataset_len = 0
        metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "f1": 0}
        for data in self.loader:
            if self.isSAM:
                images, mask, bboxes = data
                images = images
                bboxes = bboxes
                output = self.model(images, bboxes)
            else:
                images, mask = data
                images = images.to(self.device)
                with torch.no_grad():
                    output = self.model(images)

            for i in range(len(output)):
                pred = output[i].cpu().numpy()
                real = mask[i].cpu().numpy()
                dataset_len += 1
                iou, precision, recall, f1 = calculate_metrics(pred, real)
                metrics["iou"] += iou
                metrics["precision"] += precision
                metrics["recall"] += recall
                metrics["f1"] += f1

        for key in metrics.keys():
            metrics[key] /= dataset_len
        return metrics
