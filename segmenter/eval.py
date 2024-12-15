from segmenter.models.sam import SAM
from segmenter.datasets.sam import SamDataset, SAM_collate_fn
from segmenter.evaluator import Evaluator

from torch.utils.data import DataLoader
from argparse import ArgumentParser


def eval(model_name, checkpoint_path, device, data_dir, batch_size, model_type=None):
    model_name = model_name.lower()
    if model_name == "sam":
        if model_type is None:
            raise ValueError("model_type must be provided for SAM model")
        model = SAM(model_type, checkpoint_path, device)
        dataset = SamDataset.from_cvat(data_dir)
    else:
        raise NotImplementedError

    if model_name == "sam":
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=SAM_collate_fn
        )
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluator = Evaluator(model, loader, device, False)
    metrics = evaluator.evaluate()
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, choices=["sam"]
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--device", type=str, required=True, help="cuda or cpu")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="the path to the directory with images. Should contains images/ and masks/ dirs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="batch size (default 20)"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, help="model type for SAM model"
    )
    args = parser.parse_args()
    return eval(
        args.model_name,
        args.checkpoint_path,
        args.device,
        args.data_dir,
        args.batch_size,
        args.model_type,
    )


if __name__ == "__main__":
    print(main())
