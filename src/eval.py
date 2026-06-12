import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DualityDataset
from src.model_refine import ProgressiveSemanticSegmenter
from src.utils import CLASS_DEFINITIONS




class BasicSegmentationTransform:
    """Albumentations-free transform to avoid OpenCV runtime dependency."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image, mask):
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        image_t = F.interpolate(
            image_t.unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=(self.height, self.width), mode="nearest").squeeze(0).squeeze(0)

        image_t = (image_t - self.mean) / self.std
        return {"image": image_t, "mask": mask_t.long()}

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_batch_stats(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Compute intersection/union per class for a batch."""
    intersections = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)

    for class_idx in range(num_classes):
        pred_mask = preds == class_idx
        target_mask = targets == class_idx

        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()

        intersections[class_idx] += intersection
        unions[class_idx] += union

    return intersections, unions


def evaluate_on_test_set(
    checkpoint_path: str = "checkpoints/best_model.pth",
    data_dir: str = "Offroad_Segmentation_Training_Dataset",
    batch_size: int = 4,
    num_workers: int = 0,
    output_json: str = "runs/test_metrics.json",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(CLASS_DEFINITIONS)

    config = load_config()
    input_h, input_w = config["input_size"]

    test_transform = BasicSegmentationTransform(height=input_h, width=input_w)

    test_dataset = DualityDataset(data_dir, split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # IMPORTANT: best_model.pth was produced by ProgressiveSemanticSegmenter in src/train.py
    model = ProgressiveSemanticSegmenter(n_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    total_correct = 0
    total_pixels = 0
    total_intersections = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_unions = torch.zeros(num_classes, dtype=torch.float64, device=device)
    per_image_ious = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating test set")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).long()

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            batch_intersections, batch_unions = compute_batch_stats(preds, masks, num_classes)
            total_intersections += batch_intersections
            total_unions += batch_unions

            # Per-image mean IoU for distribution stats
            for i in range(images.size(0)):
                img_intersections, img_unions = compute_batch_stats(
                    preds[i : i + 1], masks[i : i + 1], num_classes
                )
                img_class_iou = (img_intersections / (img_unions + 1e-8)).cpu().numpy()
                per_image_ious.append(float(np.nanmean(img_class_iou)))

    class_iou = (total_intersections / (total_unions + 1e-8)).cpu().numpy()
    mean_iou = float(np.nanmean(class_iou))
    pixel_accuracy = float(total_correct / max(total_pixels, 1))

    class_results = {
        CLASS_DEFINITIONS[i]["name"]: {
            "class_id": i,
            "iou": float(class_iou[i]),
            "intersection": float(total_intersections[i].item()),
            "union": float(total_unions[i].item()),
        }
        for i in range(num_classes)
    }

    results = {
        "checkpoint": checkpoint_path,
        "dataset_split": "test",
        "num_samples": len(test_dataset),
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "per_image_mean_iou": {
            "mean": float(np.mean(per_image_ious)) if per_image_ious else 0.0,
            "std": float(np.std(per_image_ious)) if per_image_ious else 0.0,
            "min": float(np.min(per_image_ious)) if per_image_ious else 0.0,
            "max": float(np.max(per_image_ious)) if per_image_ious else 0.0,
        },
        "class_metrics": class_results,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def create_test_plots(results: dict, output_dir: str = "runs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(results["class_metrics"].keys())
    class_ious = [results["class_metrics"][name]["iou"] for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars = axes[0].barh(class_names, class_ious, color="steelblue", edgecolor="black")
    axes[0].set_title("Class-wise IoU (Test Set)", fontweight="bold")
    axes[0].set_xlabel("IoU")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].grid(True, axis="x", alpha=0.3)

    for bar, value in zip(bars, class_ious):
        axes[0].text(min(value + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")

    stats = results["per_image_mean_iou"]
    labels = ["Pixel Accuracy", "Mean IoU", "Per-image IoU Mean", "Per-image IoU Std"]
    values = [
        results["pixel_accuracy"],
        results["mean_iou"],
        stats["mean"],
        stats["std"],
    ]
    axes[1].bar(labels, values, color=["green", "purple", "orange", "gray"], edgecolor="black")
    axes[1].set_title("Overall Test Metrics", fontweight="bold")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "test_analysis.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    print(f"Saved test analysis plot to {plot_path}")


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Checkpoint:      {results['checkpoint']}")
    print(f"Samples:         {results['num_samples']}")
    print(f"Pixel Accuracy:  {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU:        {results['mean_iou']:.4f}")
    print("\nPer-class IoU:")

    for class_name, class_data in results["class_metrics"].items():
        print(f"  - {class_name:<16} {class_data['iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best_model.pth on test split")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--data-dir", default="Offroad_Segmentation_Training_Dataset", help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for test loader")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--output-json", default="runs/test_metrics.json", help="Output JSON for metrics")
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    results = evaluate_on_test_set(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_json=args.output_json,
    )

    print_summary(results)

    if not args.skip_plots:
        create_test_plots(results)


if __name__ == "__main__":
    main()
