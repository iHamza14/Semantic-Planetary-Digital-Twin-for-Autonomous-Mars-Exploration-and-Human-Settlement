import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import time
import json
import segmentation_models_pytorch as smp

from train import CLASS_MAPPING, NUM_CLASSES

# =========================================================
# DATASET
# =========================================================

class TestDatasetWithMasks(Dataset):
    def __init__(self, test_rgb_dir, test_seg_dir=None, transform=None):
        self.test_rgb_dir = Path(test_rgb_dir)
        self.test_seg_dir = Path(test_seg_dir) if test_seg_dir else None
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(test_rgb_dir)
             if f.endswith(('.png', '.jpg', '.jpeg'))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.test_rgb_dir / self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask = None
        
        if self.test_seg_dir:
            mask_path = self.test_seg_dir / self.images[idx]
            if mask_path.exists():
                seg_image = Image.open(mask_path).resize((256, 256), Image.NEAREST)
                seg_array = np.array(seg_image)

                mask = torch.zeros(seg_array.shape, dtype=torch.long)
                for original_class, new_class in CLASS_MAPPING.items():
                    mask[seg_array == original_class] = new_class

        return image, mask


# =========================================================
# MODEL
# =========================================================

def get_model(num_classes):
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )


# =========================================================
# FAST EVALUATION
# =========================================================

def evaluate_model(model, loader, device):

    model.eval()
    num_classes = NUM_CLASSES

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device
    )

    latencies = []

    print("Running fast evaluation...")

    with torch.no_grad():
        for images, masks in tqdm(loader):

            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            per_image_time = (end - start) / images.size(0)
            latencies.extend([per_image_time] * images.size(0))

            preds = torch.argmax(outputs, dim=1)

            if masks[0] is not None:
                masks = masks.to(device)

                # Fully vectorized confusion matrix update
                k = (masks >= 0) & (masks < num_classes)
                inds = num_classes * masks[k] + preds[k]
                confusion_matrix += torch.bincount(
                    inds,
                    minlength=num_classes**2
                ).reshape(num_classes, num_classes)

    # ================================
    # METRICS
    # ================================

    TP = confusion_matrix.diag()
    FP = confusion_matrix.sum(0) - TP
    FN = confusion_matrix.sum(1) - TP

    per_class_iou = TP.float() / (TP + FP + FN + 1e-6)
    mean_iou = per_class_iou.mean().item()

    per_class_accuracy = TP.float() / (TP + FN + 1e-6)
    pixel_accuracy = TP.sum().float() / confusion_matrix.sum().float()

    precision = TP.float() / (TP + FP + 1e-6)
    recall = TP.float() / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    latency_stats = {
        "mean": float(np.mean(latencies)),
        "max": float(np.max(latencies)),
        "min": float(np.min(latencies)),
        "std": float(np.std(latencies)),
    }

    return {
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou.cpu().numpy(),
        "per_class_accuracy": per_class_accuracy.cpu().numpy(),
        "pixel_accuracy": pixel_accuracy.item(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "latency": latency_stats,
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    BATCH_SIZE = 4
    MODEL_PATH = "runs/best_model.pth"
    TEST_RGB_DIR = "/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImages/rgb"
    TEST_SEG_DIR = "/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImagesseg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = TestDatasetWithMasks(
        TEST_RGB_DIR,
        TEST_SEG_DIR,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Test samples:", len(dataset))

    model = get_model(NUM_CLASSES).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device,weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    results = evaluate_model(model, loader, device)

    print("\n================ RESULTS ================\n")
    print("Mean IoU:", results["mean_iou"])
    print("Pixel Accuracy:", results["pixel_accuracy"])

    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
        "Ground Clutter", "Flowers", "Logs",
        "Rocks", "Landscape", "Sky"
    ]

    for i, cls in enumerate(class_names):
        print(f"{cls:15} | "
              f"IoU: {results['per_class_iou'][i]:.4f} | "
              f"Acc: {results['per_class_accuracy'][i]:.4f} | "
              f"Prec: {results['precision'][i]:.4f} | "
              f"Recall: {results['recall'][i]:.4f} | "
              f"F1: {results['f1'][i]:.4f}")

    print("\nLatency (sec per image):")
    print(results["latency"])

    os.makedirs("runs", exist_ok=True)
    with open("runs/test_results_fast.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to runs/test_results_fast.json")