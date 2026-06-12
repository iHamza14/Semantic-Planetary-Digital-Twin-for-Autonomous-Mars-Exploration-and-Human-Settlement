import io
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.model_refine import ProgressiveSemanticSegmenter as UNet
from src.utils import CLASS_DEFINITIONS, map_mask_values

from ..core.config import CHECKPOINT_PATH, DEFAULT_DEVICE, INPUT_SIZE


class InferenceService:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() and DEFAULT_DEVICE == "cuda" else "cpu"
        self.model = self._load_model()
        self.transform = A.Compose(
            [
                A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _load_model(self) -> torch.nn.Module:
        model = UNet(n_classes=len(CLASS_DEFINITIONS))
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

        state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _read_image_bytes(image_bytes: bytes) -> np.ndarray:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img)

    @staticmethod
    def _read_mask_bytes(mask_bytes: bytes) -> np.ndarray:
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        mask_raw = np.array(mask_pil)
        return map_mask_values(mask_raw)

    @staticmethod
    def _resolve_gt_from_image_path(image_path: Path) -> np.ndarray | None:
        parent_dir = str(image_path.parent)
        if "Color_Images" not in parent_dir:
            return None

        mask_dir = parent_dir.replace("Color_Images", "Segmentation")
        mask_path = Path(mask_dir) / f"{image_path.stem}.png"
        if not mask_path.exists():
            return None

        mask_raw = np.array(Image.open(mask_path))
        return map_mask_values(mask_raw)

    def predict(self, image_np: np.ndarray, mask_np: np.ndarray | None = None) -> dict:
        augmented = self.transform(image=image_np)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
            confidence = torch.max(probabilities, dim=1)[0].squeeze(0).cpu().numpy()
            mean_conf = float(np.mean(confidence) * 100)
        latency_ms = float((time.time() - start_time) * 1000)

        accuracy = None
        if mask_np is not None:
            h, w = prediction.shape
            mask_resized = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            accuracy = float(((prediction == mask_resized).sum() / prediction.size) * 100)

        return {
            "latency_ms": latency_ms,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "predicted_shape": prediction.shape,
            "prediction_map": prediction.tolist(),
        }

    def predict_from_bytes(self, image_bytes: bytes, mask_bytes: bytes | None = None) -> dict:
        image_np = self._read_image_bytes(image_bytes)
        mask_np = self._read_mask_bytes(mask_bytes) if mask_bytes else None
        return self.predict(image_np=image_np, mask_np=mask_np)

    def predict_from_path(self, image_path: Path, auto_ground_truth: bool = True) -> dict:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        mask_np = self._resolve_gt_from_image_path(image_path) if auto_ground_truth else None
        return self.predict(image_np=image_np, mask_np=mask_np)
