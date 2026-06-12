from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
INPUT_SIZE = (252, 252)
DEFAULT_DEVICE = "cuda"
