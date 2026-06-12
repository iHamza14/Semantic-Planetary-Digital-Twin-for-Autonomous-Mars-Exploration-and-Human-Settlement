"""
eval.py — Memory-efficient semantic segmentation evaluation.

All metrics (IoU, Precision, Recall, F1, mAP proxy) are derived from a single
incremental confusion matrix — O(C²) memory regardless of dataset size.
No probability tensors or full prediction arrays are ever accumulated in RAM.
"""

import os, time, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp

from train import CLASS_MAPPING, NUM_CLASSES

# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_MAP = np.array([
    [ 34, 139,  34], [ 50, 205,  50], [210, 180, 140], [139,  69,  19],
    [160,  82,  45], [255, 192, 203], [101,  67,  33], [128, 128, 128],
    [222, 184, 135], [135, 206, 235],
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def get_model(num_classes):
    return smp.Unet(encoder_name='resnet50', encoder_weights=None,
                    in_channels=3, classes=num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, rgb_dir, seg_dir=None, transform=None):
        self.rgb_dir   = Path(rgb_dir)
        self.seg_dir   = Path(seg_dir) if seg_dir else None
        self.transform = transform
        self.images    = sorted(f for f in os.listdir(rgb_dir)
                                if f.lower().endswith(('.png','.jpg','.jpeg')))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        image = Image.open(self.rgb_dir / fname).convert('RGB')
        if self.transform:
            image = self.transform(image)
        mask = None
        if self.seg_dir:
            mp = self.seg_dir / fname
            if mp.exists():
                seg  = np.array(Image.open(mp))
                mask = np.zeros(seg.shape[:2], dtype=np.int64)
                for orig_cls, new_cls in CLASS_MAPPING.items():
                    mask[seg == orig_cls] = new_cls
                mask = torch.from_numpy(mask)
        return image, mask, fname

def collate_fn(batch):
    images, masks, fnames = zip(*batch)
    return torch.stack(images), list(masks), list(fnames)


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Confusion Matrix  (O(C²) memory)
# ─────────────────────────────────────────────────────────────────────────────
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.K  = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        mask = (target >= 0) & (target < self.K)
        idx  = target[mask] * self.K + pred[mask]
        self.cm += np.bincount(idx, minlength=self.K*self.K).reshape(self.K, self.K)

    def iou_per_class(self):
        tp = np.diag(self.cm); fp = self.cm.sum(0)-tp; fn = self.cm.sum(1)-tp
        d  = tp+fp+fn
        return np.where(d > 0, tp/d, np.nan)

    def mean_iou(self):         return float(np.nanmean(self.iou_per_class()))
    def precision_per_class(self):
        tp = np.diag(self.cm); d = self.cm.sum(0)
        return np.where(d > 0, tp/d, 0.0)
    def recall_per_class(self):
        tp = np.diag(self.cm); d = self.cm.sum(1)
        return np.where(d > 0, tp/d, 0.0)
    def f1_per_class(self):
        p = self.precision_per_class(); r = self.recall_per_class(); d = p+r
        return np.where(d > 0, 2*p*r/d, 0.0)
    def pixel_accuracy(self):
        return float(np.diag(self.cm).sum() / (self.cm.sum()+1e-9))
    def mean_ap_proxy(self):
        """Lightweight mAP proxy = mean(precision * recall) over present classes."""
        p = self.precision_per_class(); r = self.recall_per_class()
        present = self.cm.sum(1) > 0
        return float(np.mean((p*r)[present])) if present.any() else 0.0
    def normalised(self):
        return self.cm / (self.cm.sum(1, keepdims=True) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def mask_to_rgb(m):
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES): rgb[m==c] = COLOR_MAP[c]
    return rgb

def denormalize(t):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return torch.clamp(t.cpu()*std+mean, 0, 1).permute(1,2,0).numpy()

def save_comparison_grid(images, preds_np, gts_np, fnames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    legend = [mpatches.Patch(color=COLOR_MAP[i]/255, label=CLASS_NAMES[i])
              for i in range(NUM_CLASSES)]
    for img, pred, gt, fname in zip(images, preds_np, gts_np, fnames):
        fig, axes = plt.subplots(1, 3, figsize=(18,6))
        axes[0].imshow(denormalize(img));    axes[0].set_title('Input');        axes[0].axis('off')
        axes[1].imshow(mask_to_rgb(gt));     axes[1].set_title('Ground Truth'); axes[1].axis('off')
        axes[2].imshow(mask_to_rgb(pred));   axes[2].set_title('Prediction');   axes[2].axis('off')
        fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cmp_{fname}'), dpi=150, bbox_inches='tight')
        plt.close()

def plot_metrics(cm_obj, scalar, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    ious  = np.nan_to_num(cm_obj.iou_per_class())
    precs = cm_obj.precision_per_class()
    recs  = cm_obj.recall_per_class()
    f1s   = cm_obj.f1_per_class()

    fig, axes = plt.subplots(2, 3, figsize=(22,14))
    fig.suptitle('Semantic Segmentation – Evaluation Metrics', fontsize=15, fontweight='bold')

    def hbar(ax, vals, title, xlabel, hi=0.6, lo=0.4):
        colors = ['green' if v>=hi else 'orange' if v>=lo else 'red' for v in vals]
        bars = ax.barh(CLASS_NAMES, vals, color=colors, edgecolor='black', height=0.6)
        ax.set_title(title, fontweight='bold'); ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1.05); ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(min(v+0.02,1.0), bar.get_y()+bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=8)

    hbar(axes[0,0], ious,  'Per-Class IoU',       'IoU')
    hbar(axes[0,1], precs, 'Per-Class Precision', 'Precision')
    hbar(axes[0,2], recs,  'Per-Class Recall',    'Recall')
    hbar(axes[1,0], f1s,   'Per-Class F1',        'F1')

    # Confusion matrix
    cm_norm = cm_obj.normalised()
    im = axes[1,1].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    axes[1,1].set_xticks(range(NUM_CLASSES))
    axes[1,1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    axes[1,1].set_yticks(range(NUM_CLASSES))
    axes[1,1].set_yticklabels(CLASS_NAMES, fontsize=7)
    axes[1,1].set_title('Normalised Confusion Matrix', fontweight='bold')
    fig.colorbar(im, ax=axes[1,1])
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1,1].text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                           fontsize=6, color='white' if cm_norm[i,j]>0.5 else 'black')

    # Summary
    r = scalar
    tag = ("Excellent" if r['mean_iou']>=0.65 else "Good" if r['mean_iou']>=0.55
           else "Fair" if r['mean_iou']>=0.45 else "Needs Work")
    summary = (
        f"PRIMARY METRICS\n{'─'*36}\n"
        f"  Mean IoU       : {r['mean_iou']:.4f}  ({tag})\n"
        f"  mAP (proxy)    : {r['map50']:.4f}\n"
        f"  Pixel Accuracy : {r['pixel_accuracy']:.4f}\n\n"
        f"LATENCY\n{'─'*36}\n"
        f"  Avg / image    : {r['avg_latency_ms']:.2f} ms\n"
        f"  Total time     : {r['total_inference_s']:.2f} s\n"
        f"  # Images       : {r['num_images']}\n\n"
        f"AGGREGATE\n{'─'*36}\n"
        f"  Avg Precision  : {np.mean(precs):.4f}\n"
        f"  Avg Recall     : {np.mean(recs):.4f}\n"
        f"  Avg F1         : {np.mean(f1s):.4f}\n"
    )
    axes[1,2].axis('off')
    axes[1,2].text(0.02, 0.97, summary, va='top', ha='left', fontsize=9,
                   family='monospace', transform=axes[1,2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'Metrics plot  -> {out_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────
def print_results(cm_obj, scalar):
    r     = scalar
    ious  = cm_obj.iou_per_class()
    precs = cm_obj.precision_per_class()
    recs  = cm_obj.recall_per_class()
    f1s   = cm_obj.f1_per_class()
    W = 80
    print('\n' + '='*W)
    print('  EVALUATION RESULTS'.center(W))
    print('='*W)
    print(f'\n  {"Metric":<25} {"Value":>10}')
    print(f'  {"─"*25} {"─"*10}')
    print(f'  {"Mean IoU":<25} {r["mean_iou"]:>10.4f}')
    print(f'  {"mAP50 (proxy)":<25} {r["map50"]:>10.4f}')
    print(f'  {"Pixel Accuracy":<25} {r["pixel_accuracy"]:>10.4f}')
    print(f'  {"Avg Latency (ms/img)":<25} {r["avg_latency_ms"]:>10.2f}')
    print(f'  {"Total Inference (s)":<25} {r["total_inference_s"]:>10.2f}')
    print(f'  {"# Images":<25} {r["num_images"]:>10}')
    print(f'\n  {"Class":<18}  {"IoU":>6}  {"Prec":>6}  {"Rec":>6}  {"F1":>6}')
    print(f'  {"─"*18}  {"─"*6}  {"─"*6}  {"─"*6}  {"─"*6}')
    for i, name in enumerate(CLASS_NAMES):
        iou = ious[i] if not np.isnan(ious[i]) else 0.0
        print(f'  {name:<18}  {iou:>6.3f}  {precs[i]:>6.3f}  {recs[i]:>6.3f}  {f1s[i]:>6.3f}')
    v = r['mean_iou']
    rating = ('EXCELLENT' if v>=0.65 else 'GOOD – competitive' if v>=0.55
              else 'FAIR – acceptable' if v>=0.45 else 'Needs improvement')
    print(f'\n  Rating: {rating}')
    print('='*W + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, visualize=True, max_vis_batches=2):
    model.eval()
    cm           = ConfusionMatrix(NUM_CLASSES)
    latencies_ms = []
    total_images = 0
    vis_count    = 0
    max_vis      = max_vis_batches * loader.batch_size

    print('\nRunning inference ...')
    for batch_idx, (images, masks, fnames) in enumerate(tqdm(loader, desc='Eval')):
        B      = images.shape[0]
        images = images.to(device, non_blocking=True)

        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == 'cuda': torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies_ms.extend([(t1-t0)*1000/B] * B)
        total_images += B

        # Argmax on GPU -> CPU int32; NEVER store softmax probabilities
        preds_np = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int32)
        images_cpu = images.cpu()
        del outputs, images
        if device.type == 'cuda': torch.cuda.empty_cache()

        if masks[0] is None:
            continue

        for i, mask in enumerate(masks):
            gt_np   = mask.numpy().astype(np.int32)
            pred_np = preds_np[i]

            # Resize pred -> GT resolution if needed (nearest, CPU only)
            if pred_np.shape != gt_np.shape:
                pred_t  = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float()
                pred_np = (F.interpolate(pred_t, size=gt_np.shape, mode='nearest')
                           .squeeze().numpy().astype(np.int32))

            cm.update(pred_np.ravel(), gt_np.ravel())

            if visualize and vis_count < max_vis:
                save_comparison_grid([images_cpu[i]], [pred_np], [gt_np],
                                     [fnames[i]], out_dir='runs/test_predictions')
                vis_count += 1

    scalar = dict(
        mean_iou            = cm.mean_iou(),
        map50               = cm.mean_ap_proxy(),
        pixel_accuracy      = cm.pixel_accuracy(),
        avg_latency_ms      = float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        total_inference_s   = float(np.sum(latencies_ms)) / 1000.0,
        num_images          = total_images,
        per_class_iou       = {CLASS_NAMES[i]: float(v) for i,v in enumerate(cm.iou_per_class())},
        per_class_precision = {CLASS_NAMES[i]: float(v) for i,v in enumerate(cm.precision_per_class())},
        per_class_recall    = {CLASS_NAMES[i]: float(v) for i,v in enumerate(cm.recall_per_class())},
        per_class_f1        = {CLASS_NAMES[i]: float(v) for i,v in enumerate(cm.f1_per_class())},
    )
    return cm, scalar


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BATCH_SIZE   = 4        # safe — only int32 argmax stored, never softmax
    MODEL_PATH   = 'runs/best_model.pth'
    TEST_RGB_DIR = '/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImages/test/rgb'
    TEST_SEG_DIR = '/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImages/test/seg'
    NUM_WORKERS  = 2
    VISUALIZE    = True
    OUT_DIR      = 'runs'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    has_masks = bool(TEST_SEG_DIR and os.path.isdir(TEST_SEG_DIR))
    if not has_masks:
        print('No test masks found – latency-only run.')
        TEST_SEG_DIR = None

    dataset = TestDataset(TEST_RGB_DIR, TEST_SEG_DIR, transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, collate_fn=collate_fn,
                         pin_memory=(device.type == 'cuda'))
    print(f'Test images: {len(dataset)}')

    if not os.path.exists(MODEL_PATH):
        print(f'Model not found: {MODEL_PATH}'); exit(1)

    ckpt  = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = get_model(NUM_CLASSES)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    print(f'Model on {next(model.parameters()).device}')
    if isinstance(ckpt, dict):
        print(f'Val IoU during training: {ckpt.get("val_iou", "N/A")}')

    cm_obj, scalar = evaluate(model, loader, device, visualize=VISUALIZE, max_vis_batches=2)
    print_results(cm_obj, scalar)

    if has_masks:
        plot_metrics(cm_obj, scalar, out_path=os.path.join(OUT_DIR, 'metrics.png'))

    os.makedirs(OUT_DIR, exist_ok=True)
    def sanitise(v):
        if isinstance(v, float) and np.isnan(v): return None
        if isinstance(v, dict): return {k: sanitise(vv) for k,vv in v.items()}
        return v
    out = {k: sanitise(v) for k,v in scalar.items()}
    out['confusion_matrix'] = cm_obj.cm.tolist()
    with open(os.path.join(OUT_DIR, 'evaluation.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Results -> {OUT_DIR}/evaluation.json')