import json
import numpy as np
import matplotlib.pyplot as plt
import os

JSON_PATH = "./segmentation/runs/evaluation.json"   # <-- change to your file
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# LOAD JSON
# ======================================================

with open(JSON_PATH, "r") as f:
    data = json.load(f)

mean_iou = data["mean_iou"]
pixel_accuracy = data["pixel_accuracy"]
map50 = data["map50"]
latency = data["avg_latency_ms"]

per_class_iou = data["per_class_iou"]
per_class_precision = data["per_class_precision"]
per_class_recall = data["per_class_recall"]
per_class_f1 = data["per_class_f1"]
confusion_matrix = np.array(data["confusion_matrix"])

class_names = list(per_class_iou.keys())

# Replace null with 0
def safe_values(d):
    return [0 if d[k] is None else d[k] for k in d]

iou_vals = safe_values(per_class_iou)
precision_vals = safe_values(per_class_precision)
recall_vals = safe_values(per_class_recall)
f1_vals = safe_values(per_class_f1)

# ======================================================
# BAR PLOTS
# ======================================================

def plot_bar(values, title, filename, color):
    plt.figure(figsize=(10, 6))
    bars = plt.barh(class_names, values, color=color)
    plt.xlim(0, 1)

    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.title(title)
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()

plot_bar(iou_vals, "Per-Class IoU", "iou.png", "orange")
plot_bar(precision_vals, "Per-Class Precision", "precision.png", "seagreen")
plot_bar(recall_vals, "Per-Class Recall", "recall.png", "orange")
plot_bar(f1_vals, "Per-Class F1 Score", "f1.png", "purple")

# ======================================================
# CONFUSION MATRIX HEATMAP
# ======================================================

plt.figure(figsize=(12, 10))
# sns.heatmap(
#     confusion_matrix,
#     xticklabels=class_names,
#     yticklabels=class_names,
#     cmap="viridis",
#     norm=None
# )

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# ======================================================
# SUMMARY PANEL
# ======================================================

plt.figure(figsize=(8, 5))
plt.axis("off")

summary_text = f"""
Model Evaluation Summary

Mean IoU:        {mean_iou:.4f}
Pixel Accuracy:  {pixel_accuracy:.4f}
mAP50:           {map50:.4f}
Latency (ms):    {latency:.3f}
Images:          {data["num_images"]}
"""

plt.text(0.1, 0.5, summary_text, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "summary.png"), dpi=200)
plt.close()

print("Plots saved to:", OUTPUT_DIR)