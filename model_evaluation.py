"""
Model Evaluation Script — PlantAI (Jalgaon Regional Model)
============================================================
Evaluates the ResNet18-based Jalgaon disease model on the validation dataset.
Outputs: Accuracy, Precision, Recall, F1 Score, Confusion Matrix (PNG).

Usage:
    python3 model_evaluation.py
"""

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "jalgaon_project/jalgaon_disease_model.pt"
DATASET_DIR = "jalgaon_project/dataset/val"
OUTPUT_CM   = "static/confusion_matrix.png"
BATCH_SIZE  = 32
DEVICE      = torch.device("cpu")

print("=" * 60)
print("  PlantAI — Model Evaluation")
print("=" * 60)

# ─── Load Model ───────────────────────────────────────────────────────────────
print(f"\n[1/4] Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"  ERROR: Model not found at {MODEL_PATH}")
    exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = checkpoint["fc.weight"].shape[0]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()
print(f"  ✓ ResNet18 loaded | Classes: {num_classes}")

# ─── Dataset ──────────────────────────────────────────────────────────────────
print(f"\n[2/4] Loading validation dataset from: {DATASET_DIR}")
if not os.path.exists(DATASET_DIR):
    print(f"  ERROR: Dataset folder not found at {DATASET_DIR}")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset    = datasets.ImageFolder(DATASET_DIR, transform=transform)
loader     = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = dataset.classes
print(f"  ✓ {len(dataset)} images | {len(class_names)} classes")

# ─── Inference ────────────────────────────────────────────────────────────────
print(f"\n[3/4] Running inference …")
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(loader):
        images  = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        print(f"  Batch {batch_idx + 1}/{len(loader)} done", end="\r")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
print(f"\n  ✓ Inference complete on {len(all_labels)} samples")

# ─── Metrics ──────────────────────────────────────────────────────────────────
print(f"\n[4/4] Computing metrics …\n")
accuracy = accuracy_score(all_labels, all_preds)

print("─" * 60)
print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
print("─" * 60)
print("\nPer-Class Report (Precision / Recall / F1-Score):\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
print("Generating Confusion Matrix …")
cm = confusion_matrix(all_labels, all_preds)

# Shorten class names for readability
short_names = [c.replace("_", " ").replace("  ", " ") for c in class_names]

fig_size = max(10, len(class_names) * 1.2)
plt.figure(figsize=(fig_size, fig_size * 0.85))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=short_names,
    yticklabels=short_names,
    linewidths=0.5,
    linecolor="#cccccc"
)
plt.title("Confusion Matrix — Jalgaon Disease Model", fontsize=14, pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=150, bbox_inches="tight")
plt.close()

print(f"  ✓ Confusion matrix saved → {OUTPUT_CM}")
print("\n✅ Evaluation complete! Use confusion_matrix.png in your report.\n")
