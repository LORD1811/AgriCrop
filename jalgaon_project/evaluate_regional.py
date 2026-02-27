import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Configuration
MODEL_PATH = "jalgaon_project/jalgaon_disease_model.pt"
VAL_DIR = "jalgaon_project/dataset/val"

# 2. Load Dataset for class names
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(VAL_DIR):
    print(f"Error: Validation directory {VAL_DIR} not found.")
    exit(1)

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = val_dataset.classes
num_classes = len(class_names)

print(f"Detected {num_classes} classes: {class_names}")

# 3. Model Setup
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file {MODEL_PATH} not found.")
    exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 4. Evaluation
all_preds = []
all_labels = []

print("Evaluating...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
y_true = np.array(all_labels)
y_pred = np.array(all_preds)
accuracy = np.sum(y_true == y_pred) / len(y_true)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 5. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Regional Confusion Matrix (Jalgaon Crops)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save to static
os.makedirs('static', exist_ok=True)
plt.savefig('static/confusion_matrix.png')
print("Updated confusion matrix saved to static/confusion_matrix.png")

# 6. Detailed Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
