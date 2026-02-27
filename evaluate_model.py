import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import CNN
import os
import torchvision.transforms.functional as TF

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load regional mapping from Jalgaon project
import pandas as pd
disease_info = pd.read_csv('jalgaon_project/jalgaon_disease_info.csv')
classes = disease_info['disease_name'].tolist()
num_classes = len(classes)

# Model setup
model = CNN.CNN(num_classes)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=device))
model.to(device)
model.eval()

# Transforms
def transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    return input_data

test_dir = 'test_images'

def get_class_from_filename(filename, classes):
    # Normalize filename: remove extension, lower case, replace underscores with spaces
    name = os.path.splitext(filename)[0].lower().replace('_', ' ')
    
    # Try to find a match in the classes
    # Classes are like 'Apple___Apple_scab'
    for i, cls in enumerate(classes):
        normalized_cls = cls.lower().replace('___', ' ').replace('_', ' ')
        if name in normalized_cls or normalized_cls in name:
            return i
    
    # Special cases for some filenames
    if 'soyaben' in name: return 25 # Soybean___healthy
    if 'starwberry' in name: 
        if 'scorch' in name: return 27
        return 28
    if 'pepper bacterial' in name: return 19 # Pepper,_bell___Bacterial_spot
    if 'apple ceder apple rust' in name: return 2
            
    return None

y_true = []
y_pred = []
images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Evaluating {len(images)} images...")

for img_name in images:
    label_idx = get_class_from_filename(img_name, classes)
    if label_idx is not None:
        try:
            img_path = os.path.join(test_dir, img_name)
            input_tensor = transform_image(img_path).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                
                y_true.append(label_idx)
                y_pred.append(predicted.item())
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

if y_true:
    # Accuracy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"Accuracy on mapped test images: {accuracy * 100:.2f}%")

    # Confusion Matrix
    # We only plot classes that are present in the test set to avoid a huge empty matrix
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm_classes = [classes[i] for i in unique_labels]
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=cm_classes, yticklabels=cm_classes)
    plt.title('Confusion Matrix (Test Images)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png')
    print("Confusion matrix saved to static/confusion_matrix.png")
    
    # Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=cm_classes))
else:
    print("No images could be automatically mapped to classes.")
