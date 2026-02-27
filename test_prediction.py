import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import sys

# Add root to sys.path to import CNN if needed
sys.path.append(os.getcwd())

def test_prediction():
    MODEL_PATH = "jalgaon_project/jalgaon_disease_model.pt"
    DISEASE_CSV = 'jalgaon_project/jalgaon_disease_info.csv'
    
    # Load Model
    device = torch.device('cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test images
    test_dir = "jalgaon_project/dataset/val/Banana_Sigatoka"
    if not os.path.exists(test_dir):
        print(f"Skipping prediction test: {test_dir} not found")
        return

    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"No images found in {test_dir}")
        return

    img_path = os.path.join(test_dir, images[0])
    image = Image.open(img_path).convert('RGB')
    input_data = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = torch.max(probabilities).item() * 100
        index = torch.argmax(output).item()
        
    print(f"Image: {img_path}")
    print(f"Predicted Index: {index}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Check CSV
    disease_info = pd.read_csv(DISEASE_CSV)
    predicted_name = disease_info.iloc[index]['disease_name']
    print(f"Predicted Disease Name: {predicted_name}")
    
    if "Sigatoka" in predicted_name:
        print("SUCCESS: Model correctly identified Banana Sigatoka")
    else:
        print(f"FAILURE: Model predicted {predicted_name} for a Sigatoka image")

if __name__ == "__main__":
    test_prediction()
