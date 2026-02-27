# 🌿 Dataset Documentation — PlantAI (Jalgaon Regional Model)

## Summary

| Property            | Details                              |
|---------------------|--------------------------------------|
| **Crop Focus**      | Banana, Cotton, Brinjal, Maize       |
| **Region**          | Jalgaon, Maharashtra, India          |
| **Total Classes**   | 8 (including Healthy variants)       |
| **Image Resolution**| 224 × 224 pixels (resized)           |
| **Source**          | PlantVillage dataset (Kaggle)        |
| **Format**          | JPEG / PNG, RGB                      |

---

## Class Distribution

| Class | Description |
|-------|-------------|
| Banana_Healthy | Healthy banana leaf |
| Cotton_Leaf_Curl_Disease | Cotton Leaf Curl Virus (CLCuV) |
| Cotton_Healthy | Healthy cotton plant |
| Brinjal_Healthy | Healthy brinjal leaf |
| Brinjal_Diseased | Brinjal leaf with disease symptoms |
| Maize_Common_Rust | Caused by *Puccinia sorghi* |
| Maize_Healthy | Healthy maize leaf |
| Developing_Category | Unlabeled / edge cases |

---

## Dataset Split

| Split      | Usage                        |
|------------|------------------------------|
| `train/`   | Model training               |
| `val/`     | Accuracy evaluation & metrics |

---

## Preprocessing Pipeline

```python
transforms.Compose([
    transforms.Resize((224, 224)),         # Standardize resolution
    transforms.ToTensor(),                 # Convert to tensor [0, 1]
    transforms.Normalize(                  # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## Data Augmentation (Applied During Training)

To prevent overfitting and improve model generalization, the following augmentations were applied to training images:

| Technique | Effect |
|-----------|--------|
| Random Horizontal/Vertical Flip | Invariance to leaf orientation |
| Random Rotation (±30°) | Robustness to capture angle |
| Color Jitter (Brightness/Contrast) | Lighting variation resilience |
| Random Crop & Resize | Scale invariance |

> This expanded the effective dataset size and dramatically improved the model's ability to generalize to real-world field images taken by farmers.

---

## Source

- **Primary**: [PlantVillage Dataset — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Adaptation**: Filtered and re-labelled for Jalgaon-specific crops by the project team.
