# 🌾 AgriCrop — AI Leaf Disease Detection for Jalgaon Farmers

> A deep learning-powered web application that identifies crop diseases from leaf photographs and recommends targeted treatments — built specifically for the crops and conditions of **Jalgaon, Maharashtra**.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🤖 **AI Diagnosis** | ResNet18 model trained on 8 Jalgaon-specific crop disease classes |
| 🛡️ **Smart Rejection** | Rejects non-leaf / unsupported crop images instead of giving wrong output |
| 📊 **Confidence Score** | Shows neural network certainty (%) for every prediction |
| 🗃️ **Prediction History** | SQLite-powered history log of all past scans |
| 💊 **Treatment Sync** | Recommends locally available supplements and fertilizers |
| 📈 **Market Prices** | Live Jalgaon mandi price data for supported crops |
| 🌙 **Dark / Light Mode** | Premium glassmorphism UI with theme toggle |

---

## 🌿 Supported Crops & Diseases

| Crop | Healthy | Disease Detected |
|---|---|---|
| 🍌 Banana | ✅ | Sigatoka (*Mycosphaerella fijiensis*) |
| 🍆 Brinjal | ✅ | Cercospora Leaf Spot |
| 🌿 Cotton | ✅ | Bacterial Blight (*Xanthomonas axonopodis*) |
| 🌽 Maize | ✅ | Common Rust (*Puccinia sorghi*) |

> ⚠️ **Note:** Images of any other crop (tomato, wheat, mango, rice etc.) will be correctly identified as **"Unrecognized Leaf"** and rejected — no wrong disease will be shown.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.8+, Flask |
| Deep Learning | PyTorch, torchvision (ResNet18) |
| Image Processing | Pillow (PIL) |
| Database | SQLite3 |
| Data Handling | Pandas, NumPy |
| Frontend | HTML5, Bootstrap 5, Vanilla CSS (glassmorphism) |

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd AgriCrop
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Open your browser at: **[http://127.0.0.1:8081](http://127.0.0.1:8081)**

---

## 📁 Project Structure

```
AgriCrop/
│
├── app.py                          # Main Flask application
├── CNN.py                          # Fallback CNN architecture
├── requirements.txt                # Python dependencies
│
├── jalgaon_project/
│   ├── jalgaon_disease_model.pt    # Trained ResNet18 model
│   ├── jalgaon_disease_info.csv    # Disease descriptions & steps
│   ├── jalgaon_supplement_info.csv # Supplement recommendations
│   ├── jalgaon_market_data.csv     # Mandi price data
│   ├── dataset/                    # Training & validation images
│   └── Jalgaon_Crop_Training.ipynb # Model training notebook
│
├── templates/
│   ├── base.html                   # Base layout (navbar, theme)
│   ├── home.html                   # Landing page
│   ├── index.html                  # Image upload page
│   ├── submit.html                 # Diagnosis result page
│   ├── market.html                 # Market price dashboard
│   ├── history.html                # Prediction history
│   └── contact-us.html             # Contact page
│
├── static/
│   ├── uploads/                    # Uploaded leaf images (runtime)
│   └── supplements/                # Supplement product images
│
├── Model/                          # Original model files & notebook
├── test_images/                    # Sample leaf images for testing
├── predictions.db                  # SQLite prediction history
├── DATASET.md                      # Full dataset documentation
├── model_evaluation.py             # Evaluation & confusion matrix script
└── evaluate_model.py               # Quick accuracy evaluator
```

---

## 🤖 How the AI Works

```
User uploads leaf image
        │
        ▼
  Resize to 224×224
        │
        ▼
  ImageNet Normalization
        │
        ▼
  ResNet18 → Softmax → Confidence %
        │
    ┌───┴────────────┐
    │                │
Confidence < 55%   Confidence ≥ 55%
    │                │
"Unrecognized      Map index → Disease Name
  Leaf" warning     + Treatment + Supplement
```

### Confidence Threshold
The model uses a **55% confidence threshold**. Any prediction below this is rejected and shown as **"Unrecognized Leaf"** — preventing false diagnoses when:
- A leaf from an unsupported crop is uploaded (tomato, rice, mango, etc.)
- A non-leaf image is uploaded
- The photo is blurry or poorly lit

---

## 📊 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet18 (Transfer Learning) |
| Input Size | 224 × 224 × 3 (RGB) |
| Output Classes | 8 |
| Confidence Threshold | 55% |
| Training Framework | PyTorch |
| Pre-trained Weights | ImageNet (fine-tuned on Jalgaon dataset) |

---

## 🧪 Testing with Sample Images

Sample leaf images are provided in the `test_images/` folder. Upload these via the AI Engine page to verify the model is working correctly:

```
test_images/
├── banana_healthy.jpg
├── banana_sigatoka.jpg
├── brinjal_healthy.jpg
├── brinjal_leafspot.jpg
├── cotton_blight.jpg
├── cotton_healthy.jpg
├── maize_rust.jpg
└── maize_healthy.jpg
```

---

## 📈 Model Evaluation

Run the evaluation script to see accuracy, F1 score, and confusion matrix:

```bash
python model_evaluation.py
```

This generates precision, recall, F1-score per class and saves a confusion matrix plot.

---

## 🗃️ Prediction History

All scans are automatically logged to `predictions.db` (SQLite).  
View the history at: **[http://127.0.0.1:8081/history](http://127.0.0.1:8081/history)**

| Column | Description |
|---|---|
| Timestamp | Date and time of scan |
| Disease Detected | Name of the detected class |
| Confidence | Model certainty (%) |
| Image | Thumbnail of uploaded leaf |

---

## 🌐 Application Pages

| Route | Page |
|---|---|
| `/` | Home / Landing page |
| `/index` | Upload leaf image for AI analysis |
| `/submit` | Diagnosis result with treatment plan |
| `/market` | Jalgaon crop market prices |
| `/history` | Past prediction history |
| `/contact` | Contact information |

---

## 📋 Dataset

See **[DATASET.md](DATASET.md)** for full documentation including:
- Class distribution
- Preprocessing pipeline
- Data augmentation techniques
- Dataset source and credits

**Dataset Source:** [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
**Adaptation:** Filtered and re-labelled for Jalgaon-specific crops and conditions.

---

## 👨‍💻 Developed By

**AgriCrop Team** — Final Year Project  
Region Focus: Jalgaon, Maharashtra 🇮🇳  

---

## 📄 License

This project is for academic and educational purposes.
