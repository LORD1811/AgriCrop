import os
import sqlite3
from datetime import datetime
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import numpy as np
import torch
import pandas as pd

# 1. Jalgaon Regional Data Configuration
MODEL_PATH = "jalgaon_project/jalgaon_disease_model.pt"
DISEASE_CSV = 'jalgaon_project/jalgaon_disease_info.csv'
SUPPLEMENT_CSV = 'jalgaon_project/jalgaon_supplement_info.csv'
MARKET_CSV = 'jalgaon_project/jalgaon_market_data.csv'
DB_PATH = 'predictions.db'

# Load Metadata
disease_info = pd.read_csv(DISEASE_CSV, encoding='utf-8')
disease_info.columns = disease_info.columns.str.strip()
supplement_info = pd.read_csv(SUPPLEMENT_CSV, encoding='utf-8')
supplement_info.columns = supplement_info.columns.str.strip()
market_data = pd.read_csv(MARKET_CSV, encoding='utf-8')
market_data.columns = market_data.columns.str.strip()

# 2. Dynamic Model Loading (ResNet18 for Jalgaon)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    if "conv1.weight" in checkpoint:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        out_features = checkpoint['fc.weight'].shape[0]
        model.fc = torch.nn.Linear(in_features, out_features)
        print(f"Jalgaon Regional Model loaded: {out_features} categories")
    else:
        import CNN
        out_features = checkpoint['dense_layers.4.weight'].shape[0] if 'dense_layers.4.weight' in checkpoint else 8
        model = CNN.CNN(out_features)
        print(f"Loaded fallback CNN architecture for Jalgaon")
    model.load_state_dict(checkpoint)
else:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)
    print("Warning: Regional model not found. Using placeholder architecture.")

model.eval()

# 3. SQLite Database Initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(disease_name, confidence, image_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO history (timestamp, disease_name, confidence, image_path) VALUES (?, ?, ?, ?)',
        (datetime.now().strftime('%d %b %Y, %I:%M %p'), disease_name, confidence, image_path)
    )
    conn.commit()
    conn.close()

init_db()

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    input_data = TF.to_tensor(image)
    input_data = normalize(input_data)
    input_data = input_data.unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = torch.max(probabilities).item() * 100
        index = torch.argmax(output).item()
    return index, int(confidence)

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image.save(file_path)

        pred, confidence = prediction(file_path)

        try:
            res = {
                'title': disease_info.iloc[pred]['disease_name'],
                'desc': disease_info.iloc[pred]['description'],
                'prevent': disease_info.iloc[pred]['Possible Steps'],
                'image_url': disease_info.iloc[pred]['image_url'],
                'sname': supplement_info.iloc[pred]['supplement name'],
                'simage': supplement_info.iloc[pred]['supplement image'],
                'buy_link': supplement_info.iloc[pred]['buy link']
            }
        except (IndexError, KeyError):
            res = {
                'title': "Developing Category",
                'desc': "We're currently expanding our data for this specific crop variety.",
                'prevent': "Consult with local agriculture experts for optimized care.",
                'image_url': "https://images.unsplash.com/photo-1594246001306-25f385c721ae",
                'sname': "Universal Nutrient Booster",
                'simage': "assets/supplements/bio_fertilizer.png",
                'buy_link': "https://www.agrostar.in/agronomy-solutions"
            }

        # Log prediction to SQLite
        log_prediction(res['title'], confidence, '/' + file_path)

        return render_template('submit.html', **res, pred=pred, confidence=confidence, user_image='/' + file_path)

@app.route('/market')
def market():
    chart_data = market_data.groupby('Commodity')['Modal_Price'].mean().to_dict()
    return render_template('market.html',
                           market_rows=market_data.to_dict('records'),
                           chart_labels=list(chart_data.keys()),
                           chart_values=list(chart_data.values()),
                           supplements=supplement_info.to_dict('records'))

@app.route('/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY id DESC LIMIT 50')
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=rows)

if __name__ == '__main__':
    app.run(debug=True, port=8081)
