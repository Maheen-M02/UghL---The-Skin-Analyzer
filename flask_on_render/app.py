# app.py
from flask import Flask, request, render_template, jsonify
import torch
from ultralytics import YOLO
import pandas as pd
import cv2
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
yolo_model = YOLO("C:/Users/Maheen/Desktop/Skin_project/yolov8_trained_skinai4.pt")

# Load U-Net model
unet_model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=6)
unet_model.load_state_dict(torch.load("C:/Users/Maheen/Desktop/Skin_project/unet_skin_segmentation.pth", map_location=torch.device('cpu')))
unet_model.eval()

# Load product data
df = pd.read_csv("C:/Users/Maheen/Desktop/Skin_project/indian_skincare_products_300_updated.csv")

SKIN_CONCERNS = ["acne", "wrinkles", "dark circles", "redness", "dry", "oily"]

def recommend_products(condition):
    filtered = df[df['concern'].str.contains(condition, case=False, na=False)]
    return filtered[['product_name', 'price', 'url']].head(5).to_dict(orient='records')

@app.route('/')
def index():
    return render_template("C:/Users/Maheen/Desktop/flask_on_render/templates/index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath).convert('RGB')
    image_np = np.array(image)

    # YOLO detection
    yolo_results = yolo_model(image_np)
    detected_classes = set()
    for r in yolo_results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id < len(SKIN_CONCERNS):
                detected_classes.add(SKIN_CONCERNS[cls_id])

    # U-Net segmentation
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(),
        ToTensorV2()
    ])
    transformed = transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0)
    with torch.no_grad():
        output = unet_model(input_tensor)[0].detach().numpy()

    # Combine detections
    all_conditions = list(detected_classes)
    all_recommendations = {}
    for cond in all_conditions:
        all_recommendations[cond] = recommend_products(cond)

    return jsonify({
        "conditions": all_conditions,
        "recommendations": all_recommendations,
        "image_path": filepath
    })

if __name__ == '__main__':
    app.run(debug=True)
