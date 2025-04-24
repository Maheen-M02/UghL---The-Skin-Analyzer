import os
import requests
import torch
import pandas as pd
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# Dropbox download helper
def download_model(url, destination):
    if not os.path.exists(destination):
        print(f"📥 Downloading model to {destination}...")
        r = requests.get(url, stream=True)
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Download complete.")
    else:
        print("✅ Model already exists.")

# --- Download models from Dropbox ---
download_model("https://www.dropbox.com/scl/fi/047moeju1vga7otmzvjdm/yolov8_trained_skinai4.pt?rlkey=zdtptu9v8pudho6djgcbkzkbp&st=pg768mh1&raw=1", "models/yolov8_trained_skinai4.pt")
download_model("https://www.dropbox.com/scl/fi/yve6s2wjcrwu1d5q14byr/unet_skin_segmentation.pth?rlkey=rb0f7yizzn0qmooj17uqxi5rk&st=rr08yblk&raw=1", "models/unet_skin_segmentation.pth")

# --- Load models ---
print("📦 Loading models...")
yolo_model = YOLO("models/yolov8_trained_skinai4.pt")
unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
unet_model.load_state_dict(torch.load("models/unet_skin_segmentation.pth", map_location="cpu"))
unet_model.eval()

# --- Load product recommendation data ---
df = pd.read_csv("indian_skincare_products_300.csv")
SKIN_CLASSES = ["acne", "redness", "dry", "oily", "dark circles", "wrinkles"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # --- YOLO Detection ---
    yolo_results = yolo_model(file_path)
    detected_conditions = []
    for result in yolo_results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id < len(SKIN_CLASSES):
                detected_conditions.append(SKIN_CLASSES[class_id])
    most_common = max(set(detected_conditions), key=detected_conditions.count) if detected_conditions else "None"

    # --- Product Recommendation ---
    recommended = df[df["concern"].str.contains(most_common, case=False, na=False)]
    products = recommended[["product_name", "price", "url"]].head(5).to_dict(orient="records")

    return jsonify({
        "condition": most_common,
        "products": products,
        "image_path": file_path
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
