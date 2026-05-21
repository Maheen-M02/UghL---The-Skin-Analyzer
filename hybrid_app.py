import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
import segmentation_models_pytorch as smp
from ultralytics import YOLO


yolo_model = YOLO("C:/Users/Maheen/Desktop/Skin_project/yolov8_trained_skinai4.pt")
unet_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
unet_model.load_state_dict(torch.load("C:/Users/Maheen/Desktop/Skin_project/unet_skin_segmentation.pth", map_location=torch.device('cpu')))
unet_model.eval()


product_df = pd.read_csv("C:/Users/Maheen/Desktop/Skin_project/indian_skincare_products_300_updated.csv")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])


app = tk.Tk()
app.title("Hybrid Skin Analyzer with Product Recommendation")
app.geometry("1000x700")

image_label = Label(app)
image_label.pack()

result_label = Label(app, text="", font=("Arial", 12), justify="left")
result_label.pack()

product_label = Label(app, text="", font=("Arial", 10), justify="left", fg="blue")
product_label.pack()


def analyze_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    display_img = Image.fromarray(img_rgb)
    display_img = display_img.resize((400, 400))
    img_tk = ImageTk.PhotoImage(display_img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    
    results = yolo_model(file_path)[0]
    detections = results.boxes.data.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    class_names = results.names

    concerns_detected = []
    for cid in class_ids:
        concern = class_names[int(cid)]
        concerns_detected.append(concern)

    
    unet_input = transform(Image.fromarray(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        mask_pred = unet_model(unet_input).squeeze().numpy()
    mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_pred, (img.shape[1], img.shape[0]))
    colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_rgb, 0.6, colored_mask, 0.4, 0)
    
    
    seg_image = Image.fromarray(blended)
    seg_image = seg_image.resize((400, 400))
    img_tk = ImageTk.PhotoImage(seg_image)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    
    unique_concerns = list(set(concerns_detected))
    result_text = "Detected Concerns:\n" + "\n".join(unique_concerns)
    result_label.configure(text=result_text)

    
    rec_text = "\nRecommended Products:\n"
    for concern in unique_concerns:
        matches = product_df[product_df['concern'].str.lower() == concern.lower()]
        for _, row in matches.iterrows():
            rec_text += f"\n{row['product_name']} - ₹{row['price']}\n{row['URL']}\n"
    product_label.configure(text=rec_text)

Button(app, text="Upload Image & Analyze", command=analyze_image, font=("Arial", 14)).pack(pady=20)

app.mainloop()
