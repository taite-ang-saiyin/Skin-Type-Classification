from ultralytics import YOLO
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

model = YOLO('runs/detect/train6/weights/best.pt')  # Path to your trained model weights

label_index = {"dry": 0, "normal": 1, "oily": 2, "combination": 3,"Damage":4}
index_label = {0: "dry", 1: "normal", 2: "oily", 3: "combination",4:"Damage"}

IMG_SIZE = 224
OUT_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="best_resnet50_skin_type.pth"):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, OUT_CLASSES)  
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(np.array(img)).unsqueeze(0)  # Add batch dimension
    return img


def predict_skin_type(model, image_path):
    img = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(img)
        predicted_class = output.argmax(1).item()
        predicted_label = index_label[predicted_class]
    return predicted_label
# Placeholder function for skin type classification
def classify_skin_type(image_path):
    
    model_path = "best_resnet50_skin_type.pth"  
    model = load_model(model_path)

    test_image = image_path 
    result = predict_skin_type(model, test_image)
    return result
# Perform inference and draw bounding boxes
@app.route('/detect_and_visualize', methods=['POST'])
def detect_and_visualize(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Perform inference
    results = model(image_path)
    
    # Check results
    if not results:
        print("No results returned from the model.")
        return

    # Access results
    detections = results[0].boxes

    # Extract bounding boxes from the 'data' attribute
    boxes = detections.data.cpu().numpy()
    class_names = results[0].names  # Dictionary mapping class indices to names

    # Iterate through all detections and print the detected class along with the bounding box
    for box in boxes:
        class_idx = int(box[5])  # Assuming that class index is in the 6th column (index 5)
        class_name = class_names[class_idx]  # Get the class name based on the index
        print(f"Detected class: {class_name}")
    # Perform skin type classification
    skin_type = classify_skin_type(image_path)
    print(f"Skin Type: {skin_type}")

if __name__ == "__main__":
    app.debug(True)
and send back the detected class and skin type