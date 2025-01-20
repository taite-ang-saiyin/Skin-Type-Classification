from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import torch
from PIL import Image
import io
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Your label-index mappings
index_label = {0: "dry", 1: "normal", 2: "oily", 3: "combination", 4: "Damage"}

IMG_SIZE = 224
OUT_CLASSES = 5

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="best_resnet50_skin_type.pth"):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, OUT_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocess the image
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

@app.route('/detect_and_visualize', methods=['POST'])
def detect_and_visualize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        img_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = output.argmax(1).item()
            predicted_label = index_label[predicted_class]

        return jsonify({"detected_class": predicted_label, "skin_type": predicted_label})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
