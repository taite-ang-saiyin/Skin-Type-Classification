import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import os

label_index = {"dry": 0, "normal": 1, "oily": 2, "combination": 3,"Damage":4}
index_label = {0: "dry", 1: "normal", 2: "oily", 3: "combination",4:"Damage"}


IMG_SIZE = 224
OUT_CLASSES = 4

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

if __name__ == '__main__':

    model_path = "best_resnet50_skin_type.pth"  
    model = load_model(model_path)

    test_image = "Trt.jpg" 

    result = predict_skin_type(model, test_image)
    print(f"Predicted skin type: {result}")
