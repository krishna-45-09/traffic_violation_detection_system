import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import sys
import os

# 🔧 Your trained model path
MODEL_PATH = "best_classification.pth"

# 🔧 Classes must match your training order
classes = ['no_helmet', 'no_violation', 'overloading']

# Load model
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# CSV file path
CSV_PATH = "predictions.csv"

def predict_and_store(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = classes[pred.item()]

    # Store result
    new_result = pd.DataFrame([{"filename": os.path.basename(img_path),
                                "violation": predicted_class}])

    # Append if file exists
    if os.path.exists(CSV_PATH):
        new_result.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        new_result.to_csv(CSV_PATH, index=False)

    print(f"{os.path.basename(img_path)} → {predicted_class} (saved to {CSV_PATH})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_and_store.py <image_path>")
    else:
        predict_and_store(sys.argv[1])