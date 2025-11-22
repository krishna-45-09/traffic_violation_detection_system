# resnet18_openalpr_append.py  (Updated for your model)
import os, sys, math, subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import cv2
import re

# Try OpenALPR Python bindings
try:
    from openalpr import Alpr
    OPENALPR_AVAILABLE = True
except:
    OPENALPR_AVAILABLE = False

# PaddleOCR fallback
try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
    PADDLE_AVAILABLE = True
except:
    PADDLE_AVAILABLE = False
    paddle_ocr = None

# ==============================
#  YOUR EXACT MODEL PATH (NEW)
# ==============================
MODEL_PATH = "best_classification.pth"

CSV_PATH = "predictions.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same class order used during training
CLASSES = ['no_helmet', 'no_violation', 'overloading']

# Confidence thresholds
VIOLATION_CONF_THRESH = 0.45
PLATE_CONF_THRESH = 0.50

# Indian plate cleanup
IND_PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,2}\d{4}$')

# ------------------------------
# Load your ResNet18 model
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ------------------------------
# Violation Prediction (TTA)
# ------------------------------
def predict_violation_tta(img):
    scales = [1.0, 1.15]
    probs_list = []

    for s in scales:
        w, h = img.size
        new_img = img.resize((int(w*s), int(h*s)))

        for flip in [False, True]:
            temp = new_img.transpose(Image.FLIP_LEFT_RIGHT) if flip else new_img
            inp = transform(temp).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(inp)
                prob = F.softmax(out, dim=1).cpu().numpy()[0]
                probs_list.append(prob)

    avg = np.mean(probs_list, axis=0)
    pred_idx = int(np.argmax(avg))

    return CLASSES[pred_idx], float(avg[pred_idx])

# ------------------------------
# OpenALPR Plate Reading
# ------------------------------
def read_plate_openalpr(image_path):
    if OPENALPR_AVAILABLE:
        try:
            alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
            if not alpr.is_loaded():
                return "", 0.0

            results = alpr.recognize_file(image_path)
            alpr.unload()

            if results['results']:
                best = results['results'][0]
                plate = best['plate']
                conf = float(best['confidence']) / 100.0
                return plate, conf

        except:
            pass

    # CLI fallback
    try:
        out = subprocess.check_output(["alpr", "-c", "us", image_path])
        out = out.decode("utf-8")

        m = re.search(r'Plate:\s*([A-Za-z0-9]+)\s+Confidence:\s*([\d\.]+)', out)
        if m:
            return m.group(1), float(m.group(2)) / 100.0
    except:
        pass

    return "", 0.0

# ------------------------------
# PaddleOCR fallback
# ------------------------------
def read_plate_paddle(image_path):
    if not PADDLE_AVAILABLE:
        return "", 0.0

    img = cv2.imread(image_path)
    if img is None:
        return "", 0.0

    ocr_res = paddle_ocr.ocr(img, cls=True)
    best_text = ""
    best_conf = 0.0

    for block in ocr_res:
        for line in block:
            text, conf = line[1][0], line[1][1]
            cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper())
            if len(cleaned) >= 4 and conf > best_conf:
                best_text = cleaned
                best_conf = conf

    return best_text, best_conf

# ------------------------------
# Combined ANPR Logic
# ------------------------------
def get_plate_number(image_path):
    plate, conf = read_plate_openalpr(image_path)

    # If OpenALPR fails -> PaddleOCR fallback
    if conf < PLATE_CONF_THRESH:
        p_txt, p_conf = read_plate_paddle(image_path)
        if p_conf > conf:
            plate, conf = p_txt, p_conf

    plate = re.sub(r'[^A-Za-z0-9]', '', plate.upper())
    return plate, conf

# ------------------------------
# Append to CSV
# ------------------------------
def append_csv(row):
    df = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, mode='w', header=True, index=False)

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")

    # 1. Violation Detection
    viol, v_conf = predict_violation_tta(img)

    # 2. Plate Number
    plate, p_conf = get_plate_number(image_path)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "image_path": os.path.abspath(image_path),
        "violation": viol,
        "violation_confidence": v_conf,
        "plate_number": plate,
        "plate_confidence": p_conf
    }

    append_csv(row)

    print("APPENDED:", row)

# ------------------------------
# Run from CLI
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resnet18_openalpr_append.py image.jpg")
        sys.exit(1)

    process_image(sys.argv[1])
