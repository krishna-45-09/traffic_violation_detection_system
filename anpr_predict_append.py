# anpr_predict_append.py
# Usage: python anpr_predict_append.py /path/to/image.jpg

import os
import sys
import pandas as pd
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import easyocr
import cv2
import numpy as np

# ---------- Config ----------
MODEL_PATH = "best_classification.pth"    # your trained model (same as your existing script). :contentReference[oaicite:3]{index=3}
CSV_PATH = "predictions.csv"              # file to append results to (will be created if missing)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep classes in same order as training
classes = ['no_helmet', 'no_violation', 'overloading']   # adjust if your class ordering differs. :contentReference[oaicite:4]{index=4}

# ---------- Model load (same transforms as training/inference) ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ---------- EasyOCR reader ----------
# Note: language list can be extended (e.g., ['en'] or ['en','hi'] for India plates)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# ---------- Helper: simple plate localization (optional) ----------
def localize_plate_opencv(image_bgr):
    """
    Try to localize license plate region using contour heuristics.
    Returns ROI image (BGR) or None if not found.
    This is a simple heuristic; EasyOCR can read the whole image as fallback.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blur, 50, 200)
    # dilate to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    h_img, w_img = gray.shape
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        # candidate size heuristics
        if area < 2000 or area > 0.8*w_img*h_img: continue
        aspect = w / float(h + 1e-6)
        # plates usually have aspect ratio between ~2 and 6 (tweak if needed)
        if 2.0 < aspect < 6.5:
            candidates.append((x,y,w,h,area))

    if not candidates:
        return None

    # choose largest candidate area
    candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
    x,y,w,h,_ = candidates[0]
    margin = 6
    x0 = max(0, x-margin); y0 = max(0, y-margin)
    x1 = min(w_img, x+w+margin); y1 = min(h_img, y+h+margin)
    roi = image_bgr[y0:y1, x0:x1]
    return roi

# ---------- OCR helper ----------
def read_plate_from_image(image_path):
    """
    Returns best plate string (or empty string if nothing found) + confidence
    Strategy:
      1) try to localize plate with simple OpenCV heuristic and OCR the ROI
      2) if that fails or OCR low confidence, run OCR on whole image and pick best candidate
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "", 0.0

    roi = localize_plate_opencv(img_bgr)
    results = []
    if roi is not None:
        # convert to RGB for easyocr
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        ocr_res = ocr_reader.readtext(roi_rgb)
        for bbox, text, conf in ocr_res:
            # filter improbable short strings
            if len(text.strip()) >= 3:
                results.append((text.strip(), conf))

    # fallback: OCR entire image
    if not results:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ocr_res = ocr_reader.readtext(img_rgb)
        for bbox, text, conf in ocr_res:
            if len(text.strip()) >= 3:
                results.append((text.strip(), conf))

    if not results:
        return "", 0.0

    # choose top by confidence (and by length to prefer plate-like tokens)
    results = sorted(results, key=lambda x: (x[1], len(x[0])), reverse=True)
    best_text, best_conf = results[0]
    # coarse cleanup (remove unlikely punctuation)
    cleaned = "".join(ch for ch in best_text if ch.isalnum())
    return cleaned, float(best_conf)

# ---------- Violation prediction ----------
def predict_violation(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_class = classes[pred.item()]
    return predicted_class

# ---------- Main: combine and append ----------
def process_and_append(image_path):
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    violation = predict_violation(image_path)
    plate_text, plate_conf = read_plate_from_image(image_path)

    # Prepare row
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "image_path": image_path,
        "violation": violation,
        "plate_number": plate_text,
        "plate_confidence": plate_conf
    }

    df = pd.DataFrame([row])
    # create file with header if not exists
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, mode='w', header=True, index=False)

    print(f"Appended -> {CSV_PATH}: image={image_path} violation={violation} plate='{plate_text}' conf={plate_conf:.2f}")

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python anpr_predict_append.py <image_path>")
        sys.exit(1)
    image = sys.argv[1]
    process_and_append(image)
