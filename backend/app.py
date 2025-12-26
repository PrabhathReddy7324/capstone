import io
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Optional: PyTorch 2.6 "safe load" fix (won't break older versions)
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([DetectionModel])
    print("Registered DetectionModel as safe global for torch.load ✅")
except Exception as e:
    print("Safe globals patch not needed / not available:", e)

# -----------------------------
# CONFIG
# -----------------------------
# Path to your trained YOLO weights (detector)
YOLO_MODEL_PATH = "d:\\capstone\\Model\\best.pt"

# Path to your trained classifier checkpoint
# (from the classification training notebook)
CLASSIFIER_PATH = "D:\\capstone\\Model\\best_mar20_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# APP
# -----------------------------
app = Flask(__name__)
CORS(app)  # allow requests from frontend (different origin)

# -----------------------------
# Load YOLO model (detector)
# -----------------------------
detector = YOLO(YOLO_MODEL_PATH)
print("Loaded YOLO detector with classes:", detector.names)

# -----------------------------
# Load classifier model
# -----------------------------
def create_classifier(num_classes: int, model_type: str = "resnet50") -> nn.Module:
    model_type = model_type.lower()
    if model_type == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif model_type == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model

# Load checkpoint
ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
clf_num_classes = ckpt["num_classes"]
clf_model_type  = ckpt.get("model_type", "resnet50")
clf_class_names = ckpt.get("class_names", None)

if clf_class_names is None:
    # fallback: numeric names
    clf_class_names = [str(i) for i in range(clf_num_classes)]

# Map specific aircraft model names to high-level categories
# (examples: bomber, fighter, transport, tanker, awacs, patrol, attack)
TYPE_CATEGORY_MAP = {
    "B-1B": "bomber",
    "B-52": "bomber",
    "C-130": "transport",
    "C-17": "transport",
    "C-5": "transport",
    "E-3": "awacs",
    "E-8": "awacs",
    "P-3C": "patrol",
    "F-15": "fighter",
    "F-16": "fighter",
    "F-22": "fighter",
    "FA-18": "fighter",
    "SU-35": "fighter",
    "SU-34": "attack",
    "SU-24": "attack",
    "TU-160": "bomber",
    "TU-22": "bomber",
    "TU-95": "bomber",
    "KC-10": "tanker",
    "KC-135": "tanker",
}

def get_category_for_type(type_name: str) -> str:
    if not type_name:
        return "unknown"
    # try exact match first, then uppercase key
    if type_name in TYPE_CATEGORY_MAP:
        return TYPE_CATEGORY_MAP[type_name]
    t_up = type_name.upper()
    return TYPE_CATEGORY_MAP.get(t_up, "unknown")

classifier = create_classifier(clf_num_classes, clf_model_type).to(DEVICE)
classifier.load_state_dict(ckpt["model_state"])
classifier.eval()

print(f"Loaded classifier: {clf_model_type} with {clf_num_classes} classes")
print("Classifier classes:", clf_class_names)

# Preprocessing for classifier (same normalization as training, no augmentations)
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # must match training IMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -----------------------------
# Helper: classify a crop
# -----------------------------
def classify_crop(pil_img: Image.Image):
    """Run the classifier on a PIL image crop and return (class_id, class_name, confidence)."""
    tensor = clf_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, cls_idx = probs.max(1)
    cls_id = int(cls_idx.item())
    conf_val = float(conf.item())
    cls_name = clf_class_names[cls_id] if cls_id < len(clf_class_names) else str(cls_id)
    return cls_id, cls_name, conf_val

# -----------------------------
# Helper: run detection + classification
# -----------------------------
def run_detection_and_classification(image_bytes, conf_thres=0.25):
    # Convert raw bytes -> OpenCV image (BGR)
    npimg = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed to read image data")

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detection (boxes only)
    det_result = detector(img_bgr, conf=conf_thres)[0]

    detections = []
    annotated = img_bgr.copy()

    for box in det_result.boxes:
        # YOLO bbox (float)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = max(0, min(int(x2), W - 1))
        y2 = max(0, min(int(y2), H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop aircraft from RGB image for classifier
        crop_rgb = img_rgb[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop_rgb)

        cls_id, cls_name, cls_conf = classify_crop(pil_crop)

        # YOLO's own prediction (for reference)
        yolo_cls_id = int(box.cls[0].item())
        yolo_conf = float(box.conf[0].item())
        yolo_cls_name = detector.names[yolo_cls_id]

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": cls_conf,
            "category": get_category_for_type(cls_name),
            "detector_class_id": yolo_cls_id,
            "detector_class_name": yolo_cls_name,
            "detector_confidence": yolo_conf,
            "bbox": [x1, y1, x2, y2],
        })

        # Draw box + classifier label on annotated image
        label = f"{cls_name} {cls_conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Encode annotated image as base64
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return detections, encoded_image


def analyze_fleet_composition_by_category(detections):
    """Return composition counts and percentages by category and by class name."""
    by_type = {}
    by_category = {}
    total = len(detections)
    for d in detections:
        t = d.get("class_name", "unknown")
        c = d.get("category", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        by_category[c] = by_category.get(c, 0) + 1

    by_type_pct = {k: {"count": v, "percentage": round((v / total * 100), 2) if total>0 else 0} for k, v in by_type.items()}
    by_cat_pct = {k: {"count": v, "percentage": round((v / total * 100), 2) if total>0 else 0} for k, v in by_category.items()}
    return {"total": total, "by_type": by_type_pct, "by_category": by_cat_pct}

# -----------------------------
# Routes
# -----------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file part"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        image_bytes = file.read()
        detections, encoded_image = run_detection_and_classification(image_bytes)
        composition = analyze_fleet_composition_by_category(detections)

        return jsonify({
            "count": len(detections),
            "detections": detections,
            "annotated_image": encoded_image,
            "fleet_composition": composition,
        })
    except Exception as e:
        print("Error in /predict:", repr(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run backend on http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
