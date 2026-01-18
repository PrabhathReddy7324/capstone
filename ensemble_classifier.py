# ============================================================================
# ENSEMBLE CLASSIFIER - ResNet50 + ViT for Aircraft Classification
# ============================================================================
# Copy these cells into your final.ipynb notebook to replace the single-model
# classifier with an ensemble approach that combines both architectures.
# ============================================================================

# ============================================================================
# CELL 1: Configuration (replace existing config cell)
# ============================================================================
import os
from pathlib import Path
import yaml
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import patches
from ultralytics.nn.tasks import DetectionModel

# PyTorch >=2.6 "safe load" fix
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([DetectionModel])
    print("Registered DetectionModel as safe global for torch.load ✅")
except Exception as e:
    print("Safe globals not needed / not available:", e)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ----- PATHS -----
DATA_ROOT = Path("/kaggle/input/capstonev3")

ORIG_YAML  = DATA_ROOT / "data.yaml"
FIXED_YAML = Path("/kaggle/working/data_fixed.yaml")

# ENSEMBLE SETTINGS - Train BOTH models
EPOCHS       = 20           # increase to 40-60 for max accuracy
BATCH_SIZE   = 64
LR           = 1e-4
IMG_SIZE     = 224
NUM_WORKERS  = 4

# Checkpoints for both models
RESNET_CKPT = Path("/kaggle/working/best_resnet50.pth")
VIT_CKPT    = Path("/kaggle/working/best_vit.pth")

# Ensemble weights (can be tuned based on validation performance)
RESNET_WEIGHT = 0.5
VIT_WEIGHT    = 0.5

YOLO_EXP_NAME = "yolov12_mar20"

# ============================================================================
# CELL 2: Model Creation Functions (replace existing)
# ============================================================================
def create_resnet50(num_classes: int):
    """Create ResNet50 classifier with pretrained weights."""
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def create_vit(num_classes: int):
    """Create ViT-B/16 classifier with pretrained weights."""
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    in_feats = model.heads.head.in_features
    model.heads.head = nn.Linear(in_feats, num_classes)
    return model


def create_model(num_classes: int, model_type: str = "resnet50"):
    """Factory function for creating either model type."""
    model_type = model_type.lower()
    if model_type == "resnet50":
        return create_resnet50(num_classes)
    elif model_type == "vit_b_16":
        return create_vit(num_classes)
    else:
        raise ValueError("model_type must be 'resnet50' or 'vit_b_16'")


# ============================================================================
# CELL 3: Training Functions (same as before)
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_model(model, model_name, train_loader, valid_loader, ckpt_path, epochs=EPOCHS):
    """Train a single model and save best checkpoint."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "model_type": model_name,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> Saved new best {model_name} (val_acc={val_acc:.4f})")
    
    print(f"{model_name} best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


# ============================================================================
# CELL 4: Train BOTH Models (replace single model training)
# ============================================================================
# Build datasets from YOLO labels (YoloCropDataset defined earlier)
train_ds = YoloCropDataset(DATA_ROOT, split="train", img_size=IMG_SIZE, augment=True)
valid_ds = YoloCropDataset(DATA_ROOT, split="valid", img_size=IMG_SIZE, augment=False)
test_ds  = YoloCropDataset(DATA_ROOT, split="test",  img_size=IMG_SIZE, augment=False)

num_classes = train_ds.num_classes
print("Classifier num_classes:", num_classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# Train ResNet50
resnet_model = create_resnet50(num_classes)
resnet_val_acc = train_model(resnet_model, "resnet50", train_loader, valid_loader, RESNET_CKPT)

# Train ViT
vit_model = create_vit(num_classes)
vit_val_acc = train_model(vit_model, "vit_b_16", train_loader, valid_loader, VIT_CKPT)

# Calculate dynamic ensemble weights based on validation accuracy
total_acc = resnet_val_acc + vit_val_acc
RESNET_WEIGHT = resnet_val_acc / total_acc
VIT_WEIGHT = vit_val_acc / total_acc
print(f"\n{'='*60}")
print(f"ENSEMBLE WEIGHTS (based on validation accuracy):")
print(f"  ResNet50: {RESNET_WEIGHT:.3f}")
print(f"  ViT:      {VIT_WEIGHT:.3f}")
print(f"{'='*60}")


# ============================================================================
# CELL 5: Ensemble Classifier Class
# ============================================================================
class EnsembleClassifier:
    """
    Ensemble classifier combining ResNet50 and ViT predictions.
    
    Supports multiple ensemble strategies:
    - 'weighted_avg': Weighted average of softmax probabilities
    - 'max_conf': Use prediction from model with highest confidence
    - 'voting': Hard voting (each model votes for its predicted class)
    """
    
    def __init__(self, resnet_ckpt, vit_ckpt, device, 
                 resnet_weight=0.5, vit_weight=0.5, strategy='weighted_avg'):
        self.device = device
        self.resnet_weight = resnet_weight
        self.vit_weight = vit_weight
        self.strategy = strategy
        
        # Load ResNet50
        ckpt = torch.load(resnet_ckpt, map_location=device)
        self.num_classes = ckpt["num_classes"]
        self.resnet = create_resnet50(self.num_classes).to(device)
        self.resnet.load_state_dict(ckpt["model_state"])
        self.resnet.eval()
        print(f"Loaded ResNet50 (val_acc={ckpt.get('val_acc', 'N/A')})")
        
        # Load ViT
        ckpt = torch.load(vit_ckpt, map_location=device)
        self.vit = create_vit(self.num_classes).to(device)
        self.vit.load_state_dict(ckpt["model_state"])
        self.vit.eval()
        print(f"Loaded ViT (val_acc={ckpt.get('val_acc', 'N/A')})")
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    @torch.no_grad()
    def predict(self, pil_img):
        """
        Predict class for a PIL image using ensemble.
        
        Returns:
            cls_id: Predicted class ID
            confidence: Ensemble confidence score
            details: Dict with individual model predictions
        """
        img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Get predictions from both models
        resnet_out = self.resnet(img)
        vit_out = self.vit(img)
        
        resnet_probs = torch.softmax(resnet_out, dim=1)
        vit_probs = torch.softmax(vit_out, dim=1)
        
        resnet_conf, resnet_cls = resnet_probs.max(1)
        vit_conf, vit_cls = vit_probs.max(1)
        
        # Ensemble prediction based on strategy
        if self.strategy == 'weighted_avg':
            # Weighted average of probabilities
            ensemble_probs = (self.resnet_weight * resnet_probs + 
                            self.vit_weight * vit_probs)
            confidence, cls_id = ensemble_probs.max(1)
            
        elif self.strategy == 'max_conf':
            # Use model with highest confidence
            if resnet_conf > vit_conf:
                cls_id, confidence = resnet_cls, resnet_conf
            else:
                cls_id, confidence = vit_cls, vit_conf
                
        elif self.strategy == 'voting':
            # Hard voting (with tie-breaker by confidence)
            if resnet_cls == vit_cls:
                cls_id = resnet_cls
                confidence = (resnet_conf + vit_conf) / 2
            else:
                # Tie: use higher confidence prediction
                if resnet_conf > vit_conf:
                    cls_id, confidence = resnet_cls, resnet_conf
                else:
                    cls_id, confidence = vit_cls, vit_conf
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        details = {
            'resnet': {'class': resnet_cls.item(), 'conf': resnet_conf.item()},
            'vit': {'class': vit_cls.item(), 'conf': vit_conf.item()},
            'strategy': self.strategy,
        }
        
        return cls_id.item(), confidence.item(), details


# ============================================================================
# CELL 6: Evaluate Ensemble on Test Set
# ============================================================================
@torch.no_grad()
def evaluate_ensemble(ensemble, test_loader, class_names):
    """Evaluate ensemble classifier on test set."""
    correct = 0
    total = 0
    resnet_correct = 0
    vit_correct = 0
    
    for images, labels in test_loader:
        for i in range(images.size(0)):
            # Convert tensor to PIL for ensemble
            img_tensor = images[i]
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            pil_img = transforms.ToPILImage()(img_denorm)
            
            label = labels[i].item()
            cls_id, conf, details = ensemble.predict(pil_img)
            
            if cls_id == label:
                correct += 1
            if details['resnet']['class'] == label:
                resnet_correct += 1
            if details['vit']['class'] == label:
                vit_correct += 1
            total += 1
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS:")
    print(f"{'='*60}")
    print(f"ResNet50 alone:     {resnet_correct}/{total} = {100*resnet_correct/total:.2f}%")
    print(f"ViT alone:          {vit_correct}/{total} = {100*vit_correct/total:.2f}%")
    print(f"ENSEMBLE ({ensemble.strategy}): {correct}/{total} = {100*correct/total:.2f}%")
    print(f"{'='*60}")
    
    return correct / total


# Create ensemble and evaluate
ensemble = EnsembleClassifier(
    RESNET_CKPT, VIT_CKPT, DEVICE,
    resnet_weight=RESNET_WEIGHT,
    vit_weight=VIT_WEIGHT,
    strategy='weighted_avg'  # or 'max_conf' or 'voting'
)

ensemble_acc = evaluate_ensemble(ensemble, test_loader, CLASS_NAMES)


# ============================================================================
# CELL 7: Updated Pipeline with Ensemble
# ============================================================================
YOLO_BEST = Path(f"yolo_mar20/{YOLO_EXP_NAME}/weights/best.pt")
detector = YOLO(str(YOLO_BEST))


def run_ensemble_pipeline(image_path, show_details=True):
    """
    Run detection + ensemble classification pipeline.
    
    Args:
        image_path: Path to input image
        show_details: If True, show individual model predictions
    """
    img = Image.open(image_path).convert("RGB")
    res = detector(str(image_path))[0]
    
    draw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = img.crop((x1, y1, x2, y2))
        
        # Ensemble prediction
        cls_id, conf, details = ensemble.predict(crop)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        
        # YOLO prediction (for comparison)
        yolo_cls = int(box.cls[0].item())
        yolo_conf = float(box.conf[0].item())
        yolo_name = CLASS_NAMES[yolo_cls] if yolo_cls < len(CLASS_NAMES) else str(yolo_cls)
        
        if show_details:
            resnet_name = CLASS_NAMES[details['resnet']['class']]
            vit_name = CLASS_NAMES[details['vit']['class']]
            label = (f"ENS:{cls_name}({conf:.2f}) | "
                    f"R:{resnet_name}({details['resnet']['conf']:.2f}) | "
                    f"V:{vit_name}({details['vit']['conf']:.2f})")
        else:
            label = f"{cls_name} ({conf:.2f}) | YOLO:{yolo_name} ({yolo_conf:.2f})"
        
        # Color coding: green if ensemble agrees with YOLO, yellow otherwise
        color = (0, 255, 0) if cls_id == yolo_cls else (0, 255, 255)
        
        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(draw, label, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 10))
    plt.imshow(draw_rgb)
    plt.axis("off")
    plt.title("YOLOv12 + Ensemble (ResNet50 + ViT)")
    plt.show()


# ============================================================================
# CELL 8: Test the Pipeline
# ============================================================================
sample_img = next((DATA_ROOT / "test" / "images").glob("*"))
print("Sample image:", sample_img)
run_ensemble_pipeline(sample_img, show_details=True)

# Try different ensemble strategies
print("\nTesting different ensemble strategies...")
for strategy in ['weighted_avg', 'max_conf', 'voting']:
    ensemble.strategy = strategy
    print(f"\nStrategy: {strategy}")
    ensemble_acc = evaluate_ensemble(ensemble, test_loader, CLASS_NAMES)
