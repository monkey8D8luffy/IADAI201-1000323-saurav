from ultralytics import YOLO

def train_model():
    # Use YOLOv8-Small for stronger feature extraction capacity than Nano
    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        patience=12,          # Stop early before background memorization peaks
        imgsz=640,
        batch=32,
        device=0,
        workers=4,
        
        # Transfer Learning & Anti-Overfitting
        freeze=10,            # CRITICAL: Freeze backbone so it doesn't forget general cars
        optimizer="AdamW",
        lr0=0.001,            # Lower learning rate to preserve pre-trained features
        cos_lr=True,
        weight_decay=0.001,
        dropout=0.2,          # Randomly drop neurons to prevent pixel memorization
        
        # Geometric & Color Augmentations
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=20.0,
        translate=0.1,
        scale=0.6,
        perspective=0.0005,   # Simulate slight camera angle changes
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.15,
    )

if __name__ == "__main__":
    train_model()