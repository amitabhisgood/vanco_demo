import os
from ultralytics import YOLO

def train_asl_model():
    print("=== INITIALIZING YOLO OBJECT DETECTION TRAINING ===")
    
    # 1. Load a pre-trained lightweight Nano model (ideal for quick CPU/GPU training)
    model = YOLO("yolov8n.pt")
    
    # 2. Define path to dataset.yaml configuration file
    yaml_path = os.path.abspath("dataset.yaml")
    
    print(f"[+] Loading dataset specifications from: {yaml_path}")
    
    # 3. Execute the training lifecycle
    # We use a small epoch value and standard image resolution for verification
    model.train(
        data=yaml_path,
        epochs=15,
        imgsz=640,
        batch=4,
        workers=2,
        name="asl_yolov8_model"
    )
    
    print("\n[+] Model training run completed successfully!")
    print("[+] Check 'runs/detect/asl_yolov8_model/weights/best.pt' for your trained network weights.")

if __name__ == "__main__":
    train_asl_model()