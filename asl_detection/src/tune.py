import os
from ultralytics import YOLO

def tune_asl_model():
    print("=== INITIALIZING YOLO TRANSFER LEARNING (TUNING) ===")
    
    # Point to the specific 'best.pt' of your previous successful training
    # Adjust this path to wherever your best model for A-C is currently saved
    base_weights = os.path.join("runs", "detect", "asl_yolov8_model", "weights", "best.pt")
    
    if not os.path.exists(base_weights):
        print(f"[!] Error: Previous model not found at {base_weights}")
        return

    model = YOLO(base_weights)
    yaml_path = os.path.abspath("dataset.yaml")
    
    # Tuning usually requires fewer epochs than training from scratch
    model.train(
        data=yaml_path,
        epochs=30,      # Slightly longer to ensure new classes are learned well
        imgsz=640,
        batch=4,
        workers=2,
        name="asl_yolov8_tuned" # New folder to avoid overwriting previous results
    )

if __name__ == "__main__":
    tune_asl_model()