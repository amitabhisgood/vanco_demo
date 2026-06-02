import cv2
import os
import time
import yaml  # Import yaml to handle configuration

def get_classes_from_yaml(yaml_path="dataset.yaml"):
    """Reads the names dictionary from the yaml file and returns a list of values."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return list(config['names'].values())

def find_working_camera():
    # ... (remains the same)
    for index in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    return None

def collect_asl_images():
    # Load classes dynamically from dataset.yaml
    classes = get_classes_from_yaml("dataset.yaml")
    base_dir = os.path.join("data", "raw")
    
    cap = find_working_camera()
    if cap is None:
        return

    print("\n=== ASL CUSTOM DATASET CAPTURE TOOL ===")
    print(f"Loaded {len(classes)} classes from dataset.yaml: {classes}")
    
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    window_name = "ASL Data Capture Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        display_frame = cv2.flip(frame, 1)
        
        # UI Overlays
        cv2.putText(display_frame, "ASL Studio - Active Capture Mode", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset = 60
        for cls in classes:
            class_path = os.path.join(base_dir, cls)
            count = len(os.listdir(class_path)) if os.path.exists(class_path) else 0
            color = (0, 255, 0) if count >= 20 else (0, 165, 255)
            cv2.putText(display_frame, f"Class {cls}: {count} images", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        
        # Check if the pressed key matches one of our dynamic classes
        pressed_char = chr(key).upper() if key < 256 else ""
        if pressed_char in classes:
            class_path = os.path.join(base_dir, pressed_char)
            timestamp = int(time.time() * 1000)
            cv2.imwrite(os.path.join(class_path, f"{pressed_char}_{timestamp}.jpg"), frame)
            print(f"[+] Saved snapshot for Class {pressed_char}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_asl_images()