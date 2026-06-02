import cv2
import os
import random
import mediapipe as mp
import yaml  # Import yaml for dynamic configuration

def get_classes_from_yaml(yaml_path="dataset.yaml"):
    """Reads the names dictionary from the yaml file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['names']  # Returns the dict: {0: 'A', 1: 'B', ...}

def generate_auto_labels():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    # Load classes dynamically
    name_map = get_classes_from_yaml("dataset.yaml")
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, ".."))
    
    raw_base_dir = os.path.join(root_dir, "data", "raw")
    train_img_dir = os.path.join(root_dir, "data", "images", "train")
    train_lbl_dir = os.path.join(root_dir, "data", "labels", "train")
    val_img_dir = os.path.join(root_dir, "data", "images", "val")
    val_lbl_dir = os.path.join(root_dir, "data", "labels", "val")

    for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(path, exist_ok=True)
        
    print("=== STARTING AUTOMATED BOUNDING BOX GENERATION ===")
    
    # Iterate through the dynamic name_map
    for class_idx, cls in name_map.items():
        class_raw_dir = os.path.join(raw_base_dir, cls)
        if not os.path.exists(class_raw_dir):
            print(f"[!] Warning: Raw data folder missing for Class {cls}. Skipping...")
            continue
            
        images = [f for f in os.listdir(class_raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nProcessing Class {cls} (Index: {class_idx}, {len(images)} images found)...")
        
        random.seed(42)
        random.shuffle(images)
        
        split_point = int(len(images) * 0.8)
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(class_raw_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if idx < split_point:
                dest_img_dir, dest_lbl_dir = train_img_dir, train_lbl_dir
            else:
                dest_img_dir, dest_lbl_dir = val_img_dir, val_lbl_dir
                
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_max, y_max, x_min, y_min = 0, 0, w, h
                    
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_max, x_min = max(x_max, x), min(x_min, x)
                        y_max, y_min = max(y_max, y), min(y_min, y)
                    
                    padding = 20
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
                    
                    x_center = ((x_min + x_max) / 2.0) / w
                    y_center = ((y_min + y_max) / 2.0) / h
                    bbox_width = (x_max - x_min) / w
                    bbox_height = (y_max - y_min) / h
                    
                    base_name = os.path.splitext(img_name)[0]
                    label_file_path = os.path.join(dest_lbl_dir, f"{base_name}.txt")
                    
                    # Write label using class_idx from YAML
                    with open(label_file_path, "w") as f:
                        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    
                    cv2.imwrite(os.path.join(dest_img_dir, img_name), frame)
            else:
                print(f" [!] Hand not detected in image: {img_name} (Skipping)")
                
    hands.close()
    print("\n[+] Structured datasets generated inside data/images/ and data/labels/")

if __name__ == "__main__":
    generate_auto_labels()