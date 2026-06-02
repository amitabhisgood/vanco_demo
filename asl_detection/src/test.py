import os
import cv2
from ultralytics import YOLO

def run_realtime_inference():
    """
    Runs real-time ASL gesture detection using optimized YOLOv8 weights via webcam.
    """
    # 1. Resolve absolute paths dynamically relative to script location
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # UPDATED: Now points to 'asl_yolov8_tuned' weights, NOT the old 'asl_yolov8_model'
    weights_path = os.path.abspath(os.path.join(src_dir, "..", "runs", "detect", "asl_yolov8_tuned", "weights", "best.pt"))
    
    if not os.path.exists(weights_path):
        print(f"[!] Error: Could not locate training weights at: {weights_path}")
        print(f"[!] Please ensure your 'tune.py' run saved the model to this location.")
        return

    print(f"[+] Loading custom ASL model weights from: {weights_path}")
    model = YOLO(weights_path)
    
    # 2. Bind to local web camera stream
    print("[-] Initializing camera feed...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows optimized framework
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)             # Generic/Unix fallback
        
    if not cap.isOpened():
        print("[!] Error: Unable to access the camera hardware stream.")
        return

    window_name = "ASL Real-Time Object Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("\n=== LIVE RECOGNITION STUDIO ACTIVE ===")
    print(" -> Hold up your sign gesture in front of the lens.")
    print(" -> Press 'Q' inside the camera window to exit safely.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Error: Frame capture failed mid-stream.")
            break

        # Mirror the screen naturally for an intuitive user experience
        display_frame = cv2.flip(frame, 1)
        
        # Run inference using a solid validation confidence threshold
        results = model(display_frame, conf=0.5, verbose=False)
        
        # Draw bounding boxes and text overlays onto the frame
        annotated_frame = results[0].plot()
        
        # Add UI context on top of the stream canvas
        cv2.putText(annotated_frame, "System Mode: Live Prediction (A-M)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, annotated_frame)
        
        # Optimized keypress handler (fixes frame lag)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()