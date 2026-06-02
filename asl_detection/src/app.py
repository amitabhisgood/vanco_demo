import streamlit as st
import cv2
import os
from ultralytics import YOLO
import numpy as np

# Set browser page layout configuration
st.set_page_config(page_title="ASL Real-Time Translator", layout="wide")

st.title("🤟 American Sign Language Real-Time Translator")
st.markdown("---")

# Sidebar - Settings Configuration panel
st.sidebar.header("Configuration Panel")
weights_path = os.path.join("runs", "detect", "asl_yolov8_model-4", "weights", "best.pt")
conf_threshold = st.sidebar.slider("Model Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

# Initialize tracking model state memory
@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_yolo_model(weights_path)

if model is None:
    st.error(f"Could not locate model weights at `{weights_path}`. Please verify training paths.")
else:
    st.sidebar.success("YOLOv8 Multi-Class Weights Loaded Successfully!")

    # Layout structure split for presentation canvas
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed Feed Stream")
        # Reactive dashboard control buttons
        start_button = st.button("▶ Start Translation Engine")
        stop_button = st.button("⏹ Stop Feed")
        
        # Place static image placeholder component matrix container
        frame_placeholder = st.empty()

    with col2:
        st.subheader("System Instructions")
        st.info(
            "1. Click the 'Start Translation Engine' button to activate webcam tracking.\n"
            "2. Ensure your hand is well-lit and fully within the camera view.\n"
            "3. Use gestures matching the alphabet letters A through H.\n"
            "4. Click 'Stop Feed' to clear resources."
        )

    # Core hardware capturing thread initialization loop execution logic
    if start_button:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Hardware streaming interface dropped frame segments.")
                break
                
            # Natural mirror rotation flip modification matrix conversion execution
            display_frame = cv2.flip(frame, 1)
            
            # Predict bounds directly onto frame channel variables
            results = model(display_frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Reformat array from BGR channel order layout tracking formats over to web-standard RGB standard format matching
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Stream the updated, processed matrix array instantly to the active web window layout block
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        frame_placeholder.empty()
        st.success("Webcam stream closed safely.")