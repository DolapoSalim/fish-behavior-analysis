import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import json
import os
from datetime import datetime
import time
import tempfile

# Page config
st.set_page_config(
    page_title="Fish Tracker",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Fish Tracker - Video Processing")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model upload
    model_file = st.file_uploader("Upload YOLO Model", type=['pt'])
    
    # Video upload
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    # Parameters
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    
    # Output options
    st.header("Output Options")
    save_video = st.checkbox("Save Annotated Video", value=True)
    save_data = st.checkbox("Save Tracking Data (JSON)", value=True)
    save_frames = st.checkbox("Save Individual Frames", value=False)

def process_video_streamlit(model_file, video_file, confidence_threshold, save_video, save_data, save_frames):
    """Process video with Streamlit interface"""
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
        tmp_model.write(model_file.read())
        model_path = tmp_model.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name
    
    # Progress containers
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_text = st.empty()
    
    # Results container
    results_container = st.container()
    
    try:
        # Initialize model and tracker
        status_text.text("Loading YOLO model...")
        model = YOLO(model_path)
        model.fuse()
        
        status_text.text("Initializing tracker...")
        tracker = DeepSort(max_age=30)
        
        # Open video
        status_text.text("Opening video file...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with results_container:
            st.success(f"Video loaded: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Setup output directory
        output_dir = tempfile.mkdtemp()
        
        # Video writer
        video_writer = None
        if save_video:
            output_video_path = os.path.join(output_dir, f"tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Data storage
        tracking_data = []
        prev_centroids = {}
        
        current_frame = 0
        start_time = time.time()
        
        # Process video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            
            # Update progress
            progress = current_frame / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {current_frame}/{total_frames} ({progress*100:.1f}%)")
            frame_text.text(f"Current frame: {current_frame}")
            
            # Run inference
            results = model(frame, conf=confidence_threshold, verbose=False)[0]
            
            detections = []
            frame_data = {
                'frame': current_frame,
                'timestamp': current_frame / fps,
                'detections': []
            }
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box
                    detections.append(([x1, y1, x2, y2], conf, int(cls)))
            
            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                cx, cy = int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2)
                
                # Draw tracking results
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)