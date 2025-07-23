import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import json
import os
import tempfile
import time
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import threading
import queue

# Configure Streamlit page
st.set_page_config(
    page_title="Fish Tracker Pro",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitFishTracker:
    def __init__(self):
        self.model = None
        self.tracker = None
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
    def load_model(self, model_path):
        """Load YOLO model with caching"""
        try:
            if self.model is None or getattr(self, 'current_model_path', '') != model_path:
                with st.spinner("Loading YOLO model..."):
                    self.model = YOLO(model_path)
                    self.model.fuse()
                    self.current_model_path = model_path
                st.success("✅ Model loaded successfully!")
                return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
        return True
    
    def process_video(self, video_path, output_dir, settings, progress_callback=None):
        """Process video with tracking"""
        try:
            # Initialize tracker
            self.tracker = DeepSort(max_age=settings['max_age'])
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output video writer
            output_video = None
            if settings['save_video']:
                output_video_path = os.path.join(output_dir, "tracked_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Data storage
            tracking_data = {
                'video_info': {
                    'fps': fps, 
                    'total_frames': total_frames, 
                    'width': width, 
                    'height': height,
                    'processed_at': datetime.now().isoformat()
                },
                'tracks': [],
                'frame_summary': []
            }
            
            prev_centroids = {}
            frame_count = 0
            processed_frames = 0
            total_detections = 0
            unique_tracks = set()
            
            start_time = time.time()
            
            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress, frame_count, total_frames)
                
                # Process every N frames
                if frame_count % settings['frame_skip'] == 0:
                    # Run inference
                    results = self.model(frame, 
                                       conf=settings['conf_threshold'],
                                       iou=settings['iou_threshold'],
                                       verbose=False)[0]
                    
                    detections = []
                    frame_detections = 0
                    
                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        confidences = results.boxes.conf.cpu().numpy()
                        classes = results.boxes.cls.cpu().numpy()
                        
                        frame_detections = len(boxes)
                        total_detections += frame_detections
                        
                        for box, conf, cls in zip(boxes, confidences, classes):
                            x1, y1, x2, y2 = box
                            detections.append(([x1, y1, x2, y2], conf, int(cls)))
                    
                    # Update tracker
                    tracks = self.tracker.update_tracks(detections, frame=frame)
                    
                    # Process tracks
                    frame_tracks = []
                    active_tracks = 0
                    
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        
                        active_tracks += 1
                        track_id = track.track_id
                        unique_tracks.add(track_id)
                        ltrb = track.to_ltrb()
                        cx, cy = int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2)
                        
                        # Draw annotations
                        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), 
                                    (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", 
                                  (int(ltrb[0]), int(ltrb[1])-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Calculate direction and speed
                        direction = None
                        speed = 0
                        distance_moved = 0
                        
                        if track_id in prev_centroids:
                            prev_cx, prev_cy = prev_centroids[track_id]
                            dx, dy = cx - prev_cx, cy - prev_cy
                            distance_moved = math.sqrt(dx*dx + dy*dy)
                            
                            # Calculate speed (pixels per frame, could convert to real units)
                            speed = distance_moved * settings['frame_skip']
                            
                            if distance_moved > settings['min_movement']:
                                angle_rad = math.atan2(dy, dx)
                                direction = (math.degrees(angle_rad) + 360) % 360
                                
                                # Draw direction arrow
                                cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (0, 255, 0), 2)
                                cv2.putText(frame, f"{int(direction)}° {speed:.1f}px/s", 
                                          (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        prev_centroids[track_id] = (cx, cy)
                        
                        # Store tracking data
                        frame_tracks.append({
                            'track_id': track_id,
                            'bbox': [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                            'centroid': [cx, cy],
                            'direction': direction,
                            'speed': speed,
                            'distance_moved': distance_moved
                        })
                    
                    # Store frame data
                    tracking_data['tracks'].append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'objects': frame_tracks
                    })
                    
                    # Store frame summary for analytics
                    tracking_data['frame_summary'].append({
                        'frame': frame_count,
                        'detections': frame_detections,
                        'active_tracks': active_tracks,
                        'timestamp': frame_count / fps
                    })
                    
                    processed_frames += 1
                
                # Save frame to output video
                if output_video is not None:
                    output_video.write(frame)
                
                frame_count += 1
            
            # Cleanup
            cap.release()
            if output_video is not None:
                output_video.release()
            
            processing_time = time.time() - start_time
            
            # Calculate analytics
            analytics = self.calculate_analytics(tracking_data, processing_time)
            tracking_data['analytics'] = analytics
            
            # Save data
            if settings['save_data']:
                data_path = os.path.join(output_dir, "tracking_data.json")
                with open(data_path, 'w') as f:
                    json.dump(tracking_data, f, indent=2)
            
            return tracking_data, analytics
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            return None, None
    
    def calculate_analytics(self, tracking_data, processing_time):
        """Calculate analytics from tracking data"""
        frame_summary = tracking_data['frame_summary']
        all_tracks = tracking_data['tracks']
        
        # Basic stats
        total_frames = len(frame_summary)
        total_detections = sum(f['detections'] for f in frame_summary)
        unique_tracks = set()
        all_speeds = []
        all_directions = []
        
        for frame_data in all_tracks:
            for obj in frame_data['objects']:
                unique_tracks.add(obj['track_id'])
                if obj['speed'] > 0:
                    all_speeds.append(obj['speed'])
                if obj['direction'] is not None:
                    all_directions.append(obj['direction'])
        
        analytics = {
            'processing_time': processing_time,
            'total_frames_processed': total_frames,
            'total_detections': total_detections,
            'unique_fish_count': len(unique_tracks),
            'average_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'average_processing_fps': total_frames / processing_time if processing_time > 0 else 0,
            'speed_stats': {
                'mean_speed': np.mean(all_speeds) if all_speeds else 0,
                'max_speed': max(all_speeds) if all_speeds else 0,
                'min_speed': min(all_speeds) if all_speeds else 0
            },
            'direction_distribution': self.calculate_direction_distribution(all_directions)
        }
        
        return analytics
    
    def calculate_direction_distribution(self, directions):
        """Calculate direction distribution in 8 compass directions"""
        if not directions:
            return {}
        
        direction_bins = {
            'North': [337.5, 22.5],
            'Northeast': [22.5, 67.5],
            'East': [67.5, 112.5],
            'Southeast': [112.5, 157.5],
            'South': [157.5, 202.5],
            'Southwest': [202.5, 247.5],
            'West': [247.5, 292.5],
            'Northwest': [292.5, 337.5]
        }
        
        distribution = {direction: 0 for direction in direction_bins.keys()}
        
        for angle in directions:
            for direction, (start, end) in direction_bins.items():
                if direction == 'North':
                    if angle >= start or angle < end:
                        distribution[direction] += 1
                        break
                else:
                    if start <= angle < end:
                        distribution[direction] += 1
                        break
        
        # Convert to percentages
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: (v/total)*100 for k, v in distribution.items()}
        
        return distribution

def main():
    # Header
    st.markdown('<h1 class="main-header">🐟 Fish Tracker Pro</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced fish tracking and analysis with YOLO + DeepSORT**")
    
    # Initialize tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = StreamlitFishTracker()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("🧠 Model Settings")
    model_file = st.sidebar.file_uploader("Upload YOLO Model (.pt)", type=['pt'])
    
    if model_file is not None:
        # Save uploaded model temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(model_file.read())
            model_path = tmp_file.name
        
        if st.session_state.tracker.load_model(model_path):
            st.sidebar.success("✅ Model loaded!")
    
    # Video upload
    st.sidebar.subheader("🎥 Video Input")
    video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    # Processing settings
    st.sidebar.subheader("🔧 Processing Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
        frame_skip = st.slider("Process Every N Frames", 1, 10, 1)
    
    with col2:
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.4, 0.05)
        max_age = st.slider("Track Max Age", 10, 100, 30, 5)
    
    min_movement = st.sidebar.slider("Min Movement (pixels)", 1, 20, 5)
    
    # Output settings
    st.sidebar.subheader("💾 Output Settings")
    save_video = st.sidebar.checkbox("Save Annotated Video", True)
    save_data = st.sidebar.checkbox("Save Tracking Data", True)
    
    # Main content area
    if video_file is not None and model_file is not None:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🚀 Processing", "📊 Analytics", "📈 Visualizations"])
        
        with tab1:
            st.header("Video Processing")
            
            # Processing button
            if st.button("🎬 Start Processing", type="primary", use_container_width=True):
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(video_file.read())
                    video_path = tmp_video.name
                
                # Create output directory
                output_dir = tempfile.mkdtemp()
                
                # Processing settings
                settings = {
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'frame_skip': frame_skip,
                    'max_age': max_age,
                    'min_movement': min_movement,
                    'save_video': save_video,
                    'save_data': save_data
                }
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()
                
                def progress_callback(progress, current_frame, total_frames):
                    progress_bar.progress(progress / 100)
                    status_text.text(f"Processing frame {current_frame:,} of {total_frames:,} ({progress:.1f}%)")
                
                # Start processing
                st.session_state.tracker.is_processing = True
                
                with st.spinner("Processing video..."):
                    tracking_data, analytics = st.session_state.tracker.process_video(
                        video_path, output_dir, settings, progress_callback
                    )
                
                if tracking_data and analytics:
                    st.success("✅ Processing completed successfully!")
                    
                    # Display metrics
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("🐟 Fish Detected", analytics['unique_fish_count'])
                        with col2:
                            st.metric("⏱️ Processing Time", f"{analytics['processing_time']:.2f}s")
                        with col3:
                            st.metric("🎯 Total Detections", analytics['total_detections'])
                        with col4:
                            st.metric("📊 Avg FPS", f"{analytics['average_processing_fps']:.2f}")
                    
                    # Store results in session state
                    st.session_state.tracking_data = tracking_data
                    st.session_state.analytics = analytics
                    st.session_state.output_dir = output_dir
                    
                    # Download buttons
                    st.subheader("📥 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    if save_data:
                        with col1:
                            data_path = os.path.join(output_dir, "tracking_data.json")
                            if os.path.exists(data_path):
                                with open(data_path, 'rb') as f:
                                    st.download_button(
                                        "📄 Download Tracking Data (JSON)",
                                        f.read(),
                                        "tracking_data.json",
                                        "application/json"
                                    )
                    
                    if save_video:
                        with col2:
                            video_path = os.path.join(output_dir, "tracked_video.mp4")
                            if os.path.exists(video_path):
                                with open(video_path, 'rb') as f:
                                    st.download_button(
                                        "🎥 Download Annotated Video",
                                        f.read(),
                                        "tracked_video.mp4",
                                        "video/mp4"
                                    )
        
        with tab2:
            st.header("📊 Analytics Dashboard")
            
            if 'analytics' in st.session_state:
                analytics = st.session_state.analytics
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🐟 Unique Fish", analytics['unique_fish_count'])
                    st.metric("⚡ Avg Speed", f"{analytics['speed_stats']['mean_speed']:.2f} px/s")
                
                with col2:
                    st.metric("🎯 Total Detections", analytics['total_detections'])
                    st.metric("🏃 Max Speed", f"{analytics['speed_stats']['max_speed']:.2f} px/s")
                
                with col3:
                    st.metric("📊 Avg Detections/Frame", f"{analytics['average_detections_per_frame']:.2f}")
                    st.metric("⏱️ Processing FPS", f"{analytics['average_processing_fps']:.2f}")
                
                # Direction distribution
                if analytics['direction_distribution']:
                    st.subheader("🧭 Movement Direction Distribution")
                    direction_df = pd.DataFrame(
                        list(analytics['direction_distribution'].items()),
                        columns=['Direction', 'Percentage']
                    )
                    
                    fig_pie = px.pie(direction_df, values='Percentage', names='Direction',
                                   title="Fish Movement Directions")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            else:
                st.info("Process a video first to see analytics.")
        
        with tab3:
            st.header("📈 Data Visualizations")
            
            if 'tracking_data' in st.session_state:
                tracking_data = st.session_state.tracking_data
                
                # Create timeline data
                frame_summary = tracking_data['frame_summary']
                timeline_df = pd.DataFrame(frame_summary)
                
                if not timeline_df.empty:
                    # Detections over time
                    fig_timeline = px.line(timeline_df, x='timestamp', y='detections',
                                         title='Fish Detections Over Time',
                                         labels={'timestamp': 'Time (seconds)', 'detections': 'Number of Fish'})
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Active tracks over time
                    fig_tracks = px.line(timeline_df, x='timestamp', y='active_tracks',
                                       title='Active Tracks Over Time',
                                       labels={'timestamp': 'Time (seconds)', 'active_tracks': 'Active Tracks'})
                    st.plotly_chart(fig_tracks, use_container_width=True)
                    
                    # Heatmap of activity
                    st.subheader("🔥 Activity Heatmap")
                    
                    # Create movement heatmap data
                    heatmap_data = []
                    for frame_data in tracking_data['tracks']:
                        for obj in frame_data['objects']:
                            heatmap_data.append({
                                'x': obj['centroid'][0],
                                'y': obj['centroid'][1],
                                'frame': frame_data['frame']
                            })
                    
                    if heatmap_data:
                        heatmap_df = pd.DataFrame(heatmap_data)
                        fig_heatmap = px.density_heatmap(heatmap_df, x='x', y='y',
                                                       title='Fish Movement Heatmap',
                                                       labels={'x': 'X Position', 'y': 'Y Position'})
                        fig_heatmap.update_yaxes(autorange="reversed")  # Flip Y axis for image coordinates
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            
            else:
                st.info("Process a video first to see visualizations.")
    
    else:
        # Welcome screen
        st.info("👆 Please upload a YOLO model and video file to get started.")
        
        st.markdown("""
        ## 🚀 Features
        
        - **Real-time Processing**: Process videos with live progress tracking
        - **Advanced Analytics**: Get detailed statistics about fish behavior
        - **Interactive Visualizations**: Explore movement patterns and trends
        - **Customizable Settings**: Fine-tune detection and tracking parameters
        - **Export Results**: Download annotated videos and tracking data
        
        ## 📋 Requirements
        
        1. **YOLO Model**: Upload your trained YOLOv8 model (.pt file)
        2. **Video File**: Upload the video you want to analyze
        3. **Configure Settings**: Adjust parameters in the sidebar
        4. **Start Processing**: Click the process button and monitor progress
        
        ## 🎯 Supported Formats
        
        - **Models**: .pt (PyTorch/YOLOv8)
        - **Videos**: .mp4, .avi, .mov, .mkv
        """)

if __name__ == "__main__":
    main()