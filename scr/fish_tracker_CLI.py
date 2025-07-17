import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import json
import os
from datetime import datetime
import time
from tqdm import tqdm

def process_video(args):
    """Process video without GUI for maximum performance"""
    
    print("Loading YOLO model...")
    model = YOLO(args.model)
    model.fuse()
    
    print("Initializing tracker...")
    tracker = DeepSort(max_age=30)
    
    print("Opening video file...")
    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    # Setup output
    os.makedirs(args.output, exist_ok=True)
    
    # Video writer
    video_writer = None
    if args.save_video:
        output_video_path = os.path.join(args.output, f"tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Output video: {output_video_path}")
    
    # Data storage
    tracking_data = []
    prev_centroids = {}
    
    # Process with progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=args.confidence, verbose=False)[0]
        
        detections = []
        frame_data = {
            'frame': frame_count,
            'timestamp': frame_count / fps,
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
            
            direction_angle = None
            if track_id in prev_centroids:
                prev_cx, prev_cy = prev_centroids[track_id]
                dx, dy = cx - prev_cx, cy - prev_cy
                
                if math.sqrt(dx*dx + dy*dy) > 5:
                    angle_rad = math.atan2(dy, dx)
                    direction_angle = (math.degrees(angle_rad) + 360) % 360
                    
                    cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id} Dir:{int(direction_angle)}°", 
                               (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            prev_centroids[track_id] = (cx, cy)
            
            # Store tracking data
            frame_data['detections'].append({
                'track_id': track_id,
                'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                'centroid': [cx, cy],
                'direction_angle': direction_angle
            })
        
        tracking_data.append(frame_data)
        
        # Save video frame
        if video_writer:
            video_writer.write(frame)
        
        # Save individual frames
        if args.save_frames:
            frame_filename = os.path.join(args.output, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        pbar.update(1)
    
    pbar.close()
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Save tracking data
    if args.save_data:
        data_filename = os.path.join(args.output, f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(data_filename, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Tracking data saved to: {data_filename}")
    
    # Processing complete
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average FPS: {frame_count / processing_time:.2f}")
    print(f"Output saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Fish Tracking with YOLO and DeepSORT")
    parser.add_argument("--model", "-m", required=True, help="Path to YOLO model file")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--save-data", action="store_true", default=True, help="Save tracking data")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")
    
    args = parser.parse_args()
    process_video(args)

if __name__ == "__main__":
    main()

# Usage example:
# python fish_tracker_cli.py --model yolo_model.pt --input video.mp4 --output results --save-video --confidence 0.6