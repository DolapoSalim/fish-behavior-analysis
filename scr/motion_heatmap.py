import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import json
from collections import defaultdict

class FishTrackerWithHeatmap:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.tracker = DeepSort(max_age=30)
        self.confidence_threshold = confidence_threshold
        
        # Data storage
        self.tracking_data = []
        self.motion_data = []
        self.heatmap_data = None
        self.prev_centroids = {}
        self.track_paths = defaultdict(list)  # Store complete paths for each track
        
    def process_video(self, video_path, output_dir):
        """Process video and generate all outputs"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Initialize heatmap
        self.heatmap_data = np.zeros((height, width), dtype=np.float32)
        
        # Setup video writer
        output_video_path = os.path.join(output_dir, f"tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Process video with progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
            
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
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                cx, cy = int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2)
                
                # Add to heatmap
                self.add_to_heatmap(cx, cy, width, height)
                
                # Add to track path
                self.track_paths[track_id].append((cx, cy, frame_count, frame_count / fps))
                
                # Draw tracking results
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Calculate motion metrics
                speed = 0
                direction_angle = None
                distance_traveled = 0
                
                if track_id in self.prev_centroids:
                    prev_cx, prev_cy = self.prev_centroids[track_id]
                    dx, dy = cx - prev_cx, cy - prev_cy
                    distance_traveled = math.sqrt(dx*dx + dy*dy)
                    speed = distance_traveled * fps  # pixels per second
                    
                    if distance_traveled > 2:  # Minimum movement threshold
                        angle_rad = math.atan2(dy, dx)
                        direction_angle = (math.degrees(angle_rad) + 360) % 360
                        
                        # Draw arrow
                        cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (0, 255, 0), 2)
                        cv2.putText(frame, f"Dir:{int(direction_angle)}°", 
                                   (cx-20, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                # Store motion data
                motion_entry = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'track_id': track_id,
                    'x': cx,
                    'y': cy,
                    'bbox_x1': float(ltrb[0]),
                    'bbox_y1': float(ltrb[1]),
                    'bbox_x2': float(ltrb[2]),
                    'bbox_y2': float(ltrb[3]),
                    'bbox_width': float(ltrb[2] - ltrb[0]),
                    'bbox_height': float(ltrb[3] - ltrb[1]),
                    'speed_pixels_per_second': speed,
                    'direction_angle': direction_angle,
                    'distance_traveled': distance_traveled
                }
                
                self.motion_data.append(motion_entry)
                
                # Update previous centroids
                self.prev_centroids[track_id] = (cx, cy)
                
                # Store frame data
                frame_data['detections'].append({
                    'track_id': track_id,
                    'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                    'centroid': [cx, cy],
                    'speed': speed,
                    'direction_angle': direction_angle,
                    'distance_traveled': distance_traveled
                })
            
            self.tracking_data.append(frame_data)
            
            # Write frame
            video_writer.write(frame)
            
            pbar.update(1)
        
        pbar.close()
        
        # Cleanup
        cap.release()
        video_writer.release()
        
        # Generate all outputs
        self.save_csv_files(output_dir)
        self.generate_heatmap(output_dir, width, height)
        self.generate_path_visualization(output_dir, width, height)
        self.generate_analytics_report(output_dir)
        
        print(f"\nProcessing complete! Results saved to: {output_dir}")
        
    def add_to_heatmap(self, cx, cy, width, height, radius=15):
        """Add fish position to heatmap with Gaussian blur"""
        # Create a small gaussian kernel
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x*x + y*y <= radius*radius
        kernel = np.zeros((2*radius+1, 2*radius+1))
        kernel[mask] = 1
        
        # Apply gaussian blur
        kernel = cv2.GaussianBlur(kernel, (15, 15), 0)
        
        # Add to heatmap
        y1, y2 = max(0, cy-radius), min(height, cy+radius+1)
        x1, x2 = max(0, cx-radius), min(width, cx+radius+1)
        
        ky1, ky2 = max(0, radius-cy), min(2*radius+1, height-cy+radius)
        kx1, kx2 = max(0, radius-cx), min(2*radius+1, width-cx+radius)
        
        if y2 > y1 and x2 > x1:
            self.heatmap_data[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
    
    def save_csv_files(self, output_dir):
        """Save all data to CSV files"""
        
        # Motion data CSV
        motion_df = pd.DataFrame(self.motion_data)
        motion_csv_path = os.path.join(output_dir, f"motion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        motion_df.to_csv(motion_csv_path, index=False)
        print(f"Motion data saved to: {motion_csv_path}")
        
        # Track summary CSV
        track_summary = []
        for track_id, path in self.track_paths.items():
            if len(path) > 1:
                # Calculate total distance
                total_distance = 0
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    total_distance += math.sqrt(dx*dx + dy*dy)
                
                # Get time span
                start_time = path[0][3]
                end_time = path[-1][3]
                duration = end_time - start_time
                
                # Calculate average speed
                avg_speed = total_distance / duration if duration > 0 else 0
                
                # Get bounding box of track
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                
                track_summary.append({
                    'track_id': track_id,
                    'first_frame': path[0][2],
                    'last_frame': path[-1][2],
                    'duration_seconds': duration,
                    'total_distance_pixels': total_distance,
                    'average_speed_pixels_per_second': avg_speed,
                    'min_x': min(x_coords),
                    'max_x': max(x_coords),
                    'min_y': min(y_coords),
                    'max_y': max(y_coords),
                    'path_length': len(path)
                })
        
        summary_df = pd.DataFrame(track_summary)
        summary_csv_path = os.path.join(output_dir, f"track_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Track summary saved to: {summary_csv_path}")
        
        # Heatmap data CSV
        heatmap_df = pd.DataFrame(self.heatmap_data)
        heatmap_csv_path = os.path.join(output_dir, f"heatmap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        heatmap_df.to_csv(heatmap_csv_path, index=False)
        print(f"Heatmap data saved to: {heatmap_csv_path}")
    
    def generate_heatmap(self, output_dir, width, height):
        """Generate motion heatmap visualization"""
        
        # Normalize heatmap
        if self.heatmap_data.max() > 0:
            normalized_heatmap = (self.heatmap_data / self.heatmap_data.max() * 255).astype(np.uint8)
        else:
            normalized_heatmap = self.heatmap_data.astype(np.uint8)
        
        # Create colormap
        heatmap_colored = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # Save heatmap image
        heatmap_path = os.path.join(output_dir, f"motion_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(heatmap_path, heatmap_colored)
        
        # Create matplotlib version
        plt.figure(figsize=(12, 8))
        plt.imshow(self.heatmap_data, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Motion Intensity')
        plt.title('Fish Motion Heatmap')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        
        heatmap_matplotlib_path = os.path.join(output_dir, f"motion_heatmap_matplotlib_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(heatmap_matplotlib_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to: {heatmap_path}")
        print(f"Matplotlib heatmap saved to: {heatmap_matplotlib_path}")
    
    def generate_path_visualization(self, output_dir, width, height):
        """Generate track path visualization"""
        
        # Create blank image
        path_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color palette for different tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0)
        ]
        
        # Draw paths
        for i, (track_id, path) in enumerate(self.track_paths.items()):
            if len(path) > 1:
                color = colors[i % len(colors)]
                
                # Draw path
                for j in range(1, len(path)):
                    cv2.line(path_image, 
                            (path[j-1][0], path[j-1][1]),
                            (path[j][0], path[j][1]),
                            color, 2)
                
                # Mark start and end points
                cv2.circle(path_image, (path[0][0], path[0][1]), 5, (0, 255, 0), -1)  # Start (green)
                cv2.circle(path_image, (path[-1][0], path[-1][1]), 5, (0, 0, 255), -1)  # End (red)
                
                # Add track ID
                cv2.putText(path_image, f"T{track_id}", 
                           (path[0][0], path[0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save path visualization
        path_vis_path = os.path.join(output_dir, f"track_paths_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path_vis_path, path_image)
        print(f"Track paths saved to: {path_vis_path}")
    
    def generate_analytics_report(self, output_dir):
        """Generate analytics report"""
        
        if not self.motion_data:
            return
        
        df = pd.DataFrame(self.motion_data)
        
        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Speed distribution
        axes[0, 0].hist(df['speed_pixels_per_second'].dropna(), bins=50, alpha=0.7)
        axes[0, 0].set_title('Speed Distribution')
        axes[0, 0].set_xlabel('Speed (pixels/second)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Direction distribution
        direction_data = df['direction_angle'].dropna()
        if len(direction_data) > 0:
            axes[0, 1].hist(direction_data, bins=36, alpha=0.7)
            axes[0, 1].set_title('Direction Distribution')
            axes[0, 1].set_xlabel('Direction (degrees)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Track activity over time
        track_counts = df.groupby('timestamp')['track_id'].nunique()
        axes[1, 0].plot(track_counts.index, track_counts.values)
        axes[1, 0].set_title('Number of Active Tracks Over Time')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Number of Tracks')
        
        # Average speed over time
        avg_speed = df.groupby('timestamp')['speed_pixels_per_second'].mean()
        axes[1, 1].plot(avg_speed.index, avg_speed.values)
        axes[1, 1].set_title('Average Speed Over Time')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Average Speed (pixels/second)')
        
        plt.tight_layout()
        
        # Save analytics report
        analytics_path = os.path.join(output_dir, f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(analytics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analytics report saved to: {analytics_path}")

def main():
    parser = argparse.ArgumentParser(description="Fish Tracking with Motion Heatmap and CSV Export")
    parser.add_argument("--model", "-m", required=True, help="Path to YOLO model file")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default="./fish_tracking_results", help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = FishTrackerWithHeatmap(args.model, args.confidence)
    
    # Process video
    tracker.process_video(args.input, args.output)

if __name__ == "__main__":
    main()

# Usage example:
# python fish_tracker_heatmap.py --model yolo_model.pt --input video.mp4 --output results --confidence 0.6