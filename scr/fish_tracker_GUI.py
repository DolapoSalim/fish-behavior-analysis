import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import threading
import json
import os
from datetime import datetime
import time

class FishTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fish Tracker - Background Processing")
        self.root.geometry("600x500")
        
        # Variables
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="./output")
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.processing = False
        self.current_frame = 0
        self.total_frames = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model selection
        ttk.Label(main_frame, text="YOLO Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Video selection
        ttk.Label(main_frame, text="Video File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_video).grid(row=1, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(main_frame, text="Confidence Threshold:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Scale(main_frame, from_=0.1, to=1.0, variable=self.confidence_threshold, 
                 orient=tk.HORIZONTAL, length=200).grid(row=3, column=1, padx=5, pady=5)
        ttk.Label(main_frame, textvariable=self.confidence_threshold).grid(row=3, column=2, padx=5, pady=5)
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.save_video = tk.BooleanVar(value=True)
        self.save_data = tk.BooleanVar(value=True)
        self.save_frames = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(options_frame, text="Save annotated video", variable=self.save_video).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Save tracking data (JSON)", variable=self.save_data).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Save individual frames", variable=self.save_frames).grid(row=0, column=2, sticky=tk.W)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="Ready to process")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.frame_label = ttk.Label(progress_frame, text="Frame: 0/0")
        self.frame_label.grid(row=1, column=1, sticky=tk.E, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", 
                                     command=self.start_processing, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="Open Output Folder", 
                  command=self.open_output_folder).grid(row=0, column=2, padx=5)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.results_text = tk.Text(results_frame, height=8, width=70)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
    
    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def open_output_folder(self):
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            os.startfile(output_path) if os.name == 'nt' else os.system(f'open "{output_path}"')
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def start_processing(self):
        if not self.model_path.get() or not self.video_path.get():
            messagebox.showerror("Error", "Please select both model and video files")
            return
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
    
    def stop_processing(self):
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("Processing stopped by user")
    
    def process_video(self):
        try:
            # Initialize model and tracker
            self.log_message("Loading YOLO model...")
            model = YOLO(self.model_path.get())
            model.fuse()
            
            self.log_message("Initializing tracker...")
            tracker = DeepSort(max_age=30)
            
            # Open video
            self.log_message("Opening video file...")
            cap = cv2.VideoCapture(self.video_path.get())
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.log_message(f"Video info: {self.total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Setup output
            output_dir = self.output_dir.get()
            os.makedirs(output_dir, exist_ok=True)
            
            # Video writer
            video_writer = None
            if self.save_video.get():
                output_video_path = os.path.join(output_dir, f"tracked_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Data storage
            tracking_data = []
            prev_centroids = {}
            
            self.current_frame = 0
            start_time = time.time()
            
            while cap.isOpened() and self.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame += 1
                
                # Update progress
                progress = (self.current_frame / self.total_frames) * 100
                self.progress_bar['value'] = progress
                self.frame_label.config(text=f"Frame: {self.current_frame}/{self.total_frames}")
                self.status_label.config(text=f"Processing... ({progress:.1f}%)")
                
                # Run inference
                results = model(frame, conf=self.confidence_threshold.get(), verbose=False)[0]
                
                detections = []
                frame_data = {
                    'frame': self.current_frame,
                    'timestamp': self.current_frame / fps,
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
                if self.save_frames.get():
                    frame_filename = os.path.join(output_dir, f"frame_{self.current_frame:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                
                # Update GUI periodically
                if self.current_frame % 30 == 0:
                    self.root.update()
            
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            
            # Save tracking data
            if self.save_data.get():
                data_filename = os.path.join(output_dir, f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(data_filename, 'w') as f:
                    json.dump(tracking_data, f, indent=2)
                self.log_message(f"Tracking data saved to: {data_filename}")
            
            # Processing complete
            processing_time = time.time() - start_time
            self.log_message(f"Processing completed in {processing_time:.2f} seconds")
            self.log_message(f"Average FPS: {self.total_frames / processing_time:.2f}")
            self.log_message(f"Output saved to: {output_dir}")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress_bar['value'] = 100
            self.status_label.config(text="Processing complete")

if __name__ == "__main__":
    root = tk.Tk()
    app = FishTrackerGUI(root)
    root.mainloop()