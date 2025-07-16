#### STEP-BY-STEP PIPELINE OVERVIEW
- Run YOLOv8 segmentation on video frames
- Extract object masks and centroids per fish
- Feed detections into a tracker (e.g., DeepSORT or ByteTrack)
- Track fish over frames and assign consistent IDs
- Estimate direction from centroid displacement
- (Optional) Use optical flow inside masks to refine motion