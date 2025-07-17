# Your YOLOv8 pipeline would look like:
from ultralytics import YOLO

# 1. Load YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # or your trained model

# 2. Run inference on video
results = model.track(source="video.mp4", save=True)

# 3. Convert to our format
predictions = []
for frame_num, result in enumerate(results):
    for box, mask in zip(result.boxes, result.masks):
        prediction = {
            "frame_number": frame_num,
            "mask": mask.data.cpu().numpy(),  # Binary mask
            "predicted_behaviour": "walking",  # Your behavior classification
            "confidence": box.conf.item()
        }
        predictions.append(prediction)

# 4. Apply our filter
filtered = filter_segmentation_predictions(
    predictions, 
    temporal_window=5,
    spatial_threshold=0.3,
    spatial_method="iou"
)

# Print results
print("Filtered Results:")
for pred in filtered_results:
    print(f"Frame {pred['frame_number']}: {pred['predicted_behaviour']} "
            f"(votes: {pred['confidence_votes']}/{pred['total_votes']}, "
            f"avg_conf: {pred['avg_confidence']:.2f})")

print("\nAdvanced Filtering:")
advanced_results = filter_segmentation_predictions_advanced(
    sample_predictions,
    confidence_threshold=0.7
)

for pred in advanced_results:
    print(f"Frame {pred['frame_number']}: {pred['predicted_behaviour']} "
            f"(ratio: {pred['confidence_ratio']:.2f})")