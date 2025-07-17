from collections import Counter
from typing import List, Dict, Any, Optional
import copy

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Dictionaries with keys 'xmin', 'xmax', 'ymin', 'ymax'
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x_left = max(box1['xmin'], box2['xmin'])
    y_top = max(box1['ymin'], box2['ymin'])
    x_right = min(box1['xmax'], box2['xmax'])
    y_bottom = min(box1['ymax'], box2['ymax'])
    
    # No intersection
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def filter_temporal_predictions(
    video_predictions: List[Dict[str, Any]], 
    temporal_window: int = 5,
    spatial_threshold: float = 0.1,
    min_votes: int = 1
) -> List[Dict[str, Any]]:
    """
    Apply temporal-spatial filtering to video predictions.
    
    Args:
        video_predictions: List of prediction dictionaries containing:
            - 'frame_number': int
            - 'xmin', 'xmax', 'ymin', 'ymax': float (bounding box coordinates)
            - 'predicted_behaviour': str
        temporal_window: Number of frames to look forward/backward
        spatial_threshold: Minimum IoU threshold for spatial overlap
        min_votes: Minimum number of votes needed for consensus
    
    Returns:
        List of filtered predictions with smoothed behaviors
    """
    if not video_predictions:
        return []
    
    # Sort predictions by frame number for efficient processing
    sorted_predictions = sorted(video_predictions, key=lambda x: x['frame_number'])
    filtered_predictions = []
    
    for current_prediction in sorted_predictions:
        current_frame = current_prediction['frame_number']
        
        # Find temporally linked predictions within the window
        temporally_linked = []
        for prediction in sorted_predictions:
            frame_diff = abs(prediction['frame_number'] - current_frame)
            if frame_diff <= temporal_window:
                temporally_linked.append(prediction)
        
        # Find spatially overlapping predictions using IoU
        spatially_overlapping = []
        for prediction in temporally_linked:
            iou = calculate_iou(current_prediction, prediction)
            if iou >= spatial_threshold:
                spatially_overlapping.append(prediction)
        
        # Get consensus behavior from overlapping predictions
        if len(spatially_overlapping) >= min_votes:
            behaviors = [pred['predicted_behaviour'] for pred in spatially_overlapping]
            behavior_counts = Counter(behaviors)
            most_common_behavior = behavior_counts.most_common(1)[0][0]
            
            # Create updated prediction with consensus behavior
            updated_prediction = copy.deepcopy(current_prediction)
            updated_prediction['predicted_behaviour'] = most_common_behavior
            updated_prediction['confidence_votes'] = behavior_counts[most_common_behavior]
            updated_prediction['total_votes'] = len(spatially_overlapping)
            
        else:
            # Not enough votes, keep original prediction
            updated_prediction = copy.deepcopy(current_prediction)
            updated_prediction['confidence_votes'] = 1
            updated_prediction['total_votes'] = 1
        
        filtered_predictions.append(updated_prediction)
    
    return filtered_predictions

def filter_video_predictions_advanced(
    video_predictions: List[Dict[str, Any]], 
    temporal_window: int = 5,
    spatial_threshold: float = 0.1,
    confidence_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Advanced filtering with confidence-based filtering.
    
    Args:
        video_predictions: List of prediction dictionaries
        temporal_window: Number of frames to look forward/backward
        spatial_threshold: Minimum IoU threshold for spatial overlap
        confidence_threshold: Minimum confidence ratio to accept consensus
    
    Returns:
        List of filtered predictions with high-confidence behaviors
    """
    filtered_predictions = filter_temporal_predictions(
        video_predictions, temporal_window, spatial_threshold
    )
    
    # Additional filtering based on confidence
    high_confidence_predictions = []
    for prediction in filtered_predictions:
        confidence_ratio = prediction['confidence_votes'] / prediction['total_votes']
        
        if confidence_ratio >= confidence_threshold:
            prediction['confidence_ratio'] = confidence_ratio
            high_confidence_predictions.append(prediction)
    
    return high_confidence_predictions

# Example usage
if __name__ == "__main__":
    # Sample video predictions
    sample_predictions = [
        {"frame_number": 1, "xmin": 100, "xmax": 200, "ymin": 50, "ymax": 150, "predicted_behaviour": "walking"},
        {"frame_number": 2, "xmin": 105, "xmax": 205, "ymin": 55, "ymax": 155, "predicted_behaviour": "running"},
        {"frame_number": 3, "xmin": 110, "xmax": 210, "ymin": 60, "ymax": 160, "predicted_behaviour": "walking"},
        {"frame_number": 4, "xmin": 115, "xmax": 215, "ymin": 65, "ymax": 165, "predicted_behaviour": "walking"},
        {"frame_number": 5, "xmin": 120, "xmax": 220, "ymin": 70, "ymax": 170, "predicted_behaviour": "walking"},
    ]
    
    # Apply filtering
    filtered_results = filter_temporal_predictions(
        sample_predictions, 
        temporal_window=3, 
        spatial_threshold=0.3
    )
    
    # Print results
    for pred in filtered_results:
        print(f"Frame {pred['frame_number']}: {pred['predicted_behaviour']} "
              f"(votes: {pred['confidence_votes']}/{pred['total_votes']})")