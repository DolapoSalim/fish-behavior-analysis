import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Union
import copy

def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two segmentation masks.
    
    Args:
        mask1, mask2: Binary numpy arrays (same shape) where True/1 = object pixel
    
    Returns:
        IoU value between 0 and 1
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape")
    
    # Convert to boolean if needed
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_mask_overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate overlap ratio (intersection / smaller_mask_area).
    Useful when objects might be partially occluded.
    
    Args:
        mask1, mask2: Binary numpy arrays
    
    Returns:
        Overlap ratio between 0 and 1
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape")
    
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    
    area1 = mask1.sum()
    area2 = mask2.sum()
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Use the smaller mask as denominator
    smaller_area = min(area1, area2)
    return intersection / smaller_area

def mask_to_centroid(mask: np.ndarray) -> tuple:
    """
    Calculate centroid (center point) of a mask.
    
    Args:
        mask: Binary numpy array
    
    Returns:
        Tuple of (y, x) coordinates of centroid
    """
    if not mask.any():
        return (0, 0)
    
    y_coords, x_coords = np.where(mask)
    centroid_y = np.mean(y_coords)
    centroid_x = np.mean(x_coords)
    
    return (centroid_y, centroid_x)

def euclidean_distance(point1: tuple, point2: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_segmentation_predictions(
    video_predictions: List[Dict[str, Any]], 
    temporal_window: int = 5,
    spatial_threshold: float = 0.1,
    spatial_method: str = "iou",
    max_centroid_distance: Optional[float] = None,
    min_votes: int = 1
) -> List[Dict[str, Any]]:
    """
    Apply temporal-spatial filtering to video predictions with segmentation masks.
    
    Args:
        video_predictions: List of prediction dictionaries containing:
            - 'frame_number': int
            - 'mask': np.ndarray (binary mask)
            - 'predicted_behaviour': str
            - 'confidence': float (optional)
        temporal_window: Number of frames to look forward/backward
        spatial_threshold: Minimum overlap threshold for spatial matching
        spatial_method: "iou" or "overlap_ratio" for spatial matching
        max_centroid_distance: Maximum distance between mask centroids (optional filter)
        min_votes: Minimum number of votes needed for consensus
    
    Returns:
        List of filtered predictions with smoothed behaviors
    """
    if not video_predictions:
        return []
    
    # Sort predictions by frame number
    sorted_predictions = sorted(video_predictions, key=lambda x: x['frame_number'])
    filtered_predictions = []
    
    for current_prediction in sorted_predictions:
        current_frame = current_prediction['frame_number']
        current_mask = current_prediction['mask']
        
        # Find temporally linked predictions within the window
        temporally_linked = []
        for prediction in sorted_predictions:
            frame_diff = abs(prediction['frame_number'] - current_frame)
            if frame_diff <= temporal_window:
                temporally_linked.append(prediction)
        
        # Find spatially overlapping predictions
        spatially_overlapping = []
        current_centroid = mask_to_centroid(current_mask)
        
        for prediction in temporally_linked:
            pred_mask = prediction['mask']
            
            # Calculate spatial overlap
            if spatial_method == "iou":
                overlap_score = calculate_mask_iou(current_mask, pred_mask)
            elif spatial_method == "overlap_ratio":
                overlap_score = calculate_mask_overlap_ratio(current_mask, pred_mask)
            else:
                raise ValueError("spatial_method must be 'iou' or 'overlap_ratio'")
            
            # Check overlap threshold
            passes_spatial_threshold = overlap_score >= spatial_threshold
            
            # Optional: Check centroid distance
            passes_centroid_filter = True
            if max_centroid_distance is not None:
                pred_centroid = mask_to_centroid(pred_mask)
                distance = euclidean_distance(current_centroid, pred_centroid)
                passes_centroid_filter = distance <= max_centroid_distance
            
            if passes_spatial_threshold and passes_centroid_filter:
                # Add overlap score for transparency
                prediction_with_score = copy.deepcopy(prediction)
                prediction_with_score['spatial_overlap_score'] = overlap_score
                spatially_overlapping.append(prediction_with_score)
        
        # Get consensus behavior from overlapping predictions
        if len(spatially_overlapping) >= min_votes:
            behaviors = [pred['predicted_behaviour'] for pred in spatially_overlapping]
            behavior_counts = Counter(behaviors)
            most_common_behavior = behavior_counts.most_common(1)[0][0]
            
            # Calculate average confidence if available
            confidences = [pred.get('confidence', 1.0) for pred in spatially_overlapping 
                          if pred['predicted_behaviour'] == most_common_behavior]
            avg_confidence = np.mean(confidences)
            
            # Create updated prediction
            updated_prediction = copy.deepcopy(current_prediction)
            updated_prediction['predicted_behaviour'] = most_common_behavior
            updated_prediction['confidence_votes'] = behavior_counts[most_common_behavior]
            updated_prediction['total_votes'] = len(spatially_overlapping)
            updated_prediction['avg_confidence'] = avg_confidence
            updated_prediction['spatial_overlap_scores'] = [
                pred['spatial_overlap_score'] for pred in spatially_overlapping
            ]
            
        else:
            # Not enough votes, keep original prediction
            updated_prediction = copy.deepcopy(current_prediction)
            updated_prediction['confidence_votes'] = 1
            updated_prediction['total_votes'] = 1
            updated_prediction['avg_confidence'] = current_prediction.get('confidence', 1.0)
            updated_prediction['spatial_overlap_scores'] = [1.0]  # Self-overlap
        
        filtered_predictions.append(updated_prediction)
    
    return filtered_predictions

def filter_segmentation_predictions_advanced(
    video_predictions: List[Dict[str, Any]], 
    temporal_window: int = 5,
    spatial_threshold: float = 0.1,
    confidence_threshold: float = 0.6,
    spatial_method: str = "iou",
    min_mask_area: int = 100
) -> List[Dict[str, Any]]:
    """
    Advanced segmentation filtering with additional quality filters.
    
    Args:
        video_predictions: List of prediction dictionaries
        temporal_window: Number of frames to look forward/backward
        spatial_threshold: Minimum overlap threshold
        confidence_threshold: Minimum confidence ratio to accept consensus
        spatial_method: "iou" or "overlap_ratio"
        min_mask_area: Minimum number of pixels for valid mask
    
    Returns:
        List of high-quality filtered predictions
    """
    # Filter out masks that are too small
    valid_predictions = []
    for pred in video_predictions:
        mask_area = pred['mask'].sum()
        if mask_area >= min_mask_area:
            valid_predictions.append(pred)
    
    # Apply basic filtering
    filtered_predictions = filter_segmentation_predictions(
        valid_predictions, 
        temporal_window=temporal_window,
        spatial_threshold=spatial_threshold,
        spatial_method=spatial_method
    )
    
    # Additional confidence-based filtering
    high_confidence_predictions = []
    for prediction in filtered_predictions:
        confidence_ratio = prediction['confidence_votes'] / prediction['total_votes']
        
        if confidence_ratio >= confidence_threshold:
            prediction['confidence_ratio'] = confidence_ratio
            high_confidence_predictions.append(prediction)
    
    return high_confidence_predictions

def create_sample_mask(height: int, width: int, center_x: int, center_y: int, radius: int) -> np.ndarray:
    """Create a circular mask for testing purposes."""
    mask = np.zeros((height, width), dtype=bool)
    y, x = np.ogrid[:height, :width]
    mask_condition = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask[mask_condition] = True
    return mask

# Example usage
if __name__ == "__main__":
    # Create sample segmentation predictions
    sample_predictions = [
        {
            "frame_number": 1, 
            "mask": create_sample_mask(480, 640, 320, 240, 50),
            "predicted_behaviour": "walking",
            "confidence": 0.9
        },
        {
            "frame_number": 2, 
            "mask": create_sample_mask(480, 640, 325, 245, 52),
            "predicted_behaviour": "running",  # Noisy prediction
            "confidence": 0.7
        },
        {
            "frame_number": 3, 
            "mask": create_sample_mask(480, 640, 330, 250, 48),
            "predicted_behaviour": "walking",
            "confidence": 0.85
        },
        {
            "frame_number": 4, 
            "mask": create_sample_mask(480, 640, 335, 255, 51),
            "predicted_behaviour": "walking",
            "confidence": 0.92
        },
        {
            "frame_number": 5, 
            "mask": create_sample_mask(480, 640, 340, 260, 49),
            "predicted_behaviour": "walking",
            "confidence": 0.88
        }
    ]
    
    # Apply segmentation filtering
    filtered_results = filter_segmentation_predictions(
        sample_predictions, 
        temporal_window=3, 
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