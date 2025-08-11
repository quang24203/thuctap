"""
Simple Object Tracker - Alternative to ByteTrack for Windows
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import time

class SimpleTracker:
    """Simple centroid-based object tracker"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Initialize tracking variables
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
    
    def register(self, detection: Dict[str, Any]) -> int:
        """Register a new object"""
        # Calculate centroid
        bbox = detection['bbox']
        centroid = self._calculate_centroid(bbox)
        
        # Store object with centroid
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'detection': detection,
            'first_seen': time.time(),
            'last_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        
        # Add track_id to detection
        detection['track_id'] = self.next_object_id
        
        object_id = self.next_object_id
        self.next_object_id += 1
        
        return object_id
    
    def deregister(self, object_id: int):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _calculate_centroid(self, bbox: List[float]) -> tuple:
        """Calculate centroid from bounding box"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracker with new detections"""
        start_time = time.time()
        
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove objects that have disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.total_frames_processed += 1
            self.total_processing_time += processing_time
            
            return []
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
        else:
            # Match detections to existing objects
            self._match_detections_to_objects(detections)
        
        # Update performance stats
        processing_time = (time.time() - start_time) * 1000
        self.total_frames_processed += 1
        self.total_processing_time += processing_time
        
        # Return tracked objects
        return self._get_tracked_objects()
    
    def _match_detections_to_objects(self, detections: List[Dict[str, Any]]):
        """Match new detections to existing tracked objects"""
        # Calculate centroids for all detections
        detection_centroids = []
        for detection in detections:
            centroid = self._calculate_centroid(detection['bbox'])
            detection_centroids.append(centroid)
        
        # Get existing object centroids
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
        
        # Compute distance matrix
        if len(object_centroids) > 0 and len(detection_centroids) > 0:
            distance_matrix = np.zeros((len(object_centroids), len(detection_centroids)))
            
            for i, obj_centroid in enumerate(object_centroids):
                for j, det_centroid in enumerate(detection_centroids):
                    distance_matrix[i, j] = self._calculate_distance(obj_centroid, det_centroid)
            
            # Simple greedy matching (not optimal but fast)
            used_detection_indices = set()
            used_object_indices = set()
            
            # Find matches
            for i in range(len(object_centroids)):
                if i in used_object_indices:
                    continue
                
                min_distance = float('inf')
                min_j = -1
                
                for j in range(len(detection_centroids)):
                    if j in used_detection_indices:
                        continue
                    
                    if distance_matrix[i, j] < min_distance and distance_matrix[i, j] < self.max_distance:
                        min_distance = distance_matrix[i, j]
                        min_j = j
                
                if min_j != -1:
                    # Match found
                    object_id = object_ids[i]
                    detection = detections[min_j]
                    
                    # Update object
                    self.objects[object_id]['centroid'] = detection_centroids[min_j]
                    self.objects[object_id]['detection'] = detection
                    self.objects[object_id]['last_seen'] = time.time()
                    self.disappeared[object_id] = 0
                    
                    # Add track_id to detection
                    detection['track_id'] = object_id
                    
                    used_object_indices.add(i)
                    used_detection_indices.add(min_j)
            
            # Mark unmatched objects as disappeared
            for i, object_id in enumerate(object_ids):
                if i not in used_object_indices:
                    self.disappeared[object_id] += 1
                    
                    # Remove if disappeared too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Register new detections
            for j, detection in enumerate(detections):
                if j not in used_detection_indices:
                    self.register(detection)
        
        else:
            # No existing objects or no detections
            for detection in detections:
                self.register(detection)
    
    def _get_tracked_objects(self) -> List[Dict[str, Any]]:
        """Get list of currently tracked objects"""
        tracked_objects = []
        
        for object_id, obj_data in self.objects.items():
            detection = obj_data['detection'].copy()
            detection['track_id'] = object_id
            detection['tracking_confidence'] = max(0.0, 1.0 - (self.disappeared[object_id] / self.max_disappeared))
            tracked_objects.append(detection)
        
        return tracked_objects
    
    def get_track_count(self) -> int:
        """Get total number of tracks created"""
        return self.next_object_id
    
    def get_active_tracks(self) -> List[int]:
        """Get list of active track IDs"""
        return list(self.objects.keys())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get tracking performance statistics"""
        avg_processing_time = (
            self.total_processing_time / max(1, self.total_frames_processed)
            if self.total_frames_processed > 0 else 0
        )
        
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "fps": 1000.0 / avg_processing_time if avg_processing_time > 0 else 0,
            "active_tracks": len(self.objects),
            "total_tracks_created": self.next_object_id
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_frames_processed = 0
        self.total_processing_time = 0.0

class MultiObjectTracker:
    """Multi-object tracker wrapper"""
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, 
                 match_thresh: float = 0.8, frame_rate: int = 30, **kwargs):
        
        # Convert parameters for SimpleTracker
        max_disappeared = track_buffer
        max_distance = 100.0  # Adjust based on your needs
        
        self.tracker = SimpleTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance
        )
        
        # Store parameters for compatibility
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracker with new detections"""
        # Filter detections by confidence threshold
        filtered_detections = [
            det for det in detections 
            if det.get('confidence', 0.0) >= self.track_thresh
        ]
        
        return self.tracker.update(filtered_detections)
    
    def get_track_count(self) -> int:
        """Get total number of tracks"""
        return self.tracker.get_track_count()
    
    def get_active_tracks(self) -> List[int]:
        """Get active track IDs"""
        return self.tracker.get_active_tracks()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.tracker.get_performance_stats()
    
    def reset_stats(self):
        """Reset statistics"""
        self.tracker.reset_stats()
