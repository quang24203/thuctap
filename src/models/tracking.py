"""
Multi-object tracking using ByteTrack algorithm or Simple Tracker
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import logging

from ..utils.simple_logger import get_logger, performance_logger

# Try to import ByteTrack, fallback to SimpleTracker
try:
    # ByteTrack imports would go here
    BYTETRACK_AVAILABLE = False  # Set to False for now due to cython-bbox issues
except ImportError:
    BYTETRACK_AVAILABLE = False

if not BYTETRACK_AVAILABLE:
    from .simple_tracker import MultiObjectTracker


@dataclass
class Detection:
    """Detection data structure"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    feature: Optional[np.ndarray] = None


@dataclass
class Track:
    """Track data structure"""
    track_id: int
    bbox: List[int]
    confidence: float
    class_id: int
    state: str = "tracked"  # "tracked", "lost", "removed"
    age: int = 0
    time_since_update: int = 0
    hit_streak: int = 0
    feature: Optional[np.ndarray] = None
    history: List[List[int]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class KalmanFilter:
    """Simple Kalman filter for object tracking"""
    
    def __init__(self):
        """Initialize Kalman filter"""
        # State: [x, y, w, h, dx, dy, dw, dh]
        self.ndim = 4
        self.dt = 1.0
        
        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt
        self.F[3, 7] = self.dt
        
        # Measurement matrix
        self.H = np.eye(4, 8)
        
        # Process noise
        self.Q = np.eye(8) * 0.1
        
        # Measurement noise
        self.R = np.eye(4) * 1.0
        
        # Initial covariance
        self.P = np.eye(8) * 1000.0
        
        # State
        self.x = np.zeros(8)
        
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]
    
    def update(self, measurement: np.ndarray):
        """Update with measurement"""
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        
        return self.x[:4]
    
    def get_state(self) -> np.ndarray:
        """Get current state [x, y, w, h]"""
        return self.x[:4]


class ByteTracker:
    """ByteTrack multi-object tracker"""
    
    def __init__(
        self,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10
    ):
        """
        Initialize ByteTracker
        
        Args:
            frame_rate: Frame rate of input video
            track_thresh: Threshold for high confidence detections
            track_buffer: Buffer for lost tracks
            match_thresh: Matching threshold for association
            min_box_area: Minimum bounding box area
        """
        self.logger = get_logger(self.__class__.__name__)
        
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.frame_id = 0
        self.track_id_count = 0
        
        # Track lists
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        # Kalman filters for each track
        self.kalman_filters: Dict[int, KalmanFilter] = {}
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections
            
        Returns:
            List of active tracks
        """
        import time
        start_time = time.time()
        
        self.frame_id += 1
        
        # Filter detections by area
        valid_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= self.min_box_area:
                valid_detections.append(det)
        
        # Split detections by confidence
        high_conf_dets = [d for d in valid_detections if d.confidence >= self.track_thresh]
        low_conf_dets = [d for d in valid_detections if d.confidence < self.track_thresh]
        
        # Update tracked tracks with high confidence detections
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
            high_conf_dets, self.tracked_tracks
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracked_tracks[track_idx]
            detection = high_conf_dets[det_idx]
            self._update_track(track, detection)
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracked_tracks[track_idx]
            if track.state == "tracked":
                track.state = "lost"
                self.lost_tracks.append(track)
        
        # Remove matched tracks from tracked list
        self.tracked_tracks = [
            track for i, track in enumerate(self.tracked_tracks) 
            if i not in unmatched_tracks
        ]
        
        # Associate unmatched detections with lost tracks
        unmatched_high_conf_dets = [high_conf_dets[i] for i in unmatched_dets]
        matched_tracks, unmatched_dets, unmatched_lost = self._associate_detections_to_tracks(
            unmatched_high_conf_dets, self.lost_tracks
        )
        
        # Update matched lost tracks
        for track_idx, det_idx in matched_tracks:
            track = self.lost_tracks[track_idx]
            detection = unmatched_high_conf_dets[det_idx]
            self._update_track(track, detection)
            track.state = "tracked"
            self.tracked_tracks.append(track)
        
        # Remove matched tracks from lost list
        self.lost_tracks = [
            track for i, track in enumerate(self.lost_tracks)
            if i not in [idx for idx, _ in matched_tracks]
        ]
        
        # Associate remaining detections with low confidence detections
        remaining_dets = [unmatched_high_conf_dets[i] for i in unmatched_dets] + low_conf_dets
        if remaining_dets:
            matched_tracks, _, _ = self._associate_detections_to_tracks(
                remaining_dets, self.lost_tracks
            )
            
            for track_idx, det_idx in matched_tracks:
                track = self.lost_tracks[track_idx]
                detection = remaining_dets[det_idx]
                self._update_track(track, detection)
                track.state = "tracked"
                self.tracked_tracks.append(track)
        
        # Create new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            detection = unmatched_high_conf_dets[det_idx]
            self._initiate_track(detection)
        
        # Remove tracks that have been lost for too long
        self._remove_old_tracks()
        
        # Predict for all active tracks
        for track in self.tracked_tracks:
            if track.track_id in self.kalman_filters:
                predicted_bbox = self.kalman_filters[track.track_id].predict()
                track.bbox = self._tlwh_to_xyxy(predicted_bbox)
        
        # Log performance
        processing_time = (time.time() - start_time) * 1000
        performance_logger.log_detection_time("tracking", processing_time)
        
        return self.tracked_tracks.copy()
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[Detection], 
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using IoU matching"""
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(track.bbox, detection.bbox)
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for t, d in zip(track_indices, det_indices):
            if iou_matrix[t, d] >= self.match_thresh:
                matches.append((t, d))
                unmatched_detections.remove(d)
                unmatched_tracks.remove(t)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_track(self, track: Track, detection: Detection):
        """Update track with new detection"""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.time_since_update = 0
        track.hit_streak += 1
        track.history.append(detection.bbox)
        
        # Update Kalman filter
        if track.track_id in self.kalman_filters:
            measurement = self._xyxy_to_tlwh(detection.bbox)
            self.kalman_filters[track.track_id].update(measurement)
    
    def _initiate_track(self, detection: Detection):
        """Create new track from detection"""
        self.track_id_count += 1
        
        track = Track(
            track_id=self.track_id_count,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            state="tracked",
            age=1,
            time_since_update=0,
            hit_streak=1,
            history=[detection.bbox]
        )
        
        self.tracked_tracks.append(track)
        
        # Initialize Kalman filter
        kf = KalmanFilter()
        measurement = self._xyxy_to_tlwh(detection.bbox)
        kf.x[:4] = measurement
        self.kalman_filters[track.track_id] = kf
    
    def _remove_old_tracks(self):
        """Remove tracks that have been lost for too long"""
        for track in self.lost_tracks[:]:
            track.time_since_update += 1
            if track.time_since_update > self.track_buffer:
                self.lost_tracks.remove(track)
                self.removed_tracks.append(track)
                if track.track_id in self.kalman_filters:
                    del self.kalman_filters[track.track_id]
        
        # Update age for all tracks
        for track in self.tracked_tracks + self.lost_tracks:
            track.age += 1
    
    def _xyxy_to_tlwh(self, bbox: List[int]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        return np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32)
    
    def _tlwh_to_xyxy(self, tlwh: np.ndarray) -> List[int]:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return [int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])]
    
    def get_track_count(self) -> int:
        """Get total number of tracks created"""
        return self.track_id_count
    
    def get_active_tracks(self) -> List[Track]:
        """Get list of active tracks"""
        return [track for track in self.tracked_tracks if track.state == "tracked"]
    
    def reset(self):
        """Reset tracker"""
        self.frame_id = 0
        self.track_id_count = 0
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.kalman_filters.clear()


class VehicleTracker:
    """Vehicle tracking wrapper with additional features"""
    
    def __init__(
        self,
        tracker_config: Dict[str, Any] = None
    ):
        """Initialize vehicle tracker"""
        self.logger = get_logger(self.__class__.__name__)
        
        # Default config
        default_config = {
            "frame_rate": 30,
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "min_box_area": 100
        }
        
        config = default_config if tracker_config is None else {**default_config, **tracker_config}
        
        # Initialize tracker
        if BYTETRACK_AVAILABLE:
            self.tracker = ByteTracker(**config)
        else:
            # Use simple tracker
            from .simple_tracker import SimpleTracker
            self.tracker = SimpleTracker(
                max_disappeared=config.get("track_buffer", 30),
                max_distance=config.get("max_distance", 100)
            )
        
        # Track entry/exit detection
        self.entry_zones = []
        self.exit_zones = []
        self.track_events = {}  # track_id -> {"entry_time": timestamp, "exit_time": timestamp}

        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
    
    def update(self, vehicle_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with vehicle detections
        
        Args:
            vehicle_detections: List of vehicle detection dictionaries
            
        Returns:
            List of tracked vehicles with IDs
        """
        # Convert detections to tracker format
        detections = []
        for det in vehicle_detections:
            detection = Detection(
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_id=det["class_id"]
            )
            detections.append(detection)
        
        # Update tracker
        import time
        start_time = time.time()

        tracks = self.tracker.update(detections)

        # Track performance
        processing_time = (time.time() - start_time) * 1000
        self.total_frames_processed += 1
        self.total_processing_time += processing_time

        # Convert tracks back to dictionary format
        tracked_vehicles = []
        for track in tracks:
            if track.state == "tracked":
                vehicle = {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "confidence": track.confidence,
                    "class_id": track.class_id,
                    "class_name": self._get_class_name(track.class_id),
                    "age": track.age,
                    "hit_streak": track.hit_streak
                }
                tracked_vehicles.append(vehicle)
        
        return tracked_vehicles
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        class_names = {
            2: "car",
            3: "motorcycle",
            5: "bus", 
            7: "truck"
        }
        return class_names.get(class_id, "vehicle")
    
    def set_entry_zone(self, zone: List[List[int]]):
        """Set entry zone polygon"""
        self.entry_zones.append(zone)
    
    def set_exit_zone(self, zone: List[List[int]]):
        """Set exit zone polygon"""
        self.exit_zones.append(zone)
    
    def detect_entry_exit(self, tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect entry/exit events for tracks"""
        events = {"entries": [], "exits": []}
        
        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            
            # Check entry zones
            for zone in self.entry_zones:
                if self._point_in_polygon(center, zone):
                    if track_id not in self.track_events:
                        self.track_events[track_id] = {"entry_time": self.tracker.frame_id}
                        events["entries"].append({
                            "track_id": track_id,
                            "time": self.tracker.frame_id,
                            "bbox": bbox
                        })
            
            # Check exit zones
            for zone in self.exit_zones:
                if self._point_in_polygon(center, zone):
                    if track_id in self.track_events and "exit_time" not in self.track_events[track_id]:
                        self.track_events[track_id]["exit_time"] = self.tracker.frame_id
                        events["exits"].append({
                            "track_id": track_id,
                            "time": self.tracker.frame_id,
                            "duration": self.tracker.frame_id - self.track_events[track_id]["entry_time"],
                            "bbox": bbox
                        })
        
        return events
    
    def _point_in_polygon(self, point: List[int], polygon: List[List[int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            "total_tracks": self.tracker.get_track_count(),
            "active_tracks": len(self.tracker.get_active_tracks()),
            "tracked_tracks": len(self.tracker.tracked_tracks),
            "lost_tracks": len(self.tracker.lost_tracks),
            "removed_tracks": len(self.tracker.removed_tracks),
            "total_entries": len([e for e in self.track_events.values() if "entry_time" in e]),
            "total_exits": len([e for e in self.track_events.values() if "exit_time" in e])
        }

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
            "fps": 1000.0 / avg_processing_time if avg_processing_time > 0 else 0
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
    
    def visualize_tracks(
        self, 
        frame: np.ndarray, 
        tracks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Visualize tracks on frame"""
        vis_frame = frame.copy()
        
        for track in tracks:
            bbox = track["bbox"]
            track_id = track["track_id"]
            class_name = track["class_name"]
            confidence = track["confidence"]
            
            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (255, 0, 0),
                2
            )
            
            # Draw track ID and info
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            cv2.putText(
                vis_frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            
            # Draw track history (trajectory)
            if track_id in self.tracker.kalman_filters:
                track_obj = None
                for t in self.tracker.tracked_tracks:
                    if t.track_id == track_id:
                        track_obj = t
                        break
                
                if track_obj and len(track_obj.history) > 1:
                    for i in range(1, len(track_obj.history)):
                        pt1 = track_obj.history[i-1]
                        pt2 = track_obj.history[i]
                        center1 = ((pt1[0] + pt1[2]) // 2, (pt1[1] + pt1[3]) // 2)
                        center2 = ((pt2[0] + pt2[2]) // 2, (pt2[1] + pt2[3]) // 2)
                        cv2.line(vis_frame, center1, center2, (0, 255, 0), 2)
        
        # Draw entry/exit zones
        for zone in self.entry_zones:
            cv2.polylines(vis_frame, [np.array(zone)], True, (0, 255, 0), 2)
            cv2.putText(vis_frame, "ENTRY", zone[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for zone in self.exit_zones:
            cv2.polylines(vis_frame, [np.array(zone)], True, (0, 0, 255), 2)
            cv2.putText(vis_frame, "EXIT", zone[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame
