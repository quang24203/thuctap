"""
Vehicle Detection Model using YOLOv8
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from ultralytics import YOLO
import logging

from ..utils.simple_logger import get_logger, performance_logger


class VehicleDetector:
    """Vehicle detection using YOLOv8"""
    
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "auto"
    ):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Input size for model (width, height)
            device: Device to run inference ('cpu', 'cuda', 'auto')
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Vehicle classes (COCO dataset classes)
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        # Performance metrics
        self.total_detections = 0
        self.total_inference_time = 0.0
        
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info(f"Loaded vehicle detection model from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: detection confidence
            - class_id: vehicle class ID
            - class_name: vehicle class name
        """
        import time
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter vehicle classes only
                        if class_id in self.vehicle_classes:
                            detection = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": float(confidence),
                                "class_id": class_id,
                                "class_name": self.vehicle_classes[class_id]
                            }
                            detections.append(detection)
            
            # Log performance
            inference_time = (time.time() - start_time) * 1000
            self.total_detections += len(detections)
            self.total_inference_time += inference_time
            performance_logger.log_detection_time("vehicle_detection", inference_time)

            self.logger.debug(f"Detected {len(detections)} vehicles in {inference_time:.2f}ms")

            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect vehicles in batch of frames
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists for each frame
        """
        import time
        start_time = time.time()
        
        try:
            # Run batch inference
            results = self.model(
                frames,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            
            batch_detections = []
            
            # Process each result
            for result in results:
                detections = []
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id in self.vehicle_classes:
                            detection = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": float(confidence),
                                "class_id": class_id,
                                "class_name": self.vehicle_classes[class_id]
                            }
                            detections.append(detection)
                
                batch_detections.append(detections)
            
            # Log performance
            inference_time = (time.time() - start_time) * 1000
            performance_logger.log_detection_time("vehicle_detection_batch", inference_time)
            
            return batch_detections
            
        except Exception as e:
            self.logger.error(f"Batch detection error: {e}")
            return [[] for _ in frames]
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Visualize detections on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                vis_frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return vis_frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "input_size": self.input_size,
            "vehicle_classes": self.vehicle_classes
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_inference_time = (
            self.total_inference_time / max(1, self.total_detections)
            if self.total_detections > 0 else 0
        )

        return {
            "total_detections": self.total_detections,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_inference_time,
            "fps": 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.total_detections = 0
        self.total_inference_time = 0.0
