"""
Main Parking System Integration
Hệ thống tích hợp tất cả các thành phần AI và quản lý bãi đỗ xe
"""

import asyncio
import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .models.vehicle_detection import VehicleDetector
from .models.license_plate import LicensePlateProcessor
from .models.tracking import VehicleTracker, Detection
from .database.operations import DatabaseManager
from .utils.config import Config
from .utils.logger import get_logger, performance_logger, system_logger


class VideoProcessor:
    """Video processing for a single camera"""
    
    def __init__(
        self,
        camera_config,
        vehicle_detector: VehicleDetector,
        license_plate_processor: LicensePlateProcessor,
        vehicle_tracker: VehicleTracker,
        db_manager: DatabaseManager
    ):
        self.camera_config = camera_config
        self.vehicle_detector = vehicle_detector
        self.license_plate_processor = license_plate_processor
        self.vehicle_tracker = vehicle_tracker
        self.db_manager = db_manager
        self.logger = get_logger(f"{self.__class__.__name__}_{camera_config.id}")
        
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Frame buffer for processing
        self.current_frame = None
        self.processed_frame = None
        
    async def start(self):
        """Start video processing"""
        self.logger.info(f"Starting video processor for camera {self.camera_config.id}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_config.rtsp_url)
        
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_config.id}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        
        self.is_running = True
        
        # Start processing loop
        await self._processing_loop()
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning(f"Failed to read frame from camera {self.camera_config.id}")
                    await asyncio.sleep(0.1)
                    continue
                
                self.current_frame = frame.copy()
                self.frame_count += 1
                
                # Process frame
                await self._process_frame(frame)
                
                # Calculate FPS
                self._update_fps()
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(1/30)  # Target 30 FPS processing
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_frame(self, frame: np.ndarray):
        """Process a single frame"""
        start_time = time.time()
        
        try:
            # 1. Detect vehicles
            vehicle_detections = self.vehicle_detector.detect(frame)
            
            # 2. Track vehicles
            tracked_vehicles = self.vehicle_tracker.update(vehicle_detections)
            
            # 3. Process license plates for tracked vehicles
            for vehicle in tracked_vehicles:
                bbox = vehicle["bbox"]
                
                # Crop vehicle region with some padding
                x1, y1, x2, y2 = bbox
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                vehicle_crop = frame[y1:y2, x1:x2]
                
                if vehicle_crop.size > 0:
                    # Detect and recognize license plate
                    license_results = self.license_plate_processor.process_frame(vehicle_crop)
                    
                    if license_results:
                        best_result = max(license_results, key=lambda x: x["ocr_confidence"])
                        vehicle["license_plate"] = best_result["text"]
                        vehicle["license_confidence"] = best_result["ocr_confidence"]
            
            # 4. Detect entry/exit events
            events = self.vehicle_tracker.detect_entry_exit(tracked_vehicles)
            
            # 5. Handle entry events
            for entry in events["entries"]:
                await self._handle_vehicle_entry(entry, tracked_vehicles)
            
            # 6. Handle exit events
            for exit_event in events["exits"]:
                await self._handle_vehicle_exit(exit_event, tracked_vehicles)
            
            # 7. Create visualization frame
            self.processed_frame = self._create_visualization(frame, tracked_vehicles, vehicle_detections)
            
            # Log performance
            processing_time = (time.time() - start_time) * 1000
            performance_logger.log_detection_time(f"frame_processing_{self.camera_config.id}", processing_time)
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    async def _handle_vehicle_entry(self, entry_event: Dict[str, Any], tracked_vehicles: List[Dict[str, Any]]):
        """Handle vehicle entry event"""
        try:
            track_id = entry_event["track_id"]
            
            # Find vehicle with this track ID
            vehicle = None
            for v in tracked_vehicles:
                if v["track_id"] == track_id:
                    vehicle = v
                    break
            
            if not vehicle:
                return
            
            license_plate = vehicle.get("license_plate", f"UNKNOWN_{track_id}")
            vehicle_type = vehicle.get("class_name", "car")
            confidence = vehicle.get("license_confidence", 0.0)
            
            # Add to database
            db_vehicle = self.db_manager.add_vehicle_entry(
                license_plate=license_plate,
                vehicle_type=vehicle_type,
                entry_camera=self.camera_config.id,
                track_id=track_id,
                confidence=confidence,
                metadata={
                    "bbox": vehicle["bbox"],
                    "entry_event": entry_event
                }
            )
            
            if db_vehicle:
                system_logger.log_vehicle_entry(license_plate, self.camera_config.id)
            
        except Exception as e:
            self.logger.error(f"Error handling vehicle entry: {e}")
    
    async def _handle_vehicle_exit(self, exit_event: Dict[str, Any], tracked_vehicles: List[Dict[str, Any]]):
        """Handle vehicle exit event"""
        try:
            track_id = exit_event["track_id"]
            
            # Find vehicle with this track ID
            vehicle = None
            for v in tracked_vehicles:
                if v["track_id"] == track_id:
                    vehicle = v
                    break
            
            if not vehicle:
                return
            
            license_plate = vehicle.get("license_plate", f"UNKNOWN_{track_id}")
            duration_frames = exit_event.get("duration", 0)
            duration_minutes = duration_frames / 30 / 60  # Assuming 30 FPS
            
            # Update database
            db_vehicle = self.db_manager.add_vehicle_exit(
                license_plate=license_plate,
                exit_camera=self.camera_config.id
            )
            
            if db_vehicle:
                system_logger.log_vehicle_exit(license_plate, self.camera_config.id, duration_minutes)
            
        except Exception as e:
            self.logger.error(f"Error handling vehicle exit: {e}")
    
    def _create_visualization(
        self, 
        frame: np.ndarray, 
        tracked_vehicles: List[Dict[str, Any]], 
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create visualization frame with detections and tracking"""
        vis_frame = frame.copy()
        
        # Draw vehicle detections
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.putText(
                vis_frame,
                f"{class_name}: {confidence:.2f}",
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Draw tracked vehicles
        for vehicle in tracked_vehicles:
            bbox = vehicle["bbox"]
            track_id = vehicle["track_id"]
            class_name = vehicle["class_name"]
            license_plate = vehicle.get("license_plate", "")
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            
            # Draw track ID
            cv2.putText(
                vis_frame,
                f"ID: {track_id}",
                (bbox[0], bbox[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            
            # Draw license plate if available
            if license_plate:
                cv2.putText(
                    vis_frame,
                    license_plate,
                    (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
        
        # Draw camera info
        cv2.putText(
            vis_frame,
            f"Camera: {self.camera_config.id} | FPS: {self.fps_counter:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw frame count
        cv2.putText(
            vis_frame,
            f"Frame: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return vis_frame
    
    def _update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # Log FPS
            performance_logger.log_fps(self.camera_config.id, self.fps_counter)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current processed frame"""
        return self.processed_frame
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.logger.info(f"Stopped video processor for camera {self.camera_config.id}")


class ParkingSystem:
    """Main parking system coordinator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize database
        self.db_manager = DatabaseManager(config.database.connection_string)
        
        # Initialize AI models
        self._initialize_models()
        
        # Video processors for each camera
        self.video_processors: Dict[str, VideoProcessor] = {}
        
        # System status
        self.is_running = False
        
    def _initialize_models(self):
        """Initialize AI models"""
        self.logger.info("Initializing AI models...")
        
        try:
            # Vehicle detection model
            self.vehicle_detector = VehicleDetector(
                model_path=self.config.models.vehicle_detection_model,
                confidence=self.config.models.confidence_threshold,
                iou_threshold=self.config.models.iou_threshold,
                input_size=tuple(self.config.models.input_size)
            )
            
            # License plate processor
            self.license_plate_processor = LicensePlateProcessor(
                detection_model_path=self.config.models.license_plate_model,
                ocr_engine=self.config.models.ocr_engine,
                detection_confidence=self.config.models.confidence_threshold,
                ocr_confidence=0.7
            )
            
            # Vehicle tracker
            self.vehicle_tracker = VehicleTracker()
            
            self.logger.info("AI models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    async def start(self):
        """Start the parking system"""
        self.logger.info("Starting Smart Parking System...")
        
        try:
            # Initialize parking slots in database
            self._initialize_parking_slots()
            
            # Start video processors for enabled cameras
            enabled_cameras = self.config.get_enabled_cameras()
            
            for camera_config in enabled_cameras:
                self.logger.info(f"Starting processor for camera {camera_config.id}")
                
                # Create video processor
                processor = VideoProcessor(
                    camera_config=camera_config,
                    vehicle_detector=self.vehicle_detector,
                    license_plate_processor=self.license_plate_processor,
                    vehicle_tracker=VehicleTracker(),  # Each camera gets its own tracker
                    db_manager=self.db_manager
                )
                
                self.video_processors[camera_config.id] = processor
                
                # Start processor in background
                asyncio.create_task(processor.start())
            
            self.is_running = True
            self.logger.info(f"Parking system started with {len(enabled_cameras)} cameras")
            
            # Start monitoring loop
            await self._monitoring_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start parking system: {e}")
            raise
    
    def _initialize_parking_slots(self):
        """Initialize parking slots in database"""
        try:
            self.db_manager.initialize_parking_slots(
                total_slots=self.config.parking.total_slots,
                zones=self.config.parking.zones
            )
            self.logger.info(f"Initialized {self.config.parking.total_slots} parking slots")
        except Exception as e:
            self.logger.error(f"Failed to initialize parking slots: {e}")
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                # Log system metrics
                active_processors = len([p for p in self.video_processors.values() if p.is_running])
                self.db_manager.log_metric("active_cameras", active_processors)
                
                # Get parking status
                status = self.db_manager.get_parking_status()
                self.db_manager.log_metric("occupancy_rate", status["occupancy_rate"])
                self.db_manager.log_metric("occupied_slots", status["occupied_slots"])
                
                # Log FPS for each camera
                for camera_id, processor in self.video_processors.items():
                    self.db_manager.log_metric(f"fps_{camera_id}", processor.fps_counter, "fps", camera_id)
                
                # Check for full parking
                if status["occupancy_rate"] >= 95:
                    system_logger.log_parking_full()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    def get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        if camera_id in self.video_processors:
            return self.video_processors[camera_id].get_current_frame()
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        camera_status = {}
        for camera_id, processor in self.video_processors.items():
            camera_status[camera_id] = {
                "is_running": processor.is_running,
                "fps": processor.fps_counter,
                "frame_count": processor.frame_count
            }
        
        return {
            "is_running": self.is_running,
            "cameras": camera_status,
            "total_cameras": len(self.video_processors),
            "active_cameras": len([p for p in self.video_processors.values() if p.is_running])
        }
    
    def stop(self):
        """Stop the parking system"""
        self.logger.info("Stopping parking system...")
        
        self.is_running = False
        
        # Stop all video processors
        for processor in self.video_processors.values():
            processor.stop()
        
        self.logger.info("Parking system stopped")
