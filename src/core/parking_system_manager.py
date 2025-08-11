"""
Smart Parking System Manager
Integrates all components into a complete system
"""

import asyncio
import threading
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

from ..models.vehicle_detection import VehicleDetector
from ..models.license_plate import LicensePlateRecognizer
try:
    from ..models.tracking import MultiObjectTracker
except ImportError:
    from ..models.simple_tracker import MultiObjectTracker
from ..database.operations import DatabaseManager
from ..utils.config import Config
from ..utils.simple_logger import get_logger
from ..web.websocket_handler import (
    notify_vehicle_entry, notify_vehicle_exit, 
    notify_parking_status_change, notify_detection_result
)

class CameraProcessor:
    """Processes video stream from a single camera"""
    
    def __init__(self, camera_config: Dict[str, Any], models: Dict[str, Any]):
        self.camera_id = camera_config['id']
        self.camera_name = camera_config['name']
        self.rtsp_url = camera_config['rtsp_url']
        self.zone_type = camera_config['zone_type']
        self.resolution = camera_config['resolution']
        self.fps = camera_config['fps']
        
        # Models
        self.vehicle_detector = models['vehicle_detector']
        self.license_plate_recognizer = models['license_plate_recognizer']
        self.tracker = models['tracker']
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Statistics
        self.frames_processed = 0
        self.detections_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        self.logger = get_logger(f"Camera_{self.camera_id}")
    
    def start(self):
        """Start camera processing"""
        self.logger.info(f"Starting camera {self.camera_id}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.is_running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        capture_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_frames, daemon=True)
        process_thread.start()
        
        return True
    
    def stop(self):
        """Stop camera processing"""
        self.logger.info(f"Stopping camera {self.camera_id}")
        self.is_running = False
        
        if self.cap:
            self.cap.release()
    
    def _capture_frames(self):
        """Capture frames from camera"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                self.logger.error(f"Error capturing frame: {e}")
                time.sleep(1)
    
    def _process_frames(self):
        """Process captured frames"""
        while self.is_running:
            try:
                # Get frame from queue
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process frame
                self._process_single_frame(frame)
                
                # Update statistics
                self.frames_processed += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.frames_processed / (current_time - self.last_fps_time)
                    self.frames_processed = 0
                    self.last_fps_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                time.sleep(0.1)
    
    def _process_single_frame(self, frame: np.ndarray):
        """Process a single frame"""
        try:
            start_time = time.time()
            
            # Detect vehicles
            detections = self.vehicle_detector.detect(frame)
            
            if detections:
                self.detections_count += len(detections)
                
                # Track vehicles
                tracked_vehicles = self.tracker.update(detections)
                
                # Process each tracked vehicle
                for vehicle in tracked_vehicles:
                    self._process_vehicle(frame, vehicle)
            
            # Log processing time
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 100:  # Log if processing takes more than 100ms
                self.logger.warning(f"Slow processing: {processing_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error in frame processing: {e}")
    
    def _process_vehicle(self, frame: np.ndarray, vehicle: Dict[str, Any]):
        """Process detected vehicle"""
        try:
            # Extract vehicle region
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_region = frame[y1:y2, x1:x2]
            
            # Recognize license plate
            license_result = self.license_plate_recognizer.recognize(vehicle_region)
            
            if license_result:
                vehicle['license_plate'] = license_result['text']
                vehicle['license_confidence'] = license_result['confidence']
                
                # Handle entry/exit based on zone type
                if self.zone_type in ['entrance', 'exit']:
                    self._handle_entry_exit(vehicle)
            
            # Send detection result via WebSocket
            asyncio.create_task(notify_detection_result({
                'camera_id': self.camera_id,
                'vehicle': vehicle,
                'timestamp': datetime.utcnow().isoformat()
            }))
            
        except Exception as e:
            self.logger.error(f"Error processing vehicle: {e}")
    
    def _handle_entry_exit(self, vehicle: Dict[str, Any]):
        """Handle vehicle entry/exit"""
        try:
            license_plate = vehicle.get('license_plate')
            if not license_plate:
                return
            
            if self.zone_type == 'entrance':
                # Vehicle entering
                asyncio.create_task(notify_vehicle_entry({
                    'license_plate': license_plate,
                    'vehicle_type': vehicle.get('class_name', 'car'),
                    'camera_id': self.camera_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'confidence': vehicle.get('confidence', 0.0)
                }))
                
            elif self.zone_type == 'exit':
                # Vehicle exiting
                asyncio.create_task(notify_vehicle_exit({
                    'license_plate': license_plate,
                    'camera_id': self.camera_id,
                    'timestamp': datetime.utcnow().isoformat()
                }))
            
        except Exception as e:
            self.logger.error(f"Error handling entry/exit: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get camera processing statistics"""
        return {
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'is_running': self.is_running,
            'current_fps': self.current_fps,
            'frames_processed': self.frames_processed,
            'detections_count': self.detections_count,
            'queue_size': self.frame_queue.qsize()
        }

class SmartParkingSystemManager:
    """Main system manager that coordinates all components"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize database
        self.db_manager = DatabaseManager(config.database.connection_string)
        
        # Initialize AI models
        self.models = self._initialize_models()
        
        # Camera processors
        self.camera_processors: Dict[str, CameraProcessor] = {}
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Statistics
        self.system_stats = {
            'total_vehicles_detected': 0,
            'total_entries': 0,
            'total_exits': 0,
            'system_uptime': 0
        }
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize AI models"""
        self.logger.info("Initializing AI models...")
        
        try:
            # Vehicle detection model
            vehicle_detector = VehicleDetector(
                model_path=self.config.models.vehicle_detection['model_path'],
                confidence=self.config.models.vehicle_detection['confidence_threshold'],
                device=self.config.models.vehicle_detection['device']
            )
            
            # License plate recognition
            license_plate_recognizer = LicensePlateRecognizer(
                detection_model=self.config.models.license_plate['detection_model'],
                ocr_engine=self.config.models.license_plate['ocr_engine'],
                confidence=self.config.models.license_plate['confidence_threshold']
            )
            
            # Multi-object tracker
            tracker = MultiObjectTracker(
                track_thresh=self.config.models.tracking['track_thresh'],
                track_buffer=self.config.models.tracking['track_buffer'],
                match_thresh=self.config.models.tracking['match_thresh'],
                frame_rate=self.config.models.tracking['frame_rate']
            )
            
            self.logger.info("AI models initialized successfully")
            
            return {
                'vehicle_detector': vehicle_detector,
                'license_plate_recognizer': license_plate_recognizer,
                'tracker': tracker
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def start(self):
        """Start the parking system"""
        self.logger.info("Starting Smart Parking System...")
        
        try:
            # Initialize database tables
            self.db_manager.initialize_parking_slots(
                self.config.parking.total_slots,
                self.config.parking.zones
            )
            
            # Initialize camera processors
            for camera_config in self.config.cameras:
                if camera_config.get('enabled', True):
                    processor = CameraProcessor(camera_config, self.models)
                    self.camera_processors[camera_config['id']] = processor
            
            # Start camera processors
            for camera_id, processor in self.camera_processors.items():
                if processor.start():
                    self.logger.info(f"Camera {camera_id} started successfully")
                else:
                    self.logger.error(f"Failed to start camera {camera_id}")
            
            # Start statistics update thread
            stats_thread = threading.Thread(target=self._update_statistics, daemon=True)
            stats_thread.start()
            
            self.is_running = True
            self.start_time = datetime.utcnow()
            
            self.logger.info("Smart Parking System started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            raise
    
    def stop(self):
        """Stop the parking system"""
        self.logger.info("Stopping Smart Parking System...")
        
        self.is_running = False
        
        # Stop camera processors
        for camera_id, processor in self.camera_processors.items():
            processor.stop()
            self.logger.info(f"Camera {camera_id} stopped")
        
        self.logger.info("Smart Parking System stopped")
    
    def _update_statistics(self):
        """Update system statistics periodically"""
        while self.is_running:
            try:
                # Update system uptime
                if self.start_time:
                    uptime = (datetime.utcnow() - self.start_time).total_seconds()
                    self.system_stats['system_uptime'] = uptime
                
                # Get parking status
                parking_status = self.db_manager.get_parking_status()
                
                # Send status update via WebSocket
                asyncio.create_task(notify_parking_status_change(parking_status))
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error updating statistics: {e}")
                time.sleep(10)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        camera_stats = {}
        for camera_id, processor in self.camera_processors.items():
            camera_stats[camera_id] = processor.get_stats()
        
        # Get model performance stats
        model_stats = {}
        if 'vehicle_detector' in self.models:
            model_stats['vehicle_detection'] = self.models['vehicle_detector'].get_performance_stats()
        if 'license_plate_recognizer' in self.models:
            model_stats['license_plate'] = self.models['license_plate_recognizer'].get_performance_stats()
        if 'tracker' in self.models:
            model_stats['tracking'] = self.models['tracker'].get_performance_stats()
        
        return {
            'system': {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': self.system_stats['system_uptime']
            },
            'cameras': camera_stats,
            'models': model_stats,
            'parking': self.db_manager.get_parking_status() if self.is_running else None
        }
    
    def restart_camera(self, camera_id: str) -> bool:
        """Restart a specific camera"""
        try:
            if camera_id in self.camera_processors:
                processor = self.camera_processors[camera_id]
                processor.stop()
                time.sleep(2)  # Wait for cleanup
                return processor.start()
            return False
        except Exception as e:
            self.logger.error(f"Error restarting camera {camera_id}: {e}")
            return False
