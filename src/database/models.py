"""
Database models for Smart Parking System using SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()


class Vehicle(Base):
    """Vehicle table model"""
    __tablename__ = 'vehicles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    license_plate = Column(String(20), unique=True, nullable=False, index=True)
    vehicle_type = Column(String(20), nullable=False)  # car, motorcycle, bus, truck
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    entry_camera = Column(String(50), nullable=False)
    exit_camera = Column(String(50), nullable=True)
    parking_zone = Column(String(50), nullable=True)
    slot_number = Column(Integer, nullable=True)
    track_id = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata
    metadata_json = Column(Text, nullable=True)  # Store additional data as JSON
    
    def __repr__(self):
        return f"<Vehicle(license_plate='{self.license_plate}', type='{self.vehicle_type}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'license_plate': self.license_plate,
            'vehicle_type': self.vehicle_type,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_camera': self.entry_camera,
            'exit_camera': self.exit_camera,
            'parking_zone': self.parking_zone,
            'slot_number': self.slot_number,
            'track_id': self.track_id,
            'confidence_score': self.confidence_score,
            'is_active': self.is_active,
            'parking_duration': self.get_parking_duration(),
            'metadata': self.get_metadata()
        }
    
    def get_parking_duration(self):
        """Get parking duration in minutes"""
        if self.entry_time:
            end_time = self.exit_time if self.exit_time else datetime.utcnow()
            duration = end_time - self.entry_time
            return duration.total_seconds() / 60
        return 0
    
    def get_metadata(self):
        """Get metadata as dictionary"""
        if self.metadata_json:
            try:
                return json.loads(self.metadata_json)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_metadata(self, metadata_dict):
        """Set metadata from dictionary"""
        self.metadata_json = json.dumps(metadata_dict)


class ParkingSlot(Base):
    """Parking slot table model"""
    __tablename__ = 'parking_slots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    slot_number = Column(Integer, unique=True, nullable=False, index=True)
    zone_name = Column(String(50), nullable=False, index=True)
    is_occupied = Column(Boolean, default=False, nullable=False)
    occupied_by = Column(String(20), nullable=True)  # license plate
    occupied_since = Column(DateTime, nullable=True)
    coordinates = Column(Text, nullable=True)  # JSON string for slot coordinates
    camera_id = Column(String(50), nullable=True)
    status = Column(String(20), default='available')  # available, occupied, reserved, maintenance
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ParkingSlot(number={self.slot_number}, zone='{self.zone_name}', occupied={self.is_occupied})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'slot_number': self.slot_number,
            'zone_name': self.zone_name,
            'is_occupied': self.is_occupied,
            'occupied_by': self.occupied_by,
            'occupied_since': self.occupied_since.isoformat() if self.occupied_since else None,
            'coordinates': self.get_coordinates(),
            'camera_id': self.camera_id,
            'status': self.status,
            'occupied_duration': self.get_occupied_duration()
        }
    
    def get_coordinates(self):
        """Get coordinates as dictionary"""
        if self.coordinates:
            try:
                return json.loads(self.coordinates)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_coordinates(self, coords_dict):
        """Set coordinates from dictionary"""
        self.coordinates = json.dumps(coords_dict)
    
    def get_occupied_duration(self):
        """Get occupied duration in minutes"""
        if self.occupied_since:
            duration = datetime.utcnow() - self.occupied_since
            return duration.total_seconds() / 60
        return 0


class Camera(Base):
    """Camera table model"""
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    rtsp_url = Column(String(500), nullable=False)
    position = Column(String(100), nullable=False)  # entrance, exit, parking_zone_a, etc.
    resolution_width = Column(Integer, default=1280)
    resolution_height = Column(Integer, default=720)
    fps = Column(Integer, default=30)
    is_enabled = Column(Boolean, default=True)
    status = Column(String(20), default='offline')  # online, offline, error
    last_frame_time = Column(DateTime, nullable=True)
    total_detections = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Camera(id='{self.camera_id}', position='{self.position}', status='{self.status}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'name': self.name,
            'rtsp_url': self.rtsp_url,
            'position': self.position,
            'resolution': [self.resolution_width, self.resolution_height],
            'fps': self.fps,
            'is_enabled': self.is_enabled,
            'status': self.status,
            'last_frame_time': self.last_frame_time.isoformat() if self.last_frame_time else None,
            'total_detections': self.total_detections,
            'error_count': self.error_count
        }


class DetectionLog(Base):
    """Detection log table for storing AI detection results"""
    __tablename__ = 'detection_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), nullable=False, index=True)
    detection_type = Column(String(20), nullable=False)  # vehicle, license_plate
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=True)
    license_plate = Column(String(20), nullable=True)
    track_id = Column(Integer, nullable=True)
    frame_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    
    def __repr__(self):
        return f"<DetectionLog(camera='{self.camera_id}', type='{self.detection_type}', confidence={self.confidence})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'detection_type': self.detection_type,
            'confidence': self.confidence,
            'bbox': [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            'class_name': self.class_name,
            'license_plate': self.license_plate,
            'track_id': self.track_id,
            'frame_timestamp': self.frame_timestamp.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'model_version': self.model_version
        }


class SystemMetrics(Base):
    """System performance metrics table"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    camera_id = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat()
        }


class ParkingEvent(Base):
    """Parking events table for tracking entry/exit events"""
    __tablename__ = 'parking_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(20), nullable=False, index=True)  # entry, exit
    license_plate = Column(String(20), nullable=False, index=True)
    vehicle_type = Column(String(20), nullable=False)
    camera_id = Column(String(50), nullable=False)
    track_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    event_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Float, nullable=True)
    metadata_json = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ParkingEvent(type='{self.event_type}', plate='{self.license_plate}', time='{self.event_time}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'license_plate': self.license_plate,
            'vehicle_type': self.vehicle_type,
            'camera_id': self.camera_id,
            'track_id': self.track_id,
            'confidence': self.confidence,
            'event_time': self.event_time.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'metadata': self.get_metadata()
        }
    
    def get_metadata(self):
        """Get metadata as dictionary"""
        if self.metadata_json:
            try:
                return json.loads(self.metadata_json)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_metadata(self, metadata_dict):
        """Set metadata from dictionary"""
        self.metadata_json = json.dumps(metadata_dict)


def create_database(connection_string: str):
    """Create database tables"""
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()
