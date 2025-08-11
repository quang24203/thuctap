"""
Configuration management for Smart Parking System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Camera configuration"""
    id: str
    name: str
    rtsp_url: str
    position: str
    resolution: List[int] = field(default_factory=lambda: [1280, 720])
    fps: int = 30
    enabled: bool = True
    zone_type: str = "parking"  # "entrance", "exit", "parking"


@dataclass
class ModelConfig:
    """AI Model configuration"""
    # Vehicle detection
    vehicle_detection: Dict[str, Any] = field(default_factory=dict)

    # License plate detection & OCR
    license_plate: Dict[str, Any] = field(default_factory=dict)

    # Multi-object tracking
    tracking: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default values after initialization"""
        if not self.vehicle_detection:
            self.vehicle_detection = {
                "model_path": "data/models/vehicle_yolov8.pt",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.4,
                "input_size": [640, 640],
                "classes": ["car", "truck", "bus", "motorcycle"],
                "device": "cuda"
            }

        if not self.license_plate:
            self.license_plate = {
                "detection_model": "data/models/license_plate_yolov8.pt",
                "ocr_engine": "paddleocr",
                "languages": ["vi", "en"],
                "confidence_threshold": 0.7,
                "device": "cuda"
            }

        if not self.tracking:
            self.tracking = {
                "algorithm": "bytetrack",
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 30,
                "max_disappeared": 30,
                "max_distance": 100
            }


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"  # "sqlite" or "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "parking_system.db"
    username: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    @property
    def connection_string(self) -> str:
        """Get database connection string"""
        if self.type == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")


@dataclass
class ParkingConfig:
    """Parking lot configuration"""
    total_slots: int = 300
    zones: List[Dict[str, Any]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: str = "logs/app.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/app_config.yaml"
        self.config_data = self._load_config()
        
        # Initialize configuration objects
        self.cameras = self._load_cameras()
        self.models = self._load_models()
        self.database = self._load_database()
        self.parking = self._load_parking()
        self.logging = self._load_logging()

        # Application settings
        self.app_settings = self.config_data.get("app_settings", {})
        self.debug = self.app_settings.get("debug", False)
        self.log_level = self.app_settings.get("log_level", "INFO")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # Create default configuration if not exists
            self._create_default_config(config_path)
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self, config_path: Path):
        """Create default configuration file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "app": {
                "debug": False,
                "log_level": "INFO",
                "secret_key": "your-secret-key-change-this"
            },
            "cameras": [
                {
                    "id": "cam_01",
                    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
                    "position": "entrance",
                    "resolution": [1920, 1080],
                    "fps": 30,
                    "enabled": True
                },
                {
                    "id": "cam_02",
                    "rtsp_url": "rtsp://admin:password@192.168.1.101:554/stream1", 
                    "position": "parking_zone_a",
                    "resolution": [1280, 720],
                    "fps": 30,
                    "enabled": True
                }
            ],
            "models": {
                "vehicle_detection_model": "data/models/vehicle_yolov8.pt",
                "license_plate_model": "data/models/license_plate_yolov8.pt",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.4,
                "input_size": [640, 640],
                "ocr_engine": "paddleocr",
                "ocr_languages": ["vi", "en"],
                "tracking_algorithm": "bytetrack",
                "max_disappeared": 30,
                "max_distance": 100.0
            },
            "database": {
                "type": "sqlite",
                "database": "data/parking_system.db",
                "host": "localhost",
                "port": 5432,
                "username": "",
                "password": ""
            },
            "parking": {
                "total_slots": 300,
                "zones": [
                    {
                        "name": "Zone A",
                        "slots": 100,
                        "coordinates": [[100, 100], [500, 400]]
                    },
                    {
                        "name": "Zone B", 
                        "slots": 200,
                        "coordinates": [[600, 100], [1200, 600]]
                    }
                ],
                "entry_points": ["cam_01"],
                "exit_points": ["cam_01"]
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    def _load_cameras(self) -> List[CameraConfig]:
        """Load camera configurations"""
        cameras = []
        for cam_config in self.config_data.get("cameras", []):
            cameras.append(CameraConfig(**cam_config))
        return cameras
    
    def _load_models(self) -> ModelConfig:
        """Load model configuration"""
        model_config = self.config_data.get("models", {})
        return ModelConfig(
            vehicle_detection=model_config.get("vehicle_detection", {}),
            license_plate=model_config.get("license_plate", {}),
            tracking=model_config.get("tracking", {})
        )
    
    def _load_database(self) -> DatabaseConfig:
        """Load database configuration"""
        db_config = self.config_data.get("database", {})

        # Only pass valid DatabaseConfig fields
        valid_fields = {
            'type', 'host', 'port', 'database', 'username', 'password',
            'pool_size', 'max_overflow', 'pool_timeout'
        }

        filtered_config = {k: v for k, v in db_config.items() if k in valid_fields}
        return DatabaseConfig(**filtered_config)
    
    def _load_parking(self) -> ParkingConfig:
        """Load parking configuration"""
        parking_config = self.config_data.get("parking", {})
        return ParkingConfig(**parking_config)

    def _load_logging(self) -> LoggingConfig:
        """Load logging configuration"""
        logging_config = self.config_data.get("logging", {})

        # Only pass valid LoggingConfig fields
        valid_fields = {'level', 'file', 'format', 'max_bytes', 'backup_count'}
        filtered_config = {k: v for k, v in logging_config.items() if k in valid_fields}

        return LoggingConfig(**filtered_config)
    
    def get_camera_by_id(self, camera_id: str) -> CameraConfig:
        """Get camera configuration by ID"""
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        raise ValueError(f"Camera {camera_id} not found")
    
    def get_enabled_cameras(self) -> List[CameraConfig]:
        """Get list of enabled cameras"""
        return [cam for cam in self.cameras if cam.enabled]
    
    def reload(self):
        """Reload configuration from file"""
        self.__init__(self.config_path)
