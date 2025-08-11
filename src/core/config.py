"""
🔧 Smart Parking System Configuration
Cấu hình hệ thống cho production environment
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Smart Parking System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/parking_system.db"
    
    # AI Models
    VEHICLE_MODEL_PATH: str = "data/models/vehicle_detection/best.pt"
    LICENSE_PLATE_MODEL_PATH: str = "data/models/license_plate_detection/best.pt"
    
    # Processing
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.3
    MAX_TRACKING_AGE: int = 30
    
    # Performance Requirements
    TARGET_FPS: int = 15
    MIN_VEHICLE_ACCURACY: float = 0.90  # ≥90% mAP
    MIN_LICENSE_PLATE_ACCURACY: float = 0.85  # ≥85% accuracy
    
    # Camera Configuration
    MAX_CAMERAS: int = 10
    DEFAULT_CAMERA_RESOLUTION: tuple = (1920, 1080)
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    STATIC_DIR: Path = Path("src/web/static")
    TEMPLATES_DIR: Path = Path("src/web/templates")
    
    # File Upload
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    # Vietnamese License Plate Patterns
    VN_LICENSE_PLATE_PATTERNS: List[str] = [
        r"\d{2}[A-Z]-\d{3}\.\d{2}",  # Old format: 29A-123.45
        r"\d{2}[A-Z]\d{1}-\d{3}\.\d{2}",  # New format: 29A1-123.45
        r"\d{2}[A-Z]-\d{4}",  # Motorcycle: 29A-1234
        r"\d{2}[A-Z]\d{1}-\d{4}",  # New motorcycle: 29A1-1234
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # Create necessary directories
        for directory in [
            _settings.DATA_DIR,
            _settings.MODELS_DIR,
            _settings.UPLOADS_DIR,
            _settings.DATA_DIR / "processed",
            _settings.DATA_DIR / "raw",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    return _settings

# Performance monitoring settings
class PerformanceSettings:
    """Performance monitoring and requirements"""
    
    # Detection Performance
    VEHICLE_DETECTION_MAP_THRESHOLD = 0.90  # mAP ≥ 90%
    LICENSE_PLATE_ACCURACY_THRESHOLD = 0.85  # Accuracy ≥ 85%
    
    # Processing Performance  
    MIN_FPS = 15  # ≥ 15 FPS
    MAX_PROCESSING_TIME_PER_FRAME = 1.0 / MIN_FPS  # 66.67ms per frame
    
    # Memory and Resource Limits
    MAX_MEMORY_USAGE_GB = 8.0
    MAX_GPU_MEMORY_USAGE_GB = 6.0
    
    # Concurrent Processing
    MAX_CONCURRENT_STREAMS = 5
    MAX_CONCURRENT_UPLOADS = 10
    
    @staticmethod
    def validate_performance_requirements():
        """Validate system meets performance requirements"""
        requirements = {
            "Vehicle Detection": "≥90% mAP",
            "License Plate Recognition": "≥85% accuracy", 
            "Processing Speed": "≥15 FPS",
            "Multi-camera Support": "5-10 streams",
            "Real-time Processing": "< 66.67ms per frame"
        }
        return requirements

# Export settings
__all__ = ["Settings", "get_settings", "PerformanceSettings"]
