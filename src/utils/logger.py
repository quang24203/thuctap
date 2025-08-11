"""
Logging setup for Smart Parking System
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_file: str = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup logger with both console and file handlers
    
    Args:
        name: Logger name (None for root logger)
        level: Logging level
        log_file: Log file path (auto-generated if None)
        max_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Auto-generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"parking_system_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler with Rich formatting
    console = Console()
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)
    logging.getLogger("opencv").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
    
    def log_fps(self, camera_id: str, fps: float):
        """Log FPS for a camera"""
        self.logger.info(f"Camera {camera_id}: {fps:.2f} FPS")
        self.metrics[f"{camera_id}_fps"] = fps
    
    def log_detection_time(self, model: str, time_ms: float):
        """Log detection time for a model"""
        self.logger.debug(f"{model} detection: {time_ms:.2f}ms")
        self.metrics[f"{model}_time"] = time_ms
    
    def log_accuracy(self, model: str, accuracy: float):
        """Log model accuracy"""
        self.logger.info(f"{model} accuracy: {accuracy:.2f}%")
        self.metrics[f"{model}_accuracy"] = accuracy
    
    def log_database_query(self, query_type: str, time_ms: float):
        """Log database query performance"""
        self.logger.debug(f"DB {query_type}: {time_ms:.2f}ms")
        self.metrics[f"db_{query_type}"] = time_ms
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()


class SystemLogger:
    """Logger for system events"""
    
    def __init__(self, logger_name: str = "system"):
        self.logger = logging.getLogger(logger_name)
    
    def log_vehicle_entry(self, license_plate: str, camera_id: str):
        """Log vehicle entry"""
        self.logger.info(f"Vehicle entered: {license_plate} via {camera_id}")
    
    def log_vehicle_exit(self, license_plate: str, camera_id: str, duration: float):
        """Log vehicle exit"""
        self.logger.info(f"Vehicle exited: {license_plate} via {camera_id}, duration: {duration:.1f}min")
    
    def log_parking_full(self):
        """Log parking lot full event"""
        self.logger.warning("Parking lot is full!")
    
    def log_camera_error(self, camera_id: str, error: str):
        """Log camera error"""
        self.logger.error(f"Camera {camera_id} error: {error}")
    
    def log_model_error(self, model_name: str, error: str):
        """Log model error"""
        self.logger.error(f"Model {model_name} error: {error}")
    
    def log_database_error(self, operation: str, error: str):
        """Log database error"""
        self.logger.error(f"Database {operation} error: {error}")


# Global logger instances
performance_logger = PerformanceLogger()
system_logger = SystemLogger()
