"""
Simple logger without rich dependency
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, 
                 format_string: Optional[str] = None):
    """Setup basic logging"""
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default format
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

# Performance logger placeholder
class PerformanceLogger:
    def log_detection_time(self, model_name: str, time_ms: float):
        logger = get_logger("Performance")
        logger.debug(f"{model_name}: {time_ms:.2f}ms")

performance_logger = PerformanceLogger()
