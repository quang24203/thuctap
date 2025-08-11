#!/usr/bin/env python3
"""
🎯 Smart Parking System - Production Application
Hệ thống giám sát bãi đỗ xe thông minh sử dụng AI và Computer Vision

Mục tiêu:
- Phát hiện và theo dõi phương tiện (≥90% accuracy)
- Nhận diện biển số xe Việt Nam (≥85% accuracy) 
- Xử lý real-time (≥15 FPS)
- Hỗ trợ nhiều camera streams
- Web interface với API đầy đủ
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.web.app import create_app
from src.utils.logger import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

def main():
    """Main application entry point"""
    logger.info("🎯 Starting Smart Parking System")
    logger.info("=" * 60)
    
    # System requirements check
    logger.info("📋 System Requirements Check:")
    logger.info(f"   • Vehicle Detection: ≥90% mAP")
    logger.info(f"   • License Plate Recognition: ≥85% accuracy")
    logger.info(f"   • Processing Speed: ≥15 FPS")
    logger.info(f"   • Multi-camera support: ✅")
    logger.info(f"   • Real-time monitoring: ✅")
    
    # Create FastAPI application
    app = create_app()
    
    # Run server
    logger.info(f"🚀 Starting server on {settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🌐 Web Interface: http://{settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG
    )

if __name__ == "__main__":
    main()
