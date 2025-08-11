"""
Database setup script
Script thiết lập cơ sở dữ liệu cho hệ thống parking
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.database.models import create_database
from src.database.operations import DatabaseManager
from src.utils.config import Config


def setup_database():
    """Setup database and create initial data"""
    print("🔧 Thiết lập cơ sở dữ liệu...")
    
    # Load configuration
    config = Config()
    
    # Create database
    print(f"📊 Tạo database: {config.database.connection_string}")
    engine = create_database(config.database.connection_string)
    
    # Initialize database manager
    db_manager = DatabaseManager(config.database.connection_string)
    
    # Initialize parking slots
    print("🅿️ Khởi tạo parking slots...")
    db_manager.initialize_parking_slots(
        total_slots=config.parking.total_slots,
        zones=config.parking.zones
    )
    
    # Add sample cameras
    print("📹 Thêm thông tin cameras...")
    from src.database.models import Camera, get_session
    
    session = get_session(engine)
    
    for camera_config in config.cameras:
        existing_camera = session.query(Camera).filter(
            Camera.camera_id == camera_config.id
        ).first()
        
        if not existing_camera:
            camera = Camera(
                camera_id=camera_config.id,
                name=f"Camera {camera_config.id}",
                rtsp_url=camera_config.rtsp_url,
                position=camera_config.position,
                resolution_width=camera_config.resolution[0],
                resolution_height=camera_config.resolution[1],
                fps=camera_config.fps,
                is_enabled=camera_config.enabled,
                status='offline'
            )
            session.add(camera)
    
    session.commit()
    session.close()
    
    print("✅ Database setup hoàn thành!")
    print(f"   - Tổng số parking slots: {config.parking.total_slots}")
    print(f"   - Số zones: {len(config.parking.zones)}")
    print(f"   - Số cameras: {len(config.cameras)}")


if __name__ == "__main__":
    setup_database()
