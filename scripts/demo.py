"""
Demo script để test hệ thống Smart Parking
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config
from src.utils.simple_logger import setup_logging, get_logger
from src.models.vehicle_detection import VehicleDetector
from src.models.license_plate import LicensePlateProcessor
from src.models.tracking import VehicleTracker, Detection
from src.database.operations import DatabaseManager


def create_demo_video(width=1280, height=720, duration_seconds=30, fps=30):
    """Tạo video demo với xe di chuyển"""
    
    total_frames = duration_seconds * fps
    frames = []
    
    for frame_idx in range(total_frames):
        # Tạo background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Gray background
        
        # Vẽ đường
        cv2.rectangle(frame, (0, height//2-50), (width, height//2+50), (80, 80, 80), -1)
        
        # Vẽ làn đường
        for x in range(0, width, 100):
            cv2.rectangle(frame, (x, height//2-5), (x+50, height//2+5), (255, 255, 255), -1)
        
        # Tạo xe di chuyển
        cars = [
            {"x": (frame_idx * 5) % (width + 200) - 100, "y": height//2 - 30, "color": (0, 0, 255)},
            {"x": (frame_idx * 3 + 300) % (width + 200) - 100, "y": height//2, "color": (255, 0, 0)},
            {"x": (frame_idx * 4 + 600) % (width + 200) - 100, "y": height//2 + 20, "color": (0, 255, 0)}
        ]
        
        for car in cars:
            if 0 <= car["x"] <= width - 80:
                # Vẽ xe
                cv2.rectangle(frame, (car["x"], car["y"]), (car["x"]+80, car["y"]+40), car["color"], -1)
                # Vẽ biển số
                cv2.rectangle(frame, (car["x"]+10, car["y"]+45), (car["x"]+70, car["y"]+55), (255, 255, 255), -1)
                cv2.putText(frame, "51A123", (car["x"]+15, car["y"]+52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        frames.append(frame)
    
    return frames


def demo_detection_only(model_path=None):
    """Demo chỉ detection không cần camera thật"""

    setup_logging(level="INFO")
    logger = get_logger("Demo")
    logger.info("🎬 Bắt đầu demo detection...")
    
    # Tạo video demo
    demo_frames = create_demo_video(duration_seconds=10)
    
    try:
        # Load model (sử dụng model mặc định nếu không có custom model)
        if model_path and Path(model_path).exists():
            detector = VehicleDetector(model_path)
        else:
            logger.warning("Sử dụng YOLOv8 model mặc định")
            detector = VehicleDetector("yolov8n.pt")  # Model nhỏ để demo
        
        logger.info("✅ Đã load model detection")
        
        # Process frames
        for i, frame in enumerate(demo_frames):
            # Detection
            detections = detector.detect(frame)
            
            # Visualize
            vis_frame = detector.visualize_detections(frame, detections)
            
            # Add info
            cv2.putText(vis_frame, f"Frame: {i+1}/{len(demo_frames)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Detections: {len(detections)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Smart Parking Demo - Detection", vis_frame)
            
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        logger.info("✅ Demo detection hoàn thành")
        
    except Exception as e:
        logger.error(f"❌ Lỗi demo: {e}")


def demo_full_system():
    """Demo hệ thống đầy đủ với database"""

    setup_logging(level="INFO")
    logger = get_logger("Demo")
    logger.info("🎬 Bắt đầu demo hệ thống đầy đủ...")
    
    try:
        # Load config
        config = Config()
        
        # Initialize database
        db_manager = DatabaseManager(config.database.connection_string)
        
        # Initialize models
        detector = VehicleDetector("yolov8n.pt")
        license_processor = LicensePlateProcessor("yolov8n.pt")  # Placeholder
        tracker = VehicleTracker()
        
        logger.info("✅ Đã khởi tạo tất cả components")
        
        # Demo data
        demo_frames = create_demo_video(duration_seconds=15)
        
        for i, frame in enumerate(demo_frames):
            # Detection
            vehicle_detections = detector.detect(frame)
            
            # Convert to tracker format
            detections = []
            for det in vehicle_detections:
                detection = Detection(
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    class_id=det["class_id"]
                )
                detections.append(detection)
            
            # Tracking
            tracked_vehicles = tracker.update(detections)
            
            # Simulate license plate recognition
            for vehicle in tracked_vehicles:
                if np.random.random() > 0.7:  # 30% chance to "recognize" plate
                    vehicle["license_plate"] = f"51A-{np.random.randint(100, 999)}.{np.random.randint(10, 99)}"
                    vehicle["license_confidence"] = np.random.uniform(0.8, 0.95)
            
            # Simulate entry/exit events
            if i > 0:  # Skip first frame
                for vehicle in tracked_vehicles:
                    # Random entry event
                    if np.random.random() > 0.95 and "license_plate" in vehicle:
                        license_plate = vehicle["license_plate"]
                        
                        # Add to database
                        db_vehicle = db_manager.add_vehicle_entry(
                            license_plate=license_plate,
                            vehicle_type="car",
                            entry_camera="demo_cam",
                            track_id=vehicle["track_id"],
                            confidence=vehicle.get("license_confidence", 0.0)
                        )
                        
                        if db_vehicle:
                            logger.info(f"🚗 Xe vào: {license_plate}")
            
            # Visualization
            vis_frame = frame.copy()
            
            # Draw tracked vehicles
            for vehicle in tracked_vehicles:
                bbox = vehicle["bbox"]
                track_id = vehicle["track_id"]
                
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(vis_frame, f"ID: {track_id}", 
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if "license_plate" in vehicle:
                    cv2.putText(vis_frame, vehicle["license_plate"], 
                               (bbox[0], bbox[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add system info
            status = db_manager.get_parking_status()
            cv2.putText(vis_frame, f"Frame: {i+1}/{len(demo_frames)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Tracked: {len(tracked_vehicles)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Active Vehicles: {status['active_vehicles']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Occupancy: {status['occupancy_rate']:.1f}%", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Smart Parking Demo - Full System", vis_frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):  # Slower playback
                break
        
        cv2.destroyAllWindows()
        
        # Show final statistics
        final_status = db_manager.get_parking_status()
        recent_vehicles = db_manager.get_recent_vehicles(limit=10)
        
        logger.info("📊 Thống kê cuối demo:")
        logger.info(f"   - Tổng xe hiện tại: {final_status['active_vehicles']}")
        logger.info(f"   - Tỷ lệ lấp đầy: {final_status['occupancy_rate']:.1f}%")
        logger.info(f"   - Xe gần đây:")
        
        for vehicle in recent_vehicles[:5]:
            logger.info(f"     {vehicle['license_plate']} - {vehicle['vehicle_type']}")
        
        logger.info("✅ Demo hệ thống hoàn thành")
        
    except Exception as e:
        logger.error(f"❌ Lỗi demo: {e}")


def demo_web_interface():
    """Demo web interface"""

    setup_logging(level="INFO")
    logger = get_logger("Demo")
    logger.info("🌐 Khởi động web interface demo...")
    
    try:
        # Import web app
        from src.web.app import create_app
        from src.utils.config import Config
        
        config = Config()
        app = create_app(config)
        
        logger.info("🚀 Khởi động web server...")
        logger.info("📱 Truy cập: http://localhost:5000")
        logger.info("⏹️  Nhấn Ctrl+C để dừng")
        
        app.run(host="0.0.0.0", port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"❌ Lỗi web demo: {e}")


def main():
    parser = argparse.ArgumentParser(description="Smart Parking System Demo")
    parser.add_argument("--mode", choices=["detection", "full", "web"], 
                       default="detection", help="Chế độ demo")
    parser.add_argument("--model", type=str, help="Đường dẫn model custom")
    
    args = parser.parse_args()
    
    print("🎯 Smart Parking System Demo")
    print("=" * 50)
    
    if args.mode == "detection":
        demo_detection_only(args.model)
    elif args.mode == "full":
        demo_full_system()
    elif args.mode == "web":
        demo_web_interface()


if __name__ == "__main__":
    main()
