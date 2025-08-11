"""
Vehicle Detection Training Script
Script huấn luyện mô hình YOLOv8 để phát hiện xe
"""

import sys
import os
from pathlib import Path
import argparse
from ultralytics import YOLO
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import setup_logger


def create_dataset_yaml(data_dir: Path, output_path: Path):
    """Tạo file dataset.yaml cho YOLO training"""
    
    dataset_config = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,  # Number of classes
        'names': {
            0: 'car',
            1: 'motorcycle', 
            2: 'bus',
            3: 'truck'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Đã tạo dataset config: {output_path}")


def train_vehicle_detection(
    data_yaml: str,
    model_size: str = "yolov8m",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "vehicle_detection",
    name: str = "train"
):
    """
    Huấn luyện mô hình YOLOv8 để phát hiện xe
    
    Args:
        data_yaml: Đường dẫn đến file dataset.yaml
        model_size: Kích thước model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Số epochs training
        batch_size: Batch size
        img_size: Kích thước ảnh input
        device: GPU device (0, 1, ... hoặc 'cpu')
        project: Tên project
        name: Tên experiment
    """
    
    logger = setup_logger()
    logger.info("🚀 Bắt đầu huấn luyện mô hình phát hiện xe...")
    
    try:
        # Load model
        model = YOLO(f"{model_size}.pt")
        logger.info(f"📦 Đã load model: {model_size}")
        
        # Start training
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            cache=True,
            amp=True,  # Automatic Mixed Precision
            patience=20,  # Early stopping patience
            
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        logger.info("✅ Hoàn thành training!")
        logger.info(f"📊 Kết quả training:")
        logger.info(f"   - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"   - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # Save best model to models directory
        best_model_path = Path(project) / name / "weights" / "best.pt"
        if best_model_path.exists():
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(best_model_path, models_dir / "vehicle_yolov8.pt")
            logger.info(f"💾 Đã lưu model tốt nhất: data/models/vehicle_yolov8.pt")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Lỗi training: {e}")
        raise


def prepare_dataset_structure(data_dir: Path):
    """Tạo cấu trúc thư mục cho dataset YOLO"""
    
    subdirs = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    for subdir in subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Đã tạo cấu trúc dataset tại: {data_dir}")
    print("📝 Hướng dẫn:")
    print("   1. Copy ảnh training vào images/train/")
    print("   2. Copy ảnh validation vào images/val/") 
    print("   3. Copy ảnh test vào images/test/")
    print("   4. Copy labels (.txt) tương ứng vào labels/train/, labels/val/, labels/test/")
    print("   5. Format labels: class_id x_center y_center width height (normalized 0-1)")


def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection Training")
    parser.add_argument("--data", type=str, default="data/vehicle_dataset", 
                       help="Đường dẫn đến thư mục dataset")
    parser.add_argument("--model", type=str, default="yolov8m",
                       choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                       help="Kích thước model YOLOv8")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Số epochs training")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Kích thước ảnh input")
    parser.add_argument("--device", type=str, default="0",
                       help="GPU device (0, 1, ... hoặc 'cpu')")
    parser.add_argument("--setup-only", action="store_true",
                       help="Chỉ tạo cấu trúc dataset, không training")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    
    if args.setup_only:
        prepare_dataset_structure(data_dir)
        return
    
    # Kiểm tra dataset
    if not data_dir.exists():
        print(f"❌ Không tìm thấy dataset: {data_dir}")
        print("💡 Sử dụng --setup-only để tạo cấu trúc dataset")
        return
    
    # Tạo dataset.yaml
    dataset_yaml = data_dir / "dataset.yaml"
    create_dataset_yaml(data_dir, dataset_yaml)
    
    # Bắt đầu training
    train_vehicle_detection(
        data_yaml=str(dataset_yaml),
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
