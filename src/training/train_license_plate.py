"""
License Plate Detection Training Script
Script huấn luyện mô hình YOLOv8 để phát hiện biển số xe
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


def create_license_plate_dataset_yaml(data_dir: Path, output_path: Path):
    """Tạo file dataset.yaml cho YOLO training biển số"""
    
    dataset_config = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # Number of classes (chỉ có 1 class: license_plate)
        'names': {
            0: 'license_plate'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Đã tạo license plate dataset config: {output_path}")


def train_license_plate_detection(
    data_yaml: str,
    model_size: str = "yolov8s",
    epochs: int = 150,
    batch_size: int = 32,
    img_size: int = 640,
    device: str = "0",
    project: str = "license_plate_detection",
    name: str = "train"
):
    """
    Huấn luyện mô hình YOLOv8 để phát hiện biển số xe
    
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
    logger.info("🚀 Bắt đầu huấn luyện mô hình phát hiện biển số...")
    
    try:
        # Load model
        model = YOLO(f"{model_size}.pt")
        logger.info(f"📦 Đã load model: {model_size}")
        
        # Start training với hyperparameters tối ưu cho biển số
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            save=True,
            save_period=10,
            cache=True,
            amp=True,
            patience=30,  # Patience cao hơn cho biển số
            
            # Hyperparameters tối ưu cho biển số xe
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Data augmentation phù hợp với biển số
            hsv_h=0.01,  # Ít thay đổi hue
            hsv_s=0.5,   # Vừa phải saturation
            hsv_v=0.3,   # Ít thay đổi brightness
            degrees=5.0, # Ít rotation
            translate=0.05, # Ít dịch chuyển
            scale=0.3,   # Ít scale
            shear=2.0,   # Một chút perspective
            perspective=0.0001,
            flipud=0.0,  # Không lật dọc
            fliplr=0.2,  # Ít lật ngang
            mosaic=0.8,  # Giảm mosaic
            mixup=0.1,   # Ít mixup
            copy_paste=0.0,
            
            # Class weights (nếu cần)
            cls=0.5,
            box=7.5,
            dfl=1.5,
            
            # NMS settings
            iou=0.7,
            conf=0.25
        )
        
        logger.info("✅ Hoàn thành training biển số!")
        logger.info(f"📊 Kết quả training:")
        logger.info(f"   - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"   - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        logger.info(f"   - Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
        logger.info(f"   - Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Save best model
        best_model_path = Path(project) / name / "weights" / "best.pt"
        if best_model_path.exists():
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(best_model_path, models_dir / "license_plate_yolov8.pt")
            logger.info(f"💾 Đã lưu model biển số tốt nhất: data/models/license_plate_yolov8.pt")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Lỗi training biển số: {e}")
        raise


def prepare_license_plate_dataset(data_dir: Path):
    """Tạo cấu trúc dataset cho biển số xe"""
    
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
    
    print(f"✅ Đã tạo cấu trúc dataset biển số tại: {data_dir}")
    print("📝 Hướng dẫn chuẩn bị data biển số:")
    print("   1. Thu thập ảnh có chứa biển số xe (tối thiểu 5000+ ảnh)")
    print("   2. Annotation biển số với bounding box")
    print("   3. Format YOLO: 0 x_center y_center width height (normalized)")
    print("   4. Chia dataset: 70% train, 20% val, 10% test")
    print("   5. Đảm bảo chất lượng ảnh tốt và đa dạng điều kiện ánh sáng")
    
    # Tạo file hướng dẫn
    guide_file = data_dir / "annotation_guide.txt"
    with open(guide_file, "w", encoding="utf-8") as f:
        f.write("""
Hướng dẫn tạo dataset biển số xe Việt Nam

1. Thu thập dữ liệu:
   - Ảnh xe có biển số rõ ràng
   - Đa dạng góc chụp: thẳng, nghiêng, xa, gần
   - Đa dạng điều kiện: ban ngày, ban đêm, mưa, nắng
   - Đa dạng loại xe: ô tô, xe máy, xe tải
   - Đa dạng loại biển số: trắng, vàng, xanh

2. Annotation:
   - Sử dụng LabelImg hoặc Roboflow
   - Chỉ annotation vùng biển số (không annotation xe)
   - Bounding box bao trọn biển số
   - Class: 0 (license_plate)

3. Định dạng YOLO:
   - File .txt cùng tên với ảnh
   - Mỗi dòng: class_id x_center y_center width height
   - Tất cả giá trị normalized từ 0-1
   - Ví dụ: 0 0.5 0.3 0.2 0.1

4. Kiểm tra chất lượng:
   - Ảnh không mờ, không bị che khuất
   - Biển số đọc được bằng mắt thường
   - Annotation chính xác, không thiếu sót

5. Dataset tốt:
   - Tối thiểu 5000 ảnh cho training
   - 1000 ảnh cho validation
   - 500 ảnh cho testing
   - Cân bằng các loại biển số và điều kiện
""")
    
    print(f"📖 Đã tạo hướng dẫn: {guide_file}")


def main():
    parser = argparse.ArgumentParser(description="License Plate Detection Training")
    parser.add_argument("--data", type=str, default="data/license_plate_dataset",
                       help="Đường dẫn đến thư mục dataset biển số")
    parser.add_argument("--model", type=str, default="yolov8s",
                       choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                       help="Kích thước model YOLOv8")
    parser.add_argument("--epochs", type=int, default=150,
                       help="Số epochs training")
    parser.add_argument("--batch", type=int, default=32,
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
        prepare_license_plate_dataset(data_dir)
        return
    
    # Kiểm tra dataset
    if not data_dir.exists():
        print(f"❌ Không tìm thấy dataset biển số: {data_dir}")
        print("💡 Sử dụng --setup-only để tạo cấu trúc dataset")
        return
    
    # Tạo dataset.yaml
    dataset_yaml = data_dir / "dataset.yaml"
    create_license_plate_dataset_yaml(data_dir, dataset_yaml)
    
    # Bắt đầu training
    train_license_plate_detection(
        data_yaml=str(dataset_yaml),
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
