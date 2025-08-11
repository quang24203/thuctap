#!/usr/bin/env python3
"""
Training script for license plate detection model using YOLOv8
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
import shutil
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO
from src.utils.config import Config
from src.utils.simple_logger import setup_logging, get_logger

def prepare_license_plate_dataset(data_dir: str, output_dir: str, train_split: float = 0.8, val_split: float = 0.15):
    """Prepare license plate dataset in YOLO format"""
    logger = get_logger("LicensePlateDatasetPreparation")
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    data_path = Path(data_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(data_path.glob(f"**/{ext}"))
    
    logger.info(f"Found {len(image_files)} license plate images")
    
    # Split dataset
    import random
    random.shuffle(image_files)
    
    n_train = int(len(image_files) * train_split)
    n_val = int(len(image_files) * val_split)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Copy files to respective directories
    for files, split_name, split_dir in [
        (train_files, "train", train_dir),
        (val_files, "val", val_dir),
        (test_files, "test", test_dir)
    ]:
        logger.info(f"Copying {len(files)} files to {split_name} set")
        
        for img_file in files:
            # Copy image
            dst_img = split_dir / "images" / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copy corresponding label file
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                dst_label = split_dir / "labels" / label_file.name
                shutil.copy2(label_file, dst_label)
    
    # Create dataset.yaml for license plate detection
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # number of classes (license_plate)
        'names': ['license_plate']
    }
    
    with open(output_path / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_config, f)
    
    logger.info(f"License plate dataset prepared successfully at {output_path}")
    return str(output_path / "dataset.yaml")

def train_license_plate_model(dataset_yaml: str, model_name: str = "yolov8n", epochs: int = 150, 
                              batch_size: int = 32, img_size: int = 416, device: str = "auto"):
    """Train YOLOv8 model for license plate detection"""
    logger = get_logger("LicensePlateTraining")
    
    logger.info(f"Starting license plate detection training with {model_name}")
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    
    # Load model
    model = YOLO(f"{model_name}.pt")
    
    # Train model with license plate specific parameters
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="data/models",
        name="license_plate_detection",
        save=True,
        save_period=20,  # Save checkpoint every 20 epochs
        val=True,
        plots=True,
        verbose=True,
        # License plate specific augmentations
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
        mixup=0.0
    )
    
    logger.info("License plate detection training completed successfully")
    
    # Validate model
    logger.info("Running validation...")
    val_results = model.val()
    
    logger.info(f"Validation mAP50: {val_results.box.map50:.4f}")
    logger.info(f"Validation mAP50-95: {val_results.box.map:.4f}")
    
    # Export model
    logger.info("Exporting model...")
    model.export(format="onnx")
    
    return results

def fine_tune_ocr_model(license_plate_images_dir: str, output_dir: str):
    """Fine-tune OCR model for Vietnamese license plates"""
    logger = get_logger("OCRFineTuning")
    
    logger.info("Fine-tuning OCR model for Vietnamese license plates")
    
    try:
        import paddleocr
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR with Vietnamese support
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            use_gpu=True,
            show_log=False
        )
        
        # Process license plate images to create training data
        images_path = Path(license_plate_images_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_path.glob(f"**/{ext}"))
        
        logger.info(f"Processing {len(image_files)} license plate images for OCR training")
        
        # Create output directory for OCR training data
        ocr_output_path = Path(output_dir) / "ocr_training"
        ocr_output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract text from images and create training dataset
        training_data = []
        for img_file in image_files:
            try:
                result = ocr.ocr(str(img_file), cls=True)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        if confidence > 0.5:  # Only use high-confidence results
                            training_data.append({
                                'image': str(img_file),
                                'text': text,
                                'confidence': confidence
                            })
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {e}")
        
        logger.info(f"Extracted {len(training_data)} text samples")
        
        # Save training data
        import json
        with open(ocr_output_path / "training_data.json", 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"OCR training data saved to {ocr_output_path}")
        
    except ImportError:
        logger.warning("PaddleOCR not available, skipping OCR fine-tuning")
    except Exception as e:
        logger.error(f"Error in OCR fine-tuning: {e}")

def evaluate_license_plate_model(model_path: str, dataset_yaml: str, device: str = "auto"):
    """Evaluate trained license plate detection model"""
    logger = get_logger("LicensePlateEvaluation")
    
    logger.info(f"Evaluating license plate model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run evaluation
    results = model.val(data=dataset_yaml, device=device)
    
    # Print results
    logger.info("License Plate Detection Evaluation Results:")
    logger.info(f"mAP50: {results.box.map50:.4f}")
    logger.info(f"mAP50-95: {results.box.map:.4f}")
    logger.info(f"Precision: {results.box.mp:.4f}")
    logger.info(f"Recall: {results.box.mr:.4f}")
    
    return results

def main():
    """Main license plate training script"""
    parser = argparse.ArgumentParser(description="Train license plate detection model")
    
    parser.add_argument("--data-dir", required=True, help="Path to license plate dataset directory")
    parser.add_argument("--output-dir", default="data/processed/license_plate_detection", 
                       help="Output directory for processed dataset")
    parser.add_argument("--model", default="yolov8n", 
                       choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                       help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=416, help="Image size")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, 0, 1, ...)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--eval-only", help="Path to model for evaluation only")
    parser.add_argument("--fine-tune-ocr", action="store_true", help="Fine-tune OCR model")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("LicensePlateTraining")
    
    logger.info("Starting license plate detection training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config = Config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Config file {args.config} not found, using defaults")
        
        # Evaluation only mode
        if args.eval_only:
            if not args.skip_prep:
                dataset_yaml = prepare_license_plate_dataset(
                    args.data_dir, 
                    args.output_dir,
                    args.train_split,
                    args.val_split
                )
            else:
                dataset_yaml = os.path.join(args.output_dir, "dataset.yaml")
            
            evaluate_license_plate_model(args.eval_only, dataset_yaml, args.device)
            return
        
        # Prepare dataset
        if not args.skip_prep:
            logger.info("Preparing license plate dataset...")
            dataset_yaml = prepare_license_plate_dataset(
                args.data_dir, 
                args.output_dir,
                args.train_split,
                args.val_split
            )
        else:
            dataset_yaml = os.path.join(args.output_dir, "dataset.yaml")
            logger.info(f"Using existing dataset: {dataset_yaml}")
        
        # Train model
        logger.info("Starting license plate detection training...")
        results = train_license_plate_model(
            dataset_yaml=dataset_yaml,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device
        )
        
        # Fine-tune OCR if requested
        if args.fine_tune_ocr:
            logger.info("Fine-tuning OCR model...")
            fine_tune_ocr_model(args.data_dir, args.output_dir)
        
        logger.info("License plate training completed successfully!")
        
        # Copy best model to models directory
        best_model_path = Path("data/models/license_plate_detection/weights/best.pt")
        if best_model_path.exists():
            target_path = Path("data/models/license_plate_yolov8.pt")
            shutil.copy2(best_model_path, target_path)
            logger.info(f"Best model copied to {target_path}")
        
    except Exception as e:
        logger.error(f"License plate training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
