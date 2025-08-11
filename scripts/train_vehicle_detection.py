#!/usr/bin/env python3
"""
Training script for vehicle detection model using YOLOv8
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

def prepare_dataset(data_dir: str, output_dir: str, train_split: float = 0.8, val_split: float = 0.15):
    """Prepare dataset in YOLO format"""
    logger = get_logger("DatasetPreparation")
    
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
    
    logger.info(f"Found {len(image_files)} images")
    
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
    
    # Create dataset.yaml
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,  # number of classes
        'names': ['car', 'truck', 'bus', 'motorcycle']
    }
    
    with open(output_path / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_config, f)
    
    logger.info(f"Dataset prepared successfully at {output_path}")
    return str(output_path / "dataset.yaml")

def train_model(dataset_yaml: str, model_name: str = "yolov8n", epochs: int = 100, 
                batch_size: int = 16, img_size: int = 640, device: str = "auto"):
    """Train YOLOv8 model"""
    logger = get_logger("ModelTraining")
    
    logger.info(f"Starting training with {model_name}")
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    
    # Load model
    model = YOLO(f"{model_name}.pt")
    
    # Train model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="data/models",
        name="vehicle_detection",
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,
        plots=True,
        verbose=True
    )
    
    logger.info("Training completed successfully")
    
    # Validate model
    logger.info("Running validation...")
    val_results = model.val()
    
    logger.info(f"Validation mAP50: {val_results.box.map50:.4f}")
    logger.info(f"Validation mAP50-95: {val_results.box.map:.4f}")
    
    # Export model
    logger.info("Exporting model...")
    model.export(format="onnx")
    
    return results

def evaluate_model(model_path: str, dataset_yaml: str, device: str = "auto"):
    """Evaluate trained model"""
    logger = get_logger("ModelEvaluation")
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run evaluation
    results = model.val(data=dataset_yaml, device=device)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"mAP50: {results.box.map50:.4f}")
    logger.info(f"mAP50-95: {results.box.map:.4f}")
    logger.info(f"Precision: {results.box.mp:.4f}")
    logger.info(f"Recall: {results.box.mr:.4f}")
    
    # Per-class results
    if hasattr(results.box, 'maps'):
        class_names = ['car', 'truck', 'bus', 'motorcycle']
        for i, (name, map_val) in enumerate(zip(class_names, results.box.maps)):
            logger.info(f"{name} mAP50: {map_val:.4f}")
    
    return results

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train vehicle detection model")
    
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", default="data/processed/vehicle_detection", 
                       help="Output directory for processed dataset")
    parser.add_argument("--model", default="yolov8n", 
                       choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                       help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, 0, 1, ...)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--eval-only", help="Path to model for evaluation only")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("VehicleDetectionTraining")
    
    logger.info("Starting vehicle detection training")
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
                dataset_yaml = prepare_dataset(
                    args.data_dir, 
                    args.output_dir,
                    args.train_split,
                    args.val_split
                )
            else:
                dataset_yaml = os.path.join(args.output_dir, "dataset.yaml")
            
            evaluate_model(args.eval_only, dataset_yaml, args.device)
            return
        
        # Prepare dataset
        if not args.skip_prep:
            logger.info("Preparing dataset...")
            dataset_yaml = prepare_dataset(
                args.data_dir, 
                args.output_dir,
                args.train_split,
                args.val_split
            )
        else:
            dataset_yaml = os.path.join(args.output_dir, "dataset.yaml")
            logger.info(f"Using existing dataset: {dataset_yaml}")
        
        # Train model
        logger.info("Starting training...")
        results = train_model(
            dataset_yaml=dataset_yaml,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device
        )
        
        logger.info("Training completed successfully!")
        
        # Copy best model to models directory
        best_model_path = Path("data/models/vehicle_detection/weights/best.pt")
        if best_model_path.exists():
            target_path = Path("data/models/vehicle_yolov8.pt")
            shutil.copy2(best_model_path, target_path)
            logger.info(f"Best model copied to {target_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
