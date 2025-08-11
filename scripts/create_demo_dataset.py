#!/usr/bin/env python3
"""
Create demo dataset for testing training pipeline
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_synthetic_vehicle_image(width=640, height=640):
    """Create synthetic image with vehicles"""
    
    # Create background
    img = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    
    # Add road
    road_y = height // 2
    cv2.rectangle(img, (0, road_y-50), (width, road_y+50), (80, 80, 80), -1)
    
    # Add lane markings
    for x in range(0, width, 100):
        cv2.rectangle(img, (x, road_y-5), (x+50, road_y+5), (255, 255, 255), -1)
    
    vehicles = []
    annotations = []
    
    # Add random vehicles
    num_vehicles = random.randint(1, 4)
    
    for i in range(num_vehicles):
        # Random vehicle position
        x = random.randint(50, width-150)
        y = random.randint(road_y-40, road_y+20)
        
        # Random vehicle size
        w = random.randint(80, 120)
        h = random.randint(40, 60)
        
        # Random vehicle type and color
        vehicle_type = random.choice([0, 0, 0, 1, 3])  # More cars
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
        color = random.choice(colors)
        
        # Draw vehicle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        
        # Add license plate
        plate_x = x + w//4
        plate_y = y + h + 5
        plate_w = w//2
        plate_h = 15
        cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h), (255, 255, 255), -1)
        cv2.putText(img, "51A123", (plate_x+5, plate_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # YOLO format annotation (normalized)
        x_center = (x + w/2) / width
        y_center = (y + h/2) / height
        norm_w = w / width
        norm_h = h / height
        
        annotations.append(f"{vehicle_type} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    return img, annotations

def create_demo_dataset(output_dir, num_images=100):
    """Create demo dataset"""
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} demo images...")
    
    for i in range(num_images):
        # Create synthetic image
        img, annotations = create_synthetic_vehicle_image()
        
        # Save image
        img_filename = f"demo_{i:04d}.jpg"
        img_path = images_dir / img_filename
        cv2.imwrite(str(img_path), img)
        
        # Save annotations
        label_filename = f"demo_{i:04d}.txt"
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for annotation in annotations:
                f.write(annotation + '\n')
        
        if (i + 1) % 20 == 0:
            print(f"Created {i + 1}/{num_images} images")
    
    print(f"✅ Demo dataset created at: {output_path}")
    print(f"   - Images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"   - Labels: {len(list(labels_dir.glob('*.txt')))}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create demo dataset")
    parser.add_argument("--output-dir", default="data/raw/vehicle_dataset", 
                       help="Output directory for dataset")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to create")
    
    args = parser.parse_args()
    
    print("🎨 Creating demo dataset for vehicle detection...")
    create_demo_dataset(args.output_dir, args.num_images)
    
    # Create dataset.yaml if not exists
    dataset_yaml = Path(args.output_dir) / "dataset.yaml"
    if not dataset_yaml.exists():
        print("📝 Creating dataset.yaml...")
        with open(dataset_yaml, 'w') as f:
            f.write(f"""# Demo Vehicle Detection Dataset

path: {args.output_dir}
train: images
val: images

nc: 4
names: ['car', 'truck', 'bus', 'motorcycle']

description: "Demo dataset for vehicle detection training"
""")
    
    print("\n🚀 Ready to train! Run:")
    print(f"python scripts/train_vehicle_detection.py --data-dir {args.output_dir} --epochs 10")

if __name__ == "__main__":
    main()
