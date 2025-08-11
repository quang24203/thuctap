#!/usr/bin/env python3
"""
Create demo dataset for license plate detection and recognition
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import random
import string

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def generate_vietnamese_license_plate():
    """Generate Vietnamese license plate text"""
    
    # Vietnamese license plate formats
    formats = [
        # Old format: 51A-123.45
        lambda: f"{random.randint(10,99)}{random.choice('ABCDEFGHKLMNPRSTUVXYZ')}-{random.randint(100,999)}.{random.randint(10,99)}",
        
        # New format: 51A-12345
        lambda: f"{random.randint(10,99)}{random.choice('ABCDEFGHKLMNPRSTUVXYZ')}-{random.randint(10000,99999)}",
        
        # Motorcycle: 51A1-123.45
        lambda: f"{random.randint(10,99)}{random.choice('ABCDEFGHKLMNPRSTUVXYZ')}{random.randint(1,9)}-{random.randint(100,999)}.{random.randint(10,99)}",
    ]
    
    return random.choice(formats)()

def create_license_plate_image(text, width=200, height=50):
    """Create synthetic license plate image"""
    
    # Create white background
    plate = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add blue border (Vietnamese style)
    cv2.rectangle(plate, (0, 0), (width-1, height-1), (255, 0, 0), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Calculate text size and position
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    # Add text
    cv2.putText(plate, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    
    return plate

def create_vehicle_with_license_plate(width=640, height=640):
    """Create synthetic image with vehicle and license plate"""
    
    # Create background
    img = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    
    # Add road
    road_y = height * 2 // 3
    cv2.rectangle(img, (0, road_y), (width, height), (80, 80, 80), -1)
    
    # Vehicle position
    car_x = random.randint(100, width-300)
    car_y = random.randint(road_y-100, road_y-20)
    car_w = random.randint(200, 280)
    car_h = random.randint(80, 120)
    
    # Draw vehicle
    vehicle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.rectangle(img, (car_x, car_y), (car_x+car_w, car_y+car_h), vehicle_color, -1)
    cv2.rectangle(img, (car_x, car_y), (car_x+car_w, car_y+car_h), (0, 0, 0), 2)
    
    # License plate position (front of car)
    plate_w = 120
    plate_h = 30
    plate_x = car_x + car_w//2 - plate_w//2
    plate_y = car_y + car_h + 5
    
    # Generate license plate text
    plate_text = generate_vietnamese_license_plate()
    
    # Create license plate
    plate_img = create_license_plate_image(plate_text, plate_w, plate_h)
    
    # Add license plate to image
    img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = plate_img
    
    # YOLO format annotation for license plate (normalized)
    x_center = (plate_x + plate_w/2) / width
    y_center = (plate_y + plate_h/2) / height
    norm_w = plate_w / width
    norm_h = plate_h / height
    
    annotation = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
    
    return img, annotation, plate_text

def create_license_plate_dataset(output_dir, num_images=100):
    """Create demo license plate dataset"""
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    ocr_labels_dir = output_path / "ocr_labels"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    ocr_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} license plate demo images...")
    
    for i in range(num_images):
        # Create synthetic image
        img, annotation, plate_text = create_vehicle_with_license_plate()
        
        # Save image
        img_filename = f"lp_demo_{i:04d}.jpg"
        img_path = images_dir / img_filename
        cv2.imwrite(str(img_path), img)
        
        # Save YOLO annotation (bounding box)
        label_filename = f"lp_demo_{i:04d}.txt"
        label_path = labels_dir / label_filename
        with open(label_path, 'w') as f:
            f.write(annotation + '\n')
        
        # Save OCR annotation (text)
        ocr_filename = f"lp_demo_{i:04d}.txt"
        ocr_path = ocr_labels_dir / ocr_filename
        with open(ocr_path, 'w') as f:
            f.write(plate_text + '\n')
        
        if (i + 1) % 20 == 0:
            print(f"Created {i + 1}/{num_images} images")
    
    print(f"✅ License plate demo dataset created at: {output_path}")
    print(f"   - Images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"   - Detection labels: {len(list(labels_dir.glob('*.txt')))}")
    print(f"   - OCR labels: {len(list(ocr_labels_dir.glob('*.txt')))}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create license plate demo dataset")
    parser.add_argument("--output-dir", default="data/raw/license_plate_dataset", 
                       help="Output directory for dataset")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to create")
    
    args = parser.parse_args()
    
    print("🚗 Creating demo dataset for license plate detection...")
    create_license_plate_dataset(args.output_dir, args.num_images)
    
    # Create dataset.yaml
    dataset_yaml = Path(args.output_dir) / "dataset.yaml"
    with open(dataset_yaml, 'w') as f:
        f.write(f"""# License Plate Detection Dataset

path: {args.output_dir}
train: images
val: images

nc: 1
names: ['license_plate']

description: "Demo dataset for Vietnamese license plate detection"
""")
    
    print("\n🚀 Ready for license plate training! Run:")
    print(f"python scripts/train_license_plate.py --data-dir {args.output_dir} --epochs 50")

if __name__ == "__main__":
    main()
