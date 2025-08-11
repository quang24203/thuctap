#!/usr/bin/env python3
"""
Auto-annotate license plates using existing OCR models
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import json
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def detect_license_plate_paddleocr(image):
    """Detect license plate using PaddleOCR"""
    try:
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Run OCR
        results = ocr.ocr(image, cls=True)
        
        license_plates = []
        
        if results and results[0]:
            for line in results[0]:
                if line:
                    # Extract bounding box and text
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = line[1][0]  # text
                    confidence = line[1][1]  # confidence
                    
                    # Check if text looks like Vietnamese license plate
                    if is_vietnamese_license_plate(text) and confidence > 0.5:
                        # Convert bbox to YOLO format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        license_plates.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'text': text,
                            'confidence': confidence
                        })
        
        return license_plates
        
    except ImportError:
        print("❌ PaddleOCR not installed. Install with: pip install paddleocr")
        return []
    except Exception as e:
        print(f"❌ Error with PaddleOCR: {e}")
        return []

def detect_license_plate_easyocr(image):
    """Detect license plate using EasyOCR"""
    try:
        import easyocr
        
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])
        
        # Run OCR
        results = reader.readtext(image)
        
        license_plates = []
        
        for (bbox, text, confidence) in results:
            # Check if text looks like Vietnamese license plate
            if is_vietnamese_license_plate(text) and confidence > 0.5:
                # Convert bbox to standard format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                license_plates.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'text': text,
                    'confidence': confidence
                })
        
        return license_plates
        
    except ImportError:
        print("❌ EasyOCR not installed. Install with: pip install easyocr")
        return []
    except Exception as e:
        print(f"❌ Error with EasyOCR: {e}")
        return []

def is_vietnamese_license_plate(text):
    """Check if text looks like Vietnamese license plate"""
    
    # Clean text
    text = text.strip().upper().replace(' ', '').replace('O', '0')
    
    # Vietnamese license plate patterns
    patterns = [
        r'^\d{2}[A-Z]\d{3}\.\d{2}$',      # 51A123.45
        r'^\d{2}[A-Z]-\d{3}\.\d{2}$',     # 51A-123.45
        r'^\d{2}[A-Z]\d{5}$',             # 51A12345
        r'^\d{2}[A-Z]-\d{5}$',            # 51A-12345
        r'^\d{2}[A-Z]\d-\d{3}\.\d{2}$',   # 51A1-123.45
        r'^\d{2}[A-Z]\d-\d{5}$',          # 51A1-12345
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    # Also check for partial matches (at least 5 characters with numbers and letters)
    if len(text) >= 5 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
        return True
    
    return False

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert bounding box to YOLO format"""
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def process_images(images_dir, output_dir, ocr_engine='paddleocr'):
    """Process images and create annotations"""
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_images_dir = output_path / "images"
    output_labels_dir = output_path / "labels"
    output_ocr_dir = output_path / "ocr_labels"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_ocr_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f'*{ext}')))
        image_files.extend(list(images_path.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    processed = 0
    detected = 0
    
    for i, img_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_file.name}")
        
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"❌ Could not load {img_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Detect license plates
            if ocr_engine == 'paddleocr':
                license_plates = detect_license_plate_paddleocr(image)
            else:
                license_plates = detect_license_plate_easyocr(image)
            
            if license_plates:
                detected += 1
                
                # Copy image to output
                output_img_path = output_images_dir / img_file.name
                cv2.imwrite(str(output_img_path), image)
                
                # Create YOLO annotation
                label_file = output_labels_dir / f"{img_file.stem}.txt"
                ocr_file = output_ocr_dir / f"{img_file.stem}.txt"
                
                with open(label_file, 'w') as f_label, open(ocr_file, 'w') as f_ocr:
                    for plate in license_plates:
                        # YOLO format
                        yolo_bbox = bbox_to_yolo(plate['bbox'], img_width, img_height)
                        f_label.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                        
                        # OCR text
                        f_ocr.write(f"{plate['text']}\n")
                
                print(f"✅ Detected {len(license_plates)} license plates")
            else:
                print("⚠️  No license plates detected")
            
            processed += 1
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Processed: {processed}/{len(image_files)} images")
    print(f"   Detected: {detected} images with license plates")
    print(f"   Success rate: {detected/processed*100:.1f}%" if processed > 0 else "   Success rate: 0%")
    
    # Create dataset.yaml
    dataset_yaml = output_path / "dataset.yaml"
    with open(dataset_yaml, 'w') as f:
        f.write(f"""# Auto-annotated License Plate Dataset

path: {output_path}
train: images
val: images

nc: 1
names: ['license_plate']

description: "Auto-annotated Vietnamese license plate dataset"
auto_annotated: true
ocr_engine: {ocr_engine}
""")
    
    return detected > 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-annotate license plates")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", default="data/raw/license_plate_dataset", 
                       help="Output directory for annotated dataset")
    parser.add_argument("--ocr-engine", choices=['paddleocr', 'easyocr'], default='paddleocr',
                       help="OCR engine to use")
    
    args = parser.parse_args()
    
    print("🔍 Auto-annotating license plates...")
    print(f"   Input: {args.images_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   OCR Engine: {args.ocr_engine}")
    
    success = process_images(args.images_dir, args.output_dir, args.ocr_engine)
    
    if success:
        print("\n🚀 Ready for training! Run:")
        print(f"python scripts/train_license_plate.py --data-dir {args.output_dir} --epochs 50")
    else:
        print("\n❌ No license plates detected. Check your images or try different OCR engine.")

if __name__ == "__main__":
    main()
