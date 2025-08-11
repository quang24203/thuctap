#!/usr/bin/env python3
"""
Check dataset format and structure
"""

import os
import sys
from pathlib import Path
import json
import xml.etree.ElementTree as ET

def check_dataset_structure(dataset_path):
    """Check dataset structure and format"""
    
    dataset_path = Path(dataset_path)
    print(f"🔍 Checking dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Check for common structures
    images_dir = None
    labels_dir = None
    
    # Look for images directory
    possible_image_dirs = ['images', 'img', 'pictures', 'data']
    for dirname in possible_image_dirs:
        if (dataset_path / dirname).exists():
            images_dir = dataset_path / dirname
            print(f"✅ Found images directory: {dirname}")
            break
    
    # Look for labels directory
    possible_label_dirs = ['labels', 'annotations', 'ann', 'txt']
    for dirname in possible_label_dirs:
        if (dataset_path / dirname).exists():
            labels_dir = dataset_path / dirname
            print(f"✅ Found labels directory: {dirname}")
            break
    
    # Check for annotation files
    annotation_files = []
    for ext in ['.json', '.xml', '.txt']:
        annotation_files.extend(list(dataset_path.glob(f'*{ext}')))
    
    if annotation_files:
        print(f"📄 Found annotation files: {[f.name for f in annotation_files[:5]]}")
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    total_images = 0
    
    if images_dir:
        for ext in image_extensions:
            total_images += len(list(images_dir.glob(f'*{ext}')))
            total_images += len(list(images_dir.glob(f'*{ext.upper()}')))
    else:
        # Check root directory
        for ext in image_extensions:
            total_images += len(list(dataset_path.glob(f'*{ext}')))
            total_images += len(list(dataset_path.glob(f'*{ext.upper()}')))
    
    print(f"🖼️  Total images found: {total_images}")
    
    # Check annotation format
    if labels_dir:
        txt_files = list(labels_dir.glob('*.txt'))
        if txt_files:
            print(f"📝 Found {len(txt_files)} YOLO format label files")
            # Check first file
            with open(txt_files[0], 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    if len(parts) == 5:
                        print(f"✅ YOLO format detected: {first_line}")
                    else:
                        print(f"⚠️  Unknown format: {first_line}")
    
    # Check for COCO format
    coco_files = list(dataset_path.glob('*.json'))
    for coco_file in coco_files:
        try:
            with open(coco_file, 'r') as f:
                data = json.load(f)
                if 'images' in data and 'annotations' in data:
                    print(f"✅ COCO format detected: {coco_file.name}")
                    print(f"   Images: {len(data['images'])}")
                    print(f"   Annotations: {len(data['annotations'])}")
                    if 'categories' in data:
                        classes = [cat['name'] for cat in data['categories']]
                        print(f"   Classes: {classes}")
        except:
            pass
    
    # Check for Pascal VOC format
    xml_files = list(dataset_path.glob('*.xml'))
    if not xml_files and labels_dir:
        xml_files = list(labels_dir.glob('*.xml'))
    
    if xml_files:
        print(f"📄 Found {len(xml_files)} XML files (Pascal VOC format?)")
        try:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()
            if root.tag == 'annotation':
                print("✅ Pascal VOC format detected")
                objects = root.findall('object')
                if objects:
                    classes = [obj.find('name').text for obj in objects]
                    print(f"   Sample classes: {list(set(classes))[:5]}")
        except:
            pass
    
    print("\n📋 Summary:")
    print(f"   Dataset path: {dataset_path}")
    print(f"   Total images: {total_images}")
    print(f"   Images directory: {images_dir.name if images_dir else 'Not found'}")
    print(f"   Labels directory: {labels_dir.name if labels_dir else 'Not found'}")
    
    return {
        'dataset_path': dataset_path,
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'total_images': total_images,
        'annotation_files': annotation_files
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check dataset structure")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    
    args = parser.parse_args()
    
    result = check_dataset_structure(args.dataset_path)
    
    print("\n🚀 Next steps:")
    if result['total_images'] > 0:
        print("1. Copy your dataset to: data/raw/vehicle_dataset/")
        print("2. Ensure YOLO format: class_id x_center y_center width height")
        print("3. Update dataset.yaml with correct classes")
        print("4. Run training: python scripts/train_vehicle_detection.py")
    else:
        print("❌ No images found. Please check dataset path.")

if __name__ == "__main__":
    main()
