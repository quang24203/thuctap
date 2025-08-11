#!/usr/bin/env python3
"""
🎬 Benchmark Video Creator
Tạo video test chất lượng cao để đánh giá hiệu năng hệ thống
"""

import cv2
import numpy as np
import os
from datetime import datetime
import random

def create_realistic_parking_video():
    """Tạo video bãi đỗ xe realistic để benchmark"""
    print("🎬 Creating realistic parking benchmark video...")
    
    # Settings for high quality
    width, height = 1280, 720  # HD quality
    fps = 30  # High FPS for benchmark
    duration = 60  # 1 minute
    total_frames = fps * duration
    
    # Create output directory
    os.makedirs('data/benchmark', exist_ok=True)
    output_path = 'data/benchmark/parking_benchmark.mp4'
    
    # Video writer with high quality
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Cannot create video file")
        return None
    
    # Parking lot layout (3x4 = 12 spaces)
    spaces_per_row = 4
    rows = 3
    total_spaces = spaces_per_row * rows
    
    # Space dimensions
    space_width = 200
    space_height = 120
    margin_x = 80
    margin_y = 100
    
    # Calculate starting positions
    start_x = (width - (spaces_per_row * space_width + (spaces_per_row - 1) * 20)) // 2
    start_y = (height - (rows * space_height + (rows - 1) * 30)) // 2
    
    # Car colors for variety
    car_colors = [
        (50, 50, 150),    # Red
        (150, 50, 50),    # Blue  
        (50, 150, 50),    # Green
        (100, 100, 100),  # Gray
        (200, 200, 200),  # Light gray
        (50, 100, 150),   # Brown
        (150, 150, 50),   # Cyan
        (100, 50, 150),   # Purple
    ]
    
    # License plate patterns
    license_patterns = [
        "30A-123.45", "51B-678.90", "29C-111.22", "43D-333.44",
        "77E-555.66", "88F-777.88", "92G-999.00", "15H-222.33",
        "36K-444.55", "59L-666.77", "72M-888.99", "84N-000.11"
    ]
    
    for frame_num in range(total_frames):
        # Create background (asphalt texture)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 60
        
        # Add some texture to background
        noise = np.random.randint(-10, 10, (height, width, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Title and info
        cv2.putText(frame, "SMART PARKING BENCHMARK", (width//2 - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw parking spaces and cars
        occupied_count = 0
        
        for row in range(rows):
            for col in range(spaces_per_row):
                space_id = row * spaces_per_row + col
                
                # Calculate space position
                x = start_x + col * (space_width + 20)
                y = start_y + row * (space_height + 30)
                
                # Draw parking space outline
                cv2.rectangle(frame, (x, y), (x + space_width, y + space_height), 
                             (255, 255, 255), 2)
                
                # Space number
                cv2.putText(frame, f"P{space_id+1:02d}", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Dynamic car presence (cars come and go)
                car_cycle = (frame_num + space_id * 200) % 600
                if car_cycle < 400:  # Car is present for 400/600 frames
                    occupied_count += 1
                    
                    # Car body with realistic proportions
                    car_x = x + 15
                    car_y = y + 15
                    car_w = space_width - 30
                    car_h = space_height - 30
                    
                    # Car color
                    color = car_colors[space_id % len(car_colors)]
                    
                    # Main car body
                    cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), 
                                 color, -1)
                    
                    # Car details (windows, etc.)
                    # Front windshield
                    cv2.rectangle(frame, (car_x + 10, car_y + 10), 
                                 (car_x + car_w - 10, car_y + 25), (100, 150, 200), -1)
                    
                    # Rear windshield  
                    cv2.rectangle(frame, (car_x + 10, car_y + car_h - 25), 
                                 (car_x + car_w - 10, car_y + car_h - 10), (100, 150, 200), -1)
                    
                    # License plate (realistic size and position)
                    plate_w = 80
                    plate_h = 20
                    plate_x = car_x + (car_w - plate_w) // 2
                    plate_y = car_y + car_h - 35
                    
                    # White license plate background
                    cv2.rectangle(frame, (plate_x, plate_y), 
                                 (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
                    
                    # License plate border
                    cv2.rectangle(frame, (plate_x, plate_y), 
                                 (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 1)
                    
                    # License plate text
                    plate_text = license_patterns[space_id % len(license_patterns)]
                    cv2.putText(frame, plate_text, (plate_x + 5, plate_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # Status indicator
                    cv2.putText(frame, "OCCUPIED", (x + 5, y + space_height + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    # Empty space
                    cv2.putText(frame, "EMPTY", (x + 20, y + space_height + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Statistics panel
        stats_y = height - 120
        cv2.rectangle(frame, (10, stats_y), (300, height - 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, stats_y), (300, height - 10), (255, 255, 255), 2)
        
        cv2.putText(frame, "PARKING STATISTICS", (20, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Spaces: {total_spaces}", (20, stats_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Occupied: {occupied_count}", (20, stats_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Available: {total_spaces - occupied_count}", (20, stats_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Timestamp and frame info
        timestamp = f"Frame: {frame_num:04d}/{total_frames} | Time: {frame_num/fps:.1f}s"
        cv2.putText(frame, timestamp, (width - 400, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # FPS indicator
        cv2.putText(frame, f"Target FPS: {fps}", (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        out.write(frame)
        
        # Progress indicator
        if frame_num % (fps * 5) == 0:  # Every 5 seconds
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    out.release()
    print(f"✅ Benchmark video created: {output_path}")
    print(f"📊 Video specs: {width}x{height} @ {fps}FPS, {duration}s duration")
    print(f"🎯 Features: {total_spaces} parking spaces, realistic cars, Vietnamese license plates")
    
    return output_path

if __name__ == "__main__":
    try:
        video_path = create_realistic_parking_video()
        if video_path:
            print(f"\n🎉 SUCCESS! Benchmark video ready!")
            print(f"📁 Location: {video_path}")
            print(f"\n🚀 Next steps:")
            print("1. Test với Universal Analysis Hub")
            print("2. Benchmark FPS performance") 
            print("3. Measure detection accuracy")
            print("4. Test license plate recognition")
            
    except Exception as e:
        print(f"❌ Error creating benchmark video: {e}")
        import traceback
        traceback.print_exc()
