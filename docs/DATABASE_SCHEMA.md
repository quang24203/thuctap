# 🗄️ Database Schema Documentation

## Overview

Smart Parking System sử dụng SQLite database để lưu trữ dữ liệu phân tích, metadata video, và thông tin phương tiện.

## Database Structure

### 1. Videos Table

Lưu trữ thông tin metadata của video được upload.

```sql
CREATE TABLE videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_path VARCHAR(500),
    file_size INTEGER,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration FLOAT,
    fps INTEGER,
    total_frames INTEGER,
    width INTEGER,
    height INTEGER,
    processing_status VARCHAR(20) DEFAULT 'pending',
    processing_start_time TIMESTAMP,
    processing_end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Columns:**
- `id`: Primary key, auto-increment
- `filename`: Tên file được lưu trong hệ thống
- `original_filename`: Tên file gốc do user upload
- `file_path`: Đường dẫn đầy đủ đến file
- `file_size`: Kích thước file (bytes)
- `upload_time`: Thời gian upload
- `duration`: Thời lượng video (seconds)
- `fps`: Frames per second
- `total_frames`: Tổng số frames
- `width`, `height`: Độ phân giải video
- `processing_status`: Trạng thái xử lý (pending, processing, completed, failed)
- `processing_start_time`: Thời gian bắt đầu xử lý
- `processing_end_time`: Thời gian kết thúc xử lý

### 2. Vehicles Table

Lưu trữ thông tin phương tiện được phát hiện.

```sql
CREATE TABLE vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vehicle_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    area INTEGER,
    center_x INTEGER,
    center_y INTEGER,
    tracking_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);
```

**Columns:**
- `id`: Primary key
- `video_id`: Foreign key tới videos table
- `frame_number`: Số thứ tự frame trong video
- `detection_time`: Thời gian phát hiện
- `vehicle_type`: Loại phương tiện (car, truck, bus, motorcycle)
- `confidence`: Độ tin cậy của detection (0.0-1.0)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Bounding box coordinates
- `area`: Diện tích bounding box
- `center_x`, `center_y`: Tọa độ trung tâm
- `tracking_id`: ID để tracking qua các frames

### 3. License_Plates Table

Lưu trữ thông tin biển số xe được nhận diện.

```sql
CREATE TABLE license_plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER,
    video_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    plate_number VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    ocr_text_raw TEXT,
    validation_status VARCHAR(20) DEFAULT 'pending',
    entry_exit_type VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE SET NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);
```

**Columns:**
- `id`: Primary key
- `vehicle_id`: Foreign key tới vehicles table (nullable)
- `video_id`: Foreign key tới videos table
- `frame_number`: Số thứ tự frame
- `plate_number`: Biển số đã được clean và validate
- `confidence`: Độ tin cậy OCR (0.0-1.0)
- `detection_time`: Thời gian phát hiện
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Bounding box của biển số
- `ocr_text_raw`: Text thô từ OCR trước khi clean
- `validation_status`: Trạng thái validation (valid, invalid, pending)
- `entry_exit_type`: Loại sự kiện (entry, exit)

### 4. Parking_Spaces Table

Lưu trữ thông tin về các ô đỗ xe.

```sql
CREATE TABLE parking_spaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    space_number VARCHAR(10) NOT NULL UNIQUE,
    polygon_coordinates TEXT NOT NULL,
    space_type VARCHAR(20) DEFAULT 'standard',
    status VARCHAR(20) DEFAULT 'available',
    last_occupied_time TIMESTAMP,
    last_vacated_time TIMESTAMP,
    total_occupancy_time INTEGER DEFAULT 0,
    occupancy_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Columns:**
- `id`: Primary key
- `space_number`: Số hiệu ô đỗ xe
- `polygon_coordinates`: Tọa độ polygon định nghĩa ô đỗ (JSON format)
- `space_type`: Loại ô đỗ (standard, handicapped, electric, vip)
- `status`: Trạng thái hiện tại (available, occupied, reserved, maintenance)
- `last_occupied_time`: Thời gian bắt đầu đỗ gần nhất
- `last_vacated_time`: Thời gian rời đi gần nhất
- `total_occupancy_time`: Tổng thời gian đã được sử dụng (seconds)
- `occupancy_count`: Số lần đã được sử dụng

### 5. Timeline_Events Table

Lưu trữ timeline các sự kiện ra/vào.

```sql
CREATE TABLE timeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    event_type VARCHAR(10) NOT NULL,
    plate_number VARCHAR(20),
    vehicle_type VARCHAR(50),
    confidence FLOAT,
    frame_number INTEGER,
    event_time TIMESTAMP NOT NULL,
    duration INTEGER,
    parking_space_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (parking_space_id) REFERENCES parking_spaces(id) ON DELETE SET NULL
);
```

**Columns:**
- `id`: Primary key
- `video_id`: Foreign key tới videos table
- `event_type`: Loại sự kiện (entry, exit)
- `plate_number`: Biển số xe
- `vehicle_type`: Loại phương tiện
- `confidence`: Độ tin cậy
- `frame_number`: Frame number trong video
- `event_time`: Thời gian sự kiện
- `duration`: Thời gian đỗ (cho exit events)
- `parking_space_id`: ID ô đỗ xe (nếu có)

## Indexes

Để tối ưu performance, các indexes sau được tạo:

```sql
-- Index cho video lookups
CREATE INDEX idx_videos_filename ON videos(filename);
CREATE INDEX idx_videos_status ON videos(processing_status);
CREATE INDEX idx_videos_upload_time ON videos(upload_time);

-- Index cho vehicle detections
CREATE INDEX idx_vehicles_video_id ON vehicles(video_id);
CREATE INDEX idx_vehicles_frame ON vehicles(frame_number);
CREATE INDEX idx_vehicles_type ON vehicles(vehicle_type);
CREATE INDEX idx_vehicles_detection_time ON vehicles(detection_time);

-- Index cho license plates
CREATE INDEX idx_license_plates_video_id ON license_plates(video_id);
CREATE INDEX idx_license_plates_plate_number ON license_plates(plate_number);
CREATE INDEX idx_license_plates_detection_time ON license_plates(detection_time);

-- Index cho timeline events
CREATE INDEX idx_timeline_events_video_id ON timeline_events(video_id);
CREATE INDEX idx_timeline_events_type ON timeline_events(event_type);
CREATE INDEX idx_timeline_events_plate ON timeline_events(plate_number);
CREATE INDEX idx_timeline_events_time ON timeline_events(event_time);

-- Index cho parking spaces
CREATE INDEX idx_parking_spaces_status ON parking_spaces(status);
CREATE INDEX idx_parking_spaces_number ON parking_spaces(space_number);
```

## Relationships

```
videos (1) ──── (N) vehicles
videos (1) ──── (N) license_plates
videos (1) ──── (N) timeline_events
vehicles (1) ──── (N) license_plates
parking_spaces (1) ──── (N) timeline_events
```

## Sample Queries

### 1. Lấy tất cả phương tiện trong video

```sql
SELECT v.*, lp.plate_number 
FROM vehicles v
LEFT JOIN license_plates lp ON v.id = lp.vehicle_id
WHERE v.video_id = ?
ORDER BY v.frame_number;
```

### 2. Thống kê theo loại phương tiện

```sql
SELECT 
    vehicle_type,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM vehicles 
WHERE video_id = ?
GROUP BY vehicle_type;
```

### 3. Timeline ra/vào theo biển số

```sql
SELECT 
    plate_number,
    event_type,
    event_time,
    vehicle_type
FROM timeline_events 
WHERE video_id = ?
ORDER BY event_time;
```

### 4. Tình trạng bãi đỗ xe

```sql
SELECT 
    status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM parking_spaces), 2) as percentage
FROM parking_spaces
GROUP BY status;
```

## Data Migration

Khi cần update schema, sử dụng migration scripts trong `scripts/migrations/`:

```bash
# Chạy migration
python scripts/migrations/001_add_tracking_id.py

# Rollback migration
python scripts/migrations/001_add_tracking_id.py --rollback
```

## Backup và Restore

```bash
# Backup database
sqlite3 data/parking_system.db ".backup data/backup_$(date +%Y%m%d_%H%M%S).db"

# Restore database
cp data/backup_20231225_143022.db data/parking_system.db
```
