# 🔌 API Documentation - Smart Parking System

## Base URL

```
http://localhost:8000
```

## Authentication

Hiện tại API không yêu cầu authentication. Trong production, nên implement JWT hoặc API key authentication.

## Content Types

- **Request**: `multipart/form-data` (cho file upload), `application/json`
- **Response**: `application/json`

## Error Handling

Tất cả API responses follow format:

```json
{
  "success": boolean,
  "message": string,
  "data": object,
  "error": string (optional)
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Endpoints

### 1. Health Check

#### GET /health

Kiểm tra trạng thái hệ thống.

**Response:**
```json
{
  "success": true,
  "message": "Smart Parking System is running",
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "uptime": "2h 15m 30s"
  }
}
```

### 2. Universal Analysis

#### POST /api/v1/universal/analyze

Upload và phân tích video hoặc hình ảnh.

**Request:**
```bash
curl -X POST \
  -F "file=@video.mp4" \
  http://localhost:8000/api/v1/universal/analyze
```

**Parameters:**
- `file` (required): Video file (MP4, AVI, MOV) hoặc image file (JPG, PNG)

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "analysis_data": {
    "video_info": {
      "duration": 15.0,
      "fps": 30,
      "total_frames": 450,
      "filename": "parking_video.mp4",
      "width": 1280,
      "height": 720
    },
    "vehicle_detection": {
      "total_vehicles_detected": 8,
      "vehicle_types": {
        "car": 5,
        "truck": 2,
        "bus": 1,
        "motorcycle": 0
      }
    },
    "license_plates": [
      {
        "plate_number": "30A-123.45",
        "confidence": 0.89,
        "vehicle_class": "car",
        "timestamp": 1640995825.123,
        "frame": 150,
        "bbox": [100, 200, 300, 250]
      }
    ],
    "timeline_events": [
      {
        "type": "entry",
        "timestamp": 1640995825.123,
        "plate_number": "30A-123.45",
        "confidence": 0.89,
        "vehicle_class": "car",
        "frame": 150
      }
    ],
    "parking_analysis": {
      "total_spaces": 15,
      "occupied_spaces": 11,
      "empty_spaces": 4,
      "occupancy_rate": 0.73,
      "peak_occupancy": 13,
      "average_stay": "45 minutes"
    }
  },
  "summary": {
    "total_vehicles_detected": 8,
    "max_vehicles_in_frame": 4,
    "avg_vehicles_per_frame": 2.8,
    "total_license_plates_detected": 5,
    "unique_license_plates": 5,
    "processing_time": "2.15s",
    "frames_processed": 45
  }
}
```

### 3. Test API

#### POST /api/v1/test-simple

Test API với demo data, không cần upload file.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/test-simple
```

**Response:**
```json
{
  "success": true,
  "message": "✅ API is working!",
  "analysis_data": {
    "video_info": {
      "duration": 10.0,
      "fps": 30,
      "total_frames": 300
    },
    "vehicle_detection": {
      "total_vehicles_detected": 5,
      "vehicle_types": {
        "car": 3,
        "truck": 1,
        "bus": 1,
        "motorcycle": 0
      }
    },
    "license_plates": [
      {
        "plate_number": "30A-123.45",
        "confidence": 0.89,
        "vehicle_class": "car",
        "timestamp": 1640995825.123,
        "frame": 150
      }
    ],
    "timeline_events": [
      {
        "type": "entry",
        "timestamp": 1640995825.123,
        "plate_number": "30A-123.45",
        "confidence": 0.89,
        "vehicle_class": "car",
        "frame": 150
      }
    ],
    "parking_analysis": {
      "total_spaces": 12,
      "occupied_spaces": 8,
      "empty_spaces": 4,
      "occupancy_rate": 0.67
    }
  }
}
```

### 4. Get Videos

#### GET /api/v1/videos

Lấy danh sách videos đã upload.

**Query Parameters:**
- `limit` (optional): Số lượng records (default: 50)
- `offset` (optional): Offset cho pagination (default: 0)
- `status` (optional): Filter theo processing status

**Request:**
```bash
curl "http://localhost:8000/api/v1/videos?limit=10&status=completed"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "videos": [
      {
        "id": 1,
        "filename": "parking_video_001.mp4",
        "original_filename": "my_parking_video.mp4",
        "duration": 15.0,
        "fps": 30,
        "total_frames": 450,
        "processing_status": "completed",
        "upload_time": "2024-01-15T10:30:00Z",
        "processing_time": "2.15s"
      }
    ],
    "total": 1,
    "limit": 10,
    "offset": 0
  }
}
```

### 5. Get Video Results

#### GET /api/v1/videos/{video_id}/results

Lấy kết quả phân tích của một video cụ thể.

**Path Parameters:**
- `video_id` (required): ID của video

**Request:**
```bash
curl http://localhost:8000/api/v1/videos/1/results
```

**Response:**
```json
{
  "success": true,
  "data": {
    "video_id": 1,
    "video_info": {
      "filename": "parking_video_001.mp4",
      "duration": 15.0,
      "fps": 30,
      "total_frames": 450
    },
    "vehicles": [
      {
        "id": 1,
        "frame_number": 150,
        "vehicle_type": "car",
        "confidence": 0.89,
        "bbox": [100, 200, 300, 400],
        "detection_time": "2024-01-15T10:30:15Z"
      }
    ],
    "license_plates": [
      {
        "id": 1,
        "plate_number": "30A-123.45",
        "confidence": 0.89,
        "frame_number": 150,
        "detection_time": "2024-01-15T10:30:15Z"
      }
    ],
    "timeline_events": [
      {
        "id": 1,
        "event_type": "entry",
        "plate_number": "30A-123.45",
        "vehicle_type": "car",
        "event_time": "2024-01-15T10:30:15Z"
      }
    ]
  }
}
```

### 6. Get License Plates

#### GET /api/v1/license-plates

Lấy danh sách biển số đã phát hiện.

**Query Parameters:**
- `limit` (optional): Số lượng records (default: 50)
- `offset` (optional): Offset cho pagination (default: 0)
- `date_from` (optional): Filter từ ngày (YYYY-MM-DD)
- `date_to` (optional): Filter đến ngày (YYYY-MM-DD)
- `plate_number` (optional): Filter theo biển số cụ thể

**Request:**
```bash
curl "http://localhost:8000/api/v1/license-plates?date_from=2024-01-01&limit=20"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "license_plates": [
      {
        "id": 1,
        "plate_number": "30A-123.45",
        "confidence": 0.89,
        "vehicle_type": "car",
        "detection_time": "2024-01-15T10:30:15Z",
        "video_id": 1,
        "frame_number": 150
      }
    ],
    "total": 1,
    "limit": 20,
    "offset": 0
  }
}
```

### 7. Get Parking Status

#### GET /api/v1/parking/status

Lấy tình trạng bãi đỗ xe hiện tại.

**Request:**
```bash
curl http://localhost:8000/api/v1/parking/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_spaces": 15,
    "occupied_spaces": 11,
    "empty_spaces": 4,
    "occupancy_rate": 0.73,
    "last_updated": "2024-01-15T10:30:00Z",
    "spaces": [
      {
        "space_number": "A01",
        "status": "occupied",
        "last_occupied_time": "2024-01-15T09:15:00Z",
        "current_vehicle": "30A-123.45"
      },
      {
        "space_number": "A02",
        "status": "available",
        "last_vacated_time": "2024-01-15T08:45:00Z"
      }
    ]
  }
}
```

### 8. Get Analytics

#### GET /api/v1/analytics/summary

Lấy thống kê tổng quan.

**Query Parameters:**
- `period` (optional): Khoảng thời gian (today, week, month, year)
- `date_from` (optional): Từ ngày (YYYY-MM-DD)
- `date_to` (optional): Đến ngày (YYYY-MM-DD)

**Request:**
```bash
curl "http://localhost:8000/api/v1/analytics/summary?period=today"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "today",
    "date_range": {
      "from": "2024-01-15T00:00:00Z",
      "to": "2024-01-15T23:59:59Z"
    },
    "summary": {
      "total_vehicles": 156,
      "unique_vehicles": 89,
      "total_entries": 78,
      "total_exits": 67,
      "current_occupancy": 11,
      "peak_occupancy": 14,
      "average_stay_duration": "2h 15m",
      "turnover_rate": 5.2
    },
    "vehicle_breakdown": {
      "car": 134,
      "truck": 15,
      "bus": 4,
      "motorcycle": 3
    },
    "hourly_traffic": [
      {"hour": 8, "entries": 12, "exits": 2},
      {"hour": 9, "entries": 15, "exits": 8},
      {"hour": 10, "entries": 18, "exits": 14}
    ]
  }
}
```

## WebSocket Endpoints

### Real-time Updates

#### WS /ws/updates

Nhận updates real-time về tình trạng bãi đỗ xe.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/updates');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

**Message Format:**
```json
{
  "type": "parking_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "space_number": "A01",
    "status": "occupied",
    "vehicle": "30A-123.45"
  }
}
```

## Rate Limiting

- **File Upload**: 10 requests per minute
- **API Calls**: 100 requests per minute
- **WebSocket**: 1 connection per IP

## File Upload Limits

- **Max file size**: 100MB
- **Supported formats**: MP4, AVI, MOV, MKV, JPG, PNG, BMP
- **Max duration**: 10 minutes (video)

## Error Codes

| Code | Description |
|------|-------------|
| 1001 | Invalid file format |
| 1002 | File too large |
| 1003 | Processing failed |
| 1004 | No vehicles detected |
| 1005 | Database error |
| 1006 | Model loading error |

## SDK Examples

### Python

```python
import requests

# Upload video
with open('parking_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/universal/analyze',
        files={'file': f}
    )
    result = response.json()
    print(result)

# Get parking status
response = requests.get('http://localhost:8000/api/v1/parking/status')
status = response.json()
print(f"Empty spaces: {status['data']['empty_spaces']}")
```

### JavaScript

```javascript
// Upload video
const formData = new FormData();
formData.append('file', videoFile);

fetch('/api/v1/universal/analyze', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Get parking status
fetch('/api/v1/parking/status')
.then(response => response.json())
.then(data => {
    console.log(`Empty spaces: ${data.data.empty_spaces}`);
});
```

### cURL

```bash
# Upload video
curl -X POST \
  -F "file=@parking_video.mp4" \
  http://localhost:8000/api/v1/universal/analyze

# Get analytics
curl "http://localhost:8000/api/v1/analytics/summary?period=today"
```
