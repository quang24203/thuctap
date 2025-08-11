# 🚗 Smart Parking System

## Quick Start

1. **Emergency Server** (if main server fails):
   ```bash
   python emergency_server.py
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Main Server**:
   ```bash
   python online_server.py
   ```

## Features
- 🧠 AI Vehicle Detection (YOLOv8)
- 🔢 License Plate Recognition (OCR)
- 🅿️ Real-time Parking Analysis
- 🌐 Web Dashboard
- 📱 Responsive UI

## API Endpoints
- `GET /` - Dashboard
- `POST /api/v1/analyze` - Upload & analyze
- `GET /api/v1/parking/status` - Parking status
- `GET /health` - Health check

## Backup & Restore
```bash
python backup_and_restore.py
```
"# thuctap" 
