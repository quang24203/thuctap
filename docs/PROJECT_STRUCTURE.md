# 📁 Project Structure

```
smart-parking-system/
├── 📁 src/                     # Source code
│   ├── 📁 core/               # Core business logic
│   ├── 📁 models/             # AI models
│   ├── 📁 web/                # Web application
│   ├── 📁 database/           # Database operations
│   ├── 📁 utils/              # Utilities
│   └── 📁 training/           # Model training
├── 📁 data/                   # Data directory
│   ├── 📁 models/             # Trained models
│   ├── 📁 videos/             # Video files
│   ├── 📁 uploads/            # Uploaded files
│   └── 📁 benchmark/          # Benchmark data
├── 📁 config/                 # Configuration files
├── 📁 docs/                   # Documentation
│   ├── 📄 TECHNICAL_REPORT.md # Technical report
│   ├── 📄 API.md              # API documentation
│   ├── 📄 INSTALLATION.md     # Installation guide
│   └── 📄 DEPLOYMENT.md       # Deployment guide
├── 📁 scripts/                # Utility scripts
├── 📄 README.md               # Main documentation
├── 📄 requirements.txt        # Dependencies
├── 📄 smart_parking_app.py    # Main application
└── 📄 LICENSE                 # License file
```

## 🎯 Key Components

### 📁 src/core/
- `parking_system_manager.py` - Main system manager
- `camera_processor.py` - Camera stream processing

### 📁 src/models/
- `vehicle_detection.py` - YOLOv8 vehicle detection
- `license_plate.py` - License plate recognition
- `tracking.py` - Vehicle tracking

### 📁 src/web/
- `app.py` - FastAPI application
- `api.py` - API endpoints
- `templates/` - HTML templates

### 📁 data/models/
- `vehicle_detection.pt` - Trained vehicle detection model
- `license_plate.pt` - Trained license plate model

## 📊 File Sizes
- Total project size: ~500MB
- Models: ~200MB
- Source code: ~50MB
- Documentation: ~10MB
- Data samples: ~240MB
