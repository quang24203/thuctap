# 🚀 Installation Guide - Hướng dẫn Cài đặt Chi tiết

## 📋 Yêu cầu Hệ thống

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 - 3.11
- **RAM**: 4GB (khuyến nghị 8GB+)
- **Storage**: 2GB trống
- **CPU**: Intel i5 hoặc AMD Ryzen 5 trở lên

### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU với 4GB+ VRAM (GTX 1060, RTX 2060+)
- **Storage**: SSD với 10GB+ trống
- **CPU**: Intel i7 hoặc AMD Ryzen 7 trở lên

## 🔧 Cài đặt Chi tiết

### Bước 1: Chuẩn bị Môi trường

#### Windows

```powershell
# Kiểm tra Python version
python --version

# Nếu chưa có Python, download từ python.org
# Hoặc cài qua Microsoft Store

# Cài Git (nếu chưa có)
# Download từ git-scm.com

# Kiểm tra GPU (tùy chọn)
nvidia-smi
```

#### Ubuntu/Linux

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Cài Python và pip
sudo apt install python3 python3-pip python3-venv -y

# Cài Git
sudo apt install git -y

# Cài các dependencies cho OpenCV
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -y

# Kiểm tra GPU (nếu có NVIDIA)
nvidia-smi
```

#### macOS

```bash
# Cài Homebrew (nếu chưa có)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Cài Python
brew install python@3.9

# Cài Git
brew install git

# Kiểm tra version
python3 --version
git --version
```

### Bước 2: Clone Repository

```bash
# Clone project
git clone https://github.com/your-username/smart-parking-system.git
cd smart-parking-system

# Kiểm tra structure
ls -la
```

### Bước 3: Tạo Virtual Environment

#### Windows

```powershell
# Tạo virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Kiểm tra activation
where python
```

#### Linux/macOS

```bash
# Tạo virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Kiểm tra activation
which python
```

### Bước 4: Cài đặt Dependencies

#### Cài đặt PyTorch (quan trọng!)

```bash
# Kiểm tra CUDA version (nếu có GPU)
nvidia-smi

# Cài PyTorch với CUDA 11.8 (khuyến nghị)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc CPU only (nếu không có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Cài đặt Dependencies chính

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Cài đặt requirements
pip install -r requirements.txt

# Verify installations
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import ultralytics; print('YOLOv8: OK')"
python -c "import paddleocr; print('PaddleOCR: OK')"
python -c "import fastapi; print('FastAPI: OK')"
```

### Bước 5: Tạo Thư mục và Cấu hình

```bash
# Tạo thư mục cần thiết
mkdir -p data/models data/uploads data/benchmark logs

# Windows
mkdir data\models data\uploads data\benchmark logs

# Copy config files
cp config/app_config.yaml.example config/app_config.yaml
cp config/model_config.yaml.example config/model_config.yaml

# Set permissions (Linux/macOS)
chmod 755 data/
chmod 755 logs/
```

### Bước 6: Khởi tạo Database

```bash
# Chạy database setup
python scripts/setup_database.py

# Verify database
ls -la data/parking_system.db

# Test database connection
python -c "
import sqlite3
conn = sqlite3.connect('data/parking_system.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\";')
tables = cursor.fetchall()
print('Tables:', tables)
conn.close()
"
```

### Bước 7: Download AI Models

```bash
# YOLOv8 models sẽ tự động download khi chạy lần đầu
# Hoặc download manual:
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLOv8n downloaded successfully')
"

# PaddleOCR models sẽ tự động download khi chạy lần đầu
python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print('PaddleOCR models downloaded successfully')
"
```

### Bước 8: Test Installation

```bash
# Test import tất cả modules
python -c "
import sys
sys.path.append('src')

try:
    from src.models.vehicle_detection import VehicleDetector
    from src.models.license_plate_recognition import LicensePlateRecognizer
    from src.web.app import create_app
    print('✅ All imports successful!')
except Exception as e:
    print(f'❌ Import error: {e}')
"

# Test database
python scripts/setup_database.py --test

# Test web server
python -c "
import uvicorn
from src.web.app import create_app
app = create_app()
print('✅ Web server can be created!')
"
```

## 🚀 Chạy Ứng dụng

### Development Mode

```bash
# Chạy với auto-reload
python smart_parking_app.py

# Hoặc chạy trực tiếp web server
python src/web/app.py

# Hoặc với uvicorn
uvicorn src.web.app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Cài gunicorn
pip install gunicorn

# Chạy với gunicorn
gunicorn src.web.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Hoặc với systemd service (Linux)
sudo cp scripts/smart-parking.service /etc/systemd/system/
sudo systemctl enable smart-parking
sudo systemctl start smart-parking
```

## 🔍 Troubleshooting

### Common Issues

#### 1. PyTorch CUDA Issues

```bash
# Kiểm tra CUDA compatibility
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Nếu CUDA không work, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. OpenCV Issues

```bash
# Linux: Missing dependencies
sudo apt install libgl1-mesa-glx libglib2.0-0

# macOS: Permission issues
brew install opencv

# Windows: Visual C++ redistributable
# Download từ Microsoft website
```

#### 3. PaddleOCR Issues

```bash
# Network issues downloading models
export HUB_HOME=./models  # Linux/macOS
set HUB_HOME=./models     # Windows

# Manual download
mkdir -p ~/.paddleocr/whl/det/ch
mkdir -p ~/.paddleocr/whl/rec/ch
# Download models từ PaddleOCR GitHub releases
```

#### 4. Permission Issues

```bash
# Linux/macOS
sudo chown -R $USER:$USER data/
sudo chown -R $USER:$USER logs/
chmod -R 755 data/
chmod -R 755 logs/

# Windows: Run as Administrator
```

#### 5. Port Already in Use

```bash
# Kiểm tra port usage
netstat -tulpn | grep :8000  # Linux
netstat -ano | findstr :8000  # Windows

# Kill process
sudo kill -9 <PID>  # Linux
taskkill /PID <PID> /F  # Windows

# Hoặc dùng port khác
python smart_parking_app.py --port 8001
```

### Performance Issues

#### 1. Slow Processing

```bash
# Kiểm tra GPU usage
nvidia-smi

# Reduce model size
# Trong config/model_config.yaml:
# model_path: "yolov8n.pt"  # Thay vì yolov8s.pt

# Reduce input resolution
# confidence_threshold: 0.6  # Tăng threshold
```

#### 2. Memory Issues

```bash
# Monitor memory usage
htop  # Linux
Task Manager  # Windows

# Reduce batch size
# Trong code, giảm batch_size parameter
```

## 🐳 Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t smart-parking .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data smart-parking

# With GPU support
docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data smart-parking
```

## ✅ Verification

Sau khi cài đặt xong, verify bằng cách:

1. **Truy cập Dashboard**: http://localhost:8000
2. **Test API**: Click "Test API" button
3. **Upload video**: Drag & drop một video file
4. **Check logs**: `tail -f logs/app.log`

Nếu tất cả hoạt động OK, bạn đã cài đặt thành công! 🎉
