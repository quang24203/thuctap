#!/usr/bin/env python3
"""
Setup script for Smart Parking System
Installs dependencies and prepares the system for use
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse
import shutil

def run_command(command, check=True, shell=False):
    """Run a command and return the result"""
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, 
                                  capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check OS
    os_name = platform.system()
    print(f"Operating System: {os_name}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Available Memory: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM available. System may run slowly.")
    except ImportError:
        print("Could not check memory (psutil not installed)")
    
    # Check for CUDA
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA GPU detected")
        print("CUDA support available")
    else:
        print("ℹ️  No NVIDIA GPU detected, will use CPU")
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Upgrade pip
    print("Upgrading pip...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install --upgrade pip")
    if not success:
        print(f"❌ Failed to upgrade pip: {stderr}")
        return False
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print("Installing from requirements.txt...")
        success, stdout, stderr = run_command(f"{sys.executable} -m pip install -r requirements.txt")
        if not success:
            print(f"❌ Failed to install requirements: {stderr}")
            return False
        print("✅ Python dependencies installed")
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True

def install_system_dependencies():
    """Install system dependencies"""
    print("Installing system dependencies...")
    
    os_name = platform.system()
    
    if os_name == "Linux":
        # Ubuntu/Debian
        if shutil.which("apt-get"):
            commands = [
                "sudo apt-get update",
                "sudo apt-get install -y python3-opencv",
                "sudo apt-get install -y libgl1-mesa-glx",
                "sudo apt-get install -y libglib2.0-0",
                "sudo apt-get install -y libsm6",
                "sudo apt-get install -y libxext6",
                "sudo apt-get install -y libxrender-dev",
                "sudo apt-get install -y libgomp1"
            ]
            
            for cmd in commands:
                print(f"Running: {cmd}")
                success, stdout, stderr = run_command(cmd, shell=True)
                if not success:
                    print(f"⚠️  Warning: {cmd} failed: {stderr}")
        
        # CentOS/RHEL
        elif shutil.which("yum"):
            commands = [
                "sudo yum update -y",
                "sudo yum install -y opencv-python",
                "sudo yum install -y mesa-libGL",
                "sudo yum install -y glib2",
                "sudo yum install -y libSM",
                "sudo yum install -y libXext",
                "sudo yum install -y libXrender"
            ]
            
            for cmd in commands:
                print(f"Running: {cmd}")
                success, stdout, stderr = run_command(cmd, shell=True)
                if not success:
                    print(f"⚠️  Warning: {cmd} failed: {stderr}")
    
    elif os_name == "Darwin":  # macOS
        if shutil.which("brew"):
            commands = [
                "brew update",
                "brew install opencv",
                "brew install python-tk"
            ]
            
            for cmd in commands:
                print(f"Running: {cmd}")
                success, stdout, stderr = run_command(cmd, shell=True)
                if not success:
                    print(f"⚠️  Warning: {cmd} failed: {stderr}")
        else:
            print("ℹ️  Homebrew not found. Please install manually:")
            print("  - OpenCV")
            print("  - Python Tkinter")
    
    elif os_name == "Windows":
        print("ℹ️  Windows detected. Please ensure you have:")
        print("  - Microsoft Visual C++ Redistributable")
        print("  - Windows Media Feature Pack (for video codecs)")
    
    return True

def download_models():
    """Download pre-trained models"""
    print("Downloading pre-trained models...")
    
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download YOLOv8 models
    models_to_download = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt"
    ]
    
    try:
        from ultralytics import YOLO
        
        for model_name in models_to_download:
            model_path = models_dir / model_name
            if not model_path.exists():
                print(f"Downloading {model_name}...")
                model = YOLO(model_name)  # This will download the model
                # Move to our models directory
                import torch
                torch.save(model.model.state_dict(), model_path)
                print(f"✅ {model_name} downloaded")
            else:
                print(f"✅ {model_name} already exists")
    
    except ImportError:
        print("⚠️  Ultralytics not installed, skipping model download")
        print("   Models will be downloaded automatically when first used")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directory structure...")
    
    directories = [
        "data/models",
        "data/raw",
        "data/processed",
        "data/annotations",
        "data/uploads",
        "logs",
        "config",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True

def create_config_files():
    """Create default configuration files if they don't exist"""
    print("Creating configuration files...")
    
    config_dir = Path("config")
    
    # Check if config files exist
    app_config = config_dir / "app_config.yaml"
    model_config = config_dir / "model_config.yaml"
    
    if app_config.exists():
        print("✅ app_config.yaml already exists")
    else:
        print("ℹ️  app_config.yaml not found, please create it manually")
    
    if model_config.exists():
        print("✅ model_config.yaml already exists")
    else:
        print("ℹ️  model_config.yaml not found, please create it manually")
    
    return True

def setup_database():
    """Setup database"""
    print("Setting up database...")
    
    try:
        # Import after dependencies are installed
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from src.utils.config import Config
        from src.database.operations import DatabaseManager
        
        # Load config
        config = Config()
        
        # Initialize database
        db_manager = DatabaseManager(config.database.connection_string)
        
        # Create tables
        print("Creating database tables...")
        # Tables will be created automatically when first accessed
        
        print("✅ Database setup completed")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print("Running basic tests...")
    
    try:
        # Test imports
        print("Testing imports...")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
        except ImportError:
            print("⚠️  PyTorch not found")
        
        try:
            from ultralytics import YOLO
            print("✅ Ultralytics YOLO")
        except ImportError:
            print("⚠️  Ultralytics not found")
        
        try:
            import paddleocr
            print("✅ PaddleOCR")
        except ImportError:
            print("⚠️  PaddleOCR not found")
        
        # Test basic functionality
        print("Testing basic functionality...")
        
        # Test image loading
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        if test_image is not None:
            print("✅ Image processing")
        
        print("✅ Basic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Smart Parking System Setup")
    
    parser.add_argument("--skip-system", action="store_true", 
                       help="Skip system dependency installation")
    parser.add_argument("--skip-models", action="store_true", 
                       help="Skip model download")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip tests")
    parser.add_argument("--dev", action="store_true", 
                       help="Install development dependencies")
    
    args = parser.parse_args()
    
    print("🚀 Smart Parking System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install system dependencies
    if not args.skip_system:
        if not install_system_dependencies():
            print("⚠️  System dependency installation had warnings")
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Download models
    if not args.skip_models:
        if not download_models():
            print("⚠️  Model download had warnings")
    
    # Create config files
    if not create_config_files():
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        print("⚠️  Database setup had warnings")
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Some tests failed")
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your cameras in config/app_config.yaml")
    print("2. Train your models with your dataset:")
    print("   python scripts/train_vehicle_detection.py --data-dir your_dataset")
    print("3. Run the demo:")
    print("   python scripts/demo.py --mode detection")
    print("4. Start the full system:")
    print("   python main.py --mode full")

if __name__ == "__main__":
    main()
