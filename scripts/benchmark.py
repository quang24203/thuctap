#!/usr/bin/env python3
"""
Benchmark script for Smart Parking System
Tests performance and accuracy of the system components
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2
import psutil
import GPUtil

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.vehicle_detection import VehicleDetector
from src.models.license_plate import LicensePlateRecognizer
from src.models.simple_tracker import MultiObjectTracker
from src.utils.config import Config
from src.utils.simple_logger import setup_logging, get_logger

class SystemBenchmark:
    """Benchmark class for testing system performance"""
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config = Config(config_path)
        self.logger = get_logger(self.__class__.__name__)
        
        # Models
        self.vehicle_detector = None
        self.license_plate_recognizer = None
        self.tracker = None
        
        # Benchmark results
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'model_performance': {},
            'accuracy_metrics': {},
            'resource_usage': {}
        }
    
    def _get_system_info(self):
        """Get system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'opencv_version': cv2.__version__
        }
        
        # GPU information
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['gpu'] = {
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'driver_version': gpu.driver
                }
            else:
                info['gpu'] = None
        except:
            info['gpu'] = None
        
        return info
    
    def initialize_models(self):
        """Initialize models for benchmarking"""
        self.logger.info("Initializing models for benchmarking...")
        
        try:
            # Vehicle detection model
            self.vehicle_detector = VehicleDetector(
                model_path=self.config.models.vehicle_detection.get('model_path', 'yolov8n.pt'),
                confidence=self.config.models.vehicle_detection.get('confidence_threshold', 0.5),
                device=self.config.models.vehicle_detection.get('device', 'cpu')
            )
            
            # License plate recognition
            self.license_plate_recognizer = LicensePlateRecognizer(
                detection_model=self.config.models.license_plate.get('detection_model', 'yolov8n.pt'),
                ocr_engine=self.config.models.license_plate.get('ocr_engine', 'paddleocr'),
                confidence=self.config.models.license_plate.get('confidence_threshold', 0.7)
            )
            
            # Multi-object tracker
            self.tracker = MultiObjectTracker(
                track_thresh=self.config.models.tracking.get('track_thresh', 0.5),
                track_buffer=self.config.models.tracking.get('track_buffer', 30),
                match_thresh=self.config.models.tracking.get('match_thresh', 0.8),
                frame_rate=self.config.models.tracking.get('frame_rate', 30)
            )
            
            self.logger.info("Models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            return False
    
    def benchmark_vehicle_detection(self, test_images_dir: str, num_iterations: int = 100):
        """Benchmark vehicle detection performance"""
        self.logger.info("Benchmarking vehicle detection...")
        
        # Load test images
        images_path = Path(test_images_dir)
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        if not image_files:
            self.logger.error("No test images found")
            return
        
        # Select random images for testing
        test_images = np.random.choice(image_files, min(num_iterations, len(image_files)), replace=True)
        
        # Benchmark metrics
        inference_times = []
        detection_counts = []
        memory_usage = []
        
        self.logger.info(f"Testing {len(test_images)} images...")
        
        for i, img_path in enumerate(test_images):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Measure memory before inference
            memory_before = psutil.virtual_memory().used / (1024**2)  # MB
            
            # Measure inference time
            start_time = time.time()
            detections = self.vehicle_detector.detect(image)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Measure memory after inference
            memory_after = psutil.virtual_memory().used / (1024**2)  # MB
            
            inference_times.append(inference_time)
            detection_counts.append(len(detections))
            memory_usage.append(memory_after - memory_before)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(test_images)} images")
        
        # Calculate statistics
        self.results['model_performance']['vehicle_detection'] = {
            'num_images': len(test_images),
            'avg_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'avg_fps': 1000.0 / np.mean(inference_times),
            'avg_detections_per_image': np.mean(detection_counts),
            'avg_memory_usage_mb': np.mean(memory_usage)
        }
        
        self.logger.info("Vehicle detection benchmark completed")
    
    def benchmark_license_plate_recognition(self, test_images_dir: str, num_iterations: int = 50):
        """Benchmark license plate recognition performance"""
        self.logger.info("Benchmarking license plate recognition...")
        
        # Load test images (should be cropped license plate images)
        images_path = Path(test_images_dir)
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        if not image_files:
            self.logger.error("No license plate test images found")
            return
        
        # Select random images for testing
        test_images = np.random.choice(image_files, min(num_iterations, len(image_files)), replace=True)
        
        # Benchmark metrics
        recognition_times = []
        success_count = 0
        
        self.logger.info(f"Testing {len(test_images)} license plate images...")
        
        for i, img_path in enumerate(test_images):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Measure recognition time
            start_time = time.time()
            result = self.license_plate_recognizer.recognize(image)
            recognition_time = (time.time() - start_time) * 1000  # ms
            
            recognition_times.append(recognition_time)
            
            if result and result.get('text'):
                success_count += 1
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(test_images)} license plates")
        
        # Calculate statistics
        self.results['model_performance']['license_plate_recognition'] = {
            'num_images': len(test_images),
            'avg_recognition_time_ms': np.mean(recognition_times),
            'std_recognition_time_ms': np.std(recognition_times),
            'min_recognition_time_ms': np.min(recognition_times),
            'max_recognition_time_ms': np.max(recognition_times),
            'success_rate': success_count / len(test_images),
            'successful_recognitions': success_count
        }
        
        self.logger.info("License plate recognition benchmark completed")
    
    def benchmark_tracking(self, video_path: str):
        """Benchmark tracking performance"""
        self.logger.info("Benchmarking tracking performance...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
        
        # Tracking metrics
        tracking_times = []
        track_counts = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect vehicles first
            detections = self.vehicle_detector.detect(frame)
            
            # Measure tracking time
            start_time = time.time()
            tracked_vehicles = self.tracker.update(detections)
            tracking_time = (time.time() - start_time) * 1000  # ms
            
            tracking_times.append(tracking_time)
            track_counts.append(len(tracked_vehicles))
            
            if frame_count % 30 == 0:
                self.logger.info(f"Processed {frame_count} frames")
            
            # Limit to 300 frames for benchmark
            if frame_count >= 300:
                break
        
        cap.release()
        
        # Calculate statistics
        self.results['model_performance']['tracking'] = {
            'num_frames': frame_count,
            'avg_tracking_time_ms': np.mean(tracking_times),
            'std_tracking_time_ms': np.std(tracking_times),
            'avg_tracks_per_frame': np.mean(track_counts),
            'max_tracks_per_frame': np.max(track_counts)
        }
        
        self.logger.info("Tracking benchmark completed")
    
    def benchmark_system_resources(self, duration_seconds: int = 60):
        """Benchmark system resource usage"""
        self.logger.info(f"Monitoring system resources for {duration_seconds} seconds...")
        
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # CPU usage
            cpu_usage.append(psutil.cpu_percent(interval=1))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage.append(memory.percent)
            
            # GPU usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage.append(gpus[0].load * 100)
                else:
                    gpu_usage.append(0)
            except:
                gpu_usage.append(0)
        
        self.results['resource_usage'] = {
            'duration_seconds': duration_seconds,
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'avg_memory_percent': np.mean(memory_usage),
            'max_memory_percent': np.max(memory_usage),
            'avg_gpu_percent': np.mean(gpu_usage),
            'max_gpu_percent': np.max(gpu_usage)
        }
        
        self.logger.info("Resource monitoring completed")
    
    def run_full_benchmark(self, test_data_dir: str):
        """Run complete benchmark suite"""
        self.logger.info("Starting full benchmark suite...")
        
        test_path = Path(test_data_dir)
        
        # Vehicle detection benchmark
        vehicle_images_dir = test_path / "vehicle_images"
        if vehicle_images_dir.exists():
            self.benchmark_vehicle_detection(str(vehicle_images_dir))
        
        # License plate recognition benchmark
        license_plate_images_dir = test_path / "license_plate_images"
        if license_plate_images_dir.exists():
            self.benchmark_license_plate_recognition(str(license_plate_images_dir))
        
        # Tracking benchmark
        test_video = test_path / "test_video.mp4"
        if test_video.exists():
            self.benchmark_tracking(str(test_video))
        
        # Resource usage benchmark
        self.benchmark_system_resources(60)
        
        self.logger.info("Full benchmark completed")
    
    def save_results(self, output_file: str):
        """Save benchmark results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_file}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("SMART PARKING SYSTEM BENCHMARK RESULTS")
        print("="*60)
        
        # System info
        print(f"CPU Cores: {self.results['system_info']['cpu_count']}")
        print(f"Memory: {self.results['system_info']['memory_total_gb']:.1f} GB")
        if self.results['system_info']['gpu']:
            print(f"GPU: {self.results['system_info']['gpu']['name']}")
        
        # Model performance
        if 'vehicle_detection' in self.results['model_performance']:
            vd = self.results['model_performance']['vehicle_detection']
            print(f"\nVehicle Detection:")
            print(f"  Average FPS: {vd['avg_fps']:.1f}")
            print(f"  Average inference time: {vd['avg_inference_time_ms']:.1f} ms")
            print(f"  Average detections per image: {vd['avg_detections_per_image']:.1f}")
        
        if 'license_plate_recognition' in self.results['model_performance']:
            lpr = self.results['model_performance']['license_plate_recognition']
            print(f"\nLicense Plate Recognition:")
            print(f"  Success rate: {lpr['success_rate']*100:.1f}%")
            print(f"  Average recognition time: {lpr['avg_recognition_time_ms']:.1f} ms")
        
        if 'tracking' in self.results['model_performance']:
            tr = self.results['model_performance']['tracking']
            print(f"\nTracking:")
            print(f"  Average tracking time: {tr['avg_tracking_time_ms']:.1f} ms")
            print(f"  Average tracks per frame: {tr['avg_tracks_per_frame']:.1f}")
        
        # Resource usage
        if 'resource_usage' in self.results:
            ru = self.results['resource_usage']
            print(f"\nResource Usage:")
            print(f"  Average CPU: {ru['avg_cpu_percent']:.1f}%")
            print(f"  Average Memory: {ru['avg_memory_percent']:.1f}%")
            print(f"  Average GPU: {ru['avg_gpu_percent']:.1f}%")
        
        print("="*60)

def main():
    """Main benchmark script"""
    parser = argparse.ArgumentParser(description="Smart Parking System Benchmark")
    
    parser.add_argument("--config", default="config/app_config.yaml", help="Configuration file")
    parser.add_argument("--test-data", required=True, help="Test data directory")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--vehicle-images", help="Vehicle detection test images")
    parser.add_argument("--license-images", help="License plate test images")
    parser.add_argument("--test-video", help="Test video for tracking")
    parser.add_argument("--iterations", type=int, default=100, help="Number of test iterations")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("Benchmark")
    
    logger.info("Starting Smart Parking System Benchmark")
    
    try:
        # Initialize benchmark
        benchmark = SystemBenchmark(args.config)
        
        if not benchmark.initialize_models():
            logger.error("Failed to initialize models")
            sys.exit(1)
        
        # Run benchmarks
        if args.vehicle_images:
            benchmark.benchmark_vehicle_detection(args.vehicle_images, args.iterations)
        
        if args.license_images:
            benchmark.benchmark_license_plate_recognition(args.license_images, args.iterations)
        
        if args.test_video:
            benchmark.benchmark_tracking(args.test_video)
        
        if not any([args.vehicle_images, args.license_images, args.test_video]):
            # Run full benchmark
            benchmark.run_full_benchmark(args.test_data)
        
        # Save and display results
        benchmark.save_results(args.output)
        benchmark.print_summary()
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
