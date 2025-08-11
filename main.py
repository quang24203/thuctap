#!/usr/bin/env python3
"""
Smart Parking System - Main Application
Hệ thống Giám sát Bãi đỗ xe Thông minh

Usage:
    python main.py [options]

Options:
    --config CONFIG_FILE    Configuration file path (default: config/app_config.yaml)
    --mode MODE            Run mode: web, processing, or full (default: full)
    --host HOST            Web server host (default: 0.0.0.0)
    --port PORT            Web server port (default: 8000)
    --debug                Enable debug mode
    --help                 Show this help message
"""

import argparse
import asyncio
import signal
import sys
import os
import threading
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.simple_logger import setup_logging, get_logger
from src.core.parking_system_manager import SmartParkingSystemManager
from src.web.api import app as fastapi_app
from src.web.app import create_flask_app
import uvicorn
from fastapi.staticfiles import StaticFiles

class SmartParkingApplication:
    """Main application class"""
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.parking_system = None
        self.web_server = None
        self.is_running = False
        
    def initialize(self):
        """Initialize the application"""
        try:
            # Load configuration
            self.config = Config(self.config_path)
            
            # Setup logging
            setup_logging(
                level=self.config.logging.level,
                log_file=self.config.logging.file,
                format_string=self.config.logging.format
            )
            
            self.logger = get_logger(self.__class__.__name__)
            self.logger.info("Smart Parking System initializing...")
            
            # Create necessary directories
            self._create_directories()
            
            self.logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize application: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "data/models",
            "data/raw", 
            "data/processed",
            "data/annotations",
            "data/uploads",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def start_processing_system(self):
        """Start the AI processing system"""
        try:
            self.logger.info("Starting AI processing system...")
            
            # Initialize parking system manager
            self.parking_system = SmartParkingSystemManager(self.config)
            
            # Start the system
            self.parking_system.start()
            
            self.logger.info("AI processing system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start processing system: {e}")
            return False
    
    def start_web_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the web server"""
        try:
            self.logger.info(f"Starting web server on {host}:{port}...")
            
            # Configure FastAPI app with static files
            fastapi_app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
            
            # Create Flask app for templates
            flask_app = create_flask_app(self.config)
            
            # Add template routes to FastAPI
            self._add_template_routes(flask_app)
            
            # Start server in a separate thread
            server_thread = threading.Thread(
                target=self._run_server,
                args=(host, port),
                daemon=True
            )
            server_thread.start()
            
            self.logger.info("Web server started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            return False
    
    def _add_template_routes(self, flask_app):
        """Add template routes to FastAPI"""
        from fastapi import Request
        from fastapi.responses import HTMLResponse
        
        @fastapi_app.get("/", response_class=HTMLResponse)
        @fastapi_app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            with flask_app.test_request_context():
                return flask_app.view_functions['dashboard']()
        
        @fastapi_app.get("/vehicles", response_class=HTMLResponse)
        async def vehicles():
            with flask_app.test_request_context():
                return flask_app.view_functions['vehicles']()

        @fastapi_app.get("/simple", response_class=HTMLResponse)
        async def simple_dashboard():
            with open("simple_dashboard.html", "r", encoding="utf-8") as f:
                return f.read()
        
        @fastapi_app.get("/parking", response_class=HTMLResponse)
        async def parking():
            with flask_app.test_request_context():
                return flask_app.view_functions['parking']()
        
        @fastapi_app.get("/analytics", response_class=HTMLResponse)
        async def analytics():
            with flask_app.test_request_context():
                return flask_app.view_functions['analytics']()
        
        @fastapi_app.get("/cameras", response_class=HTMLResponse)
        async def cameras():
            with flask_app.test_request_context():
                return flask_app.view_functions['cameras']()
        
        @fastapi_app.get("/settings", response_class=HTMLResponse)
        async def settings():
            with flask_app.test_request_context():
                return flask_app.view_functions['settings']()

        # Add new AI test routes
        @fastapi_app.get("/test-hub", response_class=HTMLResponse)
        async def test_hub():
            """AI Testing Hub with video and image analysis"""
            with flask_app.test_request_context():
                return flask_app.view_functions['dashboard']()  # Use enhanced dashboard    
    def _run_server(self, host: str, port: int):
        """Run the web server"""
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    
    def start(self, mode: str = "full", host: str = "0.0.0.0", port: int = 8000):
        """Start the application"""
        try:
            self.is_running = True
            
            if mode in ["processing", "full"]:
                if not self.start_processing_system():
                    return False
            
            if mode in ["web", "full"]:
                if not self.start_web_server(host, port):
                    return False
            
            self.logger.info(f"Smart Parking System started in {mode} mode")
            
            # Keep the main thread alive
            if mode == "full":
                self._run_main_loop()
            elif mode == "processing":
                self._run_processing_loop()
            elif mode == "web":
                self._run_web_loop()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
            self.stop()
            return True
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            self.stop()
            return False
    
    def _run_main_loop(self):
        """Main application loop"""
        self.logger.info("Application running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                # Print system status every 60 seconds
                if self.parking_system:
                    status = self.parking_system.get_system_status()
                    self.logger.info(f"System Status - Uptime: {status['system']['uptime_seconds']:.0f}s")
                
                time.sleep(60)
                
        except KeyboardInterrupt:
            pass
    
    def _run_processing_loop(self):
        """Processing-only loop"""
        self.logger.info("Processing system running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def _run_web_loop(self):
        """Web-only loop"""
        self.logger.info("Web server running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def stop(self):
        """Stop the application"""
        self.logger.info("Stopping Smart Parking System...")
        
        self.is_running = False
        
        # Stop processing system
        if self.parking_system:
            self.parking_system.stop()
        
        self.logger.info("Smart Parking System stopped")
    
    def get_status(self):
        """Get application status"""
        status = {
            "application": {
                "is_running": self.is_running,
                "config_path": self.config_path
            }
        }
        
        if self.parking_system:
            status.update(self.parking_system.get_system_status())
        
        return status

def setup_signal_handlers(app: SmartParkingApplication):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Parking System - Hệ thống Giám sát Bãi đỗ xe Thông minh"
    )
    
    parser.add_argument(
        "--config",
        default="config/app_config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--mode",
        choices=["web", "processing", "full"],
        default="full",
        help="Run mode: web (web server only), processing (AI only), or full (both)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = SmartParkingApplication(args.config)
    
    # Setup signal handlers
    setup_signal_handlers(app)
    
    # Initialize application
    if not app.initialize():
        print("Failed to initialize application")
        sys.exit(1)
    
    # Start application
    success = app.start(
        mode=args.mode,
        host=args.host,
        port=args.port
    )
    
    if not success:
        print("Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main()
