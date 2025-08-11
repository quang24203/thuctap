# Copilot Instructions for Smart Parking System

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a Smart Parking System using Computer Vision for monitoring 200-300 vehicles in parking lots.

## Key Technologies
- **YOLOv8**: Vehicle detection and license plate detection
- **PaddleOCR/EasyOCR**: Vietnamese license plate recognition  
- **ByteTrack**: Multi-object tracking for vehicles
- **Flask/FastAPI**: Web interface and API
- **OpenCV**: Image processing
- **SQLite/PostgreSQL**: Database for vehicle records

## Code Guidelines
- Use type hints for all functions
- Follow PEP 8 coding standards
- Implement error handling for all AI model operations
- Optimize for real-time processing (≥15 FPS)
- Support Vietnamese license plate formats
- Handle 200-300 concurrent vehicle tracking

## Performance Requirements
- Vehicle detection accuracy: ≥90% mAP
- License plate recognition accuracy: ≥85% 
- Processing speed: ≥15 FPS per camera stream
- Support 5-10 camera streams simultaneously

## File Structure Guidelines
- Keep AI models in `/models` directory
- Store configuration in `/config` directory  
- Web interface in `/web` directory
- Database schemas in `/database` directory
- Utilities and helpers in `/utils` directory
