#!/usr/bin/env python3
"""
🧠 Simple Real AI Server
Chạy được chắc chắn với AI thật
"""

import os
import sys
import time
import json
import webbrowser
import threading
from pathlib import Path

# Check and import dependencies
def check_dependencies():
    """Check if AI dependencies are available"""
    deps = {
        'cv2': False,
        'numpy': False, 
        'ultralytics': False,
        'paddleocr': False
    }
    
    try:
        import cv2
        deps['cv2'] = True
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV missing: pip install opencv-python")
    
    try:
        import numpy as np
        deps['numpy'] = True
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy missing: pip install numpy")
    
    try:
        from ultralytics import YOLO
        deps['ultralytics'] = True
        print("✅ YOLOv8 available")
    except ImportError:
        print("❌ YOLOv8 missing: pip install ultralytics")
    
    try:
        from paddleocr import PaddleOCR
        deps['paddleocr'] = True
        print("✅ PaddleOCR available")
    except ImportError:
        print("❌ PaddleOCR missing: pip install paddlepaddle paddleocr")
    
    return deps

# Check dependencies
print("🔍 Checking AI dependencies...")
DEPS = check_dependencies()

# Import what's available
if DEPS['cv2'] and DEPS['numpy']:
    import cv2
    import numpy as np

if DEPS['ultralytics']:
    from ultralytics import YOLO

if DEPS['paddleocr']:
    from paddleocr import PaddleOCR

import http.server
import socketserver

class SimpleRealAI:
    def __init__(self):
        self.setup_models()
    
    def setup_models(self):
        """Setup AI models if available"""
        print("🧠 Setting up AI models...")
        
        self.models_available = DEPS['cv2'] and DEPS['numpy'] and DEPS['ultralytics']
        self.ocr_available = DEPS['paddleocr']
        
        # Check trained models
        self.license_model_path = "data/models/license_plate_quick/weights/best.pt"
        self.parking_model_path = "data/models/pklot_detection/weights/best.pt"
        
        self.license_model_exists = Path(self.license_model_path).exists()
        self.parking_model_exists = Path(self.parking_model_path).exists()
        
        print(f"  📁 License Model: {'✅ Found' if self.license_model_exists else '❌ Missing'}")
        print(f"  📁 Parking Model: {'✅ Found' if self.parking_model_exists else '❌ Missing'}")
        
        self.models_loaded = False
        
        if self.models_available:
            try:
                # Load models
                if self.license_model_exists:
                    print("  📥 Loading YOUR license plate model...")
                    self.license_model = YOLO(self.license_model_path)
                    print("  ✅ License model loaded!")
                else:
                    print("  📥 Loading default YOLOv8...")
                    self.license_model = YOLO('yolov8n.pt')
                    print("  ✅ Default model loaded!")
                
                if self.parking_model_exists:
                    print("  📥 Loading YOUR parking model...")
                    self.parking_model = YOLO(self.parking_model_path)
                    print("  ✅ Parking model loaded!")
                else:
                    self.parking_model = self.license_model  # Use same model
                
                if self.ocr_available:
                    print("  📥 Loading PaddleOCR...")
                    self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                    print("  ✅ OCR loaded!")
                
                self.models_loaded = True
                print("🎯 REAL AI Models loaded successfully!")
                
            except Exception as e:
                print(f"  ⚠️ Model loading failed: {e}")
                self.models_loaded = False
        else:
            print("  ⚠️ AI dependencies missing - using simulation")
    
    def analyze_image_real(self, image_data):
        """Analyze image with real AI"""
        if not self.models_loaded:
            return self.generate_simulation_results()
        
        try:
            print("  🎬 Processing with REAL AI...")
            
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode image")
            
            # Vehicle detection with REAL AI
            print("  🚗 Running vehicle detection...")
            vehicle_results = self.parking_model(frame, conf=0.4, iou=0.45)
            
            vehicles = []
            vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            
            for result in vehicle_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id in vehicle_classes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            vehicles.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_name': self.get_class_name(class_id)
                            })
            
            print(f"  🚗 Found {len(vehicles)} vehicles")
            
            # License plate detection with REAL AI
            print("  🔢 Running license plate detection...")
            plate_results = self.license_model(frame, conf=0.3, iou=0.45)
            
            license_plates = []
            
            for result in plate_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Extract plate region
                        plate_region = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        if plate_region.size > 0 and self.ocr_available:
                            # OCR with PaddleOCR
                            ocr_results = self.ocr.ocr(plate_region, cls=True)
                            
                            if ocr_results and ocr_results[0]:
                                for line in ocr_results[0]:
                                    text = line[1][0]
                                    ocr_confidence = line[1][1]
                                    
                                    if self.validate_plate(text):
                                        license_plates.append({
                                            'plate_number': self.format_plate(text),
                                            'confidence': float(confidence * ocr_confidence),
                                            'vehicle_class': 'car',
                                            'detection_method': 'REAL AI: YOLOv8 + PaddleOCR'
                                        })
                                        break
            
            print(f"  🔢 Found {len(license_plates)} license plates")
            
            # Calculate results
            total_spaces = 20
            occupied = len(vehicles)
            empty = max(0, total_spaces - occupied)
            
            return {
                "vehicle_detection": {
                    "total_vehicles_detected": len(vehicles),
                    "detection_method": "REAL YOLOv8" + (" (YOUR trained model)" if self.parking_model_exists else " (default)")
                },
                "license_plates": license_plates,
                "parking_analysis": {
                    "total_spaces": total_spaces,
                    "occupied_spaces": occupied,
                    "empty_spaces": empty,
                    "occupancy_rate": occupied / total_spaces
                },
                "ai_info": {
                    "models_loaded": True,
                    "using_trained_models": self.license_model_exists or self.parking_model_exists,
                    "processing_method": "REAL AI"
                }
            }
            
        except Exception as e:
            print(f"  ⚠️ Real AI processing failed: {e}")
            return self.generate_simulation_results()
    
    def get_class_name(self, class_id):
        """Get class name"""
        names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return names.get(class_id, 'vehicle')
    
    def validate_plate(self, text):
        """Validate license plate"""
        import re
        text = text.replace(' ', '').replace('-', '').upper()
        has_digits = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        return has_digits and has_letters and len(text) >= 6
    
    def format_plate(self, text):
        """Format license plate"""
        import re
        text = text.replace(' ', '').replace('-', '').upper()
        match = re.match(r'(\d{2})([A-Z])(\d{3})(\d{2})', text)
        if match:
            return f"{match.group(1)}{match.group(2)}-{match.group(3)}.{match.group(4)}"
        return text
    
    def generate_simulation_results(self):
        """Generate simulation when AI not available"""
        import random
        
        plates = ["30A-123.45", "51B-678.90", "29X-456.78", "43C-789.01"]
        num_vehicles = random.randint(6, 12)
        num_plates = random.randint(3, min(num_vehicles, 6))
        
        license_plates = []
        for i in range(num_plates):
            license_plates.append({
                'plate_number': plates[i % len(plates)],
                'confidence': random.uniform(0.7, 0.9),
                'vehicle_class': random.choice(['car', 'truck', 'bus']),
                'detection_method': 'Simulation (AI not available)'
            })
        
        return {
            "vehicle_detection": {
                "total_vehicles_detected": num_vehicles,
                "detection_method": "Simulation"
            },
            "license_plates": license_plates,
            "parking_analysis": {
                "total_spaces": 20,
                "occupied_spaces": num_vehicles,
                "empty_spaces": 20 - num_vehicles,
                "occupancy_rate": num_vehicles / 20
            },
            "ai_info": {
                "models_loaded": False,
                "using_trained_models": False,
                "processing_method": "Simulation"
            }
        }

# Initialize AI
print("🚀 Initializing Simple Real AI...")
ai = SimpleRealAI()

class SimpleAIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_dashboard()
        elif self.path == '/api/status':
            self.send_json({
                "success": True,
                "ai_status": {
                    "models_loaded": ai.models_loaded,
                    "dependencies": DEPS,
                    "trained_models": {
                        "license_plate": ai.license_model_exists,
                        "parking": ai.parking_model_exists
                    }
                }
            })
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/analyze':
            self.handle_analysis()
        else:
            self.send_error(404)
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def handle_analysis(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json({"success": False, "error": "No file data"})
                return
            
            file_data = self.rfile.read(content_length)
            print(f"\n🎬 Analyzing file ({len(file_data)} bytes)")
            
            start_time = time.time()
            result = ai.analyze_image_real(file_data)
            processing_time = time.time() - start_time
            
            self.send_json({
                "success": True,
                "message": "✅ Analysis complete!",
                "processing_time": round(processing_time, 2),
                "analysis_data": result
            })
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            self.send_json({"success": False, "error": str(e)})
    
    def send_dashboard(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        ai_status = "✅ REAL AI" if ai.models_loaded else "❌ Simulation"
        model_info = "YOUR Trained Models" if (ai.license_model_exists or ai.parking_model_exists) else "Default Models"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>🧠 Simple Real AI Parking</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; }}
        .container {{ max-width: 1000px; margin: 50px auto; background: white; border-radius: 20px; padding: 40px; }}
        .btn-ai {{ background: linear-gradient(45deg, #28a745, #20c997); border: none; color: white; padding: 15px 25px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">🧠 Simple Real AI Smart Parking</h1>
        
        <div class="alert alert-{'success' if ai.models_loaded else 'warning'} text-center">
            <h5>🤖 AI Status: {ai_status}</h5>
            <p><strong>Models:</strong> {model_info}</p>
            <div class="row text-center">
                <div class="col-3">
                    <strong>OpenCV:</strong><br>
                    {'✅' if DEPS['cv2'] else '❌'}
                </div>
                <div class="col-3">
                    <strong>YOLOv8:</strong><br>
                    {'✅' if DEPS['ultralytics'] else '❌'}
                </div>
                <div class="col-3">
                    <strong>PaddleOCR:</strong><br>
                    {'✅' if DEPS['paddleocr'] else '❌'}
                </div>
                <div class="col-3">
                    <strong>Trained Models:</strong><br>
                    {'✅' if (ai.license_model_exists or ai.parking_model_exists) else '❌'}
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <input type="file" id="fileInput" class="form-control mb-3" accept="image/*">
                <button class="btn btn-ai w-100" onclick="analyzeImage()">
                    🧠 Analyze with {'REAL AI' if ai.models_loaded else 'Simulation'}
                </button>
            </div>
            <div class="col-md-4">
                <button class="btn btn-outline-info w-100 mb-2" onclick="checkStatus()">🔍 Check Status</button>
                <div class="alert alert-info">
                    <small>
                        <strong>Processing:</strong><br>
                        {'• Real YOLOv8 detection' if ai.models_loaded else '• Simulation mode'}<br>
                        {'• Real PaddleOCR' if ai.ocr_available else '• No OCR available'}<br>
                        {'• Your trained models' if (ai.license_model_exists or ai.parking_model_exists) else '• Default models'}
                    </small>
                </div>
            </div>
        </div>
        
        <div id="results" class="mt-4"></div>
    </div>
    
    <script>
        function analyzeImage() {{
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {{
                alert('Please select an image first!');
                return;
            }}
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            document.getElementById('results').innerHTML = `
                <div class="alert alert-warning">
                    <h5>🧠 Processing with {'REAL AI' if ai.models_loaded else 'Simulation'}...</h5>
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
                    </div>
                </div>
            `;
            
            fetch('/api/analyze', {{
                method: 'POST',
                body: formData
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    displayResults(data);
                }} else {{
                    document.getElementById('results').innerHTML = 
                        '<div class="alert alert-danger">❌ Error: ' + data.error + '</div>';
                }}
            }});
        }}
        
        function displayResults(data) {{
            const analysis = data.analysis_data;
            
            let html = '<div class="alert alert-success">✅ Analysis Complete!</div>';
            
            // AI Info
            html += '<div class="alert alert-info">';
            html += '<h6>🧠 Processing Info:</h6>';
            html += '<p><strong>Method:</strong> ' + analysis.ai_info.processing_method + '</p>';
            html += '<p><strong>Models Loaded:</strong> ' + (analysis.ai_info.models_loaded ? '✅ Yes' : '❌ No') + '</p>';
            html += '<p><strong>Using Trained Models:</strong> ' + (analysis.ai_info.using_trained_models ? '✅ Yes' : '❌ No') + '</p>';
            html += '<p><strong>Processing Time:</strong> ' + data.processing_time + ' seconds</p>';
            html += '</div>';
            
            // Results
            html += '<div class="row text-center mb-3">';
            html += '<div class="col-3"><h4>' + analysis.vehicle_detection.total_vehicles_detected + '</h4><small>🚗 Vehicles</small></div>';
            html += '<div class="col-3"><h4>' + analysis.license_plates.length + '</h4><small>🔢 Plates</small></div>';
            html += '<div class="col-3"><h4>' + analysis.parking_analysis.empty_spaces + '</h4><small>🅿️ Empty</small></div>';
            html += '<div class="col-3"><h4>' + (analysis.parking_analysis.occupancy_rate * 100).toFixed(1) + '%</h4><small>📊 Occupancy</small></div>';
            html += '</div>';
            
            // License plates
            if (analysis.license_plates.length > 0) {{
                html += '<h5>🔢 License Plates:</h5>';
                analysis.license_plates.forEach(plate => {{
                    html += '<div class="alert alert-light">';
                    html += '<strong>' + plate.plate_number + '</strong> ';
                    html += '<span class="badge bg-primary">' + (plate.confidence * 100).toFixed(1) + '%</span><br>';
                    html += '<small>' + plate.detection_method + '</small>';
                    html += '</div>';
                }});
            }}
            
            // Detection method
            html += '<div class="alert alert-warning">';
            html += '<h6>🔍 Detection Method:</h6>';
            html += '<p>' + analysis.vehicle_detection.detection_method + '</p>';
            html += '</div>';
            
            document.getElementById('results').innerHTML = html;
        }}
        
        function checkStatus() {{
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {{
                document.getElementById('results').innerHTML = 
                    '<div class="alert alert-info"><h5>🔍 System Status:</h5><pre>' + 
                    JSON.stringify(data, null, 2) + '</pre></div>';
            }});
        }}
    </script>
</body>
</html>"""
        
        self.wfile.write(html.encode())

def open_browser():
    time.sleep(3)
    webbrowser.open('http://localhost:8000')

def main():
    PORT = 8000
    
    print(f"\n🧠 Starting Simple Real AI Server...")
    print(f"🌐 Dashboard: http://localhost:{PORT}")
    print(f"🤖 AI Status: {'REAL AI' if ai.models_loaded else 'Simulation'}")
    print("📱 Browser will open automatically...")
    
    # Start browser
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start server
    with socketserver.TCPServer(("", PORT), SimpleAIHandler) as httpd:
        print(f"✅ Server running on port {PORT}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()
