#!/usr/bin/env python3
"""
🚀 Quick Smart Parking Server
Guaranteed to work!
"""

import http.server
import socketserver
import json
import webbrowser
import threading
import time

class QuickHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_dashboard()
        elif self.path == '/api/test':
            self.send_json({"success": True, "message": "✅ Server Working!"})
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/analyze':
            self.handle_analyze()
        else:
            self.send_error(404)
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def handle_analyze(self):
        # Simulate analysis
        time.sleep(1)
        result = {
            "success": True,
            "analysis_data": {
                "vehicle_detection": {"total_vehicles_detected": 8},
                "license_plates": [
                    {"plate_number": "30A-123.45", "confidence": 0.89, "vehicle_class": "car"},
                    {"plate_number": "51B-678.90", "confidence": 0.92, "vehicle_class": "truck"}
                ],
                "parking_analysis": {
                    "total_spaces": 20, "occupied_spaces": 14, 
                    "empty_spaces": 6, "occupancy_rate": 0.7
                }
            }
        }
        self.send_json(result)
    
    def send_dashboard(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """<!DOCTYPE html>
<html><head><title>🚗 Smart Parking Demo</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body { background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; }
.container { max-width: 1000px; margin: 50px auto; background: white; border-radius: 20px; padding: 40px; }
.btn-demo { background: linear-gradient(45deg, #28a745, #20c997); border: none; color: white; padding: 15px 25px; border-radius: 10px; margin: 10px; }
</style></head>
<body>
<div class="container">
    <h1 class="text-center mb-4">🚗 Smart Parking System</h1>
    <p class="text-center text-muted">Real-time parking management with AI</p>
    
    <div class="row">
        <div class="col-md-6">
            <input type="file" id="fileInput" class="form-control mb-3" accept="image/*,video/*">
            <button class="btn btn-demo w-100" onclick="analyzeFile()">🧠 Analyze with AI</button>
        </div>
        <div class="col-md-6">
            <button class="btn btn-success w-100 mb-2" onclick="showDemo()">🎬 Show Demo</button>
            <button class="btn btn-info w-100" onclick="testAPI()">🧪 Test API</button>
        </div>
    </div>
    
    <div id="results" class="mt-4"></div>
</div>

<script>
function analyzeFile() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) {
        alert('Please select a file first!');
        return;
    }
    
    document.getElementById('results').innerHTML = '<div class="alert alert-info">🧠 Analyzing...</div>';
    
    fetch('/api/analyze', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data.analysis_data);
        }
    });
}

function displayResults(data) {
    let html = '<div class="alert alert-success">✅ Analysis Complete!</div>';
    
    html += '<div class="row text-center mb-3">';
    html += '<div class="col-3"><h4>' + data.vehicle_detection.total_vehicles_detected + '</h4><small>🚗 Vehicles</small></div>';
    html += '<div class="col-3"><h4>' + data.license_plates.length + '</h4><small>🔢 Plates</small></div>';
    html += '<div class="col-3"><h4>' + data.parking_analysis.empty_spaces + '</h4><small>🅿️ Empty</small></div>';
    html += '<div class="col-3"><h4>' + (data.parking_analysis.occupancy_rate * 100).toFixed(1) + '%</h4><small>📊 Occupancy</small></div>';
    html += '</div>';
    
    html += '<h5>🚗 License Plates:</h5>';
    data.license_plates.forEach(plate => {
        html += '<span class="badge bg-primary me-2 mb-2" style="font-size: 1rem; padding: 8px 12px;">' + plate.plate_number + '</span>';
    });
    
    document.getElementById('results').innerHTML = html;
}

function showDemo() {
    const demoData = {
        vehicle_detection: {total_vehicles_detected: 8},
        license_plates: [
            {plate_number: "30A-123.45", confidence: 0.89, vehicle_class: "car"},
            {plate_number: "51B-678.90", confidence: 0.92, vehicle_class: "truck"},
            {plate_number: "29X-456.78", confidence: 0.85, vehicle_class: "car"}
        ],
        parking_analysis: {total_spaces: 20, occupied_spaces: 14, empty_spaces: 6, occupancy_rate: 0.7}
    };
    displayResults(demoData);
}

function testAPI() {
    fetch('/api/test')
    .then(response => response.json())
    .then(data => {
        document.getElementById('results').innerHTML = 
            '<div class="alert alert-success">✅ API Test Success!<br><pre>' + JSON.stringify(data, null, 2) + '</pre></div>';
    });
}
</script>
</body></html>"""
        
        self.wfile.write(html.encode())

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:8000')

def main():
    PORT = 8000
    print(f"🚀 Starting Quick Server on http://localhost:{PORT}")
    
    threading.Timer(2, open_browser).start()
    
    with socketserver.TCPServer(("", PORT), QuickHandler) as httpd:
        print("✅ Server running! Browser will open automatically.")
        httpd.serve_forever()

if __name__ == "__main__":
    main()
