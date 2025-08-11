# 🚀 Deployment Guide - Smart Parking System

## 📋 Tổng quan

Hướng dẫn deploy hệ thống Smart Parking lên production environment với các tùy chọn khác nhau.

---

## 🖥️ 1. Local Development

### 1.1 Quick Start

```bash
# Clone repository
git clone https://github.com/your-repo/smart-parking-system.git
cd smart-parking-system

# Install dependencies
pip install -r requirements.txt

# Run application
python smart_parking_app.py
```

### 1.2 Development Server

```bash
# Run with auto-reload
uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload

# Access dashboard
open http://localhost:8000
```

---

## 🐳 2. Docker Deployment

### 2.1 Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "smart_parking_app.py"]
```

### 2.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  smart-parking:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - smart-parking
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### 2.3 Run with Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f smart-parking

# Scale service
docker-compose up -d --scale smart-parking=3
```

---

## ☁️ 3. Cloud Deployment

### 3.1 AWS EC2

```bash
# Launch EC2 instance (Ubuntu 20.04)
# Instance type: t3.medium or larger
# Security group: Allow ports 22, 80, 443, 8000

# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Deploy application
git clone https://github.com/your-repo/smart-parking-system.git
cd smart-parking-system
docker-compose up -d
```

### 3.2 Google Cloud Platform

```bash
# Create GCP project
gcloud projects create smart-parking-system

# Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com

# Create Compute Engine instance
gcloud compute instances create smart-parking-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-2 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB

# Deploy with Cloud Run
gcloud run deploy smart-parking \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### 3.3 Azure Container Instances

```bash
# Create resource group
az group create --name smart-parking-rg --location eastus

# Create container instance
az container create \
    --resource-group smart-parking-rg \
    --name smart-parking-container \
    --image your-registry/smart-parking:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --environment-variables ENVIRONMENT=production
```

---

## 🔧 4. Production Configuration

### 4.1 Environment Variables

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Database
DATABASE_URL=sqlite:///data/parking_system.db

# Security
SECRET_KEY=your-super-secret-key
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# AI Models
MODEL_PATH=/app/models
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Performance
MAX_WORKERS=4
BATCH_SIZE=8
CACHE_TTL=300

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090
```

### 4.2 Nginx Configuration

```nginx
# nginx.conf
upstream smart_parking {
    server smart-parking:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    client_max_body_size 100M;

    location / {
        proxy_pass http://smart_parking;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://smart_parking;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /static {
        alias /app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### 4.3 SSL Certificate

```bash
# Using Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 📊 5. Monitoring & Logging

### 5.1 Health Checks

```python
# health_check.py
import requests
import time
import logging

def check_health():
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            logging.info("Health check passed")
            return True
        else:
            logging.error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    while True:
        check_health()
        time.sleep(60)  # Check every minute
```

### 5.2 Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
VEHICLE_DETECTIONS = Counter('vehicle_detections_total', 'Total vehicle detections')
LICENSE_PLATE_RECOGNITIONS = Counter('license_plate_recognitions_total', 'Total license plate recognitions')

# Start metrics server
start_http_server(9090)
```

### 5.3 Log Configuration

```yaml
# logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /app/logs/smart_parking.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: json
    filename: /app/logs/error.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  smart_parking:
    level: INFO
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

---

## 🔒 6. Security

### 6.1 Security Headers

```python
# security.py
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

# HTTPS redirect
app.add_middleware(HTTPSRedirectMiddleware)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### 6.2 Rate Limiting

```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/analyze")
@limiter.limit("10/minute")
async def analyze_file(request: Request, file: UploadFile = File(...)):
    # Analysis logic
    pass
```

---

## 🚀 7. Performance Optimization

### 7.1 Caching

```python
# caching.py
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            
            return result
        return wrapper
    return decorator
```

### 7.2 Load Balancing

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  smart-parking-1:
    build: .
    environment:
      - WORKER_ID=1

  smart-parking-2:
    build: .
    environment:
      - WORKER_ID=2

  smart-parking-3:
    build: .
    environment:
      - WORKER_ID=3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - smart-parking-1
      - smart-parking-2
      - smart-parking-3
```

---

## 📈 8. Scaling

### 8.1 Horizontal Scaling

```bash
# Scale with Docker Swarm
docker swarm init
docker stack deploy -c docker-compose.yml smart-parking

# Scale service
docker service scale smart-parking_smart-parking=5
```

### 8.2 Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smart-parking
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smart-parking
  template:
    metadata:
      labels:
        app: smart-parking
    spec:
      containers:
      - name: smart-parking
        image: your-registry/smart-parking:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: smart-parking-service
spec:
  selector:
    app: smart-parking
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 🔧 9. Troubleshooting

### 9.1 Common Issues

**Issue: High memory usage**
```bash
# Check memory usage
docker stats

# Solution: Optimize batch size
export BATCH_SIZE=4
```

**Issue: Slow processing**
```bash
# Check GPU availability
nvidia-smi

# Solution: Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0
```

**Issue: Database locks**
```bash
# Check database connections
sqlite3 data/parking_system.db ".timeout 30000"

# Solution: Use connection pooling
export DB_POOL_SIZE=10
```

### 9.2 Monitoring Commands

```bash
# Check application logs
docker-compose logs -f smart-parking

# Monitor system resources
htop

# Check network connections
netstat -tulpn | grep :8000

# Database status
sqlite3 data/parking_system.db "SELECT COUNT(*) FROM vehicles;"
```

---

**📅 Last Updated:** 2024-12-20  
**🔄 Version:** 1.0.0  
**📧 Support:** support@smartparking.com
