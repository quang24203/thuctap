"""
License Plate Detection and Recognition
"""

import cv2
import numpy as np
import torch
import re
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
import logging

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from ..utils.simple_logger import get_logger, performance_logger


class LicensePlateDetector:
    """License plate detection using YOLOv8"""
    
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.7,
        iou_threshold: float = 0.4,
        device: str = "auto"
    ):
        """
        Initialize license plate detector
        
        Args:
            model_path: Path to YOLOv8 license plate detection model
            confidence: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run inference
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 license plate detection model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info(f"Loaded license plate detection model from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load license plate model: {e}")
            raise
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of plate detections with bbox and confidence
        """
        import time
        start_time = time.time()
        
        try:
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False
            )
            
            plates = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        plate = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence)
                        }
                        plates.append(plate)
            
            # Log performance
            inference_time = (time.time() - start_time) * 1000
            performance_logger.log_detection_time("license_plate_detection", inference_time)
            
            return plates
            
        except Exception as e:
            self.logger.error(f"License plate detection error: {e}")
            return []


class LicensePlateRecognizer:
    """License plate text recognition using OCR"""
    
    def __init__(
        self,
        ocr_engine: str = "paddleocr",
        languages: List[str] = None,
        confidence: float = 0.7
    ):
        """
        Initialize license plate recognizer
        
        Args:
            ocr_engine: OCR engine to use ("paddleocr" or "easyocr")
            languages: List of languages for OCR
            confidence: Minimum confidence for OCR results
        """
        self.logger = get_logger(self.__class__.__name__)
        self.ocr_engine = ocr_engine
        self.languages = languages or ["vi", "en"]
        self.confidence = confidence
        
        # Initialize OCR engine
        self._init_ocr()
        
        # Vietnamese license plate patterns
        self.vn_patterns = [
            r"^[0-9]{2}[A-Z]-[0-9]{3}\.[0-9]{2}$",  # 51A-123.45
            r"^[0-9]{2}[A-Z]{2}-[0-9]{3}\.[0-9]{2}$", # 51AB-123.45
            r"^[0-9]{2}[A-Z]-[0-9]{4}$",            # 51A-1234
            r"^[0-9]{2}[A-Z][0-9]{3}\.[0-9]{2}$",   # 51A123.45
            r"^[0-9]{2}[A-Z][0-9]{4}$",             # 51A1234
            r"^[0-9]{2}[A-Z]{2}[0-9]{3}\.[0-9]{2}$" # 51AB123.45
        ]

        # Performance tracking
        self.total_recognitions = 0
        self.successful_recognitions = 0
    
    def _init_ocr(self):
        """Initialize OCR engine"""
        try:
            if self.ocr_engine == "paddleocr" and PADDLEOCR_AVAILABLE:
                self.ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True,
                    lang='vi',
                    use_gpu=torch.cuda.is_available()
                )
                self.logger.info("Initialized PaddleOCR")
                
            elif self.ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
                self.ocr = easyocr.Reader(
                    self.languages,
                    gpu=torch.cuda.is_available()
                )
                self.logger.info("Initialized EasyOCR")
                
            else:
                raise ImportError(f"OCR engine {self.ocr_engine} not available")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {e}")
            raise
    
    def recognize_plate(self, plate_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Recognize text from license plate image
        
        Args:
            plate_image: Cropped license plate image
            
        Returns:
            Dictionary with recognized text and confidence
        """
        import time
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_plate(plate_image)
            
            # OCR recognition
            if self.ocr_engine == "paddleocr":
                result = self.ocr.ocr(processed_image, cls=True)
                text, confidence = self._parse_paddleocr_result(result)
                
            elif self.ocr_engine == "easyocr":
                result = self.ocr.readtext(processed_image)
                text, confidence = self._parse_easyocr_result(result)
            
            else:
                return None
            
            # Post-process text
            self.total_recognitions += 1

            if text and confidence >= self.confidence:
                cleaned_text = self._clean_license_plate_text(text)

                if self._validate_vietnamese_plate(cleaned_text):
                    self.successful_recognitions += 1

                    # Log performance
                    inference_time = (time.time() - start_time) * 1000
                    performance_logger.log_detection_time("license_plate_ocr", inference_time)

                    return {
                        "text": cleaned_text,
                        "confidence": confidence,
                        "raw_text": text
                    }

            return None
            
        except Exception as e:
            self.logger.error(f"License plate recognition error: {e}")
            return None
    
    def _preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """Preprocess license plate image for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if too small
        h, w = gray.shape
        if h < 32 or w < 128:
            scale = max(32/h, 128/w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _parse_paddleocr_result(self, result: List) -> Tuple[str, float]:
        """Parse PaddleOCR result"""
        if not result or not result[0]:
            return "", 0.0
        
        texts = []
        confidences = []
        
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            texts.append(text)
            confidences.append(confidence)
        
        combined_text = "".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return combined_text, avg_confidence
    
    def _parse_easyocr_result(self, result: List) -> Tuple[str, float]:
        """Parse EasyOCR result"""
        if not result:
            return "", 0.0
        
        # Sort by x-coordinate to get correct reading order
        result.sort(key=lambda x: x[0][0][0])
        
        texts = []
        confidences = []
        
        for detection in result:
            text = detection[1]
            confidence = detection[2]
            texts.append(text)
            confidences.append(confidence)
        
        combined_text = "".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return combined_text, avg_confidence
    
    def _clean_license_plate_text(self, text: str) -> str:
        """Clean and format license plate text"""
        # Remove spaces and special characters except dash and dot
        text = re.sub(r'[^A-Z0-9.-]', '', text.upper())
        
        # Fix common OCR mistakes
        text = text.replace('O', '0')  # O -> 0
        text = text.replace('I', '1')  # I -> 1
        text = text.replace('S', '5')  # S -> 5
        text = text.replace('B', '8')  # B -> 8
        
        return text
    
    def _validate_vietnamese_plate(self, text: str) -> bool:
        """Validate Vietnamese license plate format"""
        for pattern in self.vn_patterns:
            if re.match(pattern, text):
                return True
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get OCR performance statistics"""
        success_rate = (
            self.successful_recognitions / max(1, self.total_recognitions) * 100
            if self.total_recognitions > 0 else 0
        )

        return {
            "total_recognitions": self.total_recognitions,
            "successful_recognitions": self.successful_recognitions,
            "success_rate": success_rate,
            "ocr_engine": self.ocr_engine
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.total_recognitions = 0
        self.successful_recognitions = 0


class LicensePlateProcessor:
    """Complete license plate detection and recognition pipeline"""
    
    def __init__(
        self,
        detection_model_path: str,
        ocr_engine: str = "paddleocr",
        detection_confidence: float = 0.7,
        ocr_confidence: float = 0.7
    ):
        """
        Initialize license plate processor
        
        Args:
            detection_model_path: Path to license plate detection model
            ocr_engine: OCR engine to use
            detection_confidence: Confidence threshold for detection
            ocr_confidence: Confidence threshold for OCR
        """
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize detector and recognizer
        self.detector = LicensePlateDetector(
            detection_model_path,
            confidence=detection_confidence
        )
        
        self.recognizer = LicensePlateRecognizer(
            ocr_engine=ocr_engine,
            confidence=ocr_confidence
        )
    
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process frame to detect and recognize license plates
        
        Args:
            frame: Input frame
            
        Returns:
            List of recognized license plates
        """
        results = []
        
        # Detect license plates
        plates = self.detector.detect_plates(frame)
        
        # Recognize text for each detected plate
        for plate in plates:
            bbox = plate["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Crop plate region
            plate_image = frame[y1:y2, x1:x2]
            
            if plate_image.size > 0:
                # Recognize text
                recognition = self.recognizer.recognize_plate(plate_image)
                
                if recognition:
                    result = {
                        "bbox": bbox,
                        "detection_confidence": plate["confidence"],
                        "text": recognition["text"],
                        "ocr_confidence": recognition["confidence"],
                        "raw_text": recognition["raw_text"]
                    }
                    results.append(result)
        
        return results
    
    def visualize_results(
        self, 
        frame: np.ndarray, 
        results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Visualize license plate detection and recognition results"""
        vis_frame = frame.copy()
        
        for result in results:
            bbox = result["bbox"]
            text = result["text"]
            det_conf = result["detection_confidence"]
            ocr_conf = result["ocr_confidence"]
            
            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 255),
                2
            )
            
            # Draw text
            label = f"{text} ({det_conf:.2f}, {ocr_conf:.2f})"
            cv2.putText(
                vis_frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        return vis_frame
