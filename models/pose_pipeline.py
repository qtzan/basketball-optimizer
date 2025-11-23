from ultralytics import YOLO
import os

class PosePipeline:
    def __init__(self, path='yolov8n-pose.pt'):
        # model path, model set to None intially
        self.path = path
        self.model = None
        
        # Load the model during initialization
        self._load_model()
        
    # Loads YOLOv8 with error handling
    def _load_model(self):
        try:
            self.model = YOLO(self.path) 
            print("Model loaded successfully.") 
        except FileNotFoundError: 
            raise RuntimeError(f"Model file not found: {self.path}") 
        except Exception as e: 
            raise RuntimeError(f"Failed to load model from {self.path}: {e}")