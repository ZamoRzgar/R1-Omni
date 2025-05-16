"""
Face detection and alignment utilities for micro-expression recognition.
These components are essential preprocessing steps for STSTNet to ensure
that micro-expressions are properly captured from aligned face regions.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceUtils")

class FaceDetector:
    """
    Face detection and alignment for micro-expression analysis.
    This class provides utilities to detect, crop, and align faces for
    optimal micro-expression recognition with STSTNet.
    """
    def __init__(self, 
                 detection_method: str = 'haarcascade',
                 face_model_path: Optional[str] = None,
                 eyes_model_path: Optional[str] = None):
        """
        Initialize the face detector.
        
        Args:
            detection_method: Method to use for face detection ('haarcascade', 'dnn', or 'mediapipe')
            face_model_path: Path to face detection model
            eyes_model_path: Path to eyes detection model (for alignment)
        """
        self.detection_method = detection_method
        
        # Set default model paths if not provided
        if face_model_path is None:
            face_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if eyes_model_path is None:
            eyes_model_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        # Initialize face detector
        if detection_method == 'haarcascade':
            self.face_cascade = cv2.CascadeClassifier(face_model_path)
            self.eyes_cascade = cv2.CascadeClassifier(eyes_model_path)
            
            if self.face_cascade.empty():
                logger.error(f"Failed to load face cascade from {face_model_path}")
                raise ValueError(f"Failed to load face cascade from {face_model_path}")
            
            if self.eyes_cascade.empty():
                logger.error(f"Failed to load eyes cascade from {eyes_model_path}")
                raise ValueError(f"Failed to load eyes cascade from {eyes_model_path}")
                
            logger.info(f"Initialized Haar cascade face detector")
            
        elif detection_method == 'dnn':
            try:
                # Check if OpenCV DNN face detector is available
                self.face_net = cv2.dnn.readNetFromCaffe(
                    'models/face_detector/deploy.prototxt',
                    'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
                )
                logger.info(f"Initialized DNN face detector")
            except Exception as e:
                logger.error(f"Failed to load DNN face detector: {str(e)}")
                logger.info(f"Falling back to Haar cascade face detector")
                self.detection_method = 'haarcascade'
                self.face_cascade = cv2.CascadeClassifier(face_model_path)
                self.eyes_cascade = cv2.CascadeClassifier(eyes_model_path)
                
        elif detection_method == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
                logger.info(f"Initialized MediaPipe face detector")
            except ImportError:
                logger.error(f"MediaPipe not available, falling back to Haar cascade face detector")
                self.detection_method = 'haarcascade'
                self.face_cascade = cv2.CascadeClassifier(face_model_path)
                self.eyes_cascade = cv2.CascadeClassifier(eyes_model_path)
        else:
            logger.error(f"Unknown detection method: {detection_method}")
            raise ValueError(f"Unknown detection method: {detection_method}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.detection_method == 'haarcascade':
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [(x, y, w, h) for (x, y, w, h) in faces]
            
        elif self.detection_method == 'dnn':
            height, width = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (x1, y1, x2, y2) = box.astype("int")
                    # Convert to (x, y, w, h) format
                    faces.append((x1, y1, x2 - x1, y2 - y1))
                    
            return faces
            
        elif self.detection_method == 'mediapipe':
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.face_detector.process(rgb_image)
            
            faces = []
            if results.detections:
                height, width = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    faces.append((x, y, w, h))
                    
            return faces
        
        return []
    
    def detect_eyes(self, face_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes in a face image.
        
        Args:
            face_image: Face image (BGR format from OpenCV)
            
        Returns:
            List of eye bounding boxes (x, y, w, h)
        """
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eyes_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in eyes]
    
    def align_face(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and align a face from an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Aligned face image
        """
        x, y, w, h = face_bbox
        
        # Extract face region with margin
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        face_image = image[y1:y2, x1:x2]
        
        # Try to detect eyes for alignment
        eyes = self.detect_eyes(face_image)
        
        # If we found exactly two eyes, align the face
        if len(eyes) == 2:
            # Get eye centers
            eye1_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            eye2_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            
            # Sort eyes horizontally
            if eye1_center[0] > eye2_center[0]:
                eye1_center, eye2_center = eye2_center, eye1_center
            
            # Calculate angle between eyes
            dx = eye2_center[0] - eye1_center[0]
            dy = eye2_center[1] - eye1_center[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Get center of face image
            center = (face_image.shape[1] // 2, face_image.shape[0] // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            aligned_face = cv2.warpAffine(face_image, rotation_matrix, (face_image.shape[1], face_image.shape[0]))
            
            return aligned_face
        
        # If we couldn't align, return the original face image
        return face_image
    
    def extract_micro_expression_region(self, 
                                        face_image: np.ndarray, 
                                        region: str = 'full',
                                        target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Extract a specific region of the face for micro-expression analysis.
        
        Args:
            face_image: Aligned face image (BGR format from OpenCV)
            region: Region to extract ('full', 'eyes', 'mouth', or 'forehead')
            target_size: Target size for the extracted region
            
        Returns:
            Extracted region resized to target size
        """
        height, width = face_image.shape[:2]
        
        if region == 'full':
            # Use the full face
            roi = face_image
        elif region == 'eyes':
            # Extract eye region (upper half of face)
            roi = face_image[:height//2, :]
        elif region == 'mouth':
            # Extract mouth region (lower half of face)
            roi = face_image[height//2:, :]
        elif region == 'forehead':
            # Extract forehead region (top third of face)
            roi = face_image[:height//3, :]
        else:
            logger.warning(f"Unknown region: {region}, using full face")
            roi = face_image
        
        # Resize to target size
        roi_resized = cv2.resize(roi, target_size)
        
        return roi_resized
    
    def process_image(self, 
                      image: np.ndarray, 
                      region: str = 'full',
                      target_size: Tuple[int, int] = (28, 28)) -> List[np.ndarray]:
        """
        Process an image to extract face regions for micro-expression analysis.
        
        Args:
            image: Input image (BGR format from OpenCV)
            region: Region to extract ('full', 'eyes', 'mouth', or 'forehead')
            target_size: Target size for the extracted region
            
        Returns:
            List of processed face regions
        """
        # Detect faces
        face_bboxes = self.detect_faces(image)
        
        if len(face_bboxes) == 0:
            logger.warning("No faces detected")
            return []
        
        # Process each face
        processed_faces = []
        for face_bbox in face_bboxes:
            # Align face
            aligned_face = self.align_face(image, face_bbox)
            
            # Extract region
            roi = self.extract_micro_expression_region(aligned_face, region, target_size)
            
            processed_faces.append(roi)
        
        return processed_faces
