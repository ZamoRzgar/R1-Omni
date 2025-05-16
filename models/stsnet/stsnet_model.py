"""
STSTNet (Spatial-Temporal-Spectral Transformer Network) for micro-expression recognition.
This module provides a clean implementation of STSTNet that can be integrated
into the R1-Omni multimodal emotion recognition framework.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import logging
from typing import List, Tuple, Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STSTNet")

class STSTNet(nn.Module):
    """
    STSTNet model for micro-expression recognition.
    This implementation uses a convolutional architecture with multiple pathways
    to capture spatial, temporal, and spectral features from facial micro-expressions.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """
        Initialize the STSTNet model.
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            out_channels: Number of output classes (default: 3 for negative, positive, surprise)
        """
        super(STSTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(5)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=5*5*16, out_features=out_channels)
        
        # Initialize emotion labels
        self.emotion_labels = {
            0: "negative",
            1: "positive",
            2: "surprise"
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the STSTNet model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels]
        """
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.maxpool(x1)
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x2 = self.maxpool(x2)
        x2 = self.dropout(x2)
        
        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        x3 = self.maxpool(x3)
        x3 = self.dropout(x3)
        
        x = torch.cat((x1, x2, x3), 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class STSTNetPredictor:
    """
    Wrapper class for STSTNet model to handle loading weights, preprocessing,
    and prediction for micro-expression recognition.
    """
    def __init__(self, 
                 weights_dir: str = None, 
                 device: str = "auto",
                 input_size: Tuple[int, int] = (28, 28)):
        """
        Initialize the STSTNet predictor.
        
        Args:
            weights_dir: Directory containing the model weights
            device: Device to run inference on ('auto', 'cpu', or 'cuda')
            input_size: Input image size (height, width)
        """
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Set weights directory
        if weights_dir is None:
            # First try the STSTNet_Weights directory (from the original implementation)
            default_weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STSTNet_Weights")
            if os.path.exists(default_weights_dir):
                self.weights_dir = default_weights_dir
                logger.info(f"Using weights from STSTNet_Weights directory: {default_weights_dir}")
            else:
                # Fall back to weights directory
                self.weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        else:
            self.weights_dir = weights_dir
            
        # Input size for preprocessing
        self.input_size = input_size
        
        # Initialize model
        self.model = STSTNet().to(self.device)
        
        # Load available subject models
        self.subject_models = self._load_subject_models()
        
        # Current active subject model
        self.current_subject = None
        if len(self.subject_models) > 0:
            # Use the first subject as default
            self.load_subject_model(list(self.subject_models.keys())[0])
    
    def _load_subject_models(self) -> Dict[str, str]:
        """
        Load all available subject models from the weights directory.
        
        Returns:
            Dictionary mapping subject names to weight file paths
        """
        subject_models = {}
        
        if not os.path.exists(self.weights_dir):
            logger.warning(f"Weights directory {self.weights_dir} does not exist")
            return subject_models
        
        # Get all weight files
        weight_files = [f for f in os.listdir(self.weights_dir) if f.endswith('.pth')]
        
        for weight_file in weight_files:
            subject_name = weight_file.split('.')[0]
            subject_models[subject_name] = os.path.join(self.weights_dir, weight_file)
            
        logger.info(f"Found {len(subject_models)} subject models: {list(subject_models.keys())}")
        return subject_models
    
    def load_subject_model(self, subject_name: str) -> bool:
        """
        Load a specific subject model.
        
        Args:
            subject_name: Name of the subject model to load
            
        Returns:
            True if loading was successful, False otherwise
        """
        if subject_name not in self.subject_models:
            logger.error(f"Subject model {subject_name} not found")
            return False
        
        try:
            self.model.load_state_dict(torch.load(self.subject_models[subject_name], map_location=self.device))
            self.model.eval()
            self.current_subject = subject_name
            logger.info(f"Loaded subject model {subject_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load subject model {subject_name}: {str(e)}")
            return False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for input to the STSTNet model.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed tensor of shape [1, channels, height, width]
        """
        # Resize to input size
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert from BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb / 255.0
        
        # Convert to PyTorch tensor and add batch dimension
        tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict micro-expression from a face image.
        
        Args:
            image: Face image (BGR format from OpenCV)
            
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        if self.current_subject is None:
            logger.error("No subject model loaded")
            # Return default values when no model is loaded (equal probabilities)
            return {
                "negative": 0.33,
                "positive": 0.33,
                "surprise": 0.34
            }
        
        # Preprocess image
        tensor = self.preprocess(image).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Map probabilities to emotion labels
        result = {self.model.emotion_labels[i]: float(probabilities[i]) for i in range(len(probabilities))}
        
        return result
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Predict micro-expressions from a batch of face images.
        
        Args:
            images: List of face images (BGR format from OpenCV)
            
        Returns:
            List of dictionaries mapping emotion labels to probabilities
        """
        return [self.predict(image) for image in images]
    
    def get_available_subjects(self) -> List[str]:
        """
        Get list of available subject models.
        
        Returns:
            List of subject names
        """
        return list(self.subject_models.keys())
    
    def get_emotion_labels(self) -> Dict[int, str]:
        """
        Get emotion label mapping.
        
        Returns:
            Dictionary mapping indices to emotion labels
        """
        return self.model.emotion_labels
