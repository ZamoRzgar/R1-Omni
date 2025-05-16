"""
DeblurGAN-v2 implementation using ONNX Runtime for efficient inference.
This module provides a simple interface to use a pre-trained DeblurGAN-v2 
model for motion deblurring in images and videos.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
import gdown
from typing import List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeblurGAN-v2")

class DeblurGANv2:
    """DeblurGAN-v2 implementation with ONNX Runtime for efficient inference"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize DeblurGAN-v2 with pre-trained weights
        
        Args:
            model_path: Path to the ONNX model, if None will download automatically
            device: Device to run inference on, 'auto', 'cpu', or 'cuda'
        """
        # Download or use provided model
        self.model_path = model_path if model_path is not None else self.download_model()
        
        # Select the appropriate execution provider
        providers = []
        if device == "auto":
            # Automatically select CUDA if available, else CPU
            if ort.get_device() == "GPU":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("Using CUDA for DeblurGAN-v2 inference")
            else:
                providers = ['CPUExecutionProvider']
                logger.info("Using CPU for DeblurGAN-v2 inference")
        elif device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("Using CUDA for DeblurGAN-v2 inference")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("Using CPU for DeblurGAN-v2 inference")
            
        # Check if model exists and is valid
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
            try:
                # Create ONNX runtime session
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"Successfully loaded DeblurGAN-v2 model from {self.model_path}")
                
                # Get model metadata
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                self.use_onnx = True
            except Exception as e:
                logger.error(f"Failed to load DeblurGAN-v2 model: {str(e)}")
                self.use_onnx = False
        else:
            # No valid model file - use fallback approach
            logger.warning("No valid ONNX model found. Using fallback deblurring algorithm.")
            self.use_onnx = False
            # Set default input shape for preprocessing
            self.input_shape = [1, 3, 256, 256]
        
        logger.info(f"DeblurGAN-v2 initialized with input shape: {self.input_shape}")
    
    @staticmethod
    def download_model(save_dir: Optional[str] = None) -> str:
        """
        Get path to DeblurGAN-v2 model or provide instructions for manual download
        
        Args:
            save_dir: Directory to save the model, if None uses default location
            
        Returns:
            Path to the model file (even if it doesn't exist yet)
        """
        if save_dir is None:
            # Place in the weights subdirectory of this module
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Model file path
        model_path = os.path.join(save_dir, "deblurgan_v2.onnx")
        
        # Check if model exists
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:  # Check if file is too small (placeholder)
            # Provide instructions for manual download
            logger.warning("DeblurGAN-v2 ONNX model not found or incomplete.")
            logger.warning("Please download the model manually:")
            logger.warning("1. Visit: https://github.com/VITA-Group/DeblurGANv2")
            logger.warning("2. Download the pre-trained FPNInception model")
            logger.warning(f"3. Convert it to ONNX format using torch.onnx.export")
            logger.warning(f"4. Save the ONNX model to: {model_path}")
            logger.warning("\nAlternatively, use the PyTorch implementation which can download weights directly.")
            
            # For testing purposes, we'll use a simplified deblurring approach instead
            logger.info("Using a simplified deblurring approach for testing...")
            
        return model_path
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Preprocessed image ready for model input (1, 3, H, W)
        """
        # Get target size from model input shape
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        
        # Resize to model input size
        if image.shape[0] != target_h or image.shape[1] != target_w:
            resized = cv2.resize(image, (target_w, target_h))
        else:
            resized = image
        
        # Convert BGR to RGB and normalize to [0,1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        
        # Transpose from HWC to CHW (height, width, channels) to (channels, height, width)
        chw = np.transpose(rgb, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(chw, axis=0).astype(np.float32)
        
        return batch
    
    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert model output to BGR image
        
        Args:
            output: Model output tensor
            original_shape: Original image shape (H, W, C)
            
        Returns:
            Deblurred BGR image
        """
        # Remove batch dimension and transpose back to HWC
        output = np.squeeze(output, axis=0)
        output = np.transpose(output, (1, 2, 0))
        
        # Scale to [0, 255] and clip
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Resize to original shape
        h, w = original_shape[:2]
        if output.shape[0] != h or output.shape[1] != w:
            output = cv2.resize(output, (w, h))
        
        return output
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur an image using DeblurGAN-v2 or fallback algorithm
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Deblurred BGR image
        """
        # Validate input
        if image is None or image.size == 0:
            logger.error("Empty or invalid image provided to DeblurGAN-v2")
            return image
        
        # Save original shape
        original_shape = image.shape
        
        if self.use_onnx:
            # Use ONNX model for deblurring
            try:
                # Preprocess
                input_data = self.preprocess(image)
                
                # Run inference
                outputs = self.session.run([self.output_name], {self.input_name: input_data})
                
                # Postprocess
                deblurred = self.postprocess(outputs[0], original_shape)
                return deblurred
                
            except Exception as e:
                logger.error(f"Error during ONNX inference: {str(e)}")
                logger.warning("Falling back to basic deblurring algorithm")
                # Fall back to basic deblurring
                return self._basic_deblur(image)
        else:
            # Use basic deblurring algorithm as fallback
            return self._basic_deblur(image)
    
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of images
        
        Args:
            images: List of BGR images
            
        Returns:
            List of deblurred BGR images
        """
        # Process each image individually
        # Note: We could optimize this for true batch processing in the future
        return [self.process(img) for img in images]
    
    def _basic_deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Basic deblurring algorithm as fallback when ONNX model is not available
        Uses a combination of bilateral filtering and unsharp masking
        
        Args:
            image: BGR image
            
        Returns:
            Deblurred BGR image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        # Parameters: image, diameter, sigmaColor, sigmaSpace
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        # Convert to LAB color space to enhance details
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge((cl, a, b))
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Callable interface for deblurring an image
        
        Args:
            image: BGR image
            
        Returns:
            Deblurred BGR image
        """
        return self.process(image)
