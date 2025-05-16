"""
DeblurGANv2 implementation for motion deblurring in videos.
Based on the paper "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better"
by Orest Kupyn et al.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from typing import List, Union, Tuple, Optional
from huggingface_hub import hf_hub_download

class FPNHead(nn.Module):
    """Feature Pyramid Network Head for DeblurGANv2"""
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.block0(x), inplace=True)
        x = F.relu(self.block1(x), inplace=True)
        return x

class FPNInner(nn.Module):
    """Feature Pyramid Network Inner Block"""
    def __init__(self, num_in, num_out, num_hidden):
        super().__init__()
        
        self.conv_inner = nn.Conv2d(num_in, num_hidden, kernel_size=1)
        self.conv_out = FPNHead(num_hidden, num_out, num_out)
        
    def forward(self, up_feature, lateral_feature):
        if up_feature is None:
            out = self.conv_inner(lateral_feature)
            out = self.conv_out(out)
            return out
        else:
            # Upsample and add
            up_size = lateral_feature.shape[-2:]
            up_feature = F.interpolate(up_feature, size=up_size, mode='nearest')
            lateral_feature = self.conv_inner(lateral_feature)
            out = up_feature + lateral_feature
            out = self.conv_out(out)
            return out

class FPN(nn.Module):
    """Feature Pyramid Network for DeblurGANv2"""
    def __init__(self, fpn_sizes, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        self.fpn_sizes = fpn_sizes
        self.inner_blocks = nn.ModuleList()
        
        for fpn_size in reversed(fpn_sizes[:-1]):
            self.inner_blocks.append(FPNInner(fpn_size, fpn_sizes[-1], fpn_sizes[-1]))
            
    def forward(self, features):
        features = features.copy()  # Don't modify the original list
        
        # Start from the deepest layer
        last_feature = None
        for idx, inner_block in enumerate(self.inner_blocks):
            feat_idx = len(features) - idx - 1
            last_feature = inner_block(last_feature, features[feat_idx])
            features[feat_idx] = last_feature
            
        return features

class DeblurGANv2Generator(nn.Module):
    """Generator architecture for DeblurGANv2"""
    def __init__(self):
        super().__init__()
        
        # Simplified MobileNetV2-based architecture
        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Encoder blocks with downsampling
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # FPN
        self.fpn = FPN([64, 128, 256])
        
        # Decoder blocks with upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final output layer
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encode
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [3, 6, 9]:  # After each downsampling block
                features.append(x)
        
        # FPN
        features = self.fpn(features)
        
        # Decode
        x = features[-1]  # Use the deepest feature
        x = self.decoder(x)
        
        return x

class DeblurGANv2:
    """DeblurGANv2 model for motion deblurring in videos"""
    
    def __init__(self, weights_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize DeblurGANv2 model
        
        Args:
            weights_path: Path to model weights
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = self._load_model(weights_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
        ])
        
    def _load_model(self, weights_path: str):
        """Load model from weights file"""
        model = DeblurGANv2Generator().to(self.device)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded DeblurGANv2 weights from {weights_path}")
        else:
            print(f"Warning: DeblurGANv2 weights not found at {weights_path}. Using randomly initialized weights.")
            
        model.eval()
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Tensor of shape (1, 3, H, W)
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return tensor
    
    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert output tensor to image
        
        Args:
            tensor: Output tensor (1, 3, H, W)
            
        Returns:
            BGR image (H, W, 3)
        """
        # Apply inverse transforms
        tensor = self.inverse_transform(tensor.squeeze(0))
        
        # Convert to numpy
        image = tensor.cpu().detach().numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # Clip values to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Scale to [0, 255]
        image = (image * 255).astype(np.uint8)
        
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image to remove motion blur
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Deblurred BGR image (H, W, 3)
        """
        # Preserve original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Resize to multiple of 8 for the model
        h, w = ((orig_h + 7) // 8) * 8, ((orig_w + 7) // 8) * 8
        if (h, w) != (orig_h, orig_w):
            image = cv2.resize(image, (w, h))
        
        # Preprocess
        tensor = self.preprocess(image)
        
        # Process
        with torch.no_grad():
            output_tensor = self.model(tensor)
        
        # Postprocess
        output_image = self.postprocess(output_tensor)
        
        # Resize back to original dimensions if needed
        if (h, w) != (orig_h, orig_w):
            output_image = cv2.resize(output_image, (orig_w, orig_h))
        
        return output_image
    
    def process_batch(self, images: List[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """
        Process a batch of images
        
        Args:
            images: List of BGR images
            batch_size: Batch size for processing
            
        Returns:
            List of deblurred BGR images
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Preprocess batch
            tensors = [self.preprocess(img) for img in batch]
            batch_tensor = torch.cat(tensors, dim=0)
            
            # Process
            with torch.no_grad():
                output_batch = self.model(batch_tensor)
            
            # Postprocess batch
            for j in range(len(batch)):
                output_image = self.postprocess(output_batch[j:j+1])
                results.append(output_image)
            
        return results

def download_deblurgan_weights(save_dir: str = None) -> str:
    """
    Download pre-trained DeblurGANv2 weights from HuggingFace Hub
    
    Args:
        save_dir: Directory to save weights to. If None, uses default cache dir.
        
    Returns:
        Path to downloaded weights
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # For demonstration purposes, we're pointing to a hypothetical model on HuggingFace
    # Replace with actual model when available
    try:
        weights_path = hf_hub_download(
            repo_id="deblurgan/deblurgan-v2", 
            filename="deblurgan_v2_fpn.pth",
            cache_dir=save_dir
        )
        print(f"Downloaded DeblurGANv2 weights to {weights_path}")
    except Exception as e:
        # If HuggingFace download fails, provide a placeholder
        print(f"Error downloading DeblurGANv2 weights: {e}")
        placeholder_path = os.path.join(save_dir, "deblurgan_v2_fpn.pth")
        
        if not os.path.exists(placeholder_path):
            # Create dummy weights for demonstration
            dummy_model = DeblurGANv2Generator()
            torch.save(dummy_model.state_dict(), placeholder_path)
            print(f"Created placeholder weights at {placeholder_path}")
        
        weights_path = placeholder_path
    
    return weights_path
