"""
DeblurGAN module for motion deblurring in the Multimodal Expression Analysis Framework.
"""

# Import both implementations
from .deblur_model import DeblurGANv2 as PyTorchDeblurGANv2, download_deblurgan_weights
from .onnx_deblur import DeblurGANv2 as ONNXDeblurGANv2

# Use ONNX implementation as the default
DeblurGANv2 = ONNXDeblurGANv2

# Add alias for ONNXDeblurGAN (without v2) for backward compatibility
ONNXDeblurGAN = ONNXDeblurGANv2

__all__ = ['DeblurGANv2', 'ONNXDeblurGANv2', 'PyTorchDeblurGANv2', 'ONNXDeblurGAN', 'download_deblurgan_weights']