"""
STSTNet module for micro-expression recognition in the R1-Omni Multimodal Expression Analysis Framework.
"""

from .stsnet_model import STSTNet, STSTNetPredictor
from .face_utils import FaceDetector

__all__ = ['STSTNet', 'STSTNetPredictor', 'FaceDetector']
