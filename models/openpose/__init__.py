"""
OpenPose module for body language analysis in the R1-Omni Multimodal Expression Analysis Framework.
"""

from .openpose_model import OpenPoseAnalyzer, EmotionPosture, BodyPart

__all__ = ['OpenPoseAnalyzer', 'EmotionPosture', 'BodyPart']
