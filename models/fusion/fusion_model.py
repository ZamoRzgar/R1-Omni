"""
Multimodal Fusion module for the R1-Omni Expression Analysis Framework.

This module combines outputs from multiple emotion analysis components:
- STSTNet: Micro-expression analysis from facial features
- OpenPose: Body language and posture analysis
- DeblurGAN: Pre-processing for motion blur handling

The fusion model provides weighted combinations of emotional signals from
various modalities and supports different fusion strategies.
"""

import os
import numpy as np
import pandas as pd
import logging
import torch
import cv2
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Multimodal-Fusion')

class FusionMode(Enum):
    """Fusion modes for combining multiple emotional signals."""
    WEIGHTED_AVERAGE = 0  # Simple weighted average of all signals
    CONFIDENCE_BASED = 1  # Weights based on confidence scores
    TEMPORAL_WEIGHTED = 2  # Weights adjusted based on temporal consistency
    HIERARCHICAL = 3      # Hierarchical decision making (face priority, body secondary)
    ADAPTIVE = 4          # Adaptive weighting based on signal quality

class MultimodalFusion:
    """
    Multimodal fusion for combining emotional signals from multiple sources.
    
    This class provides methods to combine outputs from facial micro-expression
    analysis (STSTNet) and body language analysis (OpenPose) to generate a more
    robust emotional assessment.
    """
    
    def __init__(self, 
                fusion_mode: Union[FusionMode, str] = FusionMode.WEIGHTED_AVERAGE,
                face_weight: float = 0.6,
                body_weight: float = 0.4,
                temporal_window: int = 5,
                device: str = "auto"):
        """
        Initialize the multimodal fusion module.
        
        Args:
            fusion_mode: Strategy for combining emotional signals (FusionMode enum or string)
            face_weight: Weight assigned to facial expressions (0-1)
            body_weight: Weight assigned to body language (0-1)
            temporal_window: Number of frames to consider for temporal analysis
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Validate weights
        assert 0 <= face_weight <= 1, "Face weight must be between 0 and 1"
        assert 0 <= body_weight <= 1, "Body weight must be between 0 and 1"
        assert abs(face_weight + body_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        # Handle string input for fusion_mode
        if isinstance(fusion_mode, str):
            fusion_mode_str = fusion_mode.upper()
            # Convert string to enum
            if fusion_mode_str == 'WEIGHTED_AVERAGE':
                self.fusion_mode = FusionMode.WEIGHTED_AVERAGE
            elif fusion_mode_str == 'CONFIDENCE_BASED':
                self.fusion_mode = FusionMode.CONFIDENCE_BASED
            elif fusion_mode_str == 'TEMPORAL_WEIGHTED':
                self.fusion_mode = FusionMode.TEMPORAL_WEIGHTED
            elif fusion_mode_str == 'HIERARCHICAL':
                self.fusion_mode = FusionMode.HIERARCHICAL
            elif fusion_mode_str == 'ADAPTIVE':
                self.fusion_mode = FusionMode.ADAPTIVE
            else:
                logger.warning(f"Unknown fusion mode: {fusion_mode_str}, using WEIGHTED_AVERAGE")
                self.fusion_mode = FusionMode.WEIGHTED_AVERAGE
        else:
            self.fusion_mode = fusion_mode
            
        self.face_weight = face_weight
        self.body_weight = body_weight
        self.temporal_window = temporal_window
        
        # Initialize history buffers for temporal analysis
        self.face_history = []
        self.body_history = []
        self.fused_history = []
        
        # Emotional category mapping
        # Map from STSTNet categorical outputs to common emotion space
        self.micro_to_common = {
            'anger': 'anger',
            'contempt': 'disgust',
            'disgust': 'disgust',
            'fear': 'fear',
            'happiness': 'happiness',
            'neutral': 'neutral',
            'sadness': 'sadness',
            'surprise': 'surprise'
        }
        
        # Map from OpenPose body language to common emotion space
        self.body_to_common = {
            'neutral': 'neutral',
            'confident': 'confidence',
            'defensive': 'fear',
            'anxious': 'anxiety',
            'relaxed': 'relaxed',
            'aggressive': 'anger',
            'submissive': 'sadness',
            'expressive': 'excitement'
        }
        
        # Common emotion categories for combined output
        self.emotion_categories = [
            'neutral', 'happiness', 'sadness', 'surprise', 
            'fear', 'anger', 'disgust', 'anxiety',
            'confidence', 'relaxed', 'excitement'
        ]
        
        # Get mode name for logging
        mode_name = self.fusion_mode.name if hasattr(self.fusion_mode, 'name') else str(self.fusion_mode)
        logger.info(f"Multimodal fusion initialized with {mode_name} mode")
    
    def combine_emotional_signals(self, 
                                 face_emotions: Optional[Dict] = None, 
                                 body_emotions: Optional[Dict] = None,
                                 frame_idx: int = 0,
                                 timestamp: float = 0.0) -> Dict:
        """
        Combine emotional signals from facial and body analysis.
        
        Args:
            face_emotions: Emotion predictions from facial analysis (STSTNet)
            body_emotions: Emotion predictions from body language (OpenPose)
            frame_idx: Frame index for temporal analysis
            timestamp: Timestamp of the current frame
            
        Returns:
            Dictionary with combined emotional assessment
        """
        # Initialize empty result
        result = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'emotions': {},
            'primary_emotion': None,
            'confidence': 0.0,
            'sources_used': []
        }
        
        # Check if we have any valid inputs
        if face_emotions is None and body_emotions is None:
            logger.warning("No valid emotional signals provided for fusion")
            return result
        
        # Add to history for temporal analysis
        if face_emotions:
            self.face_history.append(face_emotions)
            self.face_history = self.face_history[-self.temporal_window:]
            result['sources_used'].append('face')
        
        if body_emotions:
            self.body_history.append(body_emotions)
            self.body_history = self.body_history[-self.temporal_window:]
            result['sources_used'].append('body')
        
        # Initialize emotion scores dictionary
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
        
        # Process based on selected fusion mode
        if self.fusion_mode == FusionMode.WEIGHTED_AVERAGE:
            result = self._weighted_average_fusion(face_emotions, body_emotions, result, emotion_scores)
        elif self.fusion_mode == FusionMode.CONFIDENCE_BASED:
            result = self._confidence_based_fusion(face_emotions, body_emotions, result, emotion_scores)
        elif self.fusion_mode == FusionMode.TEMPORAL_WEIGHTED:
            result = self._temporal_weighted_fusion(result, emotion_scores)
        elif self.fusion_mode == FusionMode.HIERARCHICAL:
            result = self._hierarchical_fusion(face_emotions, body_emotions, result, emotion_scores)
        elif self.fusion_mode == FusionMode.ADAPTIVE:
            result = self._adaptive_fusion(face_emotions, body_emotions, result, emotion_scores)
        
        # Add to fused history
        self.fused_history.append(result)
        self.fused_history = self.fused_history[-self.temporal_window:]
        
        return result
    
    def _weighted_average_fusion(self, 
                               face_emotions: Optional[Dict], 
                               body_emotions: Optional[Dict], 
                               result: Dict,
                               emotion_scores: Dict) -> Dict:
        """
        Perform weighted average fusion of emotional signals.
        
        Args:
            face_emotions: Facial emotion predictions
            body_emotions: Body emotion predictions
            result: Current result dictionary to update
            emotion_scores: Emotion scores dictionary
            
        Returns:
            Updated result dictionary
        """
        # Process facial emotions if available
        if face_emotions:
            face_emotion = face_emotions.get('emotion', {})
            if face_emotion:
                face_name = face_emotion.get('name', '').lower()
                face_confidence = face_emotion.get('confidence', 0.0)
                
                # Map to common emotion space
                common_emotion = self.micro_to_common.get(face_name, face_name)
                if common_emotion in emotion_scores:
                    emotion_scores[common_emotion] += face_confidence * self.face_weight
        
        # Process body emotions if available
        if body_emotions:
            body_emotion = body_emotions.get('emotion', {})
            if body_emotion:
                body_name = body_emotion.get('name', '').lower()
                body_confidence = body_emotion.get('confidence', 0.0)
                
                # Map to common emotion space
                common_emotion = self.body_to_common.get(body_name, body_name)
                if common_emotion in emotion_scores:
                    emotion_scores[common_emotion] += body_confidence * self.body_weight
        
        # Store all emotion scores in result
        result['emotions'] = {k: float(v) for k, v in emotion_scores.items() if v > 0}
        
        # Find primary emotion (highest score)
        if result['emotions']:
            primary_emotion = max(result['emotions'].items(), key=lambda x: x[1])
            result['primary_emotion'] = {
                'name': primary_emotion[0],
                'confidence': primary_emotion[1]
            }
            result['confidence'] = primary_emotion[1]
        
        return result
    
    def _confidence_based_fusion(self, 
                               face_emotions: Optional[Dict], 
                               body_emotions: Optional[Dict], 
                               result: Dict,
                               emotion_scores: Dict) -> Dict:
        """
        Perform confidence-based fusion where weights are adjusted by confidence.
        
        Args:
            face_emotions: Facial emotion predictions
            body_emotions: Body emotion predictions
            result: Current result dictionary to update
            emotion_scores: Emotion scores dictionary
            
        Returns:
            Updated result dictionary
        """
        face_confidence = 0.0
        body_confidence = 0.0
        
        # Get confidence values
        if face_emotions and 'emotion' in face_emotions:
            face_confidence = face_emotions['emotion'].get('confidence', 0.0)
        
        if body_emotions and 'emotion' in body_emotions:
            body_confidence = body_emotions['emotion'].get('confidence', 0.0)
        
        # Normalize confidence values
        total_confidence = face_confidence + body_confidence
        if total_confidence > 0:
            face_weight_adjusted = face_confidence / total_confidence
            body_weight_adjusted = body_confidence / total_confidence
        else:
            face_weight_adjusted = self.face_weight
            body_weight_adjusted = self.body_weight
        
        # Process facial emotions with adjusted weight
        if face_emotions and 'emotion' in face_emotions:
            face_name = face_emotions['emotion'].get('name', '').lower()
            common_emotion = self.micro_to_common.get(face_name, face_name)
            if common_emotion in emotion_scores:
                emotion_scores[common_emotion] += face_confidence * face_weight_adjusted
        
        # Process body emotions with adjusted weight
        if body_emotions and 'emotion' in body_emotions:
            body_name = body_emotions['emotion'].get('name', '').lower()
            common_emotion = self.body_to_common.get(body_name, body_name)
            if common_emotion in emotion_scores:
                emotion_scores[common_emotion] += body_confidence * body_weight_adjusted
        
        # Store all emotion scores in result
        result['emotions'] = {k: float(v) for k, v in emotion_scores.items() if v > 0}
        
        # Find primary emotion (highest score)
        if result['emotions']:
            primary_emotion = max(result['emotions'].items(), key=lambda x: x[1])
            result['primary_emotion'] = {
                'name': primary_emotion[0],
                'confidence': primary_emotion[1]
            }
            result['confidence'] = primary_emotion[1]
        
        return result
    
    def _temporal_weighted_fusion(self, result: Dict, emotion_scores: Dict) -> Dict:
        """
        Perform temporal weighted fusion using emotion history.
        
        Args:
            result: Current result dictionary to update
            emotion_scores: Emotion scores dictionary
            
        Returns:
            Updated result dictionary
        """
        # If we don't have enough history, fall back to weighted average
        if len(self.face_history) < 2 and len(self.body_history) < 2:
            # Use the most recent data available
            face_emotions = self.face_history[-1] if self.face_history else None
            body_emotions = self.body_history[-1] if self.body_history else None
            return self._weighted_average_fusion(face_emotions, body_emotions, result, emotion_scores)
        
        # Process facial emotion history
        face_emotion_counts = defaultdict(float)
        for face_data in self.face_history:
            if face_data and 'emotion' in face_data:
                emotion_name = face_data['emotion'].get('name', '').lower()
                confidence = face_data['emotion'].get('confidence', 0.0)
                common_emotion = self.micro_to_common.get(emotion_name, emotion_name)
                face_emotion_counts[common_emotion] += confidence
        
        # Normalize facial emotion counts by history length
        if self.face_history:
            for emotion, count in face_emotion_counts.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += (count / len(self.face_history)) * self.face_weight
        
        # Process body emotion history
        body_emotion_counts = defaultdict(float)
        for body_data in self.body_history:
            if body_data and 'emotion' in body_data:
                emotion_name = body_data['emotion'].get('name', '').lower()
                confidence = body_data['emotion'].get('confidence', 0.0)
                common_emotion = self.body_to_common.get(emotion_name, emotion_name)
                body_emotion_counts[common_emotion] += confidence
        
        # Normalize body emotion counts by history length
        if self.body_history:
            for emotion, count in body_emotion_counts.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += (count / len(self.body_history)) * self.body_weight
        
        # Store all emotion scores in result
        result['emotions'] = {k: float(v) for k, v in emotion_scores.items() if v > 0}
        
        # Find primary emotion (highest score)
        if result['emotions']:
            primary_emotion = max(result['emotions'].items(), key=lambda x: x[1])
            result['primary_emotion'] = {
                'name': primary_emotion[0],
                'confidence': primary_emotion[1]
            }
            result['confidence'] = primary_emotion[1]
            
        # Add temporal consistency score
        if self.fused_history:
            previous_primary = [h.get('primary_emotion', {}).get('name') for h in self.fused_history 
                              if h.get('primary_emotion')]
            
            if previous_primary and result['primary_emotion']:
                current = result['primary_emotion']['name']
                consistency = sum(1 for e in previous_primary if e == current) / len(previous_primary)
                result['temporal_consistency'] = float(consistency)
        
        return result
    
    def _hierarchical_fusion(self, 
                           face_emotions: Optional[Dict], 
                           body_emotions: Optional[Dict], 
                           result: Dict,
                           emotion_scores: Dict) -> Dict:
        """
        Perform hierarchical fusion where face is primary and body is secondary.
        
        Args:
            face_emotions: Facial emotion predictions
            body_emotions: Body emotion predictions
            result: Current result dictionary to update
            emotion_scores: Emotion scores dictionary
            
        Returns:
            Updated result dictionary
        """
        # First check if we have valid facial emotions with high confidence
        if face_emotions and 'emotion' in face_emotions:
            face_name = face_emotions['emotion'].get('name', '').lower()
            face_confidence = face_emotions['emotion'].get('confidence', 0.0)
            
            # If face confidence is high enough, use it directly
            if face_confidence > 0.7:
                common_emotion = self.micro_to_common.get(face_name, face_name)
                emotion_scores[common_emotion] = face_confidence
                
                result['emotions'] = {k: float(v) for k, v in emotion_scores.items() if v > 0}
                result['primary_emotion'] = {
                    'name': common_emotion,
                    'confidence': face_confidence
                }
                result['confidence'] = face_confidence
                result['primary_source'] = 'face'
                return result
        
        # If face not confident enough or not available, check body
        if body_emotions and 'emotion' in body_emotions:
            body_name = body_emotions['emotion'].get('name', '').lower()
            body_confidence = body_emotions['emotion'].get('confidence', 0.0)
            
            # If body confidence is high enough, use it
            if body_confidence > 0.7:
                common_emotion = self.body_to_common.get(body_name, body_name)
                emotion_scores[common_emotion] = body_confidence
                
                result['emotions'] = {k: float(v) for k, v in emotion_scores.items() if v > 0}
                result['primary_emotion'] = {
                    'name': common_emotion,
                    'confidence': body_confidence
                }
                result['confidence'] = body_confidence
                result['primary_source'] = 'body'
                return result
        
        # If neither source is confident enough, fall back to weighted average
        return self._weighted_average_fusion(face_emotions, body_emotions, result, emotion_scores)
    
    def _adaptive_fusion(self, 
                       face_emotions: Optional[Dict], 
                       body_emotions: Optional[Dict], 
                       result: Dict,
                       emotion_scores: Dict) -> Dict:
        """
        Perform adaptive fusion based on quality of signals.
        
        Args:
            face_emotions: Facial emotion predictions
            body_emotions: Body emotion predictions
            result: Current result dictionary to update
            emotion_scores: Emotion scores dictionary
            
        Returns:
            Updated result dictionary
        """
        # Calculate quality metrics
        face_quality = 0.0
        body_quality = 0.0
        
        # Determine face quality
        if face_emotions:
            # Quality can be determined by confidence, number of detected keypoints, etc.
            if 'emotion' in face_emotions:
                face_quality = face_emotions['emotion'].get('confidence', 0.0)
            
            # If we have additional face keypoints quality info
            if 'quality' in face_emotions:
                face_quality *= face_emotions['quality']
        
        # Determine body quality
        if body_emotions:
            # Quality can be determined by confidence, number of detected keypoints, etc.
            if 'emotion' in body_emotions:
                body_quality = body_emotions['emotion'].get('confidence', 0.0)
            
            # If we have keypoints detected count
            if 'people' in body_emotions and body_emotions['people']:
                first_person = body_emotions['people'][0]
                keypoints_count = len(first_person.get('keypoints', {}))
                # Normalize by total possible keypoints (18 for body)
                keypoints_quality = min(1.0, keypoints_count / 18)
                body_quality *= keypoints_quality
        
        # Adjust weights based on quality
        total_quality = face_quality + body_quality
        if total_quality > 0:
            face_weight_adjusted = (face_quality / total_quality) * self.face_weight * 2
            body_weight_adjusted = (body_quality / total_quality) * self.body_weight * 2
            
            # Ensure weights are properly normalized
            total_weight = face_weight_adjusted + body_weight_adjusted
            if total_weight > 0:
                face_weight_adjusted /= total_weight
                body_weight_adjusted /= total_weight
            else:
                face_weight_adjusted = self.face_weight
                body_weight_adjusted = self.body_weight
        else:
            face_weight_adjusted = self.face_weight
            body_weight_adjusted = self.body_weight
        
        # Store adaptive weights in result
        result['adaptive_weights'] = {
            'face': float(face_weight_adjusted),
            'body': float(body_weight_adjusted)
        }
        
        # Process with adjusted weights (similar to weighted average)
        if face_emotions and 'emotion' in face_emotions:
            face_name = face_emotions['emotion'].get('name', '').lower()
            face_confidence = face_emotions['emotion'].get('confidence', 0.0)
            common_emotion = self.micro_to_common.get(face_name, face_name)
            if common_emotion in emotion_scores:
                emotion_scores[common_emotion] += face_confidence * face_weight_adjusted
        
        if body_emotions and 'emotion' in body_emotions:
            body_name = body_emotions['emotion'].get('name', '').lower()
            body_confidence = body_emotions['emotion'].get('confidence', 0.0)
            common_emotion = self.body_to_common.get(body_name, body_name)
            if common_emotion in emotion_scores:
                emotion_scores[common_emotion] += body_confidence * body_weight_adjusted
        
        # Store all emotion scores in result
        result['emotions'] = {k: float(v) for k, v in emotion_scores.items() if v > 0}
        
        # Find primary emotion (highest score)
        if result['emotions']:
            primary_emotion = max(result['emotions'].items(), key=lambda x: x[1])
            result['primary_emotion'] = {
                'name': primary_emotion[0],
                'confidence': primary_emotion[1]
            }
            result['confidence'] = primary_emotion[1]
        
        return result
    
    def analyze_temporal_patterns(self, window_size: int = None) -> Dict:
        """
        Analyze temporal patterns in emotion history.
        
        Args:
            window_size: Number of frames to analyze (defaults to self.temporal_window)
            
        Returns:
            Dictionary with temporal analysis results
        """
        if window_size is None:
            window_size = self.temporal_window
        
        if not self.fused_history:
            return {'error': 'No history available for temporal analysis'}
        
        # Take the most recent frames up to window_size
        history = self.fused_history[-window_size:]
        
        # Extract emotion sequences
        emotion_sequence = [h.get('primary_emotion', {}).get('name') for h in history 
                           if h.get('primary_emotion')]
        
        # Count transitions between emotions
        transitions = defaultdict(int)
        for i in range(len(emotion_sequence) - 1):
            if emotion_sequence[i] and emotion_sequence[i+1]:
                transition_key = f"{emotion_sequence[i]}_to_{emotion_sequence[i+1]}"
                transitions[transition_key] += 1
        
        # Count emotion frequencies
        emotion_counts = defaultdict(int)
        for emotion in emotion_sequence:
            if emotion:
                emotion_counts[emotion] += 1
        
        # Calculate stability (how often the emotion remains the same)
        same_count = sum(1 for i in range(len(emotion_sequence) - 1) 
                       if emotion_sequence[i] == emotion_sequence[i+1])
        
        stability = 0.0
        if len(emotion_sequence) > 1:
            stability = same_count / (len(emotion_sequence) - 1)
        
        # Determine dominant emotion
        dominant_emotion = None
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        # Calculate emotional variety
        variety = len(emotion_counts) / len(self.emotion_categories)
        
        return {
            'window_size': len(history),
            'stability': float(stability),
            'variety': float(variety),
            'dominant_emotion': dominant_emotion[0] if dominant_emotion else None,
            'dominant_confidence': dominant_emotion[1] / len(history) if dominant_emotion else 0.0,
            'emotion_frequencies': dict(emotion_counts),
            'transitions': dict(transitions)
        }
    
    def visualize_emotions(self, frame: np.ndarray, 
                         result: Dict, 
                         draw_history: bool = True,
                         history_length: int = 5) -> np.ndarray:
        """
        Visualize emotional analysis results on a frame.
        
        Args:
            frame: Input frame to draw on
            result: Emotion analysis result from fusion
            draw_history: Whether to draw emotion history
            history_length: Number of historical emotions to display
            
        Returns:
            Frame with visualization overlay
        """
        canvas = frame.copy()
        
        # Draw current primary emotion
        if result.get('primary_emotion'):
            emotion_name = result['primary_emotion']['name']
            confidence = result['primary_emotion']['confidence']
            
            # Draw text with current emotion
            text = f"{emotion_name.capitalize()}: {confidence:.2f}"
            cv2.putText(canvas, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8, (0, 255, 0), 2)
            
            # Draw sources used
            sources = ", ".join(result.get('sources_used', []))
            cv2.putText(canvas, f"Sources: {sources}", (20, 80), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        # Draw historical emotions if requested
        if draw_history and self.fused_history:
            history = self.fused_history[-history_length:]
            
            # Draw a small emotion timeline at the bottom
            height, width = canvas.shape[:2]
            bar_height = 30
            bar_width = width // (len(history) + 1)
            
            for i, hist_item in enumerate(history):
                if hist_item.get('primary_emotion'):
                    emotion = hist_item['primary_emotion']['name']
                    confidence = hist_item['primary_emotion']['confidence']
                    
                    # Get color based on emotion
                    color = self._get_emotion_color(emotion)
                    
                    # Draw rectangle with intensity based on confidence
                    x1 = i * bar_width
                    y1 = height - bar_height
                    x2 = (i + 1) * bar_width
                    y2 = height
                    
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                    cv2.putText(canvas, emotion[:3], (x1 + 5, y1 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp if available
        if 'timestamp' in result:
            timestamp = result['timestamp']
            cv2.putText(canvas, f"Time: {timestamp:.2f}s", (width - 150, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        return canvas
    
    def _get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get BGR color based on emotion name."""
        emotion_colors = {
            'neutral': (128, 128, 128),    # Gray
            'happiness': (0, 255, 255),    # Yellow
            'sadness': (139, 0, 0),        # Dark blue
            'surprise': (0, 140, 255),     # Orange
            'fear': (0, 0, 139),          # Dark red
            'anger': (0, 0, 255),         # Red
            'disgust': (0, 69, 0),        # Dark green
            'anxiety': (127, 0, 255),     # Purple
            'confidence': (255, 0, 0),    # Blue
            'relaxed': (0, 255, 0),       # Green
            'excitement': (255, 0, 255)   # Magenta
        }
        
        return emotion_colors.get(emotion.lower(), (200, 200, 200))
    
    def reset(self):
        """Reset history buffers."""
        self.face_history = []
        self.body_history = []
        self.fused_history = []
        logger.info("Multimodal fusion history reset")
