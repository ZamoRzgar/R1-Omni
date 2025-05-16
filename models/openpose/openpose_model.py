"""
OpenPose wrapper for body language analysis in the R1-Omni Multimodal Expression Analysis Framework.
This module provides a clean interface to the PyTorch OpenPose implementation for
detecting body posture and gestures relevant to emotion recognition.
"""

import os
import math
import numpy as np
import cv2
import torch
from enum import Enum
import logging
from typing import List, Tuple, Dict, Optional, Union

# Import OpenPose modules
from .src.body import Body
from .src.hand import Hand
from .src.model import bodypose_model, handpose_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OpenPose")

class EmotionPosture(Enum):
    """Emotional categories based on body posture"""
    NEUTRAL = 0
    CONFIDENT = 1
    DEFENSIVE = 2
    ANXIOUS = 3
    RELAXED = 4
    AGGRESSIVE = 5
    SUBMISSIVE = 6
    EXPRESSIVE = 7

class BodyPart(Enum):
    """Body part indices used in OpenPose"""
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_EYE = 14
    LEFT_EYE = 15
    RIGHT_EAR = 16
    LEFT_EAR = 17
    BACKGROUND = 18

class OpenPoseAnalyzer:
    """
    Wrapper for OpenPose to analyze body posture and gestures for emotion recognition.
    This class integrates the PyTorch OpenPose implementation to detect body 
    keypoints and analyze them for emotional indicators.
    """
    def __init__(self, 
                 model_dir: str = None, 
                 device: str = "auto",
                 detect_hands: bool = True):
        """
        Initialize the OpenPose analyzer.
        
        Args:
            model_dir: Directory containing model weights
            device: Device to use for inference ('auto', 'cpu', 'cuda')
            detect_hands: Whether to detect hand keypoints
        """
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set model directory
        if model_dir is None:
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        else:
            self.model_dir = model_dir
        
        # Set model paths
        self.body_model_path = os.path.join(self.model_dir, "body_pose_model.pth")
        self.hand_model_path = os.path.join(self.model_dir, "hand_pose_model.pth")
        
        # Check if models exist
        if not os.path.exists(self.body_model_path):
            logger.error(f"Body pose model not found at {self.body_model_path}")
            raise FileNotFoundError(f"Body pose model not found at {self.body_model_path}")
        
        if detect_hands and not os.path.exists(self.hand_model_path):
            logger.warning(f"Hand pose model not found at {self.hand_model_path}. Hand detection disabled.")
            detect_hands = False
        
        # Initialize models
        try:
            # Initialize body pose model
            self.body_estimation = Body(self.body_model_path)
            logger.info("Body pose model loaded successfully")
            
            # Initialize hand pose model if requested
            if detect_hands:
                self.hand_estimation = Hand(self.hand_model_path)
                logger.info("Hand pose model loaded successfully")
                self.detect_hands = True
            else:
                self.detect_hands = False
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenPose models: {str(e)}")
            raise RuntimeError(f"Failed to initialize OpenPose models: {str(e)}")
        
        # Set up parameter constants
        self.emotion_confidence_threshold = 0.5
        
    def process_image(self, image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Process an image to detect body keypoints and analyze emotional indicators.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (analysis result dictionary, visualization image)
        """
        # Create a copy for visualization
        canvas = image.copy()
        
        # Detect body keypoints
        candidate, subset = self.body_estimation(image)
        
        # Detect hand keypoints if enabled
        hands = []
        if self.detect_hands and len(subset) > 0:
            # For each person detected
            for person in range(len(subset)):
                # Get hand region from the body pose data
                # We'll use the wrist points to define hand regions
                # Access the enum values as integers for indexing
                right_wrist_idx = BodyPart.RIGHT_WRIST.value if hasattr(BodyPart.RIGHT_WRIST, 'value') else BodyPart.RIGHT_WRIST
                left_wrist_idx = BodyPart.LEFT_WRIST.value if hasattr(BodyPart.LEFT_WRIST, 'value') else BodyPart.LEFT_WRIST
                
                if int(subset[person][right_wrist_idx]) != -1 or int(subset[person][left_wrist_idx]) != -1:
                    # We'll just use the whole image for hand detection as a simplification
                    # In a real implementation, you'd crop the hand regions around the wrists
                    hand_points = self.hand_estimation(image)
                    if len(hand_points) > 0:
                        hands.append({"points": hand_points, "person_idx": person})
        
        # Import visualization utilities
        from .src import util
        
        # Visualize body keypoints
        if len(subset) > 0:
            canvas = util.draw_bodypose(canvas, candidate, subset)
            
            # Visualize hand keypoints if detected
            if self.detect_hands and hands:
                for hand_dict in hands:
                    # Format hand points for visualization
                    # The draw_handpose function expects a list of 21 keypoints, with each keypoint being [x, y]
                    try:
                        # If points is a 2D array with shape [21, 2], we need to convert it to a list of lists
                        if isinstance(hand_dict["points"], np.ndarray):
                            formatted_points = [hand_dict["points"][i].tolist() if i < len(hand_dict["points"]) else [0, 0] for i in range(21)]
                        else:
                            # If points is already a list of points, ensure it has the right structure
                            formatted_points = []
                            for i in range(21):  # Hand has 21 keypoints
                                if i < len(hand_dict["points"]):
                                    pt = hand_dict["points"][i]
                                    if isinstance(pt, list) and len(pt) >= 2:
                                        formatted_points.append(pt[:2])  # Take just the x, y coordinates
                                    else:
                                        formatted_points.append([0, 0])  # Use zeros for missing keypoints
                                else:
                                    formatted_points.append([0, 0])  # Use zeros for missing keypoints
                                    
                        # Call the visualization function with properly formatted points
                        canvas = util.draw_handpose_by_opencv(canvas, formatted_points)  # Use opencv version which is more robust
                    except Exception as e:
                        # If visualization fails, log the error and continue
                        logger.error(f"Hand visualization error: {str(e)}")
                        # Fall back to simple circle drawing
                        for pt in hand_dict["points"]:
                            if isinstance(pt, list) or isinstance(pt, np.ndarray):
                                if len(pt) >= 2:
                                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        
        # Analyze body posture for emotional indicators
        analysis_results = self.analyze_body_language(candidate, subset, hands)
        
        # Add text with the emotion analysis to the visualization
        if analysis_results and len(analysis_results["people"]) > 0:
            for i, person in enumerate(analysis_results["people"]):
                if "emotion" in person:
                    emotion_name = person["emotion"]["name"]
                    confidence = person["emotion"]["confidence"]
                    text = f"Person {i+1}: {emotion_name} ({confidence:.2f})"
                    cv2.putText(canvas, text, (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 255, 0), 2)
        
        return analysis_results, canvas

    def analyze_body_language(self, 
                             candidate: np.ndarray, 
                             subset: np.ndarray, 
                             hands: List[Dict] = None) -> Dict:
        """
        Analyze body keypoints to determine emotional indicators.
        
        Args:
            candidate: Keypoint candidates from OpenPose
            subset: Person subsets from OpenPose
            hands: Hand keypoints if available
            
        Returns:
            Dictionary with analysis results
        """
        if len(subset) == 0:
            return {"people": []}
        
        results = {"people": []}
        
        # Analyze each detected person
        for person_idx in range(len(subset)):
            person_result = {"keypoints": {}}
            person_subset = subset[person_idx]
            
            # Extract keypoints for this person
            for part_idx in range(18):  # 18 body parts
                if person_subset[part_idx] == -1:
                    continue
                
                keypoint_idx = int(person_subset[part_idx])
                if keypoint_idx >= len(candidate):
                    continue
                
                x, y, conf = candidate[keypoint_idx][0:3]
                if conf < 0.1:  # Low confidence keypoint
                    continue
                
                part_name = BodyPart(part_idx).name.lower()
                person_result["keypoints"][part_name] = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf)
                }
            
            # Skip if too few keypoints detected
            if len(person_result["keypoints"]) < 5:
                continue
            
            # Add hand analysis if available
            if hands:
                # Find hands that belong to this person
                person_hands = [h for h in hands if h.get("person_idx") == person_idx]
                if person_hands:
                    person_result["hands"] = {
                        "detected": True,
                        "points": [h["points"] for h in person_hands]
                    }
            
            # Analyze posture for emotional indicators
            emotion = self._analyze_emotion_from_posture(person_result["keypoints"])
            if emotion:
                person_result["emotion"] = emotion
            
            results["people"].append(person_result)
        
        return results
    
    def _analyze_emotion_from_posture(self, keypoints: Dict) -> Dict:
        """
        Analyze body posture to determine emotional state.
        
        Args:
            keypoints: Dictionary of body keypoints
            
        Returns:
            Dictionary with emotion name and confidence
        """
        # Default to neutral if not enough keypoints
        if len(keypoints) < 8:
            return {
                "name": EmotionPosture.NEUTRAL.name.lower(),
                "confidence": 0.3
            }
        
        # Calculate angles and positions for posture analysis
        posture_features = self._extract_posture_features(keypoints)
        
        # Initialize emotion scores
        emotion_scores = {
            EmotionPosture.NEUTRAL.name.lower(): 0.2,
            EmotionPosture.CONFIDENT.name.lower(): 0.0,
            EmotionPosture.DEFENSIVE.name.lower(): 0.0,
            EmotionPosture.ANXIOUS.name.lower(): 0.0,
            EmotionPosture.RELAXED.name.lower(): 0.0,
            EmotionPosture.AGGRESSIVE.name.lower(): 0.0,
            EmotionPosture.SUBMISSIVE.name.lower(): 0.0,
            EmotionPosture.EXPRESSIVE.name.lower(): 0.0
        }
        
        # Analyze shoulder angle and position (squared shoulders = confident)
        if "shoulder_angle" in posture_features:
            shoulder_angle = posture_features["shoulder_angle"]
            # Squared shoulders indicate confidence
            if abs(shoulder_angle) < 10:
                emotion_scores["confident"] += 0.3
            # Slumped shoulders indicate submission or anxiety
            elif shoulder_angle < -15:
                emotion_scores["submissive"] += 0.25
                emotion_scores["anxious"] += 0.2
                
        # Arms crossed or held close to body indicate defensive posture
        if "arms_distance" in posture_features:
            arms_distance = posture_features["arms_distance"]
            if arms_distance < 0.15:  # Arms close to body
                emotion_scores["defensive"] += 0.4
                emotion_scores["anxious"] += 0.2
            elif arms_distance > 0.4:  # Arms wide, open posture
                emotion_scores["confident"] += 0.2
                emotion_scores["expressive"] += 0.25
        
        # Analyze arm movement (high movement = expressive)
        if "arm_movement" in posture_features:
            arm_movement = posture_features["arm_movement"]
            if arm_movement > 0.5:
                emotion_scores["expressive"] += 0.4
                
        # Analyze upper/lower body alignment
        if "body_alignment" in posture_features:
            alignment = posture_features["body_alignment"]
            if alignment < 0.1:  # Well aligned
                emotion_scores["confident"] += 0.2
                emotion_scores["relaxed"] += 0.2
            else:  # Misaligned
                emotion_scores["anxious"] += 0.15
                
        # Analyze forward lean (aggressive) or backward lean (defensive)
        if "forward_lean" in posture_features:
            lean = posture_features["forward_lean"]
            if lean > 0.2:  # Forward lean
                emotion_scores["aggressive"] += 0.3
            elif lean < -0.2:  # Backward lean
                emotion_scores["defensive"] += 0.25
        
        # Find the highest scoring emotion
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        emotion_name = max_emotion[0]
        confidence = max_emotion[1]
        
        # Only return if confidence exceeds threshold
        if confidence >= self.emotion_confidence_threshold:
            return {
                "name": emotion_name,
                "confidence": confidence
            }
        else:
            return {
                "name": EmotionPosture.NEUTRAL.name.lower(),
                "confidence": max(0.3, confidence)  # Minimum confidence of 0.3
            }
    
    def _extract_posture_features(self, keypoints: Dict) -> Dict:
        """
        Extract relevant features from body keypoints for posture analysis.
        
        Args:
            keypoints: Dictionary of body keypoints
            
        Returns:
            Dictionary of posture features
        """
        features = {}
        
        # Calculate shoulder angle (horizontal alignment)
        if "right_shoulder" in keypoints and "left_shoulder" in keypoints:
            rs = keypoints["right_shoulder"]
            ls = keypoints["left_shoulder"]
            dx = rs["x"] - ls["x"]
            dy = rs["y"] - ls["y"]
            shoulder_angle = math.degrees(math.atan2(dy, dx))
            features["shoulder_angle"] = shoulder_angle
        
        # Calculate distance between arms and body (normalized by shoulder width)
        if ("right_shoulder" in keypoints and "right_elbow" in keypoints and
            "left_shoulder" in keypoints and "left_elbow" in keypoints):
            
            rs = keypoints["right_shoulder"]
            re = keypoints["right_elbow"]
            ls = keypoints["left_shoulder"]
            le = keypoints["left_elbow"]
            
            # Calculate shoulder width for normalization
            shoulder_width = math.sqrt((rs["x"] - ls["x"])**2 + (rs["y"] - ls["y"])**2)
            
            # Calculate distance from elbows to shoulder line
            # This helps identify crossed or close arms
            mid_shoulder_x = (rs["x"] + ls["x"]) / 2
            mid_shoulder_y = (rs["y"] + ls["y"]) / 2
            
            right_arm_dist = math.sqrt((re["x"] - mid_shoulder_x)**2 + (re["y"] - mid_shoulder_y)**2)
            left_arm_dist = math.sqrt((le["x"] - mid_shoulder_x)**2 + (le["y"] - mid_shoulder_y)**2)
            
            # Normalize by shoulder width
            normalized_arm_dist = (right_arm_dist + left_arm_dist) / (2 * shoulder_width)
            features["arms_distance"] = normalized_arm_dist
            
            # Calculate arm angles too
            right_arm_angle = math.degrees(math.atan2(re["y"] - rs["y"], re["x"] - rs["x"]))
            left_arm_angle = math.degrees(math.atan2(le["y"] - ls["y"], le["x"] - ls["x"]))
            features["right_arm_angle"] = right_arm_angle
            features["left_arm_angle"] = left_arm_angle
        
        # Calculate body alignment (verticality)
        if "neck" in keypoints and "nose" in keypoints:
            neck = keypoints["neck"]
            nose = keypoints["nose"]
            
            # Vertical alignment of head and neck
            dx = abs(nose["x"] - neck["x"])
            dy = abs(nose["y"] - neck["y"])
            if dy > 0:
                alignment = dx / dy  # Should be close to 0 for vertical alignment
                features["body_alignment"] = alignment
        
        # Estimate forward/backward lean
        if "neck" in keypoints and "nose" in keypoints and "right_hip" in keypoints and "left_hip" in keypoints:
            neck = keypoints["neck"]
            nose = keypoints["nose"]
            right_hip = keypoints["right_hip"]
            left_hip = keypoints["left_hip"]
            
            # Calculate hip center
            hip_center_x = (right_hip["x"] + left_hip["x"]) / 2
            hip_center_y = (right_hip["y"] + left_hip["y"]) / 2
            
            # Calculate spine vector (hip to neck)
            spine_dx = neck["x"] - hip_center_x
            spine_dy = neck["y"] - hip_center_y
            spine_length = math.sqrt(spine_dx**2 + spine_dy**2)
            
            # Calculate head vector (neck to nose)
            head_dx = nose["x"] - neck["x"]
            
            # Normalize to get forward/backward lean
            # Positive = forward lean, Negative = backward lean
            if spine_length > 0:
                forward_lean = head_dx / spine_length
                features["forward_lean"] = forward_lean
        
        return features
    
    def get_emotion_categories(self) -> List[str]:
        """
        Get list of possible emotion categories from body posture.
        
        Returns:
            List of emotion category names
        """
        return [e.name.lower() for e in EmotionPosture]
