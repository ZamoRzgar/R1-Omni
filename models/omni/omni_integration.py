"""
Omni-R1 Integration Module.

This module integrates multimodal fusion outputs with the HumanOmni backbone model
to generate comprehensive emotional and behavioral analysis.
"""

import os
import sys
import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image

# Add project root to path to ensure module imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logger = logging.getLogger("Omni-Integration")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Import the core HumanOmni modules directly from the local project codebase
try:
    # Direct import from the local implementation
    from humanomni import model_init, mm_infer
    from humanomni.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_AUDIO_TOKEN
    
    # Flag to indicate successful import
    HumanOmni_available = True
    logger.info("Successfully imported local HumanOmni core modules")
    
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Local HumanOmni core modules not available: {str(e)}")
    logger.warning("Using placeholder implementation for demonstration purposes")
    
    # Set flag to indicate fallback mode
    HumanOmni_available = False
    
    # Define placeholder constants
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_VIDEO_TOKEN = "<video>"
    DEFAULT_AUDIO_TOKEN = "<audio>"

class OmniIntegration:
    """
    Integrates multimodal fusion outputs with the HumanOmni backbone model.
    This class serves as the interface between the multimodal fusion components
    and the core HumanOmni model for comprehensive emotional analysis.
    """
    
    def __init__(self, 
                model_path: Optional[str] = None,
                device: Optional[str] = None,
                fallback_to_demo: bool = True):
        """
        Initialize the OmniIntegration module.
        
        Args:
            model_path: Path or name of the HumanOmni model. If None, will attempt to use a default model.
            device: Device to run the model on ('cuda' or 'cpu')
            fallback_to_demo: Whether to use demonstration mode if model initialization fails
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Flag to track model initialization status
        self.initialized = False
        
        # Initialize HumanOmni model if available
        if HumanOmni_available:
            try:
                # If model_path is None, use the default from your local implementation
                if model_path is None:
                    logger.info("No model path specified, using default HumanOmni_7B model")
                    model_name = "HumanOmni_7B" # This matches your local implementation's default
                else:
                    logger.info(f"Initializing HumanOmni model: {model_path}")
                    model_name = model_path
                    
                # Initialize with proper error handling
                try:
                    # Try to initialize from the provided path or name
                    # Try different approaches for model initialization
                    try:
                        # First try with the specified model_name
                        logger.info(f"Attempting to initialize with model_name={model_name}")
                        self.model, self.processor, self.tokenizer = model_init(
                            model_path=model_name,
                            device=self.device
                        )
                    except Exception as model_error:
                        logger.warning(f"Failed first initialization attempt: {str(model_error)}")
                        # If that fails, try letting model_init use its own default
                        logger.info("Attempting initialization with default parameters")
                        try:
                            self.model, self.processor, self.tokenizer = model_init(
                                device=self.device
                            )
                        except Exception as default_error:
                            # Re-raise the original error if both attempts fail
                            logger.error(f"Default initialization also failed: {str(default_error)}")
                            raise model_error
                    self.initialized = True
                    logger.info("HumanOmni model initialized successfully")
                except FileNotFoundError as e:
                    logger.error(f"Model not found at {model_name}: {str(e)}")
                    if fallback_to_demo:
                        logger.warning("Falling back to demonstration mode")
                        self.model = self.processor = self.tokenizer = None
                    else:
                        raise e
                except Exception as e:
                    logger.error(f"Failed to initialize HumanOmni model: {str(e)}")
                    if fallback_to_demo:
                        logger.warning("Falling back to demonstration mode")
                        self.model = self.processor = self.tokenizer = None
                    else:
                        raise e
            except Exception as e:
                logger.error(f"Error during model initialization: {str(e)}")
                if fallback_to_demo:
                    logger.warning("Falling back to demonstration mode")
                    self.model = self.processor = self.tokenizer = None
                else:
                    raise e
        else:
            logger.warning("HumanOmni modules not available, using demonstration mode")
            self.model = self.processor = self.tokenizer = None
            
        # Define default prompt templates
        self.prompt_templates = {
            "emotion_analysis": (
                "Analyze the emotional state based on the facial expressions "
                "and body language shown in the image. {additional_context}"
            ),
            "behavior_analysis": (
                "Analyze the behavior and intentions based on the body language "
                "and facial expressions shown in the image. {additional_context}"
            ),
            "custom": "{custom_prompt} {additional_context}"
        }
    
    def _prepare_images(self, images: List[np.ndarray]) -> List[Image.Image]:
        """
        Prepare images for the HumanOmni model by converting from OpenCV format to PIL.
        
        Args:
            images: List of images as numpy arrays (BGR format from OpenCV)
            
        Returns:
            List of processed images as PIL Image objects
        """
        processed_images = []
        for img in images:
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            # Append to list
            processed_images.append(pil_img)
            
        return processed_images
        
    def _prepare_video_from_frames(self, frames: List[np.ndarray]) -> List[Image.Image]:
        """
        Prepare a video representation from a sequence of frames.
        
        Args:
            frames: List of video frames as numpy arrays (BGR format from OpenCV)
            
        Returns:
            List of processed frames as PIL Image objects
        """
        # For HumanOmni, we just prepare a list of frames as PIL images
        # The model_init and mm_infer functions handle the video processing
        return self._prepare_images(frames)
    
    def _prepare_prompt(self, 
                      prompt_type: str, 
                      fusion_results: Dict[str, Any],
                      custom_prompt: Optional[str] = None) -> str:
        """
        Prepare the prompt for the HumanOmni model.
        
        Args:
            prompt_type: Type of prompt to use ('emotion_analysis', 'behavior_analysis', 'custom')
            fusion_results: Results from the multimodal fusion
            custom_prompt: Custom prompt to use if prompt_type is 'custom'
            
        Returns:
            Formatted prompt
        """
        # Extract information from fusion results
        additional_context = ""
        
        # Add detected emotions if available
        if 'primary_emotion' in fusion_results and fusion_results['primary_emotion'] is not None:
            # Make sure primary_emotion is a dictionary before accessing it
            if isinstance(fusion_results['primary_emotion'], dict):
                emotion = fusion_results['primary_emotion'].get('name', 'unknown')
                confidence = fusion_results['primary_emotion'].get('confidence', 0.0)
                additional_context += f"Detected primary emotion: {emotion} (confidence: {confidence:.2f}). "
            else:
                logger.warning("primary_emotion is not a dictionary as expected")
                additional_context += "Emotion detection results unavailable. "
        
        # Add sources used
        if 'sources_used' in fusion_results:
            sources = ', '.join(fusion_results['sources_used'])
            additional_context += f"Based on analysis from: {sources}. "
        
        # Format prompt
        if prompt_type == 'custom' and custom_prompt:
            prompt = self.prompt_templates['custom'].format(
                custom_prompt=custom_prompt,
                additional_context=additional_context
            )
        else:
            # Use default template
            prompt = self.prompt_templates.get(
                prompt_type, self.prompt_templates['emotion_analysis']
            ).format(additional_context=additional_context)
            
        return prompt
    
    def generate_analysis(self, 
                        images: List[np.ndarray],
                        fusion_results: Dict[str, Any],
                        prompt_type: str = 'emotion_analysis',
                        custom_prompt: Optional[str] = None,
                        max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate analysis using the HumanOmni model.
        
        Args:
            images: List of images as numpy arrays (BGR format)
            fusion_results: Results from the multimodal fusion
            prompt_type: Type of prompt to use
            custom_prompt: Custom prompt to use if prompt_type is 'custom'
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing the analysis results
        """
        # Prepare prompt for any case
        instruct = self._prepare_prompt(prompt_type, fusion_results, custom_prompt)
        
        # Check if HumanOmni is fully initialized
        if not self.initialized or self.model is None or self.processor is None or self.tokenizer is None:
            # Return a demonstration result if the model is not loaded
            logger.warning("HumanOmni model not initialized. Using demonstration results.")
            
            # Create a mock analysis based on fusion results
            emotions = fusion_results.get('emotions', {})
            primary_emotion = fusion_results.get('primary_emotion', {})
            emotion_name = primary_emotion.get('name', 'unknown') if isinstance(primary_emotion, dict) else 'unknown'
            
            if emotion_name != 'unknown':
                mock_analysis = f"Based on the visual analysis, I can detect signs of {emotion_name} in the subject. "
                mock_analysis += f"This is indicated by the emotional signals detected in the fusion results. "
                
                # Add details based on sources used
                sources_used = fusion_results.get('sources_used', [])
                if 'face' in sources_used:
                    mock_analysis += "The facial micro-expressions suggest emotional activation consistent with this state. "
                if 'body' in sources_used:
                    mock_analysis += "The body language reinforces this assessment, with posture and gesture patterns typical of this emotional state. "
                    
                # Add confidence information
                confidence = primary_emotion.get('confidence', 0.0) if isinstance(primary_emotion, dict) else 0.0
                if confidence > 0.7:
                    mock_analysis += "\n\nI have high confidence in this assessment based on the clarity of the emotional signals."
                elif confidence > 0.4:
                    mock_analysis += "\n\nI have moderate confidence in this assessment, though there are some mixed signals present."
                else:
                    mock_analysis += "\n\nI have low confidence in this assessment, as the emotional signals are subtle or ambiguous."
            else:
                mock_analysis = "I cannot detect clear emotional signals from the provided visual information. "
                mock_analysis += "This could be due to neutral expression, poor image quality, or absence of detectable emotional cues. "
                mock_analysis += "Additional context or different visual angles might help provide a more accurate assessment."
            
            return {
                'analysis': mock_analysis,
                'prompt': instruct,
                'success': False,
                'demo_mode': True
            }
        
        try:
            # Prepare images with the proper HumanOmni processor
            image_pil = self._prepare_images(images)[0] if images else None  # Use first image if available
            
            # Use HumanOmni's mm_infer function to generate analysis
            modal = 'image'  # Using image mode as we're processing single frames
            
            # Run inference using the proper HumanOmni API
            response = mm_infer(
                image_or_video=image_pil,
                instruct=instruct,
                model=self.model,
                tokenizer=self.tokenizer,
                modal=modal,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return {
                'analysis': response,
                'prompt': instruct,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in generate_analysis: {str(e)}")
            return {
                'analysis': f"Error generating analysis: {str(e)}",
                'prompt': instruct,
                'success': False
            }
    
    def analyze_sequence(self, 
                       image_sequence: List[np.ndarray],
                       fusion_results_sequence: List[Dict[str, Any]],
                       prompt_type: str = 'emotion_analysis',
                       custom_prompt: Optional[str] = None,
                       max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        Analyze a sequence of images and fusion results using the HumanOmni model's video capabilities.
        
        Args:
            image_sequence: Sequence of images (video frames)
            fusion_results_sequence: Sequence of fusion results corresponding to each frame
            prompt_type: Type of prompt to use
            custom_prompt: Custom prompt to use if prompt_type is 'custom'
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing the analysis results
        """
        # Prepare prompt based on aggregated fusion results
        aggregated_results = self._aggregate_fusion_results(fusion_results_sequence)
        instruct = self._prepare_prompt(prompt_type, aggregated_results, custom_prompt)
        
        # Check if HumanOmni is fully initialized
        if not self.initialized or self.model is None or self.processor is None or self.tokenizer is None:
            # Return demonstration results if model not initialized
            logger.warning("HumanOmni model not initialized for video analysis. Using demonstration results.")
            
            # Create a mock analysis similar to the one in generate_analysis
            # but adapted for video sequence
            primary_emotion = aggregated_results.get('primary_emotion', {})
            emotion_name = primary_emotion.get('name', 'unknown') if isinstance(primary_emotion, dict) else 'unknown'
            
            if emotion_name != 'unknown':
                mock_analysis = f"Based on analyzing the video sequence, I observe consistent {emotion_name} expressions in the subject. "
                mock_analysis += f"This emotional state persists across multiple frames, suggesting it is a significant response rather than a momentary reaction. "
                
                # Add temporal information
                if 'stability' in aggregated_results and aggregated_results['stability'] > 0.7:
                    mock_analysis += "\n\nThe emotion remains stable throughout the sequence, indicating a consistent emotional state. "
                elif 'transitions' in aggregated_results and aggregated_results['transitions']:
                    mock_analysis += "\n\nThere are noticeable transitions in emotional intensity throughout the sequence. "
                
                mock_analysis += "\n\nAnalyzing visual cues across the video sequence provides a more reliable assessment than single-frame analysis."
            else:
                mock_analysis = "After analyzing the video sequence, I cannot detect clear emotional patterns. "
                mock_analysis += "This could be due to neutral expressions, poor video quality, or subtlety of emotional cues. "
                mock_analysis += "The temporal information does not reveal significant emotional shifts or patterns."
            
            return {
                'analysis': mock_analysis,
                'prompt': instruct,
                'success': False,
                'demo_mode': True
            }
        
        try:
            # For HumanOmni, we need to prepare a video input from the sequence
            # Convert OpenCV format (BGR) to PIL images (RGB)
            frame_sequence = self._prepare_video_from_frames(image_sequence)
            
            # Use HumanOmni's mm_infer with video modality
            response = mm_infer(
                image_or_video=frame_sequence,  # Pass the frame sequence
                instruct=instruct,
                model=self.model,
                tokenizer=self.tokenizer,
                modal='video',  # Set modal to 'video' for sequence analysis
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return {
                'analysis': response,
                'prompt': instruct,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in video sequence analysis: {str(e)}")
            return {
                'analysis': f"Error analyzing video sequence: {str(e)}",
                'prompt': instruct,
                'success': False
            }
    
    def _aggregate_fusion_results(self, fusion_results_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate fusion results from a sequence.
        
        Args:
            fusion_results_sequence: Sequence of fusion results
            
        Returns:
            Aggregated fusion results
        """
        # Initialize aggregated results
        aggregated = {
            'primary_emotion': None,
            'emotions': {},
            'sources_used': set(),
            'confidence': 0.0
        }
        
        # Count emotion frequencies
        emotion_counts = {}
        total_confidence = 0.0
        valid_results = 0
        
        for result in fusion_results_sequence:
            # Add sources
            if 'sources_used' in result:
                aggregated['sources_used'].update(result['sources_used'])
            
            # Track emotions
            if 'primary_emotion' in result and result['primary_emotion']:
                emotion = result['primary_emotion'].get('name', 'unknown')
                confidence = result['primary_emotion'].get('confidence', 0.0)
                
                if emotion != 'unknown' and confidence > 0:
                    if emotion not in emotion_counts:
                        emotion_counts[emotion] = {'count': 0, 'confidence': 0.0}
                    
                    emotion_counts[emotion]['count'] += 1
                    emotion_counts[emotion]['confidence'] += confidence
                    total_confidence += confidence
                    valid_results += 1
            
            # Aggregate individual emotion scores
            if 'emotions' in result:
                for emotion, score in result['emotions'].items():
                    if emotion not in aggregated['emotions']:
                        aggregated['emotions'][emotion] = 0.0
                    aggregated['emotions'][emotion] += score
        
        # Normalize emotion scores
        if aggregated['emotions'] and len(fusion_results_sequence) > 0:
            for emotion in aggregated['emotions']:
                aggregated['emotions'][emotion] /= len(fusion_results_sequence)
        
        # Determine primary emotion
        if emotion_counts:
            # Find emotion with highest frequency and average confidence
            primary_emotion = max(emotion_counts.items(), key=lambda x: (x[1]['count'], x[1]['confidence']))
            
            aggregated['primary_emotion'] = {
                'name': primary_emotion[0],
                'confidence': primary_emotion[1]['confidence'] / primary_emotion[1]['count'] if primary_emotion[1]['count'] > 0 else 0.0
            }
            
            # Calculate overall confidence
            aggregated['confidence'] = total_confidence / valid_results if valid_results > 0 else 0.0
        
        # Convert sources_used to list
        aggregated['sources_used'] = list(aggregated['sources_used'])
        
        return aggregated
