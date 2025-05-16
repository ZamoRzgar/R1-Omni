"""
Combined Omni-R1 inference script with Multimodal Fusion.
This script leverages your original HumanOmni inference approach while adding multimodal fusion.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import torch
from typing import Dict, List, Any
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("R1-Omni-Combined")

# Direct HumanOmni imports (like your original inference.py)
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

# Import multimodal components
from models.deblurgan.onnx_deblur import DeblurGANv2
from models.stsnet.face_utils import FaceDetector
from models.stsnet.stsnet_model import STSTNetPredictor
from models.openpose.openpose_model import OpenPoseAnalyzer
from models.fusion.fusion_model import MultimodalFusion, FusionMode

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Combined Omni-R1 inference with Multimodal Fusion")
    
    parser.add_argument('--modal', type=str, default='video', 
                      help='Modal type (video or video_audio)')
    
    parser.add_argument('--model_path', type=str, required=True, 
                      help='Path to the HumanOmni model')
    
    parser.add_argument('--video_path', type=str, required=True, 
                      help='Path to the input video file')
    
    parser.add_argument('--instruct', type=str, required=True, 
                      help='Instruction for the HumanOmni model')
    
    parser.add_argument('--fusion_mode', type=str, default='weighted_average',
                      choices=['weighted_average', 'confidence_based', 
                               'temporal_weighted', 'hierarchical', 'adaptive'],
                      help='Fusion mode to use')
    
    parser.add_argument('--frame_interval', type=float, default=1.0,
                      help='Interval between frames to process in seconds')
    
    parser.add_argument('--bert_model', type=str, 
                      default="C:/Users/zamor/models/bert-base-uncased",
                      help='Path to BERT model for tokenization')
    
    parser.add_argument('--save_fusion_results', action='store_true',
                      help='Save the fusion results to a file')
    
    return parser.parse_args()

def extract_frames(video_path, interval=1.0):
    """Extract frames from a video at the specified interval."""
    frames = []
    timestamps = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return frames, timestamps
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video properties:")
    logger.info(f"- Duration: {duration:.2f} seconds")
    logger.info(f"- Frame rate: {fps} fps")
    logger.info(f"- Total frames: {total_frames}")
    
    # Calculate frame interval
    frame_interval = int(fps * interval)
    if frame_interval < 1:
        frame_interval = 1
    
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count % frame_interval == 0:
            frames.append(frame)
            timestamp = frame_count / fps
            timestamps.append(timestamp)
            
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames")
    
    return frames, timestamps

def run_pipeline(args):
    """Run the full pipeline combining HumanOmni with multimodal fusion."""
    # Extract frames from the video
    logger.info(f"Extracting frames from {args.video_path}")
    frames, timestamps = extract_frames(args.video_path, args.frame_interval)
    
    if not frames:
        logger.error(f"No frames extracted from video: {args.video_path}")
        return None
    
    # Initialize multimodal fusion components
    logger.info("Initializing multimodal models...")
    
    # DeblurGAN for frame enhancement
    deblurgan = DeblurGANv2()
    logger.info("Initialized DeblurGAN-v2 model")
    
    # Face detection and micro-expression analysis
    face_detector = FaceDetector(detection_method='haarcascade')
    stsnet = STSTNetPredictor()
    logger.info("Initialized STSTNet for micro-expression analysis")
    
    # Body language analysis
    openpose = OpenPoseAnalyzer()
    logger.info("Initialized OpenPose for body language analysis")
    
    # Multimodal fusion
    fusion = MultimodalFusion(fusion_mode=args.fusion_mode.upper())
    logger.info(f"Initialized Multimodal Fusion with {args.fusion_mode} mode")
    
    # Process each frame with multimodal analysis
    logger.info("Processing frames with multimodal analysis...")
    fusion_results = []
    processed_frames = []
    
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i+1}/{len(frames)} (t={timestamp:.2f}s)", end="\r")
        
        # 1. Deblur the frame
        deblurred_frame = deblurgan(frame)
        processed_frames.append(deblurred_frame)
        
        # 2. Analyze facial expressions
        face_emotions = None
        face_bboxes = face_detector.detect_faces(deblurred_frame)
        
        # Debug facial detection
        if i == 0 or i == len(frames) // 2:  # Log for first frame and middle frame
            logger.info(f"Face detection for frame {i}: detected {len(face_bboxes)} faces")
        
        if face_bboxes:
            # Use the first detected face
            x, y, w, h = face_bboxes[0]
            # Extract the face image from the bounding box
            face_img = deblurred_frame[y:y+h, x:x+w]
            
            # Only proceed if we have a valid face image
            if face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                # Save the first face for debugging if needed
                if i == 0:
                    debug_face_path = os.path.join(os.path.dirname(args.video_path), "debug_face.jpg")
                    cv2.imwrite(debug_face_path, face_img)
                    logger.info(f"Saved debug face image to {debug_face_path}")
                
                micro_expressions = stsnet.predict(face_img)
                
                # Debug microexpression detection
                if i == 0 or i == len(frames) // 2:
                    if micro_expressions:
                        logger.info(f"  Microexpressions detected: {micro_expressions}")
                    else:
                        logger.info("  No microexpressions detected")
                
                if micro_expressions:
                    face_emotions = {
                        'emotion': micro_expressions,
                        'face_detected': True,
                        'bbox': (x, y, w, h),
                        'micro_expressions': micro_expressions  # Store the full result
                    }
        
        # 3. Analyze body language
        body_results, body_visualization = openpose.process_image(deblurred_frame)
        
        # Debug body language results
        if i == 0 or i == len(frames) // 2:  # Log for first frame and middle frame
            logger.info(f"OpenPose results for frame {i}: {body_results}")
            if 'people' in body_results and len(body_results['people']) > 0:
                person = body_results['people'][0]
                logger.info(f"  Person keypoints: {len(person.get('keypoints', {}))}")
                if 'emotion' in person:
                    logger.info(f"  Emotion detected: {person['emotion']}")
                else:
                    logger.info("  No emotion detected from body language")
            else:
                logger.info("  No people detected by OpenPose")
        
        # 4. Combine results with multimodal fusion
        fusion_result = fusion.combine_emotional_signals(
            face_emotions=face_emotions,
            body_emotions=body_results,
            frame_idx=i,
            timestamp=timestamp
        )
        
        # Add to results list
        fusion_results.append(fusion_result)
    
    # Prepare a custom instruction that includes multimodal fusion results
    emotional_summary = []
    body_language_insights = []
    microexpression_patterns = []
    
    for result in fusion_results:
        # Extract primary emotions
        primary_emotion = result.get('primary_emotion', {})
        if primary_emotion and isinstance(primary_emotion, dict):
            emotion_name = primary_emotion.get('name', 'unknown')
            confidence = primary_emotion.get('confidence', 0.0)
            if emotion_name != 'unknown':
                emotional_summary.append(f"{emotion_name} (confidence: {confidence:.2f})")
        
        # Extract body language insights
        body_signals = result.get('body_signals', {})
        if body_signals:
            timestamp = result.get('timestamp', 0)
            if 'posture' in body_signals:
                body_language_insights.append(f"At {timestamp:.1f}s: {body_signals['posture']}")
            if 'gesture' in body_signals:
                body_language_insights.append(f"At {timestamp:.1f}s: {body_signals['gesture']}")
            if 'movement' in body_signals:
                body_language_insights.append(f"At {timestamp:.1f}s: {body_signals['movement']}")
        
        # Extract microexpression patterns
        if 'micro_expressions' in result:
            micro_expr = result.get('micro_expressions', {})
            if micro_expr:
                timestamp = result.get('timestamp', 0)
                for expr_name, expr_value in micro_expr.items():
                    if expr_value > 0.15:  # Lowered from 0.3 to detect more subtle expressions
                        microexpression_patterns.append(f"At {timestamp:.1f}s: {expr_name} ({expr_value:.2f})")
    
    # Create an enhanced instruction with fusion results
    enhanced_instruct = f"{args.instruct}\n\nAdditional context from multimodal analysis:\n"
    
    # Add overall emotional summary
    enhanced_instruct += "\n1. EMOTIONAL SUMMARY: "
    if emotional_summary:
        # Get the top 3 most frequently occurring emotions
        from collections import Counter
        emotion_counter = Counter([em.split(' (')[0] for em in emotional_summary])  # Extract just the emotion name
        top_emotions = emotion_counter.most_common(3)
        enhanced_instruct += ", ".join([f"{emotion}" for emotion, count in top_emotions])
        enhanced_instruct += f" (detected in {count} frames)" if len(top_emotions) > 0 else ""
        
        # Add temporal information if available
        if len(fusion_results) > 2:
            stability = sum(1 for i in range(len(fusion_results)-1) 
                          if fusion_results[i].get('primary_emotion', {}).get('name', '') == 
                             fusion_results[i+1].get('primary_emotion', {}).get('name', '')) / (len(fusion_results)-1)
            
            if stability > 0.7:
                enhanced_instruct += ". The emotional state appears stable throughout the video."
            else:
                enhanced_instruct += ". The emotional state shows changes throughout the video."
    else:
        enhanced_instruct += "unclear emotional signals"
    
    # Add body language insights
    enhanced_instruct += "\n\n2. BODY LANGUAGE OBSERVATIONS: "
    if body_language_insights:
        # Limit to top 5 most informative body language insights to avoid overwhelming the model
        selected_insights = body_language_insights[:5] if len(body_language_insights) > 5 else body_language_insights
        enhanced_instruct += "\n" + "\n".join(selected_insights)
        
        # Add overall body language pattern summary
        if len(body_language_insights) > 5:
            enhanced_instruct += f"\n...and {len(body_language_insights) - 5} more body language signals"
    else:
        enhanced_instruct += "No significant body language signals detected"
    
    # Add microexpression observations
    enhanced_instruct += "\n\n3. MICROEXPRESSION ANALYSIS: "
    if microexpression_patterns:
        # Limit to top 5 most informative microexpressions
        selected_patterns = microexpression_patterns[:5] if len(microexpression_patterns) > 5 else microexpression_patterns
        enhanced_instruct += "\n" + "\n".join(selected_patterns)
    else:
        enhanced_instruct += "No significant microexpressions detected"
        
    # Add prompt for integrated analysis
    enhanced_instruct += "\n\nPlease analyze the emotional state and intentions of the person in the video, considering BOTH their facial expressions AND body language. Pay special attention to any inconsistencies between verbal/facial cues and body language."
    
    # Now run HumanOmni directly (based on your original inference.py)
    logger.info("Running HumanOmni model with multimodal-enhanced instruction...")
    
    # Initialize BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, local_files_only=True)
    
    # Disable Torch initialization (from original inference.py)
    disable_torch_init()
    
    # Initialize model, processor and tokenizer
    model, processor, tokenizer = model_init(args.model_path)
    
    # Process video input directly using the processor (original approach)
    video_tensor = processor['video'](args.video_path)
    
    # Handle audio if needed
    if args.modal == 'video_audio' or args.modal == 'audio':
        audio = processor['audio'](args.video_path)[0]
    else:
        audio = None
    
    # Run inference with the enhanced instruction
    output = mm_infer(
        video_tensor, 
        enhanced_instruct, 
        model=model, 
        tokenizer=tokenizer, 
        modal=args.modal, 
        question=args.instruct,  # Original instruction as a question 
        bert_tokeni=bert_tokenizer, 
        do_sample=False, 
        audio=audio
    )
    
    # Return combined results
    return {
        'omni_analysis': output,
        'fusion_results': fusion_results,
        'fusion_summary': emotional_summary,
        'enhanced_instruct': enhanced_instruct
    }

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Make sure all required parameters are provided
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Set offline mode for transformers
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Run the integrated pipeline
        result = run_pipeline(args)
        
        if not result:
            logger.error("Pipeline execution failed")
            sys.exit(1)
        
        # Print the results
        print("\n" + "=" * 80)
        print("OMNI-R1 INTEGRATED ANALYSIS WITH MULTIMODAL FUSION")
        print("=" * 80)
        
        print("\nEnhanced instruction with fusion context:")
        print(result['enhanced_instruct'])
        
        print("\nAnalysis:")
        print(result['omni_analysis'])
        
        print("\n" + "=" * 80)
        
        # Save fusion results if requested
        if args.save_fusion_results:
            import json
            fusion_file = f"fusion_results_{Path(args.video_path).stem}.json"
            with open(fusion_file, 'w') as f:
                json.dump([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} 
                          for r in result['fusion_results']], f, indent=2)
            print(f"Saved fusion results to {fusion_file}")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
