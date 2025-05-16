"""
Integrated Omni-R1 inference script.
This provides a simplified interface to run the full Omni-R1 pipeline,
similar to the original inference.py script but with the integrated models.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import torch
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("R1-Omni-Integrated")

# Import modules
from models.deblurgan.onnx_deblur import DeblurGANv2
from models.stsnet.face_utils import FaceDetector
from models.stsnet.stsnet_model import STSTNetPredictor
from models.openpose.openpose_model import OpenPoseAnalyzer
from models.fusion.fusion_model import MultimodalFusion, FusionMode
from models.omni.omni_integration import OmniIntegration

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Omni-R1 integrated pipeline")
    
    parser.add_argument("--model_path", type=str, 
                      default="C:\\Users\\zamor\\R1-Omni-0.5B",
                      help="Path to the HumanOmni model")
    
    parser.add_argument("--video_path", type=str, required=True, 
                      help="Path to the input video file")
    
    parser.add_argument("--instruct", type=str, 
                      default="Analyze the emotions in this video and explain your reasoning.",
                      help="Instruction for the HumanOmni model")
    
    parser.add_argument("--fusion_mode", type=str, default="weighted_average",
                      choices=["weighted_average", "confidence_based", 
                               "temporal_weighted", "hierarchical", "adaptive"],
                      help="Fusion mode to use (default: weighted_average)")
    
    parser.add_argument("--frame_interval", type=float, default=1.0,
                      help="Interval between frames to process in seconds (default: 1.0)")
    
    parser.add_argument("--save_dir", type=str, default="outputs",
                      help="Directory to save outputs (default: outputs)")
    
    parser.add_argument("--save_video", action="store_true",
                      help="Save the processed video with annotations")
    
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
    """Run the full Omni-R1 pipeline on the input video."""
    # Make sure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Extract frames from the video
    logger.info(f"Extracting frames from {args.video_path}")
    frames, timestamps = extract_frames(args.video_path, args.frame_interval)
    
    if not frames:
        logger.error(f"No frames extracted from video: {args.video_path}")
        return None
    
    # Initialize all models
    logger.info("Initializing models...")
    
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
    
    # Omni-R1 integration
    omni = OmniIntegration(
        model_path=args.model_path,
        fallback_to_demo=True  # Fallback to demo mode if model initialization fails
    )
    logger.info(f"Initialized Omni-R1 integration")
    
    # Process each frame
    logger.info("Processing frames...")
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
        if face_bboxes:
            # Use the first detected face
            x, y, w, h = face_bboxes[0]
            # Extract the face image from the bounding box
            face_img = deblurred_frame[y:y+h, x:x+w]
            
            # Only proceed if we have a valid face image
            if face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                micro_expressions = stsnet.predict(face_img)
                if micro_expressions:
                    face_emotions = {
                        'emotion': micro_expressions,
                        'face_detected': True,
                        'bbox': (x, y, w, h)
                    }
        
        # 3. Analyze body language
        body_results, body_visualization = openpose.process_image(deblurred_frame)
        
        # 4. Combine results with multimodal fusion
        fusion_result = fusion.combine_emotional_signals(
            face_emotions=face_emotions,
            body_emotions=body_results,
            frame_idx=i,
            timestamp=timestamp
        )
        
        # Store additional data for visualization
        fusion_result['face_emotions'] = face_emotions
        fusion_result['body_results'] = body_results
        
        # Add to results list
        fusion_results.append(fusion_result)
    
    # Create video sequence for analysis
    logger.info("Generating integrated analysis...")
    
    # Use Omni-R1 to analyze the sequence
    omni_result = omni.analyze_sequence(
        image_sequence=processed_frames,
        fusion_results_sequence=fusion_results,
        custom_prompt=args.instruct
    )
    
    # Print the analysis
    print("\n" + "=" * 80)
    print("OMNI-R1 INTEGRATED ANALYSIS")
    print("=" * 80)
    
    if omni_result.get('success', False):
        print(f"\nInstruction: {args.instruct}")
        print("\nAnalysis:")
        print(omni_result['analysis'])
    else:
        print("\nAnalysis using demonstration mode:")
        print(omni_result['analysis'])
        print("\nNote: Running in demonstration mode. The full HumanOmni model wasn't loaded.")
    
    print("\n" + "=" * 80)
    
    # Generate visualizations if needed
    if args.save_video:
        # Create output paths
        video_base = os.path.basename(args.video_path)
        video_name = os.path.splitext(video_base)[0]
        output_video_path = os.path.join(args.save_dir, f"{video_name}_analyzed.mp4")
        
        # Create the annotated video
        logger.info(f"Creating annotated video: {output_video_path}")
        
        # Get video properties
        cap = cv2.VideoCapture(args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Add annotations to frames and write to video
        for i, (frame, result) in enumerate(zip(processed_frames, fusion_results)):
            # Create a visualization with the fusion results
            annotated_frame = fusion.visualize_emotions(frame, result)
            
            # Add timestamp
            timestamp = timestamps[i]
            cv2.putText(
                annotated_frame, 
                f"t={timestamp:.2f}s", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            
            # Write to video
            out.write(annotated_frame)
            
        out.release()
        print(f"Saved annotated video: {output_video_path}")
    
    # Return the results
    return {
        'omni_analysis': omni_result,
        'fusion_results': fusion_results,
        'video_path': args.video_path,
        'output_video': output_video_path if args.save_video else None
    }

def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        result = run_pipeline(args)
        if not result:
            logger.error("Pipeline execution failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
