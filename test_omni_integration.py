"""
Test script for the Omni-R1 integration module.
This demonstrates how to use the multimodal fusion outputs with the HumanOmni backbone model.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Omni-Integration-Test")

# Import modules
from models.deblurgan.onnx_deblur import DeblurGANv2
from models.stsnet.face_utils import FaceDetector
from models.stsnet.stsnet_model import STSTNetPredictor
from models.openpose.openpose_model import OpenPoseAnalyzer
from models.fusion.fusion_model import MultimodalFusion
from models.omni.omni_integration import OmniIntegration

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Omni-R1 integrated pipeline")
    
    parser.add_argument("--video", type=str, required=True, 
                      help="Path to the input video file")
    
    parser.add_argument("--fusion_mode", type=str, default="weighted_average",
                      choices=["weighted_average", "confidence_based", 
                               "temporal_weighted", "hierarchical", "adaptive"],
                      help="Fusion mode to use")
    
    parser.add_argument("--omni_model_path", type=str, 
                      default=None,
                      help="Path to the HumanOmni model. If None, will try to use default model")
    
    parser.add_argument("--frame_interval", type=float, default=1.0,
                      help="Interval between frames to process (in seconds)")
    
    parser.add_argument("--save_video", action="store_true",
                      help="Save the processed video")
    
    parser.add_argument("--save_dir", type=str, default="outputs",
                      help="Directory to save outputs")
    
    parser.add_argument("--prompt_type", type=str, 
                      default="emotion_analysis",
                      choices=["emotion_analysis", "behavior_analysis", "custom"],
                      help="Type of prompt to use for the HumanOmni model")
    
    parser.add_argument("--custom_prompt", type=str, default=None,
                      help="Custom prompt to use if prompt_type is 'custom'")
    
    return parser.parse_args()

def extract_frames(video_path: str, interval: float = 1.0) -> List[np.ndarray]:
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path: Path to the video file
        interval: Interval between frames (in seconds)
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    timestamps = []
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return frames, timestamps
    
    # Get video properties
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
    
    # Extract frames
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            timestamps.append(frame_count / fps)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {extracted_count} frames")
    
    return frames, timestamps

def save_results_to_csv(results: List[Dict[str, Any]], csv_path: str) -> pd.DataFrame:
    """
    Save fusion results to a CSV file.
    
    Args:
        results: List of fusion results
        csv_path: Path to save the CSV file
        
    Returns:
        DataFrame with the results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Extract data for CSV
    data = []
    for result in results:
        # Handle the case where primary_emotion might be None
        primary_emotion = result.get('primary_emotion')
        emotion_name = primary_emotion.get('name', 'unknown') if primary_emotion else 'unknown'
        confidence = primary_emotion.get('confidence', 0.0) if primary_emotion else 0.0
        
        row = {
            'frame_idx': result.get('frame_idx', 0),
            'timestamp': result.get('timestamp', 0.0),
            'primary_emotion': emotion_name,
            'confidence': confidence,
            'sources_used': ','.join(result.get('sources_used', [])),
        }
        
        # Add individual emotion scores
        emotions = result.get('emotions', {})
        for emotion, score in emotions.items():
            row[f'emotion_{emotion}'] = score
            
        # Add omni analysis if available
        if 'omni_analysis' in result:
            omni_result = result['omni_analysis']
            row['omni_success'] = omni_result.get('success', False)
            # Truncate analysis to avoid huge CSV cells
            analysis = omni_result.get('analysis', '')
            if len(analysis) > 500:
                analysis = analysis[:497] + '...'
            row['omni_analysis'] = analysis
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    return df


def plot_emotion_timeline(results_df: pd.DataFrame, output_path: str):
    """
    Create a timeline plot of emotions.
    
    Args:
        results_df: DataFrame with results
        output_path: Path to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    # Get timestamps
    timestamps = results_df['timestamp'].values
    
    # Get all emotion columns
    emotion_cols = [col for col in results_df.columns if col.startswith('emotion_')]
    
    # Make sure we have at least some data to plot
    if len(emotion_cols) == 0 or results_df.empty:
        # Create a simple blank plot with a message if no data
        plt.text(0.5, 0.5, "No emotion data detected", 
                ha='center', va='center', fontsize=14, color='gray')
    else:
        # Create a stacked area plot
        emotion_data = results_df[emotion_cols].values
        labels = [col.replace('emotion_', '') for col in emotion_cols]
        
        # Plot the data if we have any
        if emotion_data.size > 0:
            plt.stackplot(timestamps, emotion_data.T, labels=labels, alpha=0.7)
        
        # Overlay primary emotion as a line if we have confidence values
        plt.plot(timestamps, results_df['confidence'], 'k-', linewidth=2, label='Confidence')
        
        # Add annotations for emotion changes if we have emotions detected
        prev_emotion = None
        for i, row in results_df.iterrows():
            current_emotion = row['primary_emotion']
            # Only annotate if there's an actual emotion (not 'unknown') and it changed
            if current_emotion != prev_emotion and current_emotion != 'unknown':
                confidence = row['confidence']
                # Only annotate if confidence is above zero
                if confidence > 0:
                    plt.annotate(current_emotion, 
                            xy=(row['timestamp'], confidence),
                            xytext=(0, 10), textcoords='offset points',
                            fontsize=8, rotation=45)
                prev_emotion = current_emotion
    
    plt.title('Emotion Analysis Timeline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion Intensity')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def create_frame_samples(frames: List[np.ndarray], results: List[Dict[str, Any]], output_path: str):
    """
    Create a grid of sample frames with annotations.
    
    Args:
        frames: List of frames
        results: List of results
        output_path: Path to save the output image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select sample frames (first, one-third, two-thirds, last)
    n_frames = len(frames)
    if n_frames < 4:
        indices = list(range(n_frames))
    else:
        indices = [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
    
    # Create a grid of frames with annotations
    sample_frames = []
    for idx in indices:
        if idx < len(frames) and idx < len(results):
            frame = frames[idx].copy()
            result = results[idx]
            
            # Add emotion annotation
            primary_emotion = result.get('primary_emotion')
            if primary_emotion and isinstance(primary_emotion, dict):
                emotion = primary_emotion.get('name', 'unknown')
                confidence = primary_emotion.get('confidence', 0.0)
                timestamp = result.get('timestamp', 0.0)
                
                # Add text annotation if we have a valid emotion
                if emotion != 'unknown' and confidence > 0:
                    text = f"{emotion} ({confidence:.2f}) @ {timestamp:.1f}s"
                else:
                    text = f"No emotion detected @ {timestamp:.1f}s"
                    
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
                
                # Draw face bounding box if available
                if 'face_emotions' in result and result['face_emotions'] and 'bbox' in result['face_emotions']:
                    x, y, w, h = result['face_emotions']['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add Omni-R1 analysis annotation if available
            if 'omni_analysis' in result:
                omni_result = result['omni_analysis']
                if omni_result.get('success', False):
                    # Add a truncated analysis (first line only)
                    analysis = omni_result.get('analysis', '')
                    if analysis:
                        # Take just the first sentence or first 50 characters
                        first_line = analysis.split('.')[0] + '...' if '.' in analysis[:100] else analysis[:50] + '...'
                        cv2.putText(frame, first_line, (10, frame.shape[0] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            sample_frames.append(frame)
    
    # Create a grid
    if sample_frames:
        n_cols = min(2, len(sample_frames))  # Max 2 columns
        n_rows = (len(sample_frames) + n_cols - 1) // n_cols
        
        # Get frame dimensions
        h, w = sample_frames[0].shape[:2]
        
        # Create a blank canvas
        grid_h = n_rows * h
        grid_w = n_cols * w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Place frames on the grid
        for i, frame in enumerate(sample_frames):
            r = i // n_cols
            c = i % n_cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
        
        # Save the grid
        cv2.imwrite(output_path, grid)
        logger.info(f"Saved frame samples to {output_path}")
    else:
        logger.warning(f"No frames to save")


def create_video(frames: List[np.ndarray], results: List[Dict[str, Any]], 
              input_path: str, output_path: str):
    """
    Create a video with annotations.
    
    Args:
        frames: List of frames
        results: List of results
        input_path: Path to input video (for properties)
        output_path: Path to output video
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get video properties
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for i, (frame, result) in enumerate(zip(frames, results)):
        # Create a copy to avoid modifying the original
        annotated = frame.copy()
        
        # Add timestamp
        timestamp = result.get('timestamp', 0.0)
        cv2.putText(annotated, f"t={timestamp:.2f}s", (10, height - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add primary emotion
        primary_emotion = result.get('primary_emotion')
        if primary_emotion and isinstance(primary_emotion, dict):
            emotion = primary_emotion.get('name', 'unknown')
            confidence = primary_emotion.get('confidence', 0.0)
            
            if emotion != 'unknown' and confidence > 0:
                cv2.putText(annotated, f"{emotion} ({confidence:.2f})", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw face bounding box if available
                if 'face_emotions' in result and result['face_emotions'] and 'bbox' in result['face_emotions']:
                    x, y, w, h = result['face_emotions']['bbox']
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add Omni-R1 status indicator
        if 'omni_analysis' in result and result['omni_analysis'].get('success', False):
            cv2.putText(annotated, "Omni-R1 Integrated", (width - 200, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Write frame to video
        out.write(annotated)
    
    # Release VideoWriter
    out.release()
    logger.info(f"Saved video to {output_path}")


def perform_temporal_analysis(fusion_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze temporal patterns in fusion results.
    
    Args:
        fusion_results: List of fusion results over time
        
    Returns:
        Dictionary with temporal analysis results
    """
    # Initialize analysis dict
    analysis = {
        'window_size': min(5, len(fusion_results)),
        'stability': 0.0,  # How stable emotions are
        'variety': 0.0,    # How many different emotions
        'dominant_emotion': None,
        'dominant_confidence': 0.0,
        'emotion_frequencies': {},
        'transitions': {}
    }
    
    # Count emotions and transitions
    prev_emotion = None
    for result in fusion_results:
        primary_emotion = result.get('primary_emotion')
        
        if primary_emotion and primary_emotion.get('name') != 'unknown':
            emotion = primary_emotion['name']
            confidence = primary_emotion['confidence']
            
            # Update frequencies
            if emotion not in analysis['emotion_frequencies']:
                analysis['emotion_frequencies'][emotion] = 0
            analysis['emotion_frequencies'][emotion] += 1
            
            # Update transitions
            if prev_emotion is not None and prev_emotion != emotion:
                transition_key = f"{prev_emotion}->{emotion}"
                if transition_key not in analysis['transitions']:
                    analysis['transitions'][transition_key] = 0
                analysis['transitions'][transition_key] += 1
            
            prev_emotion = emotion
            
            # Update dominant emotion
            if analysis['emotion_frequencies'][emotion] > analysis['dominant_confidence']:
                analysis['dominant_emotion'] = emotion
                analysis['dominant_confidence'] = analysis['emotion_frequencies'][emotion]
    
    # Calculate variety and stability
    if analysis['emotion_frequencies']:
        analysis['variety'] = len(analysis['emotion_frequencies']) / len(fusion_results)
        
        if analysis['dominant_emotion'] is not None:
            dominant_count = analysis['emotion_frequencies'][analysis['dominant_emotion']]
            analysis['stability'] = dominant_count / len(fusion_results)
    
    return analysis


def process_video(args):
    """
    Process a video with the full Omni-R1 pipeline.
    """
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Extract frames from video
    logger.info(f"Extracting frames from {args.video}")
    frames, timestamps = extract_frames(args.video, args.frame_interval)
    
    if not frames:
        logger.error(f"No frames extracted from video: {args.video}")
        return
    
    # Initialize models
    logger.info(f"Initializing models")
    
    # DeblurGAN for deblurring
    deblurgan = DeblurGANv2()
    logger.info(f"Initialized DeblurGAN-v2 model")
    
    # Face detector and STSTNet for micro-expression analysis
    face_detector = FaceDetector(detection_method='haarcascade')
    stsnet = STSTNetPredictor()  
    logger.info(f"Initialized STSTNet for micro-expression analysis")
    
    # OpenPose for body language analysis
    openpose = OpenPoseAnalyzer()
    logger.info(f"Initialized OpenPose analyzer for body language")
    
    # Multimodal fusion
    fusion = MultimodalFusion(fusion_mode=args.fusion_mode.upper())
    logger.info(f"Initialized Multimodal Fusion with {args.fusion_mode} mode")
    
    # Omni-R1 integration
    omni = OmniIntegration(
        model_path=args.omni_model_path,
        fallback_to_demo=True  # Always fall back to demo mode if model loading fails
    )
    logger.info(f"Initialized Omni-R1 integration")
    
    # Process frames
    logger.info(f"Processing frames for multimodal analysis")
    fusion_results = []
    
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i+1}/{len(frames)} (t={timestamp:.2f}s)", end="\r")
        
        # Deblur the frame
        processed_frame = deblurgan(frame)
        
        # Analyze facial expressions
        face_emotions = None
        face_bboxes = face_detector.detect_faces(processed_frame)
        if face_bboxes:
            # Use the first detected face
            x, y, w, h = face_bboxes[0]
            # Extract the face image from the bounding box
            face_img = processed_frame[y:y+h, x:x+w]
            
            # Only proceed if we have a valid face image
            if face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                micro_expressions = stsnet.predict(face_img)
                if micro_expressions:
                    face_emotions = {
                        'emotion': micro_expressions,
                        'face_detected': True,
                        'bbox': (x, y, w, h)
                    }
        
        # Analyze body language
        body_results, body_visualization = openpose.process_image(processed_frame)
        
        # Combine results with multimodal fusion
        fusion_result = fusion.combine_emotional_signals(
            face_emotions=face_emotions,
            body_emotions=body_results,
            frame_idx=i,
            timestamp=timestamp
        )
        
        # Store the face_emotions and body_results directly in the fusion_result
        # for easier access in visualization
        fusion_result['face_emotions'] = face_emotions
        fusion_result['body_results'] = body_results
        
        # Create visualization
        visualization = fusion.visualize_emotions(
            body_visualization if body_visualization is not None else processed_frame,
            fusion_result
        )
        
        # Add to results list
        fusion_results.append(fusion_result)
    
    # Perform temporal analysis
    temporal_analysis = perform_temporal_analysis(fusion_results)
    logger.info(f"Temporal analysis: {temporal_analysis}")
    
    # Integrate with Omni-R1
    # Sample the frames to reduce processing load
    sample_indices = [0, len(frames)//2, -1] if len(frames) > 3 else list(range(len(frames)))
    sample_frames = [frames[i] for i in sample_indices if i < len(frames)]
    sample_results = [fusion_results[i] for i in sample_indices if i < len(fusion_results)]
    
    # Generate Omni-R1 analysis
    omni_result = omni.analyze_sequence(
        image_sequence=sample_frames,
        fusion_results_sequence=sample_results,
        prompt_type=args.prompt_type,
        custom_prompt=args.custom_prompt
    )
    
    # Add Omni-R1 analysis to all frames for consistent display
    for result in fusion_results:
        result['omni_analysis'] = omni_result
    
    # Save results to CSV
    csv_path = os.path.join(args.save_dir, 'fusion_results.csv')
    results_df = save_results_to_csv(fusion_results, csv_path)
    
    # Create emotion timeline plot
    plot_path = os.path.join(args.save_dir, 'emotion_timeline.png')
    plot_emotion_timeline(results_df, plot_path)
    
    # Create frame samples
    samples_path = os.path.join(args.save_dir, 'emotion_samples.png')
    create_frame_samples(frames, fusion_results, samples_path)
    
    # Create video with annotations
    if args.save_video:
        video_path = os.path.join(args.save_dir, 'omni_integrated_analysis.mp4')
        create_video(frames, fusion_results, args.video, video_path)
    
    logger.info(f"Omni-R1 integrated analysis completed successfully!")
    
    # Return the final results and visualizations
    return {
        'fusion_results': fusion_results,
        'omni_analysis': omni_result,
        'temporal_analysis': temporal_analysis,
        'visualizations': {
            'csv_path': csv_path,
            'plot_path': plot_path,
            'samples_path': samples_path,
            'video_path': video_path if args.save_video else None
        }
    }


def main():
    """
    Main entry point for the script.
    """
    args = parse_arguments()
    
    try:
        results = process_video(args)
        
        # Print a summary of the results
        omni_analysis = results['omni_analysis']
        print("\n" + "=" * 80)
        print("OMNI-R1 INTEGRATED ANALYSIS SUMMARY")
        print("=" * 80)
        
        if omni_analysis['success']:
            print(f"Prompt: {omni_analysis['prompt']}\n")
            print(f"Analysis:\n{omni_analysis['analysis']}")
        else:
            print("Omni-R1 analysis failed. See logs for details.")
            
        print("\n" + "=" * 80)
        print("OUTPUT FILES")
        print("=" * 80)
        for name, path in results['visualizations'].items():
            if path:
                print(f"{name}: {path}")
                
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

