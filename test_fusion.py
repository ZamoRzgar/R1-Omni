"""
Test script for the Multimodal Fusion module in the R1-Omni Expression Analysis Framework.

This script demonstrates how to combine outputs from multiple emotional analysis
components (STSTNet for facial micro-expressions, OpenPose for body language) 
to generate comprehensive emotional assessments.

Usage:
    python test_fusion.py --video [path_to_video] --fusion_mode [mode] --save_video

Options:
    --video: Path to input video file (default: data/test_videos/video.mp4)
    --fusion_mode: Fusion strategy to use (default: weighted_average)
        Options: weighted_average, confidence_based, temporal_weighted, 
                hierarchical, adaptive
    --save_video: Save output video with visualizations
    --output_dir: Directory to save results (default: outputs)
    --sample_rate: Number of frames to sample per second (default: 1)
    --face_weight: Weight for facial expressions (0-1, default: 0.6)
    --body_weight: Weight for body language (0-1, default: 0.4)
    --temporal_window: Number of frames for temporal analysis (default: 5)
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Fusion-Test')

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import R1-Omni modules
from models.deblurgan import ONNXDeblurGAN
from models.stsnet import STSTNetPredictor, FaceDetector
from models.openpose import OpenPoseAnalyzer
from models.fusion import MultimodalFusion, FusionMode

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the multimodal fusion module')
    
    parser.add_argument('--video', type=str, default='data/test_videos/video.mp4',
                        help='Path to input video file')
    parser.add_argument('--fusion_mode', type=str, default='weighted_average',
                        choices=['weighted_average', 'confidence_based', 
                                'temporal_weighted', 'hierarchical', 'adaptive'],
                        help='Fusion strategy to use')
    parser.add_argument('--save_video', action='store_true',
                        help='Save output video with visualizations')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save results')
    parser.add_argument('--sample_rate', type=float, default=1,
                        help='Number of frames to sample per second')
    parser.add_argument('--face_weight', type=float, default=0.6,
                        help='Weight for facial expressions (0-1)')
    parser.add_argument('--body_weight', type=float, default=0.4,
                        help='Weight for body language (0-1)')
    parser.add_argument('--temporal_window', type=int, default=5,
                        help='Number of frames for temporal analysis')
    
    return parser.parse_args()

def extract_frames(video_path: str, sample_rate: float = 1) -> Tuple[List[np.ndarray], float, float]:
    """
    Extract frames from a video at a specified sample rate.
    
    Args:
        video_path: Path to the video file
        sample_rate: Number of frames to extract per second
        
    Returns:
        Tuple of (list of frames, frame rate, duration)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    print(f"Video properties:")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Frame rate: {fps} fps")
    print(f"- Total frames: {frame_count}")
    
    # Calculate sampling interval
    if sample_rate <= 0:
        sample_rate = fps  # Use original frame rate
    
    interval = int(fps / sample_rate)
    interval = max(1, interval)  # Ensure interval is at least 1
    
    frames = []
    frame_positions = []
    
    # Extract frames at the specified interval
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_positions.append(i / fps)  # Store timestamp in seconds
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames")
    return frames, frame_positions, fps

def save_results_to_csv(results: List[Dict], output_path: str):
    """
    Save emotion analysis results to a CSV file.
    
    Args:
        results: List of emotion analysis result dictionaries
        output_path: Path to save the CSV file
    """
    # Extract data for CSV
    data = []
    for result in results:
        # Basic info
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
        
        # Add all emotion scores
        for emotion, score in result.get('emotions', {}).items():
            row[f'emotion_{emotion}'] = score
        
        # Add temporal consistency if available
        if 'temporal_consistency' in result:
            row['temporal_consistency'] = result['temporal_consistency']
        
        # Add adaptive weights if available
        if 'adaptive_weights' in result:
            for source, weight in result['adaptive_weights'].items():
                row[f'weight_{source}'] = weight
        
        data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")
    
    return df

def create_emotion_timeline_plot(results_df: pd.DataFrame, output_path: str):
    """
    Create a plot showing the emotion timeline.
    
    Args:
        results_df: DataFrame with emotion analysis results
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot primary emotion over time
    timestamps = results_df['timestamp']
    
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
    plt.ylabel('Emotion Confidence')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved plot to {output_path}")

def save_frame_samples(frames: List[np.ndarray], results: List[Dict], output_path: str, num_samples: int = 6):
    """
    Save a grid of frame samples with emotion annotations.
    
    Args:
        frames: List of video frames
        results: List of emotion analysis results
        output_path: Path to save the output image
        num_samples: Number of frame samples to include
    """
    if not frames or not results:
        return
    
    # Select frames at regular intervals
    indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    
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
            
            sample_frames.append(frame)
    
    # Calculate grid dimensions
    n = len(sample_frames)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # Create an empty grid
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    # Fill the grid with frames
    for i, frame in enumerate(sample_frames):
        r, c = i // cols, i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    
    # Save the grid
    cv2.imwrite(output_path, grid)
    logger.info(f"Saved frame samples to {output_path}")

def main():
    """Main function for testing the multimodal fusion module."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract frames from the video
    frames, timestamps, fps = extract_frames(args.video, args.sample_rate)
    
    # Initialize deblurring model
    deblurgan = ONNXDeblurGAN()
    logger.info("Initialized DeblurGAN-v2 model")
    
    # Initialize facial expression analysis
    face_detector = FaceDetector()
    stsnet = STSTNetPredictor()
    logger.info("Initialized STSTNet for micro-expression analysis")
    
    # Initialize body language analysis
    openpose = OpenPoseAnalyzer()
    logger.info("Initialized OpenPose analyzer for body language")
    
    # Get fusion mode from arguments
    fusion_mode_map = {
        'weighted_average': FusionMode.WEIGHTED_AVERAGE,
        'confidence_based': FusionMode.CONFIDENCE_BASED,
        'temporal_weighted': FusionMode.TEMPORAL_WEIGHTED,
        'hierarchical': FusionMode.HIERARCHICAL,
        'adaptive': FusionMode.ADAPTIVE
    }
    fusion_mode = fusion_mode_map.get(args.fusion_mode, FusionMode.WEIGHTED_AVERAGE)
    
    # Initialize multimodal fusion
    fusion = MultimodalFusion(
        fusion_mode=fusion_mode,
        face_weight=args.face_weight,
        body_weight=args.body_weight,
        temporal_window=args.temporal_window
    )
    logger.info(f"Initialized Multimodal Fusion with {args.fusion_mode} mode")
    
    # Process frames
    logger.info("Processing frames for multimodal analysis")
    processed_frames = []
    fusion_results = []
    
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i+1}/{len(frames)} (t={timestamp:.2f}s)", end='\r')
        
        # Apply deblurring if needed
        processed_frame = deblurgan.deblur_image(frame)
        
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
        
        processed_frames.append(visualization)
        fusion_results.append(fusion_result)
    
    # Analyze temporal patterns
    temporal_analysis = fusion.analyze_temporal_patterns()
    logger.info(f"Temporal analysis: {temporal_analysis}")
    
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'fusion_results.csv')
    results_df = save_results_to_csv(fusion_results, csv_path)
    
    # Create emotion timeline plot
    plot_path = os.path.join(args.output_dir, 'emotion_timeline.png')
    create_emotion_timeline_plot(results_df, plot_path)
    
    # Save frame samples
    samples_path = os.path.join(args.output_dir, 'emotion_samples.png')
    save_frame_samples(processed_frames, fusion_results, samples_path)
    
    # Save video if requested
    if args.save_video and processed_frames:
        video_path = os.path.join(args.output_dir, 'multimodal_analysis.mp4')
        
        # Get frame dimensions
        h, w = processed_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps / args.sample_rate, (w, h))
        
        for frame in processed_frames:
            writer.write(frame)
        
        writer.release()
        logger.info(f"Saved video to {video_path}")
    
    logger.info("Multimodal fusion testing completed successfully!")

if __name__ == "__main__":
    main()
