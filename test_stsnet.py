"""
Test script for STSTNet component of the Multimodal Expression Analysis Framework.
This script demonstrates the use of STSTNet for micro-expression recognition
on video frames after deblurring with DeblurGAN.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from models.deblurgan import DeblurGANv2
from models.stsnet import STSTNetPredictor, FaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STSTNet-Test")

def parse_args():
    parser = argparse.ArgumentParser(description="Test STSTNet for micro-expression recognition")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--sampling_rate", type=int, default=4, help="Number of frames to sample per second")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--subject", type=str, default=None, help="Specific subject model to use")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--skip_deblur", action="store_true", help="Skip deblurring step")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    return parser.parse_args()

def extract_frames(video_path, sampling_rate=1, max_frames=None):
    """Extract frames from a video at specified sampling rate"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video properties:")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Frame rate: {fps} fps")
    print(f"- Total frames: {frame_count}")
    
    # Calculate sampling interval
    interval = int(fps / sampling_rate)
    if interval < 1:
        interval = 1
    
    # Extract frames
    frames = []
    timestamps = []
    frame_indices = []
    
    count = 0
    with tqdm(total=frame_count, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at regular intervals
            if count % interval == 0:
                frames.append(frame)
                timestamps.append(count / fps)
                frame_indices.append(count)
                
                # Check if we've reached the maximum number of frames
                if max_frames is not None and len(frames) >= max_frames:
                    break
            
            count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    return frames, timestamps, frame_indices

def visualize_microexpressions(frame, face_regions, predictions, timestamp, output_path):
    """Create a visualization of the detected micro-expressions"""
    # Make a copy of the frame for drawing
    vis_frame = frame.copy()
    
    # Draw each detected face and its micro-expression
    for i, (face_region, prediction) in enumerate(zip(face_regions, predictions)):
        # Check if prediction is empty
        if not prediction:
            logger.warning(f"No prediction available for face {i+1}")
            continue
            
        # Find the emotion with highest probability
        try:
            top_emotion = max(prediction.items(), key=lambda x: x[1])
            emotion_name = top_emotion[0]
            emotion_prob = top_emotion[1]
        except ValueError:
            # Handle empty prediction dictionary
            logger.warning(f"Empty prediction for face {i+1}")
            emotion_name = "unknown"
            emotion_prob = 0.0
        
        # Draw bounding box around face region (if face_region contains the bounding box)
        if hasattr(face_region, 'bbox'):
            x, y, w, h = face_region.bbox
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw emotion label and probability
        if hasattr(face_region, 'bbox'):
            label = f"{emotion_name}: {emotion_prob:.2f}"
            x, y = face_region.bbox[0], face_region.bbox[1]
            cv2.putText(vis_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If we don't have bounding box info, just add the label to the bottom of the image
            label = f"Person {i+1}: {emotion_name}: {emotion_prob:.2f}"
            y_pos = vis_frame.shape[0] - 30 * (i+1)
            cv2.putText(vis_frame, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add timestamp
    cv2.putText(vis_frame, f"Time: {timestamp:.2f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save visualization
    cv2.imwrite(output_path, vis_frame)
    
    return vis_frame

def save_results_to_file(timestamps, all_predictions, output_path):
    """Save micro-expression recognition results to a CSV file"""
    with open(output_path, 'w') as f:
        # Write header
        f.write("Timestamp,Person,Negative,Positive,Surprise\n")
        
        # Write data
        for t, frame_predictions in zip(timestamps, all_predictions):
            for i, predictions in enumerate(frame_predictions):
                negative = predictions.get('negative', 0.0)
                positive = predictions.get('positive', 0.0)
                surprise = predictions.get('surprise', 0.0)
                f.write(f"{t:.2f},{i},{negative:.4f},{positive:.4f},{surprise:.4f}\n")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize face detector
    face_detector = FaceDetector(detection_method='haarcascade')
    logger.info("Initialized face detector")
    
    # Initialize STSTNet predictor
    stsnet_predictor = STSTNetPredictor(device=args.device)
    logger.info("Initialized STSTNet predictor")
    
    # Print available subject models
    available_subjects = stsnet_predictor.get_available_subjects()
    logger.info(f"Available subject models: {available_subjects}")
    
    # Load specific subject model if requested
    if args.subject is not None:
        if args.subject in available_subjects:
            stsnet_predictor.load_subject_model(args.subject)
            logger.info(f"Loaded subject model: {args.subject}")
        else:
            logger.warning(f"Subject model {args.subject} not found, using default")
    
    # Initialize DeblurGAN model (if not skipping deblurring)
    if not args.skip_deblur:
        deblur_model = DeblurGANv2(device=args.device)
        logger.info("Initialized DeblurGAN-v2 model")
    
    # Extract frames from video
    frames, timestamps, frame_indices = extract_frames(
        args.video, 
        args.sampling_rate, 
        args.max_frames
    )
    
    # Process frames
    all_face_regions = []
    all_predictions = []
    
    logger.info("Processing frames for micro-expression recognition")
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i+1}/{len(frames)} (t={timestamp:.2f}s)", end="\r")
        
        # Deblur the frame if not skipping
        if not args.skip_deblur:
            deblurred_frame = deblur_model.process(frame)
        else:
            deblurred_frame = frame
        
        # Detect and process faces
        face_regions = face_detector.process_image(deblurred_frame)
        
        # Skip if no faces detected
        if not face_regions:
            all_face_regions.append([])
            all_predictions.append([])
            continue
        
        # Predict micro-expressions
        predictions = stsnet_predictor.predict_batch(face_regions)
        
        # Save results
        all_face_regions.append(face_regions)
        all_predictions.append(predictions)
        
        # Create visualization if requested
        if args.visualize:
            output_path = os.path.join(args.output_dir, f"frame_{i:04d}.jpg")
            visualize_microexpressions(
                deblurred_frame, 
                face_regions, 
                predictions, 
                timestamp, 
                output_path
            )
    
    print() # New line after progress reporting
    
    # Save all results to CSV
    results_path = os.path.join(args.output_dir, "microexpression_results.csv")
    save_results_to_file(timestamps, all_predictions, results_path)
    logger.info(f"Saved results to {results_path}")
    
    # Create emotion over time plot
    if len(all_predictions) > 0 and len(all_predictions[0]) > 0:
        # Get data for first detected face
        negative_values = []
        positive_values = []
        surprise_values = []
        
        for frame_predictions in all_predictions:
            if frame_predictions:
                predictions = frame_predictions[0]  # First face
                negative_values.append(predictions.get('negative', 0.0))
                positive_values.append(predictions.get('positive', 0.0))
                surprise_values.append(predictions.get('surprise', 0.0))
            else:
                # No face detected in this frame
                negative_values.append(0.0)
                positive_values.append(0.0)
                surprise_values.append(0.0)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, negative_values, 'r-', label='Negative')
        plt.plot(timestamps, positive_values, 'g-', label='Positive')
        plt.plot(timestamps, surprise_values, 'b-', label='Surprise')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Probability')
        plt.title('Micro-expression Recognition Over Time')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(args.output_dir, "microexpression_plot.png")
        plt.savefig(plot_path)
        logger.info(f"Saved plot to {plot_path}")
    
    logger.info("STSTNet testing completed successfully!")

if __name__ == "__main__":
    main()
