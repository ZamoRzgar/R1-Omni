"""
Test script for OpenPose component of the Multimodal Expression Analysis Framework.
This script demonstrates the use of OpenPose for body language analysis,
which can be combined with DeblurGAN and STSTNet for comprehensive emotion recognition.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from models.deblurgan import DeblurGANv2
from models.openpose import OpenPoseAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OpenPose-Test")

def parse_args():
    parser = argparse.ArgumentParser(description="Test OpenPose for body language analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--sampling_rate", type=int, default=2, help="Number of frames to sample per second")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--skip_deblur", action="store_true", help="Skip deblurring step")
    parser.add_argument("--save_video", action="store_true", help="Save processed video")
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

def save_video(frames, output_path, fps=30):
    """Save frames as a video"""
    if not frames:
        logger.error("No frames to save")
        return
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    with tqdm(total=len(frames), desc="Saving video") as pbar:
        for frame in frames:
            out.write(frame)
            pbar.update(1)
    
    out.release()
    logger.info(f"Video saved to {output_path}")

def save_results_to_file(timestamps, all_results, output_path):
    """Save emotion analysis results to a CSV file"""
    with open(output_path, 'w') as f:
        # Write header
        f.write("Timestamp,Person,Emotion,Confidence\n")
        
        # Write data
        for t, frame_results in zip(timestamps, all_results):
            if "people" not in frame_results:
                continue
                
            for i, person in enumerate(frame_results["people"]):
                if "emotion" in person:
                    emotion = person["emotion"]["name"]
                    confidence = person["emotion"]["confidence"]
                    f.write(f"{t:.2f},{i},{emotion},{confidence:.4f}\n")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize OpenPose analyzer
    openpose = OpenPoseAnalyzer(device=args.device)
    logger.info("Initialized OpenPose analyzer")
    
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
    processed_frames = []
    all_results = []
    
    logger.info("Processing frames for body language analysis")
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i+1}/{len(frames)} (t={timestamp:.2f}s)", end="\r")
        
        # Deblur the frame if not skipping
        if not args.skip_deblur:
            processed_frame = deblur_model.process(frame)
        else:
            processed_frame = frame
        
        # Analyze body language
        results, visualization = openpose.process_image(processed_frame)
        
        # Save results
        all_results.append(results)
        processed_frames.append(visualization)
    
    print() # New line after progress reporting
    
    # Save all results to CSV
    results_path = os.path.join(args.output_dir, "body_language_results.csv")
    save_results_to_file(timestamps, all_results, results_path)
    logger.info(f"Saved results to {results_path}")
    
    # Save processed video if requested
    if args.save_video and processed_frames:
        video_path = os.path.join(args.output_dir, "body_analysis_video.mp4")
        save_video(processed_frames, video_path)
    
    # Create body language emotion over time plot
    emotion_data = {}
    people_count = 0
    
    # Collect emotion data from results
    for frame_idx, (timestamp, frame_results) in enumerate(zip(timestamps, all_results)):
        if "people" not in frame_results:
            continue
            
        for person_idx, person in enumerate(frame_results["people"]):
            if "emotion" not in person:
                continue
                
            person_key = f"Person {person_idx+1}"
            emotion = person["emotion"]["name"]
            confidence = person["emotion"]["confidence"]
            
            if person_key not in emotion_data:
                emotion_data[person_key] = {
                    "timestamps": [],
                    "emotions": [],
                    "confidences": []
                }
                people_count = max(people_count, person_idx+1)
            
            emotion_data[person_key]["timestamps"].append(timestamp)
            emotion_data[person_key]["emotions"].append(emotion)
            emotion_data[person_key]["confidences"].append(confidence)
    
    # Create plot if we have data
    if emotion_data:
        plt.figure(figsize=(12, 4 * people_count))
        
        for idx, (person_key, data) in enumerate(emotion_data.items()):
            plt.subplot(people_count, 1, idx+1)
            
            # Convert emotions to numeric values for plotting
            emotion_categories = openpose.get_emotion_categories()
            emotion_values = [emotion_categories.index(e) for e in data["emotions"]]
            
            # Create scatter plot with size based on confidence
            sizes = [c * 100 for c in data["confidences"]]
            plt.scatter(data["timestamps"], emotion_values, s=sizes, alpha=0.7)
            
            # Add best fit line to show trend
            if len(data["timestamps"]) > 1:
                z = np.polyfit(data["timestamps"], emotion_values, 1)
                p = np.poly1d(z)
                plt.plot(data["timestamps"], p(data["timestamps"]), "r--", alpha=0.5)
            
            plt.yticks(range(len(emotion_categories)), emotion_categories)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Emotion")
            plt.title(f"{person_key} - Body Language Emotions Over Time")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(args.output_dir, "body_language_plot.png")
        plt.savefig(plot_path)
        logger.info(f"Saved plot to {plot_path}")
    
    # Create a combined plot with sample frames
    if processed_frames:
        # Select a subset of frames for visualization
        num_samples = min(5, len(processed_frames))
        sample_indices = np.linspace(0, len(processed_frames)-1, num_samples, dtype=int)
        
        # Create figure
        plt.figure(figsize=(15, 4 * num_samples))
        
        for i, idx in enumerate(sample_indices):
            frame = processed_frames[idx]
            timestamp = timestamps[idx]
            
            # Convert from BGR to RGB for matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(rgb_frame)
            plt.title(f"Frame at t={timestamp:.2f}s")
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        vis_path = os.path.join(args.output_dir, "body_language_samples.png")
        plt.savefig(vis_path)
        logger.info(f"Saved frame samples to {vis_path}")
    
    logger.info("OpenPose testing completed successfully!")

if __name__ == "__main__":
    main()
