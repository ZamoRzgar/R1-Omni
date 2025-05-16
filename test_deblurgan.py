"""
Test script for DeblurGAN component of the Multimodal Expression Analysis Framework.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.deblurgan.deblur_model import DeblurGANv2, download_deblurgan_weights
from utils.video_processing import extract_frames, save_video, create_comparison_grid

def parse_args():
    parser = argparse.ArgumentParser(description="Test DeblurGAN-v2 for motion deblurring")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--sampling_rate", type=int, default=4, help="Number of frames to sample per second")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--save_comparison", action="store_true", help="Save comparison grid")
    parser.add_argument("--device", type=str, default="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu", 
                        help="Device to use (cuda/cpu)")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download DeblurGAN weights
    weights_path = download_deblurgan_weights()
    
    # Initialize DeblurGAN model
    deblur_model = DeblurGANv2(weights_path, device=args.device)
    print(f"DeblurGAN-v2 model loaded on {args.device}")
    
    # Extract frames from video
    print(f"Extracting frames from {args.video}")
    frames, timestamps = extract_frames(args.video, args.sampling_rate, args.max_frames)
    print(f"Extracted {len(frames)} frames")
    
    # Process frames
    print("Processing frames with DeblurGAN-v2")
    deblurred_frames = []
    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}", end="\r")
        deblurred_frame = deblur_model.process(frame)
        deblurred_frames.append(deblurred_frame)
    print(f"\nProcessed {len(deblurred_frames)} frames")
    
    # Save processed video
    output_video_path = os.path.join(args.output_dir, "deblurred_video.mp4")
    save_video(deblurred_frames, output_video_path)
    print(f"Deblurred video saved to {output_video_path}")
    
    # Save original video (with same sampling rate for comparison)
    original_video_path = os.path.join(args.output_dir, "original_video.mp4")
    save_video(frames, original_video_path)
    print(f"Original video (sampled) saved to {original_video_path}")
    
    # Create comparison grid
    if args.save_comparison:
        # Select a subset of frames for comparison
        num_samples = min(4, len(frames))
        fig = create_comparison_grid(frames, deblurred_frames, timestamps, num_samples)
        
        # Save comparison grid
        comparison_path = os.path.join(args.output_dir, "comparison_grid.png")
        fig.savefig(comparison_path)
        print(f"Comparison grid saved to {comparison_path}")
        
        # Display difference visualization for a middle frame
        middle_idx = len(frames) // 2
        from utils.video_processing import visualize_frame_difference
        diff_fig = visualize_frame_difference(frames[middle_idx], deblurred_frames[middle_idx])
        
        # Save difference visualization
        diff_path = os.path.join(args.output_dir, "difference_visualization.png")
        diff_fig.savefig(diff_path)
        print(f"Difference visualization saved to {diff_path}")
    
    print("DeblurGAN-v2 testing completed successfully!")

if __name__ == "__main__":
    main()
