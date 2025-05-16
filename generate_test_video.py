"""
Generate a test video with motion blur for testing DeblurGAN.
This script creates a synthetic video with controlled motion blur
to test the deblurring capabilities of DeblurGAN.
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate test video with motion blur")
    parser.add_argument("--output", type=str, default="data/test_videos/blurred_test_video.mp4", 
                        help="Path to save output video")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of video in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--resolution", type=str, default="640x480", 
                        help="Resolution of video (WxH)")
    parser.add_argument("--blur_intensity", type=float, default=0.7, 
                        help="Intensity of motion blur (0.0-1.0)")
    parser.add_argument("--webcam_id", type=int, default=0, 
                        help="Webcam ID to use (if available)")
    parser.add_argument("--use_webcam", action="store_true", 
                        help="Use webcam instead of generating synthetic video")
    
    return parser.parse_args()

def apply_motion_blur(image, intensity=0.7, direction=None):
    """Apply motion blur to an image"""
    # Choose random direction if not specified
    if direction is None:
        angle = np.random.uniform(0, 360)
    else:
        angle = direction
    
    # Convert angle to radians
    rad = np.deg2rad(angle)
    
    # Calculate kernel size based on intensity
    kernel_size = max(3, int(10 * intensity) * 2 + 1)
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Fill kernel along the direction
    for i in range(kernel_size):
        x = int(center + (i - center) * np.sin(rad))
        y = int(center + (i - center) * np.cos(rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    # Apply motion blur
    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred

def generate_synthetic_video(args):
    """Generate a synthetic video with a moving object and motion blur"""
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Calculate number of frames
    num_frames = int(args.duration * args.fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
    
    # Generate frames
    for i in tqdm(range(num_frames), desc="Generating frames"):
        # Create a blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw moving text to simulate facial expressions
        time_factor = i / num_frames
        
        # Apply different expressions at different times
        if time_factor < 0.25:
            text = "Neutral"
            color = (0, 0, 0)  # Black
        elif time_factor < 0.5:
            text = "Happy"
            color = (0, 255, 0)  # Green
        elif time_factor < 0.75:
            text = "Sad"
            color = (0, 0, 255)  # Red
        else:
            text = "Surprised"
            color = (255, 0, 0)  # Blue
        
        # Calculate position (moving in a circle)
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        angle = 2 * np.pi * time_factor
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        # Draw a face-like shape
        cv2.circle(frame, (x, y), 50, color, -1)  # Head
        
        # Draw eyes
        eye_offset = 15
        cv2.circle(frame, (x - eye_offset, y - 15), 10, (255, 255, 255), -1)  # Left eye
        cv2.circle(frame, (x + eye_offset, y - 15), 10, (255, 255, 255), -1)  # Right eye
        
        # Draw mouth (different shapes for different expressions)
        if "Happy" in text:
            cv2.ellipse(frame, (x, y + 15), (25, 10), 0, 0, 180, (255, 255, 255), -1)
        elif "Sad" in text:
            cv2.ellipse(frame, (x, y + 25), (25, 10), 0, 180, 360, (255, 255, 255), -1)
        elif "Surprised" in text:
            cv2.circle(frame, (x, y + 15), 15, (255, 255, 255), -1)
        else:
            cv2.line(frame, (x - 20, y + 15), (x + 20, y + 15), (255, 255, 255), 3)
        
        # Add text label
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Apply motion blur if specified
        if args.blur_intensity > 0:
            # Calculate blur direction based on movement
            if i > 0:
                # Calculate motion direction
                prev_angle = 2 * np.pi * ((i - 1) / num_frames)
                prev_x = int(center_x + radius * np.cos(prev_angle))
                prev_y = int(center_y + radius * np.sin(prev_angle))
                
                # Calculate angle between previous and current position
                dx, dy = x - prev_x, y - prev_y
                blur_angle = np.rad2deg(np.arctan2(dy, dx))
                
                # Apply directional motion blur
                frame = apply_motion_blur(frame, intensity=args.blur_intensity, direction=blur_angle)
            else:
                # Random direction for first frame
                frame = apply_motion_blur(frame, intensity=args.blur_intensity)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    
    print(f"Synthetic video saved to {args.output}")

def capture_webcam_video(args):
    """Capture video from webcam and apply motion blur"""
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Open webcam
    cap = cv2.VideoCapture(args.webcam_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam {args.webcam_id}")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual webcam resolution: {actual_width}x{actual_height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (actual_width, actual_height))
    
    # Calculate number of frames
    num_frames = int(args.duration * args.fps)
    
    # Capture frames
    for _ in tqdm(range(num_frames), desc="Capturing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply motion blur if specified
        if args.blur_intensity > 0:
            frame = apply_motion_blur(frame, intensity=args.blur_intensity)
        
        # Write frame to video
        out.write(frame)
        
        # Display frame (press 'q' to quit early)
        cv2.imshow("Capturing Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Webcam video saved to {args.output}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate video
    if args.use_webcam:
        capture_webcam_video(args)
    else:
        generate_synthetic_video(args)

if __name__ == "__main__":
    main()
