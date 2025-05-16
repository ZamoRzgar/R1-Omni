"""
Video processing utilities for the Multimodal Expression Analysis Framework.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_frames(video_path: str, 
                  sampling_rate: int = 1, 
                  max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract frames from video at specified sampling rate
    
    Args:
        video_path: Path to video file
        sampling_rate: Number of frames to extract per second
        max_frames: Maximum number of frames to extract (None for all)
        
    Returns:
        Tuple of (frames, timestamps)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
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
    
    # Calculate frame interval based on sampling rate
    frame_interval = int(fps / sampling_rate)
    if frame_interval < 1:
        frame_interval = 1
    
    frames = []
    timestamps = []
    
    # Limit total frames if specified
    total_samples = frame_count // frame_interval
    if max_frames is not None:
        total_samples = min(total_samples, max_frames)
    
    # Extract frames
    for i in tqdm(range(0, frame_count, frame_interval), total=total_samples, desc="Extracting frames"):
        if max_frames is not None and len(frames) >= max_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        timestamps.append(i / fps)
    
    cap.release()
    return frames, timestamps

def save_video(frames: List[np.ndarray], 
              output_path: str, 
              fps: int = 30, 
              show_progress: bool = True) -> str:
    """
    Save frames as video
    
    Args:
        frames: List of frames (BGR format)
        output_path: Path to save video
        fps: Frames per second
        show_progress: Whether to show progress bar
        
    Returns:
        Path to saved video
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get video properties
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    iterator = tqdm(frames, desc="Saving video") if show_progress else frames
    for frame in iterator:
        out.write(frame)
    
    out.release()
    return output_path

def create_comparison_grid(original_frames: List[np.ndarray], 
                         processed_frames: List[np.ndarray], 
                         timestamps: List[float],
                         num_samples: int = 4,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a comparison grid of original vs processed frames
    
    Args:
        original_frames: List of original frames
        processed_frames: List of processed frames
        timestamps: List of timestamps for frames
        num_samples: Number of frame pairs to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure with comparison grid
    """
    # Ensure we have equal numbers of frames
    assert len(original_frames) == len(processed_frames) == len(timestamps), "Frames and timestamps must have same length"
    
    # Select evenly spaced samples
    indices = np.linspace(0, len(original_frames) - 1, num_samples, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    
    for i, idx in enumerate(indices):
        # Original frame
        axes[i, 0].imshow(cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original (t={timestamps[idx]:.2f}s)")
        axes[i, 0].axis('off')
        
        # Processed frame
        axes[i, 1].imshow(cv2.cvtColor(processed_frames[idx], cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Deblurred (t={timestamps[idx]:.2f}s)")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_frame_difference(original_frame: np.ndarray, processed_frame: np.ndarray) -> plt.Figure:
    """
    Visualize difference between original and processed frame
    
    Args:
        original_frame: Original frame
        processed_frame: Processed frame
        
    Returns:
        Matplotlib figure with visualization
    """
    # Convert to RGB for display
    orig_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Calculate difference
    diff = cv2.absdiff(original_frame, processed_frame)
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    
    # Create heat map of difference
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_heat = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    diff_heat_rgb = cv2.cvtColor(diff_heat, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title("Original Frame")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(proc_rgb)
    axes[0, 1].set_title("Processed Frame")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(diff_rgb)
    axes[1, 0].set_title("Absolute Difference")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_heat_rgb)
    axes[1, 1].set_title("Difference Heat Map")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
