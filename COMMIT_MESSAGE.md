# Windows Compatibility for R1-Omni Multimodal Emotion Analyzer

## Summary
Added Windows compatibility to the R1-Omni multimodal emotion recognition framework. Modified hardcoded Linux paths, created helper scripts for model download, and documented setup process.

## Changes
- Updated config.json with Windows-compatible paths for component models
- Modified inference.py and humanomni_arch.py to use local model paths
- Added download_models.py to fetch required component models
- Added comprehensive setup instructions

## Detailed Changes
1. Path modifications:
   - Updated vision tower path in config.json
   - Updated audio tower path in config.json
   - Fixed BERT model paths in inference.py and humanomni_arch.py

2. Added utility scripts:
   - download_models.py: Downloads required component models (siglip, whisper)
   - download_bert.py: Downloads and saves BERT model locally

3. Dependencies documented in requirements_windows.txt

## Setup Instructions
See README_WINDOWS.md for complete setup instructions and dependency list.

## Testing
Successfully tested on Windows with sample videos.
Command: `python inference.py --model_path "C:\path\to\R1-Omni-0.5B" --video_path "path\to\video.mp4" --instruct "Analyze the emotions in this video and explain your reasoning."`
