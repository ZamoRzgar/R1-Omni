# R1-Omni on Windows: Setup Guide

This guide outlines how to set up and run the R1-Omni multimodal emotion analysis framework on Windows systems.

## Requirements

- Python 3.8+ (tested with Python 3.13)
- CUDA-compatible GPU
- ~30GB disk space for all model components

## Setup Steps

### 1. Environment Setup

Create and activate a virtual environment:
```bash
python -m venv r1omni_env
r1omni_env\Scripts\activate
```

Install required dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python librosa soundfile
pip install einops moviepy==1.0.3 ipdb h5py
pip install timm safetensors sentencepiece peft accelerate
pip install decord ffmpeg-python scikit-image
```

### 2. Download Component Models

Create directories for component models:
```bash
mkdir -p C:\Users\{username}\models\bert-base-uncased
mkdir -p C:\Users\{username}\models\siglip-base-patch16-224
mkdir -p C:\Users\{username}\models\whisper-large-v3
```

Run the provided download script:
```bash
python download_models.py
```

Download BERT model:
```bash
python download_bert.py
```

### 3. Download R1-Omni Model

Download from Hugging Face:
- [R1-Omni-0.5B](https://huggingface.co/StarJiaxing/R1-Omni-0.5B)

### 4. Configure Model Paths

Update the config.json file in your downloaded R1-Omni model directory:
1. Change `mm_audio_tower` to your local whisper model path
2. Change `mm_vision_tower` to your local siglip model path

Example:
```json
"mm_audio_tower": "C:/Users/{username}/models/whisper-large-v3",
"mm_vision_tower": "C:/Users/{username}/models/siglip-base-patch16-224",
```

### 5. Run Inference

```bash
python inference.py --model_path "C:\path\to\R1-Omni-0.5B" --video_path "path\to\video.mp4" --instruct "Analyze the emotions in this video and explain your reasoning."
```

## Troubleshooting

- If you encounter path-related errors, check for remaining Linux paths in the codebase
- Windows paths should use forward slashes in Python code (C:/Users/...)
- Some third-party modules like moviepy may require specific versions
