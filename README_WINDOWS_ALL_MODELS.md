# R1-Omni: Multimodal Emotional Analysis Framework

## Overview

R1-Omni is a comprehensive multimodal framework that combines the powerful HumanOmni language model with specialized vision models to provide in-depth emotional analysis from video input. The system leverages multiple specialized models to detect and analyze both facial microexpressions and body language, providing a holistic understanding of emotional states.

## System Architecture

The R1-Omni framework consists of the following key components:

### Core Components

1. **HumanOmni Model**: A powerful language model specialized in emotional analysis and reasoning
2. **DeblurGAN-v2**: For enhancing frame quality by removing motion blur
3. **STSTNet**: Specialized in detecting facial microexpressions
4. **OpenPose**: For analyzing body posture and gestures
5. **Multimodal Fusion Module**: Combines outputs from all models for comprehensive analysis

### Data Flow

```
Video Input
   ↓
1. Frame Extraction
   ↓
2. Frame Enhancement (DeblurGAN-v2)
   ↓
3. Parallel Processing
   ├── Facial Expression Analysis (STSTNet)
   └── Body Language Analysis (OpenPose)
   ↓
4. Multimodal Fusion
   ↓
5. HumanOmni Analysis
   ↓
Comprehensive Emotional Assessment
```

## Installation and Setup

### Prerequisites

- Windows 10/11
- Python 3.10+ (3.11 recommended)
- CUDA-compatible GPU (for optimal performance)
- Git

### Environment Setup

1. **Clone the repository**:
   ```
   git clone https://github.com/ZamoRzgar/R1-Omni.git
   cd R1-Omni
   ```

2. **Set up virtual environment**:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements_windows.txt
   ```

### Model Downloads

1. **Download HumanOmni model**:
   ```
   python download_models.py --model humanomni
   ```

2. **Download BERT tokenizer** (required for text processing):
   ```
   python download_bert.py
   ```

3. **Download component models** (OpenPose, STSTNet, DeblurGAN):
   ```
   Recommend to download manualy on their github repositories
   ```

## Running the Model

### Basic Usage

The main interface for the R1-Omni system is the `omni_fusion_inference.py` script, which combines all models to analyze a video:

```
python omni_fusion_inference.py \
    --model_path "path/to/humanomni/model" \
    --video_path "path/to/video.mp4" \
    --instruct "Analyze the emotions in this video and explain your reasoning."
```

### Windows-Specific Usage

On Windows, use the following format with proper path escaping:

```
python omni_fusion_inference.py --model_path "C:\Users\username\R1-Omni-0.5B" --video_path "C:\Users\username\Videos\sample.mp4" --instruct "Analyze the emotions in this video and explain your reasoning."
```

### Advanced Options

- **Fusion Mode**: Control how the models are combined
  ```
  --fusion_mode weighted_average|confidence_based|temporal_weighted|hierarchical|adaptive
  ```

- **Frame Interval**: Adjust how many frames to sample per second
  ```
  --frame_interval 0.5
  ```

- **Save Fusion Results**: Save detailed multimodal analysis to a JSON file
  ```
  --save_fusion_results
  ```

### Example Command

```
python omni_fusion_inference.py --model_path "C:\Users\username\R1-Omni-0.5B" --video_path "C:\Users\username\Videos\sample.mp4" --instruct "Analyze the emotions in this video and explain your reasoning."
```

## Multimodal Integration Architecture

The integration architecture follows a modular design that allows each component to operate independently while sharing data through a central fusion module:

### Multimodal Fusion Strategies

The framework supports multiple fusion strategies:

1. **Weighted Average**: Simple weighted combination of signals
2. **Confidence-Based**: Weights adjusted based on detection confidence
3. **Temporal Weighted**: Considers patterns over time
4. **Hierarchical**: Prioritizes facial expressions with body language as secondary
5. **Adaptive**: Dynamically adjusts weights based on signal quality

### Enhanced Prompt Structure

The system automatically enhances prompts to the HumanOmni model with multimodal context:

```
Your original instruction
-----
Additional context from multimodal analysis:

1. EMOTIONAL SUMMARY: [primary emotions detected]

2. BODY LANGUAGE OBSERVATIONS: 
At 1.2s: [specific posture observation]
At 2.5s: [specific gesture observation]
...

3. MICROEXPRESSION ANALYSIS:
At 0.8s: [specific microexpression] (confidence score)
...

Please analyze the emotional state and intentions...
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Error: `ModuleNotFoundError: No module named 'transformers'`
   - Solution: Ensure you have activated your virtual environment and installed all dependencies:
   - We also have fall back methods when using the models if they don't load properly
      They will fallback to a simpler version of the model.
      This is to ensure the system can run even if some models don't load properly.
     ```
     pip install transformers torch opencv-python
     ```

2. **CUDA Issues**
   - Error: CUDA out of memory or CUDA not available
   - Solution: Try running with CPU only by setting environment variable:
     ```
     set CUDA_VISIBLE_DEVICES=-1
     ```

3. **Model Path Issues**
   - Error: Model weights not found
   - Solution: Ensure the path to the model is correct and contains the necessary files

4. **OpenPose/STSTNet Detection Problems**
   - Issue: "No significant body language signals detected"
   - Solution: Ensure video has good lighting, clear visibility of the person, and the person is not too far from the camera

### Performance Optimization

- For faster processing, reduce frame sampling rate with `--frame_interval 2.0`
- If experiencing memory issues, try processing shorter video clips

## License

This project contains multiple components with different licenses:

- HumanOmni: [Original License]
- OpenPose: CMU License
- DeblurGAN-v2: MIT License
- STSTNet: [Original License]

## Citation

If you use this work in your research, please cite the original papers for each component used and this repository.

## Acknowledgments

This project integrates multiple open-source models and frameworks. We thank all the original authors for their valuable contributions.