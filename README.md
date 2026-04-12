# Pre-contact Detection System

An automatic, real-time assault detection system for body-worn cameras. Built using pose estimation, optical flow analysis, and GRU-based temporal modeling.

## Overview

This system analyzes video streams to automatically detect assault behavior in real time, providing a binary **THREAT / NONE** classification with EMA-smoothed hazard scoring and persistence gating.

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Documentation](#documentation)
- [License](#license)

## Features

### Core Capabilities
- **Real-time Detection**: Processes video at 10 FPS (downsampled from 30 FPS)
- **Binary Threat Detection**: Single THREAT level with EMA smoothing and persistence gating
- **Pose-based Features**: 59-dimensional feature vector from body keypoints and optical flow
- **Optical Flow Analysis**: Camera-compensated radial/tangential motion decomposition
- **Temporal Modeling**: GRU neural network for sequence analysis

### Technical Features
- **Focal Loss**: Handles class imbalance and emphasizes hard examples
- **Video-level Train/Val Split**: Prevents data leakage
- **Stratified Splitting**: Maintains class balance across varying video lengths
- **Carry-forward Imputation**: Handles missing keypoints gracefully
- **EMA Smoothing**: Reduces detection flicker
- **Persistence Logic**: Reduces false alarms

## System Requirements

### Hardware
- **Minimum**: CPU with 4+ cores, 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM, 16GB RAM
- **Edge Deployment**: Raspberry Pi 5 + Hailo-8 AI HAT+ (M.2 NPU, ~10-12 Hz inference)

### Software
- Python 3.8+
- CUDA 11.x+ (optional, for GPU acceleration)
- OpenCV 4.5+
- PyTorch 2.0+
- Ultralytics YOLOv8

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Pre-contactDetection.git
cd Pre-contactDetection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Models

```bash
# Create models directory
mkdir -p models

# Download YOLOv8-Pose model (PC / Ultralytics backend)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt -O models/yolov8m-pose.pt
```

### 5. Raspberry Pi 5 Setup (Hailo-8 NPU)

No extra model download is needed — the HEF is bundled with hailo-rpi5-examples:

```bash
# Install HailoRT (follow Hailo's Pi 5 setup guide)
# The HEF is already present at:
ls /home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef

# Copy project to Pi, then run:
python -m src.infer            # live camera (uses Config.for_pi() automatically)
```

The `pose_backend` switches automatically: `Config.for_pi()` selects the Hailo NPU; `Config.for_pc()` selects Ultralytics. See [Section 2.4 of DESIGN.md](docs/DESIGN.md#24-raspberry-pi-5-deployment-hailo-8-npu) for full technical details.

## Quick Start

### Training

```bash
# Prepare dataset (see Data Preparation section)
# Train model
python -m src.train
```

### Evaluation

```bash
python -m src.evaluate holdout
```

### Inference

```bash
# Inference on a video file
python -m src.infer path/to/video.mp4

# Live detection on Raspberry Pi 5 with Pi Camera Module
python -m src.infer 0 --picamera2 --pi

# Simulate live detection on Pi 5 using pre-recorded videos
python -m src.infer --eval-dir simulationvideo_dir/ --pi --simulate-live
```

## Usage

### Data Preparation

#### Training Data Structure
```
data/
├── safe/           # Safe behavior videos (trimmed)
│   ├── s1.mp4
│   ├── s2.mp4
│   └── ...
└── attack/         # Attack videos (trimmed from onset to end of violence)
    ├── a1.mp4
    ├── a2.mp4
    └── ...
```

**Important**: Training attack videos must be **trimmed to start from onset** (wind-up phase). Every frame should contain attack behavior.

#### Holdout Data Structure
```
holdout/
├── labels.json     # Ground truth annotations
├── attack/         # Full attack videos (normal → onset → end of violence)
│   ├── h1.mp4
│   ├── h2.mp4
│   └── ...
└── safe/          # Safe behavior videos
    ├── s1.mp4
    ├── s2.mp4
    └── ...
```

#### labels.json Format
```json
{
  "h1.mp4": {
    "category": "attack",
    "onset_frame": 126,    // Attack wind-up starts
    "attack_frame": 150    // Physical contact occurs
  },
  "h2.mp4": {
    "category": "attack",
    "onset_frame": 89,
    "attack_frame": 112
  }
}
```

### Configuration

Edit `src/config.py` to customize parameters:

```python
# Model window
window_len: int = 5              # 0.5s temporal window
safe_window_stride: int = 2      # Stride for safe videos
attack_window_stride: int = 1    # Stride for attack videos (preserves coherence)

# Focal Loss
focal_gamma: float = 2.0         # Focus on hard examples
focal_alpha: float = 0.75        # Attack class weight

# GRU
gru_hidden: int = 64             # Hidden units
dropout: float = 0.25            # Dropout rate
lr: float = 1e-3                 # Learning rate
batch_size: int = 64
epochs: int = 40

# Warning threshold
early_thresh: float = 0.2        # Initial THREAT threshold (tuned during training)
```

## Training

### Training Pipeline

1. **Data Loading**: Videos processed at 10 FPS with pose detection
2. **Feature Extraction**: 59-dimensional features per frame (+ 59 validity masks = 118-dim input)
3. **Windowing**: 5-frame sliding windows (safe: stride=2, attack: stride=1)
4. **Video-level Split**: 85% train, 15% validation (no data leakage)
5. **Training**: Focal Loss optimization for 40 epochs
6. **Threshold Tuning**: Sweep [0.30-0.70] to maximize F1 score
7. **Model Saving**: Best model saved based on F1 score with timestamp filename

### Training Command

```bash
python -m src.train
```

### Training Output

From `outputs/logs/train_20260406_212358.log`:

```
================================================================================
FEATURE CONFIGURATION VERIFICATION
================================================================================
Model input dimension: 118 (features + masks)
Number of raw features: 59
Number of FEATURE_NAMES: 59
✓ Feature names match actual features
================================================================================

Focal Loss: gamma=2.0, alpha=0.75
  (class imbalance ratio: 3.38:1)

Dataset window balance:
  safe_videos=161 steps=17117 windows=8276 stride=2
  attack_videos=175 steps=3102 windows=2402 stride=1
  safe/attack window ratio = 3.45

Stratified video-level split (class-balanced):
  Total videos: 335
  Train videos: 288 (safe=137, attack=151)
  Val videos: 47 (safe=24, attack=23)

Train set balance:
  Total windows: 8940 (83.7% of all windows)
  Safe windows: 6898 (77.2%)
  Attack windows: 2042 (22.8%)
  Imbalance ratio: 3.38:1

epoch 1/40 train=0.0326 val=0.0239 best_val=0.0239
epoch 4/40 train=0.0158 val=0.0158 best_val=0.0158
  ★ New best — F1=0.921 Rec=0.925 FP%=2.2% @ thresh=0.65 [best-F1] — Model saved!
epoch 6/40 train=0.0116 val=0.0129 best_val=0.0129
...
epoch 40/40 train=0.0030 val=0.0195 best_val=0.0129

================================================================================
Training completed at 2026-04-06 22:04:42
Best validation loss: 0.0129
Best F1: 0.921  Recall: 0.925  at threshold 0.65 (epoch 5)
Model saved to: outputs/checkpoints/hazard_gru_20260406_212358.pt (selected by best F1)
================================================================================

  Threshold comparison (val set, n_neg=1378):
  thresh   Prec    Rec     F1     FP%  note
  ----------------------------------------------
    0.30  0.500  0.986  0.664  25.76%
    0.35  0.559  0.983  0.713  20.25%
    0.40  0.630  0.981  0.767  15.02%
    0.45  0.693  0.978  0.811  11.32%
    0.50  0.773  0.972  0.861   7.47%
    0.55  0.825  0.958  0.887   5.30%
    0.60  0.881  0.950  0.914   3.34%
    0.65  0.917  0.925  0.921   2.18%  ★ selected (best F1)
    0.70  0.931  0.856  0.891   1.67%

Top 5 Most Important Features:
  1. expansion_proximity: 24.7% (F1 drop: +0.2999)
  2. torso_height_px: 20.8% (F1 drop: +0.2526)
  3. divergence_torso: 9.9% (F1 drop: +0.1201)
  4. divergence_lower: 7.3% (F1 drop: +0.0886)
  5. acceleration_proximity: 7.1% (F1 drop: +0.0867)
```

### Output Files

```
outputs/
├── checkpoints/
│   ├── hazard_gru_YYYYMMDD_HHMMSS.pt  # Best model weights (timestamped)
│   └── meta.json                       # Training metadata and threshold
├── plots/
│   ├── pr_curve_YYYYMMDD_HHMMSS.png    # Precision-Recall curve
│   └── roc_curve_YYYYMMDD_HHMMSS.png   # ROC curve
└── logs/
    ├── train_YYYYMMDD_HHMMSS.log
    ├── evaluate_YYYYMMDD_HHMMSS.log
    └── feature_importance_YYYYMMDD_HHMMSS.json
```

## Evaluation

### Evaluation Command

```bash
python -m src.evaluate holdout
```

### Evaluation Output

From `outputs/logs/evaluate_20260406_221217.log`:

```
================================================================================
HOLDOUT EVALUATION - Pre-contact Detection
Holdout directory: holdout/
Using threshold: 0.65 (from training optimization)
================================================================================

--- Processing Attack Videos ---
Processing h13.mp4... DETECTED (max_hazard=0.872)
Processing h14.mp4... DETECTED (max_hazard=0.819)
Processing h15.mp4... DETECTED (max_hazard=0.884)
...

--- Processing Safe Videos ---
Processing h130_part7.mp4... OK (max_hazard=0.520)
Processing h126.mp4... OK (max_hazard=0.332)
Processing h127_part6.mp4... OK (max_hazard=0.109)
...

================================================================================
ATTACK VIDEOS — Threat Detection Analysis
================================================================================
h13.mp4              | Onset@   0 Attack@  32 Detect@  18 Lead= +14 [LEAD]
h14.mp4              | Onset@ 140 Attack@ 164 Detect@ 159 Lead=  +5 [LEAD]
h17.mp4              | Onset@  78 Attack@  93 Detect@  90 Lead=  +3 [LEAD]
h19.mp4              | Onset@   9 Attack@  38 Detect@  39 Lead=  -1 [LATE]
h28.mp4              | Onset@  53 Attack@  72 Detect@  78 Lead=  -6 [LATE]
...

Detection Rate : 100.0% (60/60)
Missed Rate    : 0.0% (0/60)
FP (pre-onset) : 0.0% (0/60)

--- THREAT Detection (threshold=0.65) ---
Pre-contact Rate   : 76.7% (46/60 detected attacks)
  Mean Lead Time   : 7.9 frames (0.26s)
  Median Lead Time : 5.0 frames (0.17s)

================================================================================
SAFE VIDEOS
================================================================================

False Positive Rate (safe) : 0.0% (0/27)
True Negative Rate         : 100.0% (27/27)

================================================================================
OVERALL SUMMARY
================================================================================
Total Videos : 87  (attacks=60  safe=27)

Threshold (THREAT) : 0.65

Attack Detection  : 100.0%
FP (safe)         : 0.0%

Detection Performance:
  Pre-contact : 76.7% | Mean Lead: 7.9 frames (0.26s)
================================================================================
```

## Project Structure

```
Pre-contactDetection/
├── src/
│   ├── config.py              # Configuration parameters + FEATURE_NAMES
│   ├── model.py               # GRU model definition
│   ├── dataset.py             # Dataset building and windowing
│   ├── train.py               # Training script with Focal Loss
│   ├── evaluate.py            # Holdout evaluation
│   ├── infer.py               # Real-time inference
│   ├── pose_detector.py       # YOLOv8 pose detection — dual backend (Ultralytics PC / Hailo Pi)
│   ├── features.py            # Feature extraction (pose + flow, 59-dim)
│   ├── flow.py                # Optical flow computation
│   ├── tracker.py             # Simple bounding box tracker
│   ├── pose_utils.py          # Pose keypoint utilities
│   ├── feature_importance.py  # Permutation importance evaluation
│   ├── utils.py               # Seed management utilities
│   └── video_io.py            # Video reading utilities
├── data/
│   ├── safe/                  # Training safe videos
│   └── attack/                # Training attack videos (trimmed)
├── holdout/
│   ├── labels.json            # Ground truth annotations
│   ├── safe/                  # Holdout safe videos
│   └── attack/                # Holdout attack videos
├── models/
│   └── yolov8m-pose.pt        # YOLOv8-Pose weights (PC training/inference)
│   # Pi uses: /home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef
├── outputs/
│   ├── checkpoints/           # Model checkpoints (timestamped)
│   ├── plots/                 # PR and ROC curves
│   └── logs/                  # Training/evaluation/importance logs
├── docs/
│   └── DESIGN.md              # Detailed design specification
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Performance

### Current Metrics (Holdout Set — 87 videos: 60 attack, 27 safe)

Evaluated at threshold **0.65** (auto-tuned during training).

#### Attack Detection (60 attack videos)

| Metric | Value |
|--------|-------|
| Detection Rate | 100.0% (60/60) |
| Missed Rate | 0.0% (0/60) |
| FP (pre-onset) | 0.0% (0/60) |

#### Safe Video False Positive Rate (27 safe videos)

| Metric | Value |
|--------|-------|
| False Positive Rate | 0.0% (0/27) |
| True Negative Rate | 100.0% (27/27) |

#### Pre-Contact Detection (of 60 detected attacks)

| Metric | Value |
|--------|-------|
| Pre-contact Detection Rate | 76.7% (46/60) |
| Mean Lead Time | 7.9 frames (0.26s) |
| Median Lead Time | 5.0 frames (0.17s) |

*Pre-contact detection = alert triggered before the labeled `attack_frame` (first physical contact).*

### Validation Set Best Metrics (epoch 5, threshold=0.65)

| Accuracy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| 0.967    | 0.917     | 0.925  | **0.921** |

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Input Features | 59 dimensions (+ 59 validity masks = 118-dim input) |
| Window Length | 5 frames (0.5s) |
| GRU Hidden Units | 64 |
| Total Parameters | ~24K |
| Inference Speed | ~10-12 Hz (Pi 5 + Hailo-8) |
| Model Size | <1 MB |

## Documentation

- **[DESIGN.md](docs/DESIGN.md)**: Comprehensive design specification
  - System architecture
  - Feature engineering details (59 features + 59 validity masks)
  - Model architecture
  - Training methodology
  - Algorithm details

## Development

### Running Tests

```bash
# Train model
python -m src.train

# Holdout evaluation
python -m src.evaluate holdout/

# Test inference on a video file
python -m src.infer path/to/video.mp4

# Test inference with simulated live pacing (matches real-time Pi5 behavior)
python -m src.infer path/to/video.mp4 --pi --simulate-live

# Test inference on a pre-recorded dataset directory
python -m src.infer --eval-dir simulationvideo_dir/ --pi --simulate-live

# Live detection on Raspberry Pi 5 with Pi Camera Module
python -m src.infer 0 --picamera2 --pi
```

### Adding New Features

1. Add feature computation in `src/features.py`
2. Update feature dimension in returned tuple
3. Retrain model (feature dimension will auto-adjust)

### Customizing Model

1. Edit `HazardGRU` class in `src/model.py`
2. Update hyperparameters in `src/config.py`
3. Retrain model

## Troubleshooting

### Common Issues

**Q: "Cannot open video" error**
- Check video codec (use H.264/MP4)
- Verify file path is correct
- Ensure OpenCV is installed with video support

**Q: Low detection rate**
- Check video quality (need clear person visibility)
- Verify lighting conditions (avoid extreme darkness)
- Ensure person is reasonably sized in frame (not too far)

**Q: High false positive rate**
- Increase THREAT threshold in `config.py`
- Increase `early_persist` value for more stability
- Retrain with more diverse safe videos

**Q: GPU not being used**
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

## Future Improvements

- [ ] Attention mechanism for improved temporal modeling
- [ ] Multi-person tracking and detection
- [ ] Audio features integration
- [ ] Transfer learning from larger datasets
- [x] Edge deployment on Raspberry Pi 5 with Hailo-8 NPU
- [ ] IMU-based ego-motion compensation
- [ ] Real-time visualization GUI

## Citation

If you use this project in your research, please cite:

```bibtex
@software{precontact_detection_2025,
  author = {Your Name},
  title = {Pre-contact Detection System for Assault Warning},
  year = {2026},
  url = {https://github.com/yourusername/Pre-contactDetection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics for pose estimation
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: April 2026
