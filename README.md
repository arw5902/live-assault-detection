# Real-Time Physical Threat Detection System

An automatic, real-time assault detection system for body-worn cameras. Built using pose estimation, optical flow analysis, and temporal modeling (GRU, LSTM, and Transformer).

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

### Core capabilities

The system processes video at 10 FPS, downsampled from a 30 FPS source. Each
frame yields a 59-dimensional feature vector built from body keypoints and
optical flow, with camera motion removed by a background homography before the
flow is decomposed into radial and tangential components. A recurrent network
(GRU by default, with LSTM and Transformer variants available) scores a
5-frame window, and the resulting hazard score is smoothed with an EMA and
gated by a persistence counter to give a binary THREAT or NONE output.

### Training and evaluation

Training uses focal loss to handle the roughly 3.4:1 imbalance between safe and
attack windows. The train/validation split is made at the video level and
stratified by class, so windows from one clip cannot appear in both sets.
Missing keypoints are carried forward from their last valid value during
dataset construction, and a parallel validity mask is fed to the model so it
can distinguish a real measurement from an imputed one.

## System Requirements

### Hardware
- Minimum: CPU with 4+ cores, 8GB RAM
- Recommended: NVIDIA GPU with 4GB+ VRAM, 16GB RAM
- Edge deployment: Raspberry Pi 5 + Hailo-8 AI HAT+ (M.2 NPU, ~10 Hz end-to-end inference)

### Software
- Python 3.8+
- CUDA 11.x+ (optional, for GPU acceleration)
- OpenCV 4.5+
- PyTorch 2.0+
- Ultralytics YOLOv8

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/arw5902/live-assault-detection.git
cd live-assault-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
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

No extra model download is needed - the HEF is bundled with hailo-rpi5-examples:

```bash
# Install HailoRT (follow Hailo's Pi 5 setup guide)
# The HEF is already present at:
ls /home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef

# Copy project to Pi, then run:
python -m src.infer # live camera (uses Config.for_pi() automatically)
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
python -m src.evaluate holdout/
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
├── safe/ # Safe behavior videos (trimmed)
│ ├── s1.mp4
│ ├── s2.mp4
│ └── ...
└── attack/ # Attack videos (trimmed from onset to end of violence)
 ├── a1.mp4
 ├── a2.mp4
 └── ...
```

Training attack videos must be trimmed to start from the onset of the wind-up
phase, so that every frame in the clip contains attack behavior.

#### Holdout Data Structure
```
holdout/
├── labels.json # Ground truth annotations
├── attack/ # Full attack videos (normal -> onset -> end of violence)
│ ├── h1.mp4
│ ├── h2.mp4
│ └── ...
└── safe/ # Safe behavior videos
 ├── s1.mp4
 ├── s2.mp4
 └── ...
```

#### labels.json Format
```json
{
 "h1.mp4": {
 "category": "attack",
 "onset_frame": 126, // Attack wind-up starts
 "attack_frame": 150 // Physical contact occurs
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
window_len: int = 5 # 0.5s temporal window
safe_window_stride: int = 2 # Stride for safe videos
attack_window_stride: int = 1 # Stride for attack videos (preserves coherence)

# Focal Loss
focal_gamma: float = 2.0 # Focus on hard examples
focal_alpha: float = 0.75 # Attack class weight

# GRU
gru_hidden: int = 64 # Hidden units
dropout: float = 0.25 # Dropout rate
lr: float = 1e-3 # Learning rate
batch_size: int = 64
epochs: int = 40

# Warning threshold
early_thresh: float = 0.2 # Initial THREAT threshold (tuned during training)
```

## Training

### Training Pipeline

1. **Data Loading**: Videos processed at 10 FPS with pose detection
2. **Feature Extraction**: 59-dimensional features per frame (+ 59 validity masks = 118-dim input)
3. **Windowing**: 5-frame sliding windows (safe: stride=2, attack: stride=1)
4. **Video-level Split**: 80% train, 20% validation (no data leakage)
5. **Training**: Focal Loss optimization for 40 epochs
6. **Threshold Tuning**: Sweep [0.30-0.70] to maximize F1 score
7. **Model Saving**: Best model saved based on F1 score with timestamp filename

### Training Command

```bash
python -m src.train
```

### Training Output

Representative training output:

```
Training 2026-04-06 21:58:11  log -> outputs/logs/train_20260406_215811.log
seed=42  cuda=deterministic

Dataset window balance:
  safe_videos=161 steps=17117 windows=8276 stride=2
  attack_videos=175 steps=3102 windows=2402 stride=1
  safe/attack window ratio = 3.45

Stratified video-level split (class-balanced):
  Total videos: 336
  Train videos: 269 (safe=129, attack=140)
  Val videos: 67 (safe=32, attack=35)

Train set balance:
  Total windows: 8542 (80.0% of all windows)
  Safe windows: 6620 (77.5%)
  Attack windows: 1922 (22.5%)
  Imbalance ratio: 3.44:1

Val set balance:
  Total windows: 2136 (20.0% of all windows)
  Safe windows: 1378 (64.5%)
  Attack windows: 758 (35.5%)

Class imbalance ratio (safe/attack windows): 3.44

Features: input_dim=118 (features+masks)  raw=59  FEATURE_NAMES=59
Model: GRU  |  Parameters: 35,393
Focal Loss: gamma=2.0, alpha=0.75
  (class imbalance ratio: 3.44:1)

epoch 1/40 train=0.0326 val=0.0239 best_val=0.0239
epoch 5/40 train=0.0158 val=0.0158 best_val=0.0158
  * New best - F1=0.921 Rec=0.925 FP%=2.2% @ thresh=0.65 [best-F1] - Model saved!
  Val @ thresh=0.65 [best-F1]: Acc=0.967 Prec=0.917 Rec=0.925 F1=0.921 FP%=2.2%
...
epoch 40/40 train=0.0030 val=0.0195 best_val=0.0129

Training done 2026-04-06 22:04:42
Best validation loss: 0.0129
Best F1: 0.921  Recall: 0.925  at threshold 0.65 (epoch 5)
Model saved to: outputs/checkpoints/hazard_gru_20260406_215811.pt (selected by best F1)

Generating PR and ROC curves...

 Threshold comparison (val set, n_neg=1378):
 thresh Prec Rec F1 FP% note
 ----------------------------------------------
 0.30 0.500 0.986 0.664 25.76%
 0.35 0.559 0.983 0.713 20.25%
 0.40 0.630 0.981 0.767 15.02%
 0.45 0.693 0.978 0.811 11.32%
 0.50 0.773 0.972 0.861 7.47%
 0.55 0.825 0.958 0.887 5.30%
 0.60 0.881 0.950 0.914 3.34%
 0.65 0.917 0.925 0.921 2.18% * selected (best F1)
 0.70 0.931 0.856 0.891 1.67%

Computing feature importance...

Top 5 Most Important Features:
 1. expansion_proximity: 24.7% (F1 drop: +0.2999)
 2. torso_height_px: 20.8% (F1 drop: +0.2526)
 3. divergence_torso: 9.9% (F1 drop: +0.1201)
 4. divergence_lower: 7.3% (F1 drop: +0.0886)
 5. acceleration_proximity: 7.1% (F1 drop: +0.0867)
```

Note that the threshold and the saved checkpoint are both selected on the
validation set, so the validation figures above are optimistic. Use the
holdout results below when reporting performance.

### Output Files

```
outputs/
├── checkpoints/
│ ├── hazard_gru_YYYYMMDD_HHMMSS.pt # Best model weights (timestamped)
│ └── meta.json # Training metadata and threshold
├── plots/
│ ├── pr_curve_YYYYMMDD_HHMMSS.png # Precision-Recall curve
│ └── roc_curve_YYYYMMDD_HHMMSS.png # ROC curve
└── logs/
 ├── train_YYYYMMDD_HHMMSS.log
 ├── evaluate_YYYYMMDD_HHMMSS.log
 └── feature_importance_YYYYMMDD_HHMMSS.json
```

## Evaluation

### Evaluation Command

```bash
python -m src.evaluate holdout/
```

### Evaluation Output

Representative evaluation output:

```
================================================================================
HOLDOUT EVALUATION
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
ATTACK VIDEOS - Threat Detection Analysis
================================================================================
h13.mp4 | Onset@ 0 Attack@ 32 Detect@ 18 Lead= +14 [LEAD]
h14.mp4 | Onset@ 140 Attack@ 164 Detect@ 159 Lead= +5 [LEAD]
h17.mp4 | Onset@ 78 Attack@ 93 Detect@ 90 Lead= +3 [LEAD]
h19.mp4 | Onset@ 9 Attack@ 38 Detect@ 39 Lead= -1 [LATE]
h28.mp4 | Onset@ 53 Attack@ 72 Detect@ 78 Lead= -6 [LATE]
...

Detection Rate : 100.0% (60/60)
Missed Rate : 0.0% (0/60)
FP (pre-onset) : 0.0% (0/60)

--- THREAT Detection (threshold=0.65) ---
Pre-contact Rate : 76.7% (46/60 detected attacks)
 Mean Lead Time : 7.9 frames (0.26s)
 Median Lead Time : 5.0 frames (0.17s)

================================================================================
SAFE VIDEOS
================================================================================

False Positive Rate (safe) : 0.0% (0/27)
True Negative Rate : 100.0% (27/27)

================================================================================
OVERALL SUMMARY
================================================================================
Total Videos : 87 (attacks=60 safe=27)

Threshold (THREAT) : 0.65

Attack Detection : 100.0%
FP (safe) : 0.0%

Detection Performance:
 Pre-contact : 76.7% | Mean Lead: 7.9 frames (0.26s)
================================================================================
```

## Project Structure

```
live-assault-detection/
├── src/
│ ├── config.py # Configuration parameters + FEATURE_NAMES
│ ├── model.py # HazardGRU / HazardLSTM / HazardTransformer
│ ├── dataset.py # Dataset building and windowing
│ ├── train.py # Training script with Focal Loss
│ ├── evaluate.py # Holdout evaluation
│ ├── infer.py # Real-time inference
│ ├── pose_detector.py # YOLOv8 pose detection - dual backend (Ultralytics PC / Hailo Pi)
│ ├── features.py # Feature extraction (pose + flow, 59-dim)
│ ├── flow.py # Optical flow computation
│ ├── tracker.py # Last-known bounding box holder
│ ├── pose_utils.py # Pose keypoint utilities
│ ├── feature_importance.py # Permutation importance evaluation
│ ├── _plot.py # Shared PR/ROC plotting helpers (used by train.py, evaluate.py)
│ ├── utils.py # Seeding and stdout logging helpers
│ ├── video_io.py # Video reading utilities
│ └── copy_mp4s.py # Helper for collecting dataset clips
├── data/
│ ├── safe/ # Training safe videos
│ └── attack/ # Training attack videos (trimmed)
├── holdout/
│ ├── labels.json # Ground truth annotations
│ ├── safe/ # Holdout safe videos
│ └── attack/ # Holdout attack videos
├── models/
│ └── yolov8m-pose.pt # YOLOv8-Pose weights (PC training/inference)
│ # Pi uses: /home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef
├── outputs/
│ ├── checkpoints/ # Model checkpoints (timestamped)
│ ├── plots/ # PR and ROC curves
│ └── logs/ # Training/evaluation/importance logs
├── docs/
│ ├── DESIGN.md # Detailed design specification
│ ├── PIPELINE_DIAGRAM.md # Pipeline figure guide
│ ├── FEATURE_IMPORTANCE_GUIDE.md # Feature-importance tooling notes
│ └── REPRODUCIBILITY.md # Reproducibility notes
├── README.md # This file
├── requirements.txt # Python dependencies
└── .gitignore
```

## Performance

### Current Metrics (Holdout Set - 87 videos: 60 attack, 27 safe)

Evaluated at threshold 0.65, tuned on the validation set during training.
These figures come from `src/evaluate.py`, which thresholds the raw per-frame
hazard score. The deployed path in `src/infer.py` additionally applies EMA
smoothing and persistence gating, so its lead times are shorter.

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

Pre-contact detection means the alert fired before the labeled `attack_frame`,
the first frame of physical contact. With 27 safe videos, a 0.0% false positive
rate carries a 95% upper confidence bound near 13%.

### Validation Set Best Metrics (epoch 5, threshold=0.65)

Both the checkpoint and the threshold were selected against this set, so these
numbers describe model selection rather than held-out performance.

| Accuracy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| 0.967    | 0.917     | 0.925  | **0.921** |

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Input Features | 59 dimensions (+ 59 validity masks = 118-dim input) |
| Window Length | 5 frames (0.5s) |
| GRU Hidden Units | 64 |
| Total Parameters | 35,393 (GRU) |
| Inference Speed | ~10 Hz end-to-end (Pi 5 + Hailo-8) |
| Model Size | <1 MB |

## Documentation

- [DESIGN.md](docs/DESIGN.md): system architecture, feature engineering, model
 architecture, training methodology, and algorithm details.
- [PIPELINE_DIAGRAM.md](docs/PIPELINE_DIAGRAM.md): end-to-end pipeline figure.
- [FEATURE_IMPORTANCE_GUIDE.md](docs/FEATURE_IMPORTANCE_GUIDE.md): how the
 permutation-importance tooling works and how to read its output.
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): seeding and sources of run-to-run
 variation.

## Development

### Running the pipeline

The repository has no automated test suite. The commands below are the manual
checks used during development.

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

### Adding a feature

1. Add the computation in `src/features.py` and extend the returned vector.
2. Add a matching entry to `FEATURE_NAMES` in `src/config.py`, and update the
   length assertions in both files.
3. Retrain. The model input dimension is derived from the data.

### Changing the model

1. Edit the relevant class in `src/model.py` (`HazardGRU`, `HazardLSTM`, or `HazardTransformer`)
2. Update hyperparameters in `src/config.py`
3. Retrain with `python -m src.train --model {gru|lstm|transformer}` (default: `gru`)

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

- [ ] Multi-person tracking and detection
- [ ] Transfer learning from larger datasets
- [x] Edge deployment on Raspberry Pi 5 with Hailo-8 NPU
- [ ] IMU-based ego-motion compensation
- [ ] Real-time visualization GUI

## Citation

If you use this project in your research, please cite:

```bibtex
@software{live_assault_detection_2026,
 author = {A.R.W.},
 title = {Real-Time Physical Threat Detection System},
 year = {2026},
 url = {https://github.com/arw5902/live-assault-detection}
}
```

## License

MIT. A LICENSE file has not been added to the repository yet.

## Acknowledgments

- YOLOv8 by Ultralytics for pose estimation
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools

## Contact

For questions or issues, please open an issue on GitHub.
