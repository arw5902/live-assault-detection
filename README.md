# Pre-contact Detection System

A real-time assault detection system that provides early warning alerts before physical contact occurs. Built using pose estimation, optical flow analysis, and GRU-based temporal modeling.

## 🎯 Overview

This system analyzes video streams to detect assault behavior **before contact happens**, providing multi-level warnings:
- **PRE-CONTACT**: Early warning (target: 0.05-0.1s before contact)
- **HIGH**: Elevated threat level
- **CRITICAL**: Imminent contact

The system achieves 96.7% detection rate with 55.9% pre-contact warning rate and 14.8% false positive rate on safe videos.

## 📋 Table of Contents

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

## ✨ Features

### Core Capabilities
- **Real-time Detection**: Processes video at 10 FPS (downsampled from 30 FPS)
- **Multi-level Alerts**: Three escalating warning levels
- **High Accuracy**: 96.7% detection rate, 14.8% false positive rate on safe videos
- **Early Warning**: 55.9% of detected attacks warned before contact (mean lead 0.8 frames)
- **Pose-based Features**: 51-dimensional feature vector from body keypoints and optical flow
- **Optical Flow Analysis**: Radial/tangential motion decomposition
- **Temporal Modeling**: GRU neural network for sequence analysis

### Technical Features
- **Focal Loss**: Handles class imbalance and emphasizes hard examples
- **Video-level Train/Val Split**: Prevents data leakage
- **Stratified Splitting**: Maintains class balance across varying video lengths
- **Carry-forward Imputation**: Handles missing keypoints gracefully
- **EMA Smoothing**: Reduces detection flicker
- **Persistence Logic**: Reduces false alarms

## 🖥️ System Requirements

### Hardware
- **Minimum**: CPU with 4+ cores, 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM, 16GB RAM
- **Edge Deployment**: Raspberry Pi 5 + Hailo-8 AI HAT+ (M.2 NPU, ~3–4 Hz inference)

### Software
- Python 3.8+
- CUDA 11.x+ (optional, for GPU acceleration)
- OpenCV 4.5+
- PyTorch 2.0+
- Ultralytics YOLOv8

## 🚀 Installation

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

## 🏃 Quick Start

### Inference on Video

```bash
python -m src.infer path/to/video.mp4
```

Output:
```
t=0.50s hazard=0.123 level=NONE
t=0.60s hazard=0.234 level=NONE
t=0.70s hazard=0.456 level=PRE-CONTACT
t=0.80s hazard=0.678 level=HIGH
t=0.90s hazard=0.823 level=CRITICAL
```

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

## 📚 Usage

### Data Preparation

#### Training Data Structure
```
data/
├── safe/           # Safe behavior videos (trimmed)
│   ├── s1.mp4
│   ├── s2.mp4
│   └── ...
└── attack/         # Attack videos (trimmed from onset to contact)
    ├── a1.mp4
    ├── a2.mp4
    └── ...
```

**Important**: Training attack videos must be **trimmed to start from onset** (wind-up phase). Every frame should contain attack behavior.

#### Holdout Data Structure
```
holdout/
├── labels.json     # Ground truth annotations
├── attack/         # Full attack videos (normal → onset → contact)
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
safe_window_stride: int = 3      # Stride for safe videos
attack_window_stride: int = 1    # Stride for attack videos (preserves coherence)

# Focal Loss
focal_gamma: float = 4.0         # Focus on hard examples (was 3); higher=more aggressive
focal_alpha: float = 0.25        # Attack class weight (was 0.75)

# GRU
gru_hidden: int = 64             # Hidden units
dropout: float = 0.25            # Dropout rate
lr: float = 1e-3                 # Learning rate
batch_size: int = 64
epochs: int = 40

# Warning thresholds
early_thresh: float = 0.2        # Initial PRE-CONTACT threshold (tuned during training)
high_thresh: float = 0.60        # HIGH warning
critical_thresh: float = 0.80    # CRITICAL warning
```

## 🎓 Training

### Training Pipeline

1. **Data Loading**: Videos processed at 10 FPS with pose detection
2. **Feature Extraction**: 51-dimensional features per frame (+ 51 validity masks = 102-dim input)
3. **Windowing**: 5-frame sliding windows (safe: stride=3, attack: stride=1)
4. **Video-level Split**: 85% train, 15% validation (no data leakage)
5. **Training**: Focal Loss optimization for 40 epochs
6. **Threshold Tuning**: Sweep [0.15-0.80] to maximize F1 score
7. **Model Saving**: Best model saved based on F1 score with timestamp filename

### Training Command

```bash
python -m src.train
```

### Training Output

From `outputs/logs/train_20260214_151253.log`:

```
================================================================================
FEATURE CONFIGURATION VERIFICATION
================================================================================
Model input dimension: 102 (features + masks)
Number of raw features: 51
Number of FEATURE_NAMES: 51
✓ Feature names match actual features
================================================================================

Focal Loss parameters: gamma=4.00, alpha=0.250
  (gamma: higher=more focus on hard examples)
  (alpha: higher=prioritize attack class/recall)
  (class imbalance ratio: 1.45:1)

Dataset window balance:
  safe_videos=95 steps=10219 windows=3308 stride=3
  attack_videos=149 steps=2826 windows=2230 stride=1
  safe/attack window ratio = 1.48

Stratified video-level split (class-balanced):
  Total videos: 243
  Train videos: 203 (safe=78, attack=125)
  Val videos: 40 (safe=17, attack=23)

Train set balance:
  Total windows: 4622 (83.5% of all windows)
  Safe windows: 2736 (59.2%)
  Attack windows: 1886 (40.8%)
  Imbalance ratio: 1.45:1

epoch 1/40 train=0.0132 val=0.0089 best_val=0.0089
epoch 4/40 train=0.0060 val=0.0069 best_val=0.0069
  New best F1: 0.896 at threshold 0.50 - Model saved!
  Val Metrics @ thresh=0.50: Acc=0.924 Prec=0.920 Rec=0.872 F1=0.896
epoch 9/40 train=0.0031 val=0.0054 best_val=0.0054
  New best F1: 0.921 at threshold 0.55 - Model saved!
  Val Metrics @ thresh=0.55: Acc=0.942 Prec=0.940 Rec=0.904 F1=0.921
...
epoch 29/40 train=0.0009 val=0.0095 best_val=0.0054
  New best F1: 0.936 at threshold 0.55 - Model saved!
epoch 30/40 train=0.0009 val=0.0100 best_val=0.0054
  Val Metrics @ thresh=0.55: Acc=0.953 Prec=0.952 Rec=0.922 F1=0.936
...
epoch 40/40 train=0.0005 val=0.0163 best_val=0.0054

================================================================================
Training completed at 2026-02-14 15:36:45
Best F1 score: 0.936 at threshold 0.55 (epoch 30)
Model saved to: outputs/checkpoints/hazard_gru_20260214_151253.pt
================================================================================
```

### Output Files

```
outputs/
├── checkpoints/
│   └── hazard_gru_YYYYMMDD_HHMMSS.pt  # Best model weights (timestamped)
└── logs/
    ├── train_YYYYMMDD_HHMMSS.log
    ├── evaluate_YYYYMMDD_HHMMSS.log
    └── feature_importance_YYYYMMDD_HHMMSS.json
```

## 📊 Evaluation

### Evaluation Command

```bash
python -m src.evaluate holdout
```

### Evaluation Output

From `outputs/logs/evaluate_20260215_125032.log` (HIGH/CRITICAL levels omitted):

```
================================================================================
HOLDOUT EVALUATION - Pre-contact Detection
Holdout directory: holdout
Using threshold: 0.55 (from training optimization)
================================================================================

--- Processing Attack Videos ---
Processing h13.mp4...        DETECTED (max_hazard=0.866)
Processing h14.mp4...        DETECTED (max_hazard=0.805)
Processing h15.mp4...        DETECTED (max_hazard=0.724)
...
Processing h133_part7.mp4... MISSED   (max_hazard=0.502)
Processing h133_part8.mp4... DETECTED (max_hazard=0.832)
Processing h133_part9.mp4... DETECTED (max_hazard=0.762)

--- Processing Safe Videos ---
Processing h130_part3.mp4... FP (max_hazard=0.623)
Processing h126.mp4...       OK (max_hazard=0.335)
Processing h7.mp4...         OK (max_hazard=0.463)
Processing h114.mp4...       FP (max_hazard=0.646)
Processing h115.mp4...       FP (max_hazard=0.600)
Processing h1_part1.mp4...   FP (max_hazard=0.880)
...

================================================================================
ATTACK VIDEOS - Multi-Level Warning Analysis
================================================================================
h13.mp4    | Onset@   0 Attack@  32 Lead=  +5 [PRE-CONTACT] | Warnings: PC@  27
h14.mp4    | Onset@ 140 Attack@ 164 Lead=  +2 [PRE-CONTACT] | Warnings: PC@ 162
h17.mp4    | Onset@  78 Attack@  93 Lead=  -6 [LATE]        | Warnings: PC@  99
h20.mp4    | Onset@  15 Attack@  36 Lead=  +0 [ON-TIME]     | Warnings: PC@  36
h23.mp4    | Onset@  57 Attack@  74 Lead= +14 [PRE-CONTACT] | Warnings: PC@  60
h130_part10.mp4 | FP @ 90 (before onset@ 156) first=0.692 max=0.875
h133_part7.mp4  | MISSED (max_hazard=0.502)
...

Detection Rate: 96.7% (59/61)
Missed Rate: 1.6% (1/61)
False Positives (pre-onset detections): 1.6% (1/61)

--- PRE-CONTACT Level (threshold=0.55) ---
Pre-contact Warning Rate: 55.9% (33/59 detected attacks)
  Mean Lead Time: 0.8 frames (0.03s)
  Median Lead Time: 2.0 frames (0.07s)
  Pre-contact warnings only: 7.4 frames (0.25s)

================================================================================
SAFE VIDEOS
================================================================================
h130_part3.mp4  | FP @ frame 189 (max_hazard=0.623)
h114.mp4        | FP @ frame  45 (max_hazard=0.646)
h115.mp4        | FP @ frame 120 (max_hazard=0.600)
h1_part1.mp4    | FP @ frame  84 (max_hazard=0.880)

False Positive Rate (safe videos): 14.8% (4/27)
True Negative Rate: 85.2% (23/27)
```

## 📁 Project Structure

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
│   ├── features.py            # Feature extraction (pose + flow, 51-dim)
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
│   └── logs/                  # Training/evaluation/importance logs
├── docs/
│   └── DESIGN.md              # Detailed design specification
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── .gitignore
```

## 🎯 Performance

### Current Metrics (Holdout Set — 88 videos: 61 attack, 27 safe)

Evaluated at threshold **0.55** (auto-tuned during training).

#### Attack Detection (61 attack videos)

| Metric | Value |
|--------|-------|
| Detection Rate | **96.7%** (59/61) |
| Missed Rate | 1.6% (1/61) |
| False Positives (pre-onset detections) | 1.6% (1/61) |

#### Safe Video False Positive Rate (27 safe videos)

| Metric | Value |
|--------|-------|
| False Positive Rate | **14.8%** (4/27) |
| True Negative Rate | 85.2% (23/27) |

#### Pre-Contact Detection (of 59 detected attacks)

| Metric | Value |
|--------|-------|
| Pre-contact Detection Rate | **55.9%** (33/59) |
| Mean Lead Time | 0.8 frames (0.03s) |
| Median Lead Time | 2.0 frames (0.07s) |

*Pre-contact detection = alert triggered before the labeled `attack_frame` (first physical contact).*

### Validation Set Best Metrics (epoch 30, threshold=0.55)

| Accuracy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| 0.953    | 0.952     | 0.922  | **0.936** |

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Input Features | 51 dimensions (+ 51 validity masks = 102-dim input) |
| Window Length | 5 frames (0.5s) |
| GRU Hidden Units | 64 |
| Total Parameters | ~24K |
| Inference Speed | ~10 FPS (CPU) |
| Model Size | <1 MB |

## 📖 Documentation

- **[DESIGN.md](docs/DESIGN.md)**: Comprehensive design specification
  - System architecture
  - Feature engineering details (51 features + 51 validity masks)
  - Model architecture
  - Training methodology
  - Algorithm details

## 🛠️ Development

### Running Tests

```bash
# Test inference on sample video
python -m src.infer data/attack/sample.mp4

# Test training on small dataset
python -m src.train

# Test evaluation
python -m src.evaluate holdout
```

### Adding New Features

1. Add feature computation in `src/features.py`
2. Update feature dimension in returned tuple
3. Retrain model (feature dimension will auto-adjust)

### Customizing Model

1. Edit `HazardGRU` class in `src/model.py`
2. Update hyperparameters in `src/config.py`
3. Retrain model

## 🐛 Troubleshooting

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
- Increase PRE-CONTACT threshold in `config.py`
- Increase `early_persist` value for more stability
- Retrain with more diverse safe videos

**Q: GPU not being used**
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

## 🔮 Future Improvements

- [ ] Attention mechanism for improved temporal modeling
- [ ] Bidirectional GRU for better context
- [ ] Multi-person tracking and detection
- [ ] Audio features integration
- [ ] Transfer learning from larger datasets
- [x] Edge deployment on Raspberry Pi 5 with Hailo-8 NPU
- [ ] Real-time visualization GUI
- [ ] REST API for deployment

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{precontact_detection_2025,
  author = {Your Name},
  title = {Pre-contact Detection System for Assault Warning},
  year = {2026},
  url = {https://github.com/yourusername/Pre-contactDetection}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for pose estimation
- PyTorch team for the deep learning framework
- OpenCV community for computer vision tools

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: February 2026
