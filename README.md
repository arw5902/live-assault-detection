# Pre-contact Detection System

A real-time assault detection system that provides early warning alerts before physical contact occurs. Built using pose estimation, optical flow analysis, and GRU-based temporal modeling.

## 🎯 Overview

This system analyzes video streams to detect assault behavior **before contact happens**, providing multi-level warnings:
- **PRE-CONTACT**: Early warning (target: 0.5-1.0s before contact)
- **HIGH**: Elevated threat level
- **CRITICAL**: Imminent contact

The system achieves 94.6% detection rate with 62.9% pre-contact warning rate and 0% false positives on safe videos.

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
- **High Accuracy**: 94.6% detection rate, 0% false positives on safe videos
- **Early Warning**: Mean lead time of 2.4 frames (0.24s) before contact
- **Pose-based Features**: 30-dimensional feature vector from body keypoints
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
- **Deployment**: Raspberry Pi 5 compatible (with Hailo-8L accelerator)

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

# Download YOLOv8-Pose model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt -O models/yolov8m-pose.pt
```

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
python -m src.evaluate ../AssaultDetection-main/holdout
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
safe_window_stride: int = 5      # Stride for safe videos
attack_window_stride: int = 1    # Stride for attack videos (preserves coherence)

# Focal Loss
focal_gamma: float = 3.0         # Focus on hard examples (2.0=standard, 3.0=aggressive)
focal_alpha: float = 0.75        # Attack class weight (higher=prioritize recall)

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
2. **Feature Extraction**: 30-dimensional features per frame
3. **Windowing**: 5-frame sliding windows (safe: stride=5, attack: stride=1)
4. **Video-level Split**: 85% train, 15% validation (no data leakage)
5. **Training**: Focal Loss optimization for 40 epochs
6. **Threshold Tuning**: Sweep [0.15-0.80] to maximize F1 score
7. **Model Saving**: Best model saved based on F1 score

### Training Command

```bash
python -m src.train
```

### Training Output

```
Dataset window balance:
  safe_videos=95 steps=10219 windows=2004 stride=5
  attack_videos=118 steps=1988 windows=1516 stride=1
  safe/attack window ratio = 1.32

Stratified video-level split (class-balanced):
  Total videos: 213
  Train videos: 181 (safe=81, attack=100)
  Val videos: 32 (safe=14, attack=18)

Train set balance:
  Total windows: 2989 (85.0% of all windows)
  Safe windows: 1702 (56.9%)
  Attack windows: 1287 (43.1%)
  Imbalance ratio: 1.32:1

Focal Loss parameters: gamma=3.00, alpha=0.75

epoch 1/40 train=0.0772 val=0.0408 best_val=0.0408
  New best F1: 0.732 at threshold 0.50 - Model saved!
epoch 5/40 train=0.0221 val=0.0273 best_val=0.0273
  Val Metrics @ thresh=0.50: Acc=0.885 Prec=0.608 Rec=0.989 F1=0.753
...
epoch 40/40 train=0.0046 val=0.0048 best_val=0.0262

Best F1 score: 0.833 at threshold 0.55 (epoch 37)
```

### Output Files

```
outputs/
├── checkpoints/
│   ├── hazard_gru.pt          # Best model weights
│   └── meta.json              # Model metadata (input_dim, best_threshold)
└── logs/
    └── train_20260204_151139.log
```

## 📊 Evaluation

### Evaluation Command

```bash
python -m src.evaluate ../AssaultDetection-main/holdout
```

### Evaluation Output

```
HOLDOUT EVALUATION - Pre-contact Detection
Holdout directory: ../AssaultDetection-main/holdout
Using threshold: 0.50 (from training optimization)

--- Processing Attack Videos ---
Processing h1.mp4... DETECTED (max_hazard=0.856)
Processing h2.mp4... DETECTED (max_hazard=0.923)
...

--- Processing Safe Videos ---
Processing s1.mp4... OK (max_hazard=0.123)
Processing s2.mp4... OK (max_hazard=0.087)
...

ATTACK VIDEOS - Multi-Level Warning Analysis
h1.mp4               | Onset@ 126 Attack@ 150 Lead=  +8 [PRE-CONTACT] | Warnings: PC@ 142 H@ 145 C@ 148
h2.mp4               | Onset@  89 Attack@ 112 Lead=  +5 [PRE-CONTACT] | Warnings: PC@ 107 H@ 110 C@  -1
...

Detection Rate: 94.6% (35/37)
False Positive Rate (safe videos): 0.0% (0/20)
False Positive Rate (pre-onset): 5.4% (2/37)

--- PRE-CONTACT Level (threshold=0.50) ---
Pre-contact Warning Rate: 62.9% (22/35 detected attacks)
  Mean Lead Time: 2.4 frames (0.24s)
  Median Lead Time: 3.0 frames (0.30s)
  Pre-contact warnings only: 8.8 frames (0.88s)

--- HIGH Level (threshold=0.60) ---
HIGH Warning Rate: 94.3% (33/35 detected attacks)
  Mean Lead Time: 3.4 frames (0.34s)

--- CRITICAL Level (threshold=0.80) ---
CRITICAL Warning Rate: 97.1% (34/35 detected attacks)
  Mean Lead Time: -0.5 frames (-0.05s)
```

## 📁 Project Structure

```
Pre-contactDetection/
├── src/
│   ├── config.py              # Configuration parameters
│   ├── model.py               # GRU model definition
│   ├── dataset.py             # Dataset building and windowing
│   ├── train.py               # Training script with Focal Loss
│   ├── evaluate.py            # Holdout evaluation
│   ├── infer.py               # Real-time inference
│   ├── pose_detector.py       # YOLOv8 pose detection wrapper
│   ├── features.py            # Feature extraction (pose + flow)
│   ├── flow.py                # Optical flow computation
│   ├── tracker.py             # Simple bounding box tracker
│   ├── pose_utils.py          # Pose keypoint utilities
│   └── video_io.py            # Video reading utilities
├── data/
│   ├── safe/                  # Training safe videos
│   └── attack/                # Training attack videos (trimmed)
├── models/
│   └── yolov8m-pose.pt       # YOLOv8-Pose weights
├── outputs/
│   ├── checkpoints/           # Model checkpoints
│   └── logs/                  # Training/evaluation logs
├── docs/
│   └── DESIGN.md             # Detailed design specification
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── .gitignore
```

## 🎯 Performance

### Current Metrics (Holdout Set)

| Metric | Value |
|--------|-------|
| Detection Rate | 94.6% (35/37) |
| Pre-contact Warning Rate | 62.9% (22/35) |
| Mean Lead Time | 2.4 frames (0.24s) |
| Median Lead Time | 3.0 frames (0.30s) |
| False Positive Rate (Safe) | 0.0% (0/20) |
| False Positive Rate (Pre-onset) | 5.4% (2/37) |

### Performance by Warning Level

| Level | Threshold | Detection Rate | Mean Lead Time |
|-------|-----------|----------------|----------------|
| PRE-CONTACT | 0.50 | 62.9% | 2.4 frames (0.24s) |
| HIGH | 0.60 | 94.3% | 3.4 frames (0.34s) |
| CRITICAL | 0.80 | 97.1% | -0.5 frames (-0.05s) |

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Input Features | 30 dimensions |
| Window Length | 5 frames (0.5s) |
| GRU Hidden Units | 64 |
| Total Parameters | ~160,995 |
| Inference Speed | ~10 FPS (CPU) |
| Model Size | <1 MB |

## 📖 Documentation

- **[DESIGN.md](docs/DESIGN.md)**: Comprehensive design specification
  - System architecture
  - Feature engineering details
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
python -m src.evaluate ../AssaultDetection-main/holdout
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
- [ ] Model quantization for edge deployment
- [ ] Real-time visualization GUI
- [ ] REST API for deployment

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{precontact_detection_2025,
  author = {Your Name},
  title = {Pre-contact Detection System for Assault Warning},
  year = {2025},
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
