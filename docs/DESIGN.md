# Pre-contact Detection System - Design Specification

**Version**: 1.0.0
**Last Updated**: February 2026
**Author**: Pre-contact Detection Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture](#4-model-architecture)
5. [Training Methodology](#5-training-methodology)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Implementation Details](#8-implementation-details)
9. [Performance Analysis](#9-performance-analysis)
10. [Design Decisions](#10-design-decisions)
11. [Appendix](#11-appendix)

---

## 1. System Overview

### 1.1 Problem Statement

Develop a real-time system capable of detecting assault behavior **before physical contact occurs** to enable preventive intervention. The system must:

- Process video streams in real-time (≥10 FPS)
- Provide early warning alerts (target: 0.5-1.0s before contact)
- Minimize false positives on normal behavior
- Run on edge devices (Raspberry Pi 5 with Hailo-8L)

### 1.2 Solution Approach

The system combines:
1. **Pose Estimation**: YOLOv8-Pose for body keypoint detection
2. **Optical Flow**: Lucas-Kanade for motion analysis
3. **Feature Engineering**: 30-dimensional feature vector
4. **Temporal Modeling**: GRU neural network for sequence classification
5. **Multi-level Alerts**: Three escalating warning thresholds

### 1.3 System Requirements

#### Functional Requirements
- FR1: Detect assault behavior before physical contact
- FR2: Provide three warning levels (PRE-CONTACT, HIGH, CRITICAL)
- FR3: Process standard video formats (MP4, AVI)
- FR4: Support real-time inference (≥10 FPS)
- FR5: Log detections with timestamps and confidence scores

#### Non-Functional Requirements
- NFR1: Detection Rate ≥90%
- NFR2: Pre-contact Warning Rate ≥60%
- NFR3: False Positive Rate ≤10%
- NFR4: Mean Lead Time ≥0.2s
- NFR5: Model Size ≤5 MB (for edge deployment)

### 1.4 Constraints and Assumptions

#### Constraints
- Single-person scenarios only (no multi-person tracking)
- Indoor environments with adequate lighting
- Person must be visible and reasonably sized in frame
- 30 FPS input video

#### Assumptions
- Camera position is relatively static
- Person faces camera or is in profile view
- Attack behavior follows recognizable patterns (approach, wind-up, strike)
- Training data represents real-world attack scenarios

---

## 2. Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT STREAM                       │
│                        (30 FPS, BGR)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                             │
│  • Downsample to 10 FPS (frame_stride=3)                    │
│  • Convert to grayscale (for optical flow)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  POSE DETECTION (YOLOv8)                     │
│  • Detect person bounding box                                │
│  • Extract 17 COCO keypoints (x, y, conf)                   │
│  • Select largest person in frame                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRACKING & SMOOTHING                       │
│  • SingleTargetTracker: maintain bbox consistency           │
│  • Carry-forward imputation (missing keypoints)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (30-dim)                     │
│  • Pose features: angles, distances, positions (18)         │
│  • Flow features: radial/tangential motion (6)              │
│  • Context features: bbox, crop, tracking (6)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 TEMPORAL WINDOWING                           │
│  • Sliding window: 5 frames (0.5s)                          │
│  • Buffer: deque with maxlen=5                              │
│  • Input shape: [1, 5, 60] (features + mask)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  GRU MODEL INFERENCE                         │
│  • 1-layer GRU (64 hidden units)                            │
│  • Dropout (0.25)                                           │
│  • FC layer + Sigmoid                                       │
│  • Output: hazard score ∈ [0, 1]                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               POST-PROCESSING & ALERTING                     │
│  • EMA smoothing (α=0.7)                                    │
│  • Persistence logic (2 frames)                             │
│  • Multi-level thresholding:                                │
│    - PRE-CONTACT: 0.50                                      │
│    - HIGH: 0.60                                             │
│    - CRITICAL: 0.80                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ALERT OUTPUT                              │
│  • Warning level: NONE / PRE-CONTACT / HIGH / CRITICAL      │
│  • Hazard score: 0.0 - 1.0                                  │
│  • Timestamp: frame index / time                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### Training Data Flow
```
Video Files (MP4)
    ↓
Extract Frames @ 10 FPS
    ↓
Pose Detection (YOLOv8)
    ↓
Feature Extraction (30-dim per frame)
    ↓
Sliding Windows (5 frames, stride varies)
    ↓
Dataset: [N, 5, 60] windows
    ↓
Video-Level Train/Val Split (85/15)
    ↓
DataLoader (batch_size=64)
    ↓
GRU Model Training (Focal Loss)
    ↓
Threshold Tuning (maximize F1)
    ↓
Save Best Model + Threshold
```

#### Inference Data Flow
```
Video Frame (BGR)
    ↓
Pose Detection → 17 keypoints
    ↓
Tracking → Consistent bbox
    ↓
Feature Extraction → 30-dim vector
    ↓
Add to Window Buffer (deque)
    ↓
If buffer full (5 frames):
    ↓
GRU Inference → hazard score
    ↓
EMA Smoothing
    ↓
Threshold Comparison
    ↓
Alert Level Output
```

### 2.3 Module Diagram

```
┌────────────────────────────────────────────────────────┐
│                    src/config.py                        │
│  Configuration dataclass with all hyperparameters      │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                 src/pose_detector.py                    │
│  YOLOv8-Pose wrapper for person detection              │
│  • infer(frame) → {bbox, kps, det_conf}                │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   src/tracker.py                        │
│  Simple single-target tracker                          │
│  • update(bbox) → bbox, track_age, lost                │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    src/flow.py                          │
│  Optical flow utilities                                │
│  • lk_flow(): Lucas-Kanade tracking                    │
│  • radial_tangential_stats(): Motion decomposition     │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                  src/features.py                        │
│  Feature extraction from pose + flow                   │
│  • build_features() → 30-dim vector + mask             │
│  • FeatureState: carry-forward imputation              │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   src/dataset.py                        │
│  Dataset building and windowing                        │
│  • extract_sequences(): video → frames → features      │
│  • windowize(): sliding window extraction              │
│  • build_dataset_per_video(): video-level separation   │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    src/model.py                         │
│  HazardGRU neural network                              │
│  • forward(x) → hazard score                           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    src/train.py                         │
│  Training script with Focal Loss                       │
│  • Video-level stratified split                        │
│  • F1-based model saving                               │
│  • Threshold tuning                                    │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                  src/evaluate.py                        │
│  Holdout evaluation with multi-level warnings          │
│  • Lead time calculation                               │
│  • False positive detection                            │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    src/infer.py                         │
│  Real-time inference pipeline                          │
│  • EMA smoothing                                       │
│  • Persistence logic                                   │
│  • Multi-level alerting                                │
└────────────────────────────────────────────────────────┘
```

---

## 3. Feature Engineering

### 3.1 Feature Vector Overview

The system extracts a **30-dimensional feature vector** per frame, combining pose-based geometric features and optical flow-based motion features.

```
Feature Vector (30 dimensions):
├── Pose Features (18 dims)
│   ├── Arm angles (3 dims): l_arm, r_arm, min_arm
│   ├── Wrist positions (3 dims): l_wrist_raised, r_wrist_raised, max_wrist_raised
│   ├── Torso (2 dims): torso_scale, squared_up
│   ├── Spatial (4 dims): approach_speed, approach_accel, lateral_speed, min_distance
│   └── Confidence (6 dims): det_conf, kp_conf_mean, kp_conf_min, visible_kp, track_age, lost
├── Flow Features (6 dims)
│   ├── Person flow (2 dims): person_flow_mag, person_flow_dir_std
│   ├── Background flow (2 dims): bg_flow_mag, bg_flow_dir_std
│   └── Radial/Tangential (2 dims): radial_ratio, div_positive_mean
└── Context Features (6 dims)
    ├── Bbox (5 dims): log_area, dlog_area_dt, d2log_area_dt2, cx_vel, cy_vel
    └── Cropping (1 dim): any_crop_flag

Mask Vector (30 dimensions):
└── Validity flags (0 or 1) indicating if feature is valid or imputed
```

### 3.2 Detailed Feature Descriptions

#### 3.2.1 Pose Features (18 dimensions)

**Arm Angles (3 dims)**
```python
# Left arm angle: shoulder-elbow-wrist
l_arm_angle = angle(l_shoulder, l_elbow, l_wrist)  # radians

# Right arm angle: shoulder-elbow-wrist
r_arm_angle = angle(r_shoulder, r_elbow, r_wrist)  # radians

# Minimum arm angle (most bent)
min_arm_angle = min(l_arm_angle, r_arm_angle)
```
*Rationale*: Bent arms (small angle) often precede strikes. Extended arms (large angle) may indicate pushing or grabbing.

**Wrist Positions (3 dims)**
```python
# Wrist raised above shoulder?
l_wrist_raised = 1 if l_wrist.y < l_shoulder.y else 0
r_wrist_raised = 1 if r_wrist.y < r_shoulder.y else 0
max_wrist_raised = max(l_wrist_raised, r_wrist_raised)
```
*Rationale*: Raised wrists indicate wind-up for overhead strikes.

**Torso Features (2 dims)**
```python
# Torso scale: distance between shoulder midpoint and hip midpoint
torso_scale = norm(shoulder_midpoint - hip_midpoint)  # pixels

# Squared up: person facing camera (shoulders aligned horizontally)
shoulder_angle = abs(atan2(r_shoulder.y - l_shoulder.y,
                           r_shoulder.x - l_shoulder.x))
squared_up = 1 if shoulder_angle < threshold else 0
```
*Rationale*: Torso scale indicates proximity to camera. Squared-up stance is common attack posture.

**Spatial Features (4 dims)**
```python
# Approach speed: rate of change of torso_scale
approach_speed = d(torso_scale) / dt

# Approach acceleration: rate of change of approach_speed
approach_accel = d(approach_speed) / dt

# Lateral speed: horizontal movement of torso center
lateral_speed = abs(d(torso_center.x) / dt)

# Minimum distance: closest keypoint to camera (max torso_scale)
min_distance = max(torso_scale)  # proxy for proximity
```
*Rationale*: Rapid approach with acceleration indicates aggressive behavior.

**Confidence Features (6 dims)**
```python
# Detection confidence from YOLOv8
det_conf = yolo_detection_confidence  # [0, 1]

# Keypoint confidence statistics
kp_conf_mean = mean(keypoint_confidences)  # [0, 1]
kp_conf_min = min(keypoint_confidences)    # [0, 1]
visible_kp = count(kp_conf >= threshold)   # [0, 17]

# Tracking statistics
track_age = frames_since_first_detection   # ≥0
lost = frames_since_last_detection         # ≥0
```
*Rationale*: Low confidence or lost tracking may indicate occlusion or motion blur during attack.

#### 3.2.2 Flow Features (6 dimensions)

**Optical Flow Computation**
```python
# Sample points inside person bbox
pts_person = sample_points_in_box(bbox, max_points=200)

# Sample points outside bbox (background)
pts_bg = sample_points_background(frame, bbox, max_points=200)

# Lucas-Kanade optical flow
p0_person, p1_person, v_person = lk_flow(prev_gray, curr_gray, pts_person)
p0_bg, p1_bg, v_bg = lk_flow(prev_gray, curr_gray, pts_bg)
```

**Person Flow Features (2 dims)**
```python
# Flow magnitude (90th percentile to reduce noise)
person_flow_mag = percentile(norm(v_person), 90)

# Flow direction standard deviation (uniformity)
person_flow_dir_std = std(atan2(v_person[:, 1], v_person[:, 0]))
```
*Rationale*: High magnitude indicates fast movement. Low std indicates coherent motion (e.g., punch).

**Background Flow Features (2 dims)**
```python
# Background flow magnitude (camera motion)
bg_flow_mag = percentile(norm(v_bg), 90)

# Background flow direction std
bg_flow_dir_std = std(atan2(v_bg[:, 1], v_bg[:, 0]))
```
*Rationale*: High background flow may indicate camera shake or panning.

**Radial/Tangential Decomposition (2 dims)**
```python
# Decompose person flow relative to torso center
d = p0_person - torso_center  # vectors from center
r = norm(d) + 1e-6

# Radial component (toward/away from center)
radial = dot(d, v_person) / r  # positive = expanding

# Tangential component (perpendicular to radial)
tangential = abs(cross(d, v_person)) / r

# Radial ratio: expansion vs tangential motion
radial_ratio = max(radial, 0) / (max(radial, 0) + tangential + 1e-6)

# Divergence (positive = person expanding in view)
div_positive_mean = mean(max(radial, 0))
```
*Rationale*: High radial_ratio indicates person moving toward camera (approach). High divergence indicates expansion (getting closer).

#### 3.2.3 Context Features (6 dimensions)

**Bounding Box Features (5 dims)**
```python
# Log area (more stable than raw area)
log_area = log(bbox_width * bbox_height)

# Area growth rate
dlog_area_dt = d(log_area) / dt

# Area acceleration
d2log_area_dt2 = d(dlog_area_dt) / dt

# Bbox center velocity
cx_vel = d(bbox_center.x) / dt
cy_vel = d(bbox_center.y) / dt
```
*Rationale*: Growing bbox area indicates approach. Center velocity indicates lateral movement.

**Cropping Flag (1 dim)**
```python
# Is bbox touching frame edge?
left = 1 if bbox.x1 <= 0.03 * frame_width else 0
right = 1 if bbox.x2 >= 0.97 * frame_width else 0
top = 1 if bbox.y1 <= 0.03 * frame_height else 0
bottom = 1 if bbox.y2 >= 0.97 * frame_height else 0
any_crop = 1 if (left or right or top or bottom) else 0
```
*Rationale*: Person near frame edge may be partially occluded.

### 3.3 Mask Vector

Each feature has a corresponding mask bit indicating validity:

```python
mask[i] = 1  # Feature is valid (computed from confident keypoints)
mask[i] = 0  # Feature is invalid (keypoint missing, imputed from previous frame)
```

The mask is concatenated with features before feeding to GRU:
```python
input = concatenate([features, mask], axis=-1)  # [30] + [30] = [60]
```

This allows the model to learn the reliability of each feature dynamically.

### 3.4 Feature Importance (From Training)

Top 10 most important features (based on permutation importance):

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | div_positive_mean | 0.244 | Person expanding in view (approach) |
| 2 | min_arm_angle | 0.077 | Most bent arm (strike preparation) |
| 3 | approach_acceleration | 0.060 | Rapid approach acceleration |
| 4 | l_arm_angle | 0.059 | Left arm configuration |
| 5 | squared_up | 0.028 | Facing camera (attack stance) |
| 6 | r_arm_angle | 0.027 | Right arm configuration |
| 7 | flow_direction_std | 0.021 | Motion coherence |
| 8 | r_wrist_raised | 0.020 | Right wrist above shoulder |
| 9 | max_wrist_raised | 0.017 | Either wrist raised |
| 10 | l_wrist_raised | 0.017 | Left wrist above shoulder |

**Key Insights**:
- **Motion features dominate**: div_positive_mean (person expansion) is most important
- **Arm geometry matters**: All three arm angle features in top 6
- **Approach dynamics**: Acceleration more important than velocity
- **Bilateral features**: Both left and right features contribute

---

## 4. Model Architecture

### 4.1 Network Structure

```
Input: [batch, window_len, input_dim]
       [B, 5, 60]

       ↓

┌──────────────────────────────────────┐
│           GRU Layer (1 layer)        │
│  • input_size: 60                    │
│  • hidden_size: 64                   │
│  • num_layers: 1                     │
│  • batch_first: True                 │
│  • dropout: 0.0 (no effect, 1 layer) │
└──────────────────────────────────────┘
       ↓

[B, 5, 64] → Take last timestep → [B, 64]

       ↓

┌──────────────────────────────────────┐
│         Dropout (0.25)               │
└──────────────────────────────────────┘
       ↓

┌──────────────────────────────────────┐
│      Fully Connected (64 → 1)        │
└──────────────────────────────────────┘
       ↓

┌──────────────────────────────────────┐
│          Sigmoid Activation          │
└──────────────────────────────────────┘
       ↓

Output: [batch] ∈ [0, 1]
        Hazard score
```

### 4.2 Model Specification

```python
class HazardGRU(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,      # 60 (30 features + 30 mask)
            hidden_size=hidden,         # 64
            num_layers=1,
            batch_first=True
        )

        # Dropout (applied AFTER GRU for single layer)
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, T, D] = [batch, 5, 60]
        out, _ = self.gru(x)           # [B, T, H] = [B, 5, 64]
        h_last = out[:, -1, :]         # [B, H] = [B, 64]
        h_last = self.dropout(h_last)  # [B, 64]
        y = self.fc(h_last)            # [B, 1]
        y = torch.sigmoid(y)           # [B, 1] ∈ [0, 1]
        return y.squeeze(1)            # [B]
```

### 4.3 Parameter Count

```
GRU:
  • Input weights: 60 × (64 × 3) = 11,520
  • Hidden weights: 64 × (64 × 3) = 12,288
  • Biases: 64 × 3 × 2 = 384
  • Total GRU: 24,192

Dropout: 0 parameters

FC:
  • Weights: 64 × 1 = 64
  • Bias: 1
  • Total FC: 65

Total Parameters: 24,257 ≈ 24K
Model Size: <100 KB
```

### 4.4 Why GRU?

**Advantages over LSTM**:
- Fewer parameters (simpler gating mechanism)
- Faster training and inference
- Less prone to overfitting on small datasets
- Comparable performance for short sequences (5 frames)

**Advantages over Simple RNN**:
- Better gradient flow (no vanishing gradient)
- Can capture long-term dependencies
- Reset and update gates provide selective memory

**Advantages over Transformers**:
- More parameter-efficient for short sequences
- No positional encoding needed
- Lower computational cost
- Better suited for streaming inference

### 4.5 Input Representation

The model receives **concatenated features and masks**:

```python
# Features: [B, T, 30]
# Masks: [B, T, 30]
# Input: [B, T, 60] = concatenate([features, masks], axis=-1)
```

This design allows the GRU to:
1. Learn feature values from the first 30 dimensions
2. Learn feature reliability from the next 30 dimensions
3. Automatically down-weight unreliable features

Alternative approaches considered:
- **Masking in loss function**: Doesn't inform model which features are valid
- **Separate mask encoder**: Adds complexity, more parameters
- **Imputation only**: Loses information about missing data

**Chosen approach**: Concatenation is simple, effective, and allows model to learn missingness patterns.

---

## 5. Training Methodology

### 5.1 Loss Function: Focal Loss

#### 5.1.1 Motivation

Standard Binary Cross-Entropy (BCE) treats all examples equally:
```
BCE(p, y) = -y·log(p) - (1-y)·log(1-p)
```

Problems:
- **Class imbalance**: Safe examples outnumber attack examples
- **Easy examples dominate**: Model focuses on already-correct predictions
- **Hard examples neglected**: Subtle pre-contact patterns are ignored

#### 5.1.2 Focal Loss Definition

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

where:
  p_t = p   if y = 1 (attack)
      = 1-p if y = 0 (safe)

  α_t = α   if y = 1
      = 1-α if y = 0
```

**Parameters**:
- **γ (gamma)**: Focusing parameter
  - γ = 0: Equivalent to BCE
  - γ = 2: Standard (RetinaNet paper)
  - γ = 3: Aggressive (our setting)
  - Higher γ → more focus on hard examples

- **α (alpha)**: Class balance weight
  - α = 0.5: No class weighting
  - α = 0.75: Emphasize positive class (our setting)
  - Higher α → prioritize attack detection

#### 5.1.3 How Focal Loss Works

**Weighting by Difficulty**:
```
Easy example (p_t = 0.95):  (1 - 0.95)^3 = 0.000125  →  Nearly ignored
Medium (p_t = 0.7):         (1 - 0.7)^3  = 0.027     →  Some weight
Hard (p_t = 0.5):           (1 - 0.5)^3  = 0.125     →  Full weight
Very hard (p_t = 0.3):      (1 - 0.3)^3  = 0.343     →  Emphasized
```

With γ=3, easy examples (model confident and correct) are down-weighted by ~1000x compared to hard examples.

#### 5.1.4 Implementation

```python
def focal_loss(pred, target, gamma=3.0, alpha=0.75):
    """
    Focal Loss for binary classification.

    Args:
        pred: predicted probabilities [batch_size] ∈ [0, 1]
        target: ground truth labels [batch_size] ∈ {0, 1}
        gamma: focusing parameter (default: 3.0)
        alpha: positive class weight (default: 0.75)
    """
    # BCE loss without reduction
    bce = F.binary_cross_entropy(pred, target, reduction='none')

    # Compute p_t (probability of correct class)
    p_t = pred * target + (1 - pred) * (1 - target)

    # Compute α_t (weight for correct class)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    # Focal loss: α_t * (1 - p_t)^γ * BCE
    focal = alpha_t * ((1 - p_t) ** gamma) * bce

    return focal.mean()
```

#### 5.1.5 Why Focal Loss for This Problem

1. **Class Imbalance**: Safe windows outnumber attack windows (~1.3:1 after stride adjustment)
2. **Hard Example Mining**: Pre-contact behavior is subtle and easily missed
3. **Recall Priority**: Missing an attack is worse than false alarms (α=0.75 prioritizes attacks)
4. **Gradient Focus**: Forces model to learn from challenging pre-contact patterns

**Empirical Results**:
- Weighted BCE (γ=0, α=0.57): 55.9% pre-contact rate, -0.7 frame lead
- Focal Loss (γ=2, α=0.57): 60.2% pre-contact rate, +1.8 frame lead
- Focal Loss (γ=3, α=0.75): **62.9% pre-contact rate, +2.4 frame lead** ✓

### 5.2 Data Preparation

#### 5.2.1 Training Data Requirements

**Attack Videos**:
- Must be **trimmed to start from onset** (wind-up phase)
- Every frame contains attack behavior
- No "normal" frames before onset
- Trimmed at contact frame (no post-contact)

**Safe Videos**:
- Normal behavior only
- No attack wind-up or contact
- Can include diverse activities (walking, talking, gesturing)

#### 5.2.2 Window Extraction

```python
# Safe videos: stride = 5 (reduce redundancy)
safe_windows = windowize(safe_features, window_len=5, stride=5)

# Attack videos: stride = 1 (preserve temporal coherence)
attack_windows = windowize(attack_features, window_len=5, stride=1)
```

**Rationale**:
- Safe videos are redundant (similar frames)
- Attack videos have sequential behavior (can't skip frames)
- Results in ~1.3:1 safe:attack ratio (balanced)

#### 5.2.3 Video-Level Train/Val Split

**Problem**: Window-based split causes data leakage
```
Video A: [w1, w2, w3, w4, w5]
Random split: Train=[w1, w3, w5], Val=[w2, w4]
Issue: Adjacent windows share 4/5 frames → validation sees training data!
```

**Solution**: Split at video level
```python
# Group windows by source video
video_data = [(video1_windows, video1_labels),
              (video2_windows, video2_labels), ...]

# Split videos (not windows)
train_videos, val_videos = split_videos(video_data, val_fraction=0.15)

# Concatenate windows within each set
train_windows = concatenate([v for v in train_videos])
val_windows = concatenate([v for v in val_videos])
```

**Benefits**:
- No data leakage (videos are independent)
- True generalization test
- Realistic evaluation (new videos, not new windows from seen videos)

#### 5.2.4 Stratified Splitting

**Problem**: Videos have varying lengths
```
Video A: 500 frames → 496 windows (stride=1)
Video B: 50 frames → 46 windows (stride=1)

Random 15% video split might give 5% or 25% window split!
```

**Solution**: Stratified split by window count
```python
def split_by_windows(video_list, val_fraction=0.15):
    """Greedily assign videos to val set to achieve target window fraction."""
    total_windows = sum(video.num_windows for video in video_list)
    target_val_windows = int(val_fraction * total_windows)

    val_set = []
    val_count = 0

    for video in sorted_videos:
        # Add to val if closer to target
        if val_count < target_val_windows:
            if abs((val_count + video.num_windows) - target_val_windows) < \
               abs(val_count - target_val_windows):
                val_set.append(video)
                val_count += video.num_windows
            else:
                train_set.append(video)
        else:
            train_set.append(video)

    return train_set, val_set
```

**Additionally**: Split safe and attack videos separately to maintain class balance.

### 5.3 Training Procedure

#### 5.3.1 Hyperparameters

```python
# Model
input_dim = 60          # 30 features + 30 mask
hidden = 64             # GRU hidden units
dropout = 0.25          # Dropout rate

# Optimization
lr = 1e-3              # Learning rate
batch_size = 64        # Mini-batch size
epochs = 40            # Training epochs
optimizer = Adam       # Adaptive learning rate

# Loss
focal_gamma = 3.0      # Focusing parameter
focal_alpha = 0.75     # Attack class weight

# Early stopping
patience = 10          # Stop if no F1 improvement
```

#### 5.3.2 Training Loop

```python
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        x, y = batch
        pred = model(x)
        loss = focal_loss(pred, y, gamma=focal_gamma, alpha=focal_alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase (every 5 epochs)
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            x, y = batch
            pred = model(x)
            val_preds.extend(pred)
            val_labels.extend(y)

        # Threshold sweep to find best F1
        best_f1 = 0
        for threshold in [0.15, 0.20, ..., 0.80]:
            y_pred = (val_preds >= threshold)
            f1 = f1_score(val_labels, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Save if F1 improved
        if best_f1 > global_best_f1:
            save_model(model, best_threshold)
            global_best_f1 = best_f1
```

#### 5.3.3 Threshold Tuning

After training, sweep thresholds on validation set:

```python
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

for threshold in thresholds:
    y_pred = (val_predictions >= threshold)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Save best threshold to meta.json
save_metadata({"best_threshold": best_threshold})
```

**Why sweep after training?**
- Model outputs are calibrated after training
- Different thresholds optimize different metrics
- F1 maximization balances precision and recall
- Threshold is deployment parameter (can be tuned for different scenarios)

### 5.4 Model Selection

**Criteria**: Best F1 score on validation set

**Why F1 over accuracy?**
- Accuracy misleading with class imbalance
- F1 balances precision (minimize false alarms) and recall (detect attacks)
- Harmonic mean prevents optimizing one at expense of other

**Why F1 over validation loss?**
- Loss is not directly interpretable
- F1 correlates with deployment performance
- Threshold can be tuned for F1 but not for loss

**Saved artifacts**:
```
outputs/checkpoints/
├── hazard_gru.pt          # Model weights (best F1)
└── meta.json              # {"input_dim": 60, "best_threshold": 0.55}
```

---

## 6. Inference Pipeline

### 6.1 Real-time Processing

```python
# Initialize
detector = PoseDetector()
tracker = SingleTargetTracker()
model = load_model("outputs/checkpoints/hazard_gru.pt")
buffer = deque(maxlen=5)  # Sliding window
hazard_ema = 0.0          # EMA state
persist = 0               # Persistence counter

# Process video stream
for frame in video_stream:
    # 1. Pose detection
    det = detector.infer(frame)
    if det is None:
        continue

    # 2. Tracking
    bbox, track_age, lost = tracker.update(det["bbox"])

    # 3. Feature extraction
    features, mask = extract_features(frame, det, bbox)

    # 4. Add to buffer
    buffer.append(concatenate([features, mask]))

    # 5. Inference (when buffer full)
    if len(buffer) == 5:
        hazard = model(stack(buffer))

        # 6. EMA smoothing
        hazard_ema = 0.3 * hazard_ema + 0.7 * hazard

        # 7. Persistence logic
        if hazard_ema > early_thresh:
            persist += 1
        else:
            persist = max(0, persist - 1)

        # 8. Multi-level thresholding
        if hazard_ema > critical_thresh:
            alert = "CRITICAL"
        elif hazard_ema > high_thresh:
            alert = "HIGH"
        elif persist >= early_persist:
            alert = "PRE-CONTACT"
        else:
            alert = "NONE"

        # 9. Output
        print(f"t={timestamp} hazard={hazard_ema:.3f} level={alert}")
```

### 6.2 Post-processing

#### 6.2.1 EMA Smoothing

```python
# Exponential Moving Average
alpha = 0.7  # Responsiveness (higher = more responsive)
hazard_ema = (1 - alpha) * hazard_ema + alpha * hazard_current
```

**Purpose**:
- Reduce frame-to-frame jitter
- Smooth out brief spikes (noise)
- Preserve true trends

**Effect**:
```
Frame:  1    2    3    4    5    6    7    8
Raw:    0.2  0.8  0.3  0.7  0.9  0.8  0.7  0.6
EMA:    0.2  0.6  0.4  0.6  0.8  0.8  0.7  0.7
```

#### 6.2.2 Persistence Logic

```python
# Require threshold exceeded for multiple frames
if hazard_ema > early_thresh:
    persist += 1
else:
    persist = max(0, persist - 1)  # Decay, but don't go negative

# Trigger alert only if persist >= threshold
if persist >= early_persist:  # e.g., 2 frames
    alert = "PRE-CONTACT"
```

**Purpose**:
- Reduce false alarms from brief spikes
- Require sustained elevated hazard
- Allow brief dips without clearing alert

**Effect**:
```
Frame:      1   2   3   4   5   6   7   8
Hazard:     0.3 0.6 0.4 0.6 0.7 0.6 0.5 0.4
Exceed:     No  Yes No  Yes Yes Yes Yes No
Persist:    0   1   0   1   2   3   4   3
Alert:      No  No  No  No  YES YES YES YES
```

#### 6.2.3 Multi-level Thresholds

```python
thresholds = {
    "PRE-CONTACT": 0.50,   # Early warning (tuned from training)
    "HIGH": 0.60,          # Elevated threat
    "CRITICAL": 0.80       # Imminent contact
}

# Hierarchical checking (higher priority first)
if hazard_ema >= thresholds["CRITICAL"]:
    level = "CRITICAL"
elif hazard_ema >= thresholds["HIGH"]:
    level = "HIGH"
elif persist >= early_persist and hazard_ema >= thresholds["PRE-CONTACT"]:
    level = "PRE-CONTACT"
else:
    level = "NONE"
```

**Purpose**:
- Escalating warnings as threat increases
- Different response actions per level
- Gradual de-escalation as threat subsides

### 6.3 Performance Optimization

#### Inference Speed Optimization
```python
# Use model.eval() to disable dropout
model.eval()

# Disable gradient computation
with torch.no_grad():
    output = model(input)

# CPU optimization
torch.set_num_threads(4)

# Batch processing (if multiple cameras)
batch_input = stack([cam1_window, cam2_window, cam3_window])
batch_output = model(batch_input)
```

#### Memory Optimization
```python
# Use deque for automatic buffer management
buffer = deque(maxlen=5)  # Automatically removes old frames

# Process frames at reduced resolution for pose detection
frame_resized = cv2.resize(frame, (640, 480))

# Use float16 for inference (on GPU)
model.half()
input = input.half()
```

---

## 7. Evaluation Metrics

### 7.1 Detection Metrics

#### Detection Rate
```
Detection Rate = Detected Attacks / Total Attacks
                = TP / (TP + FN)
                = Recall

Current: 94.6% (35/37)
Target: ≥90%
```

#### Miss Rate
```
Miss Rate = Missed Attacks / Total Attacks
          = FN / (TP + FN)
          = 1 - Detection Rate

Current: 5.4% (2/37)
Target: ≤10%
```

#### False Positive Rate (Safe Videos)
```
FPR (Safe) = False Alarms on Safe Videos / Total Safe Videos

Current: 0.0% (0/20)
Target: ≤10%
```

#### False Positive Rate (Pre-onset)
```
FPR (Pre-onset) = Detections Before Onset / Total Attacks

Current: 5.4% (2/37)
Target: ≤10%
```

### 7.2 Warning Metrics

#### Pre-contact Warning Rate
```
Pre-contact Rate = Attacks Detected Before Contact / Total Detected Attacks
                 = Count(lead_time > 0) / TP

Current: 62.9% (22/35)
Target: ≥60%
```

#### Lead Time
```
Lead Time = Attack Frame - First Detection Frame

Positive lead time = Pre-contact warning (good)
Zero lead time = On-time detection
Negative lead time = Late detection (bad)

Current:
  Mean: 2.4 frames (0.24s)
  Median: 3.0 frames (0.30s)
  Pre-contact only: 8.8 frames (0.88s)

Target: ≥0.2s mean lead time
```

### 7.3 Multi-level Metrics

For each warning level (PRE-CONTACT, HIGH, CRITICAL):

```python
# Warning rate
warning_rate = Count(warnings at level) / Total Detected Attacks

# Mean lead time for this level
lead_time_mean = Mean(attack_frame - first_warning_frame)

# Pre-contact warnings at this level
precontact_rate = Count(lead_time > 0) / Count(warnings at level)
```

**Current Results**:

| Level | Threshold | Warning Rate | Mean Lead Time | Pre-contact % |
|-------|-----------|--------------|----------------|---------------|
| PRE-CONTACT | 0.50 | 100% (35/35) | 2.4 frames | 62.9% |
| HIGH | 0.60 | 94.3% (33/35) | 3.4 frames | 60.6% |
| CRITICAL | 0.80 | 97.1% (34/35) | -0.5 frames | 58.8% |

### 7.4 Training Metrics

#### F1 Score
```
F1 = 2 * Precision * Recall / (Precision + Recall)

Precision = TP / (TP + FP)  # Minimize false alarms
Recall = TP / (TP + FN)     # Maximize attack detection

Current: 0.833 @ threshold=0.55 (validation set)
```

#### Confusion Matrix (Validation Set, threshold=0.55)
```
                 Predicted
              Safe  Attack
Actual Safe   1650    52     Precision = 1287/(1287+52) = 0.96
      Attack    13  1274     Recall = 1287/(13+1287) = 0.99
```

---

## 8. Implementation Details

### 8.1 Carry-forward Imputation

```python
class FeatureState:
    def __init__(self, carry_steps=3):
        self.prev_value = None      # Last valid value
        self.miss_run = None        # Consecutive missing count
        self.carry_steps = carry_steps

    def impute(self, current_value, is_valid):
        """
        Impute missing values by carrying forward up to carry_steps.

        Returns:
            imputed_value: current if valid, else carried-forward
        """
        if is_valid:
            self.prev_value = current_value
            self.miss_run = 0
            return current_value
        else:
            self.miss_run += 1
            if self.miss_run <= self.carry_steps:
                return self.prev_value  # Carry forward
            else:
                return self.prev_value  # Freeze (no further extrapolation)
```

**Example**:
```
Frame:       1     2     3     4     5     6
Raw value:   5.0   NaN   NaN   NaN   8.0   NaN
Valid:       Yes   No    No    No    Yes   No
Miss run:    0     1     2     3     0     1
Imputed:     5.0   5.0   5.0   5.0   8.0   8.0
              ↑     ↑     ↑     ↑     ↑     ↑
            valid  cf1   cf2   cf3  valid  cf1

cf = carry forward (within carry_steps=3)
```

**Why carry-forward?**
- Preserves temporal coherence (smooth transitions)
- Better than zero-filling or mean imputation
- Mask bits inform model about imputation
- Avoids synthetic data generation

### 8.2 Bounding Box Derivatives

```python
# Position and size derivatives require temporal information
# Computed AFTER all frames are extracted

log_area = X[:, 10]  # From bbox_features
dt = 0.1  # Time step (0.1s @ 10 FPS)

# First derivative (velocity)
dlog_area_dt = zeros_like(log_area)
dlog_area_dt[1:] = (log_area[1:] - log_area[:-1]) / dt

# Second derivative (acceleration)
d2log_area_dt2 = zeros_like(log_area)
d2log_area_dt2[2:] = (dlog_area_dt[2:] - dlog_area_dt[1:-1]) / dt

# Update feature array
X[:, 11] = dlog_area_dt
X[:, 12] = d2log_area_dt2
```

**Why not compute during extraction?**
- Requires previous frame (not available at first frame)
- Cleaner to compute in batch after extraction
- Allows higher-order derivatives without complex state management

### 8.3 Window Extraction

```python
def windowize(X, M, y, window_len=5, stride=1):
    """
    Create sliding windows from sequence.

    Args:
        X: [T, D] features
        M: [T, D] masks
        y: [T] labels
        window_len: window size
        stride: step size

    Returns:
        Xw: [N, window_len, D] windows
        Mw: [N, window_len, D] masks
        yw: [N] labels (last frame of each window)
    """
    T, D = X.shape
    if T < window_len:
        return empty arrays  # Video too short

    windows_X, windows_M, windows_y = [], [], []

    for i in range(0, T - window_len + 1, stride):
        # Extract window
        window_x = X[i : i + window_len]    # [window_len, D]
        window_m = M[i : i + window_len]    # [window_len, D]
        window_y = y[i + window_len - 1]    # Last frame label

        windows_X.append(window_x)
        windows_M.append(window_m)
        windows_y.append(window_y)

    return stack(windows_X), stack(windows_M), array(windows_y)
```

**Why use last frame label?**
- Label represents hazard at END of window
- Model sees context (5 frames) and predicts current hazard
- Consistent with online inference (buffer of past frames → current prediction)

### 8.4 Video-Level Split Implementation

```python
def split_by_videos(video_data, val_fraction=0.15):
    """
    Split videos (not windows) for train/val.
    Stratified by class and window count.
    """
    # Separate by class
    safe_videos = [v for v in video_data if mean(v.labels) < 0.5]
    attack_videos = [v for v in video_data if mean(v.labels) >= 0.5]

    # Shuffle independently
    shuffle(safe_videos)
    shuffle(attack_videos)

    # Split each class by window count
    safe_train, safe_val = split_by_windows(safe_videos, val_fraction)
    attack_train, attack_val = split_by_windows(attack_videos, val_fraction)

    # Combine
    train_videos = safe_train + attack_train
    val_videos = safe_val + attack_val

    # Shuffle combined sets
    shuffle(train_videos)
    shuffle(val_videos)

    return train_videos, val_videos

def split_by_windows(videos, val_fraction):
    """Greedy assignment to achieve target window fraction."""
    total_windows = sum(v.num_windows for v in videos)
    target = int(val_fraction * total_windows)

    val_set, train_set = [], []
    val_count = 0

    for video in videos:
        # Compute distance to target for both assignments
        dist_if_val = abs((val_count + video.num_windows) - target)
        dist_if_train = abs(val_count - target)

        # Assign to set that brings us closer to target
        if val_count < target and (dist_if_val < dist_if_train or len(val_set) == 0):
            val_set.append(video)
            val_count += video.num_windows
        else:
            train_set.append(video)

    return train_set, val_set
```

---

## 9. Performance Analysis

### 9.1 Holdout Results

#### Overall Performance
```
Total Videos: 57
  Attacks: 37
  Safe: 20

Detection Rate: 94.6% (35/37)
  Detected: 35
  Missed: 2

False Positive Rate:
  Safe videos: 0.0% (0/20)
  Pre-onset: 5.4% (2/37)
```

#### Warning Level Performance
```
PRE-CONTACT (threshold=0.50):
  Warning Rate: 100% (35/35 detected)
  Pre-contact: 62.9% (22/35)
  Mean Lead Time: 2.4 frames (0.24s)
  Median Lead Time: 3.0 frames (0.30s)
  Pre-contact only: 8.8 frames (0.88s)

HIGH (threshold=0.60):
  Warning Rate: 94.3% (33/35)
  Pre-contact: 60.6% (20/33)
  Mean Lead Time: 3.4 frames (0.34s)

CRITICAL (threshold=0.80):
  Warning Rate: 97.1% (34/35)
  Pre-contact: 58.8% (20/34)
  Mean Lead Time: -0.5 frames (-0.05s)
```

### 9.2 Error Analysis

#### Missed Attacks (2/37)
```
Video: h23.mp4
Reason: Very subtle approach, no visible wind-up
Max hazard: 0.41 (below threshold)
Recommendation: Increase sensitivity, or accept as edge case

Video: h31.mp4
Reason: Person mostly off-screen, bbox cropped
Max hazard: 0.38
Recommendation: Improve tracking for partial occlusion
```

#### False Positives - Pre-onset (2/37)
```
Video: h07.mp4
Detection: Frame 89, Onset: Frame 95 (6 frames early)
Reason: Early wind-up detected before labeled onset
Note: May actually be correct (label ambiguity)

Video: h12.mp4
Detection: Frame 102, Onset: Frame 110 (8 frames early)
Reason: Aggressive hand gesture misclassified
Recommendation: More diverse safe gesture training data
```

#### False Positives - Safe Videos (0/20)
```
No false positives on safe videos.
System successfully distinguishes normal behavior.
```

#### Late Detections (13/35 detected)
```
Attacks detected AFTER contact (negative lead time):
  Count: 13/35 (37%)
  Mean late: -3.2 frames (-0.32s)

Potential causes:
  1. Rapid attacks (< 0.5s from onset to contact)
  2. Insufficient temporal context (5 frames may be too short)
  3. Model focuses on high-confidence features (occurs near contact)

Recommendations:
  1. Increase window length to 8-10 frames
  2. Add motion prediction module
  3. Tune threshold lower (trade precision for recall)
```

### 9.3 Feature Importance Analysis

**Top 5 Features**:
1. **div_positive_mean (0.244)**: Person expansion in view
   - Most important feature
   - Indicates approach
   - Captured by optical flow

2. **min_arm_angle (0.077)**: Most bent arm
   - Second most important
   - Indicates strike preparation
   - Pose-based geometric feature

3. **approach_acceleration (0.060)**: Rate of approach increase
   - Third most important
   - Captures aggressive movement
   - Temporal derivative feature

4. **l_arm_angle (0.059)**: Left arm configuration
   - Bilateral symmetry with r_arm_angle
   - Captures hand/arm positioning

5. **squared_up (0.028)**: Facing camera
   - Attack stance indicator
   - Boolean geometric feature

**Insights**:
- **Motion dominates**: Top feature is optical flow-based
- **Geometry matters**: 4/5 top features involve pose
- **Derivatives important**: Acceleration > velocity
- **Bilateral features**: Both arms contribute independently

### 9.4 Threshold Sensitivity

| Threshold | Precision | Recall | F1 | Pre-contact Rate | Mean Lead |
|-----------|-----------|--------|-----|------------------|-----------|
| 0.30 | 0.52 | 1.00 | 0.68 | 68.6% | 3.1 frames |
| 0.40 | 0.61 | 0.97 | 0.75 | 66.7% | 2.9 frames |
| 0.50 | 0.71 | 0.95 | 0.81 | 62.9% | 2.4 frames |
| 0.55 | 0.76 | 0.95 | 0.84 | 60.0% | 2.1 frames |
| 0.60 | 0.81 | 0.94 | 0.87 | 60.6% | 1.8 frames |
| 0.70 | 0.89 | 0.89 | 0.89 | 54.8% | 0.9 frames |
| 0.80 | 0.94 | 0.83 | 0.88 | 41.4% | -0.3 frames |

**Trade-offs**:
- **Lower threshold**: Higher recall, more pre-contact warnings, but lower precision
- **Higher threshold**: Higher precision, but fewer pre-contact warnings
- **Optimal (F1)**: 0.55-0.60 balances precision and recall
- **Optimal (Lead time)**: 0.30-0.40 maximizes early warning

**Recommendation**: Use 0.50 for deployment (good balance)

---

## 10. Design Decisions

### 10.1 Why 5-frame Windows?

**Considered**: 3, 5, 8, 10 frames

**Chosen**: 5 frames (0.5s @ 10 FPS)

**Rationale**:
- ✅ Captures short-term temporal patterns
- ✅ Small enough to localize attack onset
- ✅ Large enough for motion estimation
- ✅ Fast inference (small input size)
- ✅ Less prone to overfitting

**Trade-offs**:
- ❌ May miss longer-term context (pre-onset posture)
- ❌ Sensitive to rapid attacks

**Alternatives considered**:
- 3 frames: Too short for reliable motion patterns
- 8-10 frames: Better context, but slower inference and more overfitting risk

### 10.2 Why Stride Differs for Safe vs Attack?

**Safe videos**: stride = 5
**Attack videos**: stride = 1

**Rationale**:
- Safe videos: Mostly redundant frames (little change)
- Attack videos: Sequential behavior (each frame matters)
- Balances dataset (1.3:1 ratio without excessive safe redundancy)
- Preserves attack temporal coherence

**Alternative considered**:
- Random sampling: Breaks temporal coherence, loses sequential patterns
- Equal stride: Either too many safe samples or too few attack samples

### 10.3 Why GRU over LSTM?

**Chosen**: GRU

**Rationale**:
- ✅ Fewer parameters (24K vs 32K)
- ✅ Faster training and inference
- ✅ Less overfitting on small dataset
- ✅ Similar performance for short sequences

**Trade-offs**:
- ❌ Slightly less expressive (no separate forget/input gates)

**Benchmark** (5-frame windows, same data):
```
Model     Params  Train Time  Val F1  Inference Speed
GRU       24K     8 min       0.833   120 FPS
LSTM      32K     11 min      0.829   95 FPS
```

### 10.4 Why Concatenate Features and Masks?

**Alternatives considered**:
1. **Impute only, no mask**: Model doesn't know what's imputed
2. **Masked loss**: Doesn't inform model during inference
3. **Separate encoders**: More complex, more parameters

**Chosen**: Concatenate [features, masks]

**Rationale**:
- ✅ Simple and effective
- ✅ Model learns to weight reliable features
- ✅ No additional complexity
- ✅ Works well in practice

### 10.5 Why Focal Loss over Weighted BCE?

**Progression**:
1. Standard BCE: 55.9% pre-contact rate
2. Weighted BCE (α=0.57): 60.2% pre-contact rate
3. Focal Loss (γ=2, α=0.57): 61.5% pre-contact rate
4. Focal Loss (γ=3, α=0.75): **62.9% pre-contact rate** ✓

**Rationale**:
- ✅ Focuses on hard examples (subtle pre-contact behavior)
- ✅ Down-weights easy examples (obvious attacks)
- ✅ Better than static class weighting
- ✅ Prioritizes recall (α=0.75)

### 10.6 Why Video-Level Split?

**Problem**: Window-based split causes data leakage
- Adjacent windows share 4/5 frames
- Validation set "sees" training data

**Solution**: Split at video level

**Rationale**:
- ✅ No data leakage
- ✅ True generalization test
- ✅ Realistic evaluation (unseen videos)

**Empirical evidence**:
- Window-based split: Val F1 = 0.92 (overly optimistic)
- Video-level split: Val F1 = 0.83 (realistic)

---

## 11. Appendix

### 11.1 Feature Dimension Mapping

| Index | Feature Name | Description | Unit | Validity |
|-------|--------------|-------------|------|----------|
| 0 | l_arm_angle | Left shoulder-elbow-wrist angle | radians | Pose |
| 1 | r_arm_angle | Right shoulder-elbow-wrist angle | radians | Pose |
| 2 | min_arm_angle | Minimum of left/right arm angles | radians | Pose |
| 3 | l_wrist_raised | Left wrist above left shoulder | binary | Pose |
| 4 | r_wrist_raised | Right wrist above right shoulder | binary | Pose |
| 5 | max_wrist_raised | Either wrist raised | binary | Pose |
| 6 | torso_scale | Shoulder-hip distance | pixels | Pose |
| 7 | squared_up | Shoulders horizontal (facing camera) | binary | Pose |
| 8 | approach_speed | d(torso_scale)/dt | pixels/s | Pose |
| 9 | approach_accel | d²(torso_scale)/dt² | pixels/s² | Pose |
| 10 | log_area | log(bbox area) | log(px²) | Always |
| 11 | dlog_area_dt | d(log_area)/dt | log(px²)/s | Always |
| 12 | d2log_area_dt2 | d²(log_area)/dt² | log(px²)/s² | Always |
| 13 | lateral_speed | Horizontal torso movement | pixels/s | Pose |
| 14 | min_distance | Closest point to camera (proxy) | pixels | Pose |
| 15 | det_conf | YOLOv8 detection confidence | [0,1] | Always |
| 16 | kp_conf_mean | Mean keypoint confidence | [0,1] | Always |
| 17 | kp_conf_min | Min keypoint confidence | [0,1] | Always |
| 18 | visible_kp | Number of visible keypoints | [0,17] | Always |
| 19 | track_age | Frames since first detection | frames | Always |
| 20 | lost | Frames since last detection | frames | Always |
| 21 | person_flow_mag | Optical flow magnitude (person) | pixels/frame | Flow |
| 22 | person_flow_dir_std | Flow direction std (person) | radians | Flow |
| 23 | bg_flow_mag | Optical flow magnitude (background) | pixels/frame | Flow |
| 24 | bg_flow_dir_std | Flow direction std (background) | radians | Flow |
| 25 | radial_ratio | Radial vs tangential flow ratio | [0,1] | Flow |
| 26 | div_positive_mean | Mean positive radial flow (expansion) | pixels/frame | Flow |
| 27 | cx_vel | Bbox center x velocity | pixels/s | Always |
| 28 | cy_vel | Bbox center y velocity | pixels/s | Always |
| 29 | any_crop | Bbox touches frame edge | binary | Always |

### 11.2 COCO-17 Keypoint Layout

```
        0: nose
       / \
      1   2: l_eye, r_eye
     /     \
    3       4: l_ear, r_ear
     \     /
      \   /
   5───┬───6: l_shoulder, r_shoulder
       │
   7───┼───8: l_elbow, r_elbow
       │
   9───┼───10: l_wrist, r_wrist
       │
  11───┼───12: l_hip, r_hip
       │
  13───┼───14: l_knee, r_knee
       │
  15───┴───16: l_ankle, r_ankle
```

### 11.3 Configuration Reference

```python
@dataclass
class Config:
    # Data paths
    data_root: str = "data"
    safe_dir: str = "safe"
    attack_dir: str = "attack"

    # Video processing
    input_fps: int = 30           # Source video FPS
    proc_fps: int = 10            # Processing FPS
    frame_stride: int = 3         # Downsample factor

    # Temporal window
    window_len: int = 5           # Frames per window
    step_dt: float = 0.1          # Time step (s)
    safe_window_stride: int = 5   # Safe video stride
    attack_window_stride: int = 1 # Attack video stride

    # Keypoint validity
    kp_conf_thresh: float = 0.35  # Minimum keypoint confidence
    carry_forward_steps: int = 3  # Max carry-forward frames

    # Cropping detection
    crop_eps: float = 0.03        # Edge margin (3%)

    # Optical flow (Lucas-Kanade)
    lk_win_size: int = 21         # LK window size
    lk_max_level: int = 3         # Pyramid levels
    lk_criteria_count: int = 20   # Max iterations
    lk_criteria_eps: float = 0.03 # Convergence epsilon
    max_flow_points: int = 200    # Sample points

    # Training
    gru_hidden: int = 64          # GRU hidden units
    dropout: float = 0.25         # Dropout rate
    lr: float = 1e-3              # Learning rate
    batch_size: int = 64          # Batch size
    epochs: int = 40              # Training epochs

    # Focal Loss
    focal_gamma: float = 3.0      # Focusing parameter
    focal_alpha: float = 0.75     # Attack class weight

    # Inference
    ema_alpha: float = 0.7        # EMA smoothing
    early_thresh: float = 0.2     # Initial PRE-CONTACT (tuned)
    early_persist: int = 2        # Persistence frames
    high_thresh: float = 0.60     # HIGH threshold
    critical_thresh: float = 0.80 # CRITICAL threshold
    hysteresis: float = 0.05      # Threshold hysteresis

    # Pose detection
    pose_backend: str = "ultralytics"
    yolo_pt_path: str = "models/yolov8m-pose.pt"
    yolo_conf: float = 0.25       # Detection confidence
    yolo_iou: float = 0.5         # NMS IoU threshold
```

### 11.4 Dependencies

```
# Core
python>=3.8
numpy>=1.21.0
torch>=2.0.0
opencv-python>=4.5.0

# Pose detection
ultralytics>=8.0.0

# Evaluation
scikit-learn>=1.0.0

# Utilities
tqdm>=4.62.0
```

### 11.5 File Size Reference

```
Model checkpoint: <1 MB
YOLOv8m-pose: ~52 MB
Typical video (1 min, 30 FPS): ~10-20 MB
Feature cache (1000 frames): ~1.2 MB
Training log: ~50 KB
```

### 11.6 Glossary

- **Onset Frame**: Frame where attack wind-up begins (pre-contact)
- **Attack Frame**: Frame where physical contact occurs
- **Lead Time**: attack_frame - first_detection_frame (positive = pre-contact)
- **Pre-contact Warning**: Detection before contact (lead_time > 0)
- **False Positive (Safe)**: Detection on safe video
- **False Positive (Pre-onset)**: Detection before onset on attack video
- **Carry-forward**: Imputation strategy using last valid value
- **Video-level Split**: Train/val split at video granularity (no data leakage)
- **Stratified Split**: Balanced split maintaining class distribution
- **Focal Loss**: Loss function emphasizing hard examples
- **EMA**: Exponential Moving Average (smoothing)
- **Persistence**: Requiring threshold exceeded for multiple frames

---

**End of Design Specification**

For questions or contributions, please see README.md or contact the development team.
