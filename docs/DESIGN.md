# Pre-contact Detection System - Design Specification

**Version**: 1.0.0
**Last Updated**: February 2026
**Author**: Pre-contact Detection Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
   - [2.4 Raspberry Pi 5 Deployment (Hailo-8 NPU)](#24-raspberry-pi-5-deployment-hailo-8-npu)
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
3. **Feature Engineering**: 51-dimensional feature vector (+ 51 validity masks = 102-dim model input)
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
│              FEATURE EXTRACTION (51-dim)                     │
│  • Reliability/metadata: det_conf, kp stats, crop (9)       │
│  • Bbox/approach: log_area, derivatives, center, vel (8)    │
│  • Upper body pose: wrist/elbow distances and angles (8)    │
│  • Lower body pose: knee/ankle distances and stance (7)     │
│  • Optical flow: torso/lower ROI + background (10)          │
│  • Posture: torso compression, wrist asym, face vis (3)     │
│  • Dynamics: wrist vel/accel, log_scale (3)                 │
│  • Interaction: approach_rate, expansion, accel (3)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 TEMPORAL WINDOWING                           │
│  • Sliding window: 5 frames (0.5s)                          │
│  • Buffer: deque with maxlen=5                              │
│  • Input shape: [1, 5, 102] (51 features + 51 masks)        │
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
Extract Frames @ 10 FPS (frame_stride=3)
    ↓
Pose Detection (YOLOv8)
    ↓
Feature Extraction (51-dim per frame) + Validity Masks (51-dim)
    ↓
Post-processing: fill dlog_area_dt, d2log_area_dt2, wrist derivatives
    ↓
Sliding Windows (5 frames, stride varies)
    ↓
Dataset: [N, 5, 102] windows  (51 features + 51 masks concatenated)
    ↓
Video-Level Train/Val Split (85/15)
    ↓
DataLoader (batch_size=64)
    ↓
GRU Model Training (Focal Loss)
    ↓
Threshold Tuning (maximize F1)
    ↓
Save Best Model (timestamped) + Threshold in meta.json
```

#### Inference Data Flow
```
Video Frame (BGR)
    ↓
Pose Detection → 17 keypoints
    ↓
Tracking → Consistent bbox
    ↓
Feature Extraction → 51-dim vector + 51-dim mask
    ↓
Fill temporal derivatives frame-by-frame (dlog_area_dt, wrist vel/accel)
    ↓
Add interaction features → 51-dim final vector
    ↓
Add to Window Buffer (deque, maxlen=5)
    ↓
If buffer full (5 frames):
    ↓
GRU Inference → hazard score  [input: 1×5×102]
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
│  • build_features() → 48-dim vector + mask (base)      │
│  • add_interaction_features() → 51-dim final           │
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

## 2.4 Raspberry Pi 5 Deployment (Hailo-8 NPU)

### 2.4.1 Hardware Configuration

- Raspberry Pi 5 (8 GB RAM)
- Hailo-8 AI accelerator via M.2 HAT+
- USB or CSI camera, 640×480 @ 30 FPS input video

### 2.4.2 HEF Model

The system uses the pre-compiled `yolov8m_pose.hef` from the **hailo-rpi5-examples** resource bundle (no recompilation required):

```
/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef
```

This HEF targets 640×640 input and emits 9 raw convolutional output tensors — 3 per detection stride — with **no** post-processing fused inside the HEF. All decoding (DFL, NMS, keypoint formula) is performed on the host CPU.

| Tensor | Shape | dtype | Content |
|--------|-------|-------|---------|
| `conv59` | (1, 80, 80, 64) | uint8 | DFL box regression — stride 8 |
| `conv60` | (1, 80, 80, 1) | uint8 | Person confidence — stride 8 |
| `conv61` | (1, 80, 80, 51) | uint16 | Keypoints 17×(x,y,vis) — stride 8 |
| `conv75` | (1, 40, 40, 64) | uint8 | DFL box regression — stride 16 |
| `conv76` | (1, 40, 40, 1) | uint8 | Person confidence — stride 16 |
| `conv77` | (1, 40, 40, 51) | uint16 | Keypoints 17×(x,y,vis) — stride 16 |
| `conv90` | (1, 20, 20, 64) | uint8 | DFL box regression — stride 32 |
| `conv91` | (1, 20, 20, 1) | uint8 | Person confidence — stride 32 |
| `conv92` | (1, 20, 20, 51) | uint16 | Keypoints 17×(x,y,vis) — stride 32 |

### 2.4.3 Dequantisation

HailoRT returns **raw quantised integers** (uint8 for box/conf, uint16 for keypoints). Per-tensor quantisation parameters are read from the HEF at startup using the HailoRT API:

```python
for info in hef.get_output_vstream_infos():
    qi = info.quant_info
    self._out_quant[info.name] = (float(qi.qp_scale), float(qi.qp_zp))
```

Dequantisation: **float = (raw\_int − zero\_point) × scale**

| Tensor group | Typical scale | Typical zp | Dequantised range |
|---|---|---|---|
| Conf C=1 (uint8) | ≈ 1/255 ≈ 0.0039 | 0 | [0.0, 1.0] — post-sigmoid |
| Box DFL C=64 (uint8) | ≈ 0.07–0.08 | ≈ 73–87 | float logits |
| Keypoints C=51 (uint16) | ≈ 0.0005 | ≈ 17 000–19 600 | ≈ [−7, +5] logits |

### 2.4.4 Decode Differences vs. Standard Ultralytics

Three decoding rules differ from the standard Ultralytics path:

**1. Confidence — sigmoid already fused by the Hailo compiler**

The Hailo compiler embeds sigmoid into the conf output: the dequantised scale ≈ 1/255 means values are already in [0, 1]. Applying `sigmoid()` a second time would push every zero-valued background anchor from 0.0 → 0.5, causing all ≈8 000 background anchors to fire simultaneously and produce a wildly jumping bbox. The dequantised value is compared directly to `conf_thresh`:

```python
conf_flat = conf_raw.reshape(N).astype(np.float32)   # already [0, 1]
mask = conf_flat > conf_thresh
```

**2. Keypoint x, y — no sigmoid; multiply-and-offset formula only**

The YOLOv8-pose `cv4` conv head outputs raw logits for x and y (no activation in the head). The correct host-side decode is:

```
kp_x = (kp_x_logit × 2.0 + grid_x) × stride
kp_y = (kp_y_logit × 2.0 + grid_y) × stride
```

Dequantised logits in [−7, +5] produce pixel coordinates spanning the full image.
Applying `sigmoid()` before the ×2 multiply clamps all keypoints to a ≈64 px band around the anchor centre (2 grid cells at stride 32), causing the **"crowded skeleton"** artefact where all 17 joints collapse to a tiny cluster at the anchor location.

**3. Keypoint visibility — sigmoid applied**

```python
kp_vis = sigmoid(vis_logit)   # vis is a raw logit; sigmoid IS needed
```

### 2.4.5 Config Preset

`Config.for_pi()` sets all Pi-specific overrides; `Config.for_pc()` remains unchanged for training and evaluation:

```python
@classmethod
def for_pi(cls) -> "Config":
    """Raspberry Pi 5 + Hailo-8 HAT+ preset."""
    return cls(
        pose_backend  = "hailo",
        yolo_hef_path = "/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef",
        early_persist = 1,   # 1 step ≈ sufficient at ~3-4 Hz inference rate
    )
```

Run inference on the Pi with:

```bash
python -m src.infer                         # live camera
python -m src.infer path/to/video.mp4       # recorded video
```

---

## 3. Feature Engineering

### 3.1 Feature Vector Overview

The system extracts a **51-dimensional feature vector** per frame. Each feature has a corresponding validity mask, giving a **102-dimensional model input** ([features ∥ masks]).

```
Feature Vector (51 dimensions):
├── Reliability / Metadata (9 dims)      indices 0-8
│   det_conf, kp_conf_mean, kp_conf_min, visible_kp_count,
│   anyc, crop_left, crop_right, crop_top, crop_bottom
│
├── Bbox / Approach (8 dims)             indices 9-16
│   log_area, dlog_area_dt*, d2log_area_dt2*,
│   bbox_center_x, bbox_center_y, bbox_center_vx, bbox_center_vy, bbox_aspect
│
├── Upper Body Geometry (8 dims)         indices 17-24
│   dist_l_wrist_l_shoulder, dist_r_wrist_r_shoulder,
│   dist_l_wrist_torso, dist_r_wrist_torso,
│   shoulder_width, hip_width, angle_l_elbow, angle_r_elbow
│
├── Lower Body Geometry (7 dims)         indices 25-31
│   angle_l_knee, angle_r_knee,
│   dist_l_ankle_l_hip, dist_r_ankle_r_hip,
│   dist_l_ankle_torso, dist_r_ankle_torso, stance_width
│
├── Optical Flow – Torso ROI (4 dims)    indices 32-35
│   translation_signed_torso, translation_torso,
│   divergence_torso, div_ratio_torso
│
├── Optical Flow – Lower ROI (4 dims)    indices 36-39
│   translation_signed_lower, translation_lower,
│   divergence_lower, div_ratio_lower
│
├── Optical Flow – Background (2 dims)   indices 40-41
│   bg_flow_coherence, flow_ok
│
├── Posture (3 dims)                     indices 42-44
│   torso_compression_ratio, wrist_height_asymmetry, face_visibility
│
├── Dynamics (3 dims)                    indices 45-47
│   max_wrist_extension_velocity, max_wrist_extension_accel, log_scale
│
└── Interaction Features (3 dims)        indices 48-50
    approach_rate, expansion_proximity, acceleration_proximity

* dlog_area_dt (index 10) and d2log_area_dt2 (index 11) are 0.0 placeholders in
  build_features(); filled from frame history by dataset.py / evaluate.py / infer.py.
* max_wrist_extension_velocity (index 45) and max_wrist_extension_accel (index 46)
  are likewise 0.0 placeholders filled from frame history.

Mask Vector (51 dimensions): validity flags (1 = valid, 0 = invalid/imputed)
Model Input = [features ∥ masks] → 102 dimensions
```

### 3.2 Detailed Feature Descriptions

#### 3.2.1 Reliability / Metadata (9 dims, indices 0–8)

Detection quality indicators that tell the GRU how much to trust the other features in this frame.

```python
det_conf         # YOLOv8 detection confidence [0,1]
kp_conf_mean     # Mean keypoint confidence [0,1]
kp_conf_min      # Min keypoint confidence [0,1]
visible_kp_count # Count of keypoints above confidence threshold [0,17]
anyc             # 1 if bbox touches any frame edge (cropped)
crop_left        # 1 if bbox touches left edge
crop_right       # 1 if bbox touches right edge
crop_top         # 1 if bbox touches top edge
crop_bottom      # 1 if bbox touches bottom edge
```
*Rationale*: Reliability flags allow the GRU to down-weight detections with low confidence or heavy cropping, which frequently occur just before contact.

#### 3.2.2 Bbox / Approach (8 dims, indices 9–16)

The size, position, and motion of the person's bounding box — the primary proximity and approach signal that requires no pose keypoints.

```python
log_area         # log(bbox_width × bbox_height) — more stable than raw area
dlog_area_dt     # d(log_area)/dt — bbox growth rate (approach speed proxy)
d2log_area_dt2   # d²(log_area)/dt² — bbox growth acceleration
bbox_center_x    # Normalised bbox centre x [0,1]
bbox_center_y    # Normalised bbox centre y [0,1]
bbox_center_vx   # Bbox centre x velocity (frame-to-frame)
bbox_center_vy   # Bbox centre y velocity (frame-to-frame)
bbox_aspect      # bbox_width / bbox_height
```
*Note*: `dlog_area_dt` (index 10) and `d2log_area_dt2` (index 11) are 0.0 placeholders in `build_features()`; filled from frame history by `dataset.py` / `evaluate.py` / `infer.py`.

*Rationale*: Expanding bbox (`dlog_area_dt > 0`) is the simplest approach signal. Acceleration distinguishes sudden rushes from slow walks.

#### 3.2.3 Upper Body Geometry (8 dims, indices 17–24)

Normalised skeletal distances and joint angles describing arm configuration — captures punch, grab, and strike wind-up postures.

```python
dist_l_wrist_l_shoulder  # Left wrist–shoulder distance (normalised)
dist_r_wrist_r_shoulder  # Right wrist–shoulder distance
dist_l_wrist_torso       # Left wrist–torso-centre distance
dist_r_wrist_torso       # Right wrist–torso-centre distance
shoulder_width           # Shoulder–shoulder distance
hip_width                # Hip–hip distance
angle_l_elbow            # Left elbow angle (shoulder–elbow–wrist, radians)
angle_r_elbow            # Right elbow angle
```
*Rationale*: Wrist extension (large dist_wrist_torso) and bent elbows (small angle) are common pre-strike signatures. Shoulder width provides a body-size normaliser.

#### 3.2.4 Lower Body Geometry (7 dims, indices 25–31)

Normalised leg joint angles and ankle distances describing lower-body stance — captures charging, kicking, and lunging preparation.

```python
angle_l_knee         # Left knee angle (hip–knee–ankle, radians)
angle_r_knee         # Right knee angle
dist_l_ankle_l_hip   # Left ankle–hip distance
dist_r_ankle_r_hip   # Right ankle–hip distance
dist_l_ankle_torso   # Left ankle–torso-centre distance
dist_r_ankle_torso   # Right ankle–torso-centre distance
stance_width         # Ankle–ankle distance
```
*Rationale*: Bent knees and wide stance indicate charging or lunging preparation. Ankle–torso distance captures leg extension toward the officer.

#### 3.2.5 Optical Flow – Torso ROI (4 dims, indices 32–35)

Translation-decomposed flow features for the upper-body region — separately quantifies how fast the torso is approaching the camera (translation) and how much the upper body is expanding within the frame (divergence).

```python
# See Section 3.3 for the full algorithm. Summary:
#   1. Background flow subtracted from ROI flow (camera-motion compensation)
#   2. Median of compensated ROI flow = translational component (v_med)
#   3. Residual = compensated flow − v_med = deformation / expansion
#   4. translation_signed: radial projection of v_med onto each point's outward dir
#   5. divergence: mean positive radial of residual (pure expansion signal)
#   6. div_ratio: divergence / (divergence + mean_residual_mag + ε)

translation_signed_torso  # Positive = approaching camera, negative = retreating
translation_torso         # ||v_med|| — speed of translational motion
divergence_torso          # mean(radial_resid > 0) — body expanding in frame
div_ratio_torso           # divergence / (divergence + resid_mag + ε)
                          #   ≈1.0 → residual is pure outward expansion
                          #   ≈0.0 → residual is tangential / random
```
*Rationale*: `translation_signed_torso` cleanly measures the approach component; `divergence_torso` measures local body expansion (arm extension, torso lean). Separating the two avoids the naive radial decomposition's conflation of translation with expansion. See Section 3.3 for full derivation.

#### 3.2.6 Optical Flow – Lower ROI (4 dims, indices 36–39)

The same translation-decomposed flow analysis applied to a separate lower-body region — detects leg-driven attacks (kicks, tackles) that are invisible in the torso ROI.

```python
# Same two-stage decomposition as torso ROI (see Section 3.3),
# but applied to a rectangle offset downward to cover hips/legs,
# and using the lower ROI's own centre for decomposition.

translation_signed_lower  # Positive = legs approaching camera (kick/tackle run-up)
translation_lower         # ||v_med|| — translational speed of lower body
divergence_lower          # mean(radial_resid > 0) — lower body expanding in frame
div_ratio_lower           # Expansion fraction of lower-body residual flow
```
*Rationale*: Separate lower-body tracking captures kicks and tackles that produce distinct leg-forward flow patterns not visible in the torso ROI. Using the lower ROI's own centre is critical: a forward leg stride from the torso centre would appear mostly tangential, but from the lower-body centre it correctly appears as a strong outward (radial) motion. See Section 3.3.4.

#### 3.2.7 Optical Flow – Background (2 dims, indices 40–41)

Camera-motion characterisation and flow validity flag — used both for background subtraction (upstream) and as context features telling the model how much camera movement is present.

```python
bg_flow_coherence  # mean(v_bg) / (||mean(v_bg)|| + ε)  — coherence of background motion
flow_ok            # 1 if LK tracking succeeded for this frame, 0 otherwise
```
*Note*: Raw background flow magnitude is intentionally excluded — it was a dataset confounder (body-worn cameras from different environments had different typical background motion levels). Background vectors are subtracted from ROI flow before decomposition; `bg_flow_coherence` captures whether camera motion is translational (coherent) vs shaky.

#### 3.2.8 Posture (3 dims, indices 42–44)

Body shape and orientation features that capture attack-preparation stances not expressed by raw skeletal distances alone — torso lean, arm asymmetry, and face-on orientation.

```python
torso_compression_ratio  # Ratio of torso height to width; drops when person leans forward
wrist_height_asymmetry   # |left_wrist_y - right_wrist_y| / torso_height
face_visibility          # Keypoint confidence of nose/eyes as proxy for face-on orientation
```
*Rationale*: Forward lean, asymmetric wrist height, and face-on orientation are common attack-preparation postures not captured by raw distances.

#### 3.2.9 Dynamics (3 dims, indices 45–47)

Temporal derivatives of wrist extension and a pose-derived proximity estimate — captures the speed and acceleration of a strike motion, and provides a size-stable distance signal.

```python
max_wrist_extension_velocity  # Max per-wrist extension speed (d(dist_wrist_torso)/dt)
max_wrist_extension_accel     # Max per-wrist extension acceleration
log_scale                     # log(torso_height_px) — pose-derived apparent size;
                              #   scale-invariant to arm raises unlike log_area
```
*Note*: `max_wrist_extension_velocity` (index 45) and `max_wrist_extension_accel` (index 46) are 0.0 placeholders in `build_features()`; filled from frame history alongside the bbox derivatives.

*Rationale*: Wrist velocity and acceleration capture the dynamics of a strike wind-up. `log_scale` provides a proximity estimate invariant to arm raises (unlike bbox area which grows when arms extend).

#### 3.2.10 Interaction Features (3 dims, indices 48–50)

Multiplicative combinations of motion and proximity signals — encodes the principle that the same movement is only threatening when the person is already close.

```python
approach_rate         # max(dlog_scale_dt, 0) × max(translation_signed_torso, 0)
                      #   zero for distant persons and for retreating persons
expansion_proximity   # divergence_torso × proximity_weight
                      #   where proximity_weight = exp(log_scale × 0.1)
acceleration_proximity # max_wrist_extension_accel × proximity_weight
```
*Rationale*: Motion features are only threatening when the person is close. These interaction terms gate the strongest approach/strike signals by proximity, reducing false positives from distant rapid motion.

### 3.3 Optical Flow Algorithm: Translation-Decomposed Radial Analysis

#### 3.3.1 Design Motivation

Body-worn cameras create a specific optical flow problem that naive radial/tangential decomposition cannot solve cleanly.

**The naive radial decomposition problem:**

When a person walks directly toward the camera, every tracked point inside their bounding box generates an outward flow vector (expansion). This is the desired signal. However, the person also has a global translational component — the entire bounding box shifts downward as the person approaches (the apparent size grows *and* the bounding box centroid moves). For points on the *left* and *right* sides of the torso, a purely downward translation contributes to their tangential component (perpendicular to the radial direction), inflating the tangential term and suppressing the radial ratio. The result: a direct approach looks *less* radial than it really is, because the translational component is partly misclassified as tangential.

**The fix: separate translation from expansion first.**

The person's body motion in the image can be decomposed into two components:

1. **Global translation** — the entire ROI moving as a rigid body (camera approaching or person walking toward camera, lateral sway, etc.)
2. **Local deformation (expansion)** — points spreading outward from the ROI centre as the person fills more of the frame

By separating these before computing the radial decomposition, each component can be measured cleanly and independently.

---

#### 3.3.2 Algorithm: Two-Stage Flow Decomposition

**Stage 1 — Background compensation:**

```
v_bg = LK flow of ~200 points sampled outside the person's bbox (+ margin)
v_bg_med = median(v_bg)          # Robust estimate of camera translation
v_roi_compensated = v_roi - v_bg_med  # Remove camera motion from ROI flow
```

`v_bg_med` is the median background flow vector: a robust estimate of rigid camera motion. Subtracting it from each ROI flow vector removes camera shake, panning, and walking-induced background shift before any further analysis.

---

**Stage 2 — Translation/expansion separation within the ROI:**

After background compensation, the remaining ROI flow `v` still contains both:
- The person's own translational motion relative to the camera (walking toward/away)
- Local body expansion (person growing larger in frame)

These are separated using the median of the compensated ROI flow:

```
v_med = median(v_roi_compensated)      # = person's translational component
v_resid = v_roi_compensated - v_med    # = residual deformation (expansion)
```

The **median is robust** to the non-uniform nature of body flow: limbs move differently from the torso, but the dominant rigid-body translation dominates the median.

---

**Stage 3 — Feature extraction from each component:**

From the **translational component** (`v_med`):

```
# Project median flow onto each point's outward radial direction:
d = p0 - center        # vector from ROI centre to each tracked point
r = ||d|| + ε

radial_translation = (d · v_med) / r    # per-point radial projection of translation

translation_signed = mean(radial_translation)
  # Positive → ROI uniformly expanding via translation = person approaching camera
  # Negative → ROI uniformly contracting = person retreating

translation_mag = ||v_med||             # Speed of translational motion
```

`translation_signed` is the cleanest approach signal: it is positive if and only if the entire ROI is moving outward (away from its centre), which occurs when the person is getting closer. It is negative when retreating.

From the **residual component** (`v_resid`):

```
radial_resid = (d · v_resid) / r       # per-point radial of deformation flow

divergence = mean(radial_resid[radial_resid > 0])
  # Average magnitude of outward-only residual radial components
  # Positive when body surface points are locally spreading outward
  # Zero when there is no net expansion in the residual

resid_mag = mean(||v_resid||)
div_ratio = divergence / (divergence + resid_mag + ε)
  # ≈ 1.0 → residual flow is predominantly outward expansion
  # ≈ 0.0 → residual is tangential (arm waving, rotation) or zero
```

`divergence` captures **intra-body expansion**: e.g., a punch extends one arm outward from the torso centre, generating a strong positive radial residual in the direction of the strike. `div_ratio` normalises by total residual magnitude to distinguish a genuine expansion from random limb movement.

---

#### 3.3.3 Why This Decomposition Is Better Than Alternatives

| Approach | Translation signal | Expansion signal | Camera-motion robust | Lateral rejection |
|----------|--------------------|-----------------|---------------------|-------------------|
| Raw flow magnitude | Conflated | Conflated | ❌ No | ❌ No |
| Bbox area growth | Indirect (1D) | Indirect (1D) | Partial | ❌ No |
| Simple radial/tangential on raw flow | Polluted by translation | Polluted by translation | Partial | Partial |
| Dense divergence field (∂fx/∂x + ∂fy/∂y) | ❌ Absent | Noisy | ❌ No | ❌ No |
| **This approach** | ✅ Clean (`translation_signed`) | ✅ Clean (`divergence`) | ✅ Yes (2-stage) | ✅ Yes (`div_ratio`) |

Key advantages:
- **No confusion between translation and expansion**: a person walking toward the camera at constant size (distant, zoomed out) gives high `translation_signed` but low `divergence`. A person standing still while throwing a punch gives low `translation_signed` but high `divergence`. Both are correctly captured.
- **Two independent threat indicators**: the GRU can learn that high `translation_signed` *combined with* high `divergence` is the most dangerous pattern (fast approach + arm extension), while either alone may be benign.
- **Numerically stable**: using median (not mean) for the translation estimate is robust to outlier flow vectors from limb tips and clothing edges.

---

#### 3.3.4 Multi-ROI Strategy

Two separate ROIs are processed independently:

**Torso ROI** (`translation_signed_torso`, `translation_torso`, `divergence_torso`, `div_ratio_torso`):
- Square region centred on the torso centre (shoulder-hip midpoint), side = 1.2 × torso height
- Centre of decomposition = torso centre
- Captures: approach speed, upper-body expansion, punch/grab wind-up (arm extends outward from torso centre)

**Lower ROI** (`translation_signed_lower`, `translation_lower`, `divergence_lower`, `div_ratio_lower`):
- Rectangle **anchored at the actual hip midpoint**: `hip_mid_y = mean(l_hip.y, r_hip.y)` when both hips are visible; falls back to `torso_c[1] + 0.5 × torso_s` when hips are occluded
- Width = 1.5 × torso_s, centred horizontally on the torso; extends **2.0 × torso_s downward** from `hip_mid_y` (covers hips → ankles)
- Centre of decomposition = lower ROI centre (not torso centre)
- Captures: leg/hip motion, kick preparation (leg extends outward from lower-body centre), tackle approach
- Note: the previous design anchored the lower ROI 0.4 × torso_s below torso centre, causing ~67% overlap with the torso ROI and failing to cover the legs; anchoring at `hip_mid_y` eliminates the overlap and ensures feet coverage

Using the **lower ROI's own centre** is important: if the torso centre were used instead, a forward leg stride (kick preparation) would appear mostly tangential (the leg moves downward, which is perpendicular to the radial direction from the torso), losing the approach signal. With the lower-body centre, the same leg stride correctly appears as a strong outward (radial) motion.

---

#### 3.3.5 Full Per-Frame Pipeline

```
Frame t-1, Frame t
     ↓
Pose detection → torso_center, lower_center, bbox
     ↓
Sample ~200 background points (outside bbox + 20px margin)
     ↓
Sample ~200 points inside torso ROI
Sample ~200 points inside lower ROI
     ↓
Lucas-Kanade tracking:  v_bg, v_torso, v_lower
     ↓
Background estimation:  v_bg_med = median(v_bg)
Background coherence:   bg_coh = ||mean(v_bg)|| / (mean(||v_bg||) + ε)
     ↓
Background subtraction:
    v_torso_comp = v_torso - v_bg_med
    v_lower_comp = v_lower - v_bg_med
     ↓
radial_tangential_stats(p0_torso, v_torso_comp, torso_center)
    → translation_signed_torso, translation_torso,
      divergence_torso, div_ratio_torso
     ↓
radial_tangential_stats(p0_lower, v_lower_comp, lower_center)
    → translation_signed_lower, translation_lower,
      divergence_lower, div_ratio_lower
     ↓
Output: 10 flow features (indices 32–41)
```

### 3.4 Mask Vector

Each feature has a corresponding mask bit indicating validity:

```python
mask[i] = 1  # Feature is valid (computed from confident keypoints)
mask[i] = 0  # Feature is invalid (keypoint missing, imputed from previous frame)
```

The mask is concatenated with features before feeding to GRU:
```python
input = concatenate([features, mask], axis=-1)  # [51] + [51] = [102]
```

This allows the model to learn the reliability of each feature dynamically.

### 3.5 Feature Importance (From Training)

Feature importance is computed via permutation importance after training (see `src/feature_importance.py`). The permutation importance measures the drop in F1 score when each feature is randomly shuffled.

**Top 5 features by permutation importance (measured):**

1. **`expansion_proximity`** (index 49) — 11.1%: `divergence_torso × proximity_weight`; the person is not just expanding in frame but doing so from close range, making this the single strongest combined threat signal.
2. **`acceleration_proximity`** (index 50) — 9.5%: `max_wrist_extension_accel × proximity_weight`; sudden wrist acceleration gated by proximity — separates a genuine close-range strike from a distant gesture.
3. **`translation_lower`** (index 37) — 9.2%: translational speed of the lower body ROI; fast lower-body movement toward the camera is the primary indicator of a charge, kick, or tackle run-up.
4. **`max_wrist_extension_accel`** (index 46) — 8.6%: acceleration of wrist extension toward the torso; captures the explosive snap of a strike wind-up that slower velocity features miss.
5. **`translation_torso`** (index 33) — 8.3%: translational speed of the upper body ROI; measures how quickly the torso is closing distance on the camera, independent of body expansion.

The top 5 account for ~46.7% of total importance. The dominance of the two interaction features (ranks 1–2) confirms that **proximity-gated motion** is the most discriminative pattern — the same movement that is benign at distance becomes the strongest attack signal when close.

Run `python -m src.train` to regenerate the full ranked importance list for the current model.

---

## 4. Model Architecture

### 4.1 Network Structure

```
Input: [batch, window_len, input_dim]
       [B, 5, 102]

       ↓

┌──────────────────────────────────────┐
│           GRU Layer (1 layer)        │
│  • input_size: 102                   │
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
            input_size=input_dim,      # 102 (51 features + 51 masks)
            hidden_size=hidden,         # 64
            num_layers=1,
            batch_first=True
        )

        # Dropout (applied AFTER GRU for single layer)
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, T, D] = [batch, 5, 102]
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
  • Input weights: 102 × (64 × 3) = 19,584
  • Hidden weights: 64 × (64 × 3) = 12,288
  • Biases: 64 × 3 × 2 = 384
  • Total GRU: 32,256

Dropout: 0 parameters

FC:
  • Weights: 64 × 1 = 64
  • Bias: 1
  • Total FC: 65

Total Parameters: 32,321 ≈ 32K
Model Size: <1 MB
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
# Features: [B, T, 51]
# Masks: [B, T, 51]
# Input: [B, T, 102] = concatenate([features, masks], axis=-1)
```

This design allows the GRU to:
1. Learn feature values from the first 51 dimensions
2. Learn feature reliability from the next 51 dimensions
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
input_dim = 102         # 51 features + 51 masks
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
├── hazard_gru_YYYYMMDD_HHMMSS.pt   # Model weights (best F1, timestamped)
└── meta.json                        # {"input_dim": 102, "best_threshold": 0.55, "model_file": "hazard_gru_...pt"}
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
        Freeze at last known value when feature is invalid.
        miss_run is tracked informationally but does not alter behaviour.

        Returns:
            imputed_value: current if valid, else last known value (frozen)
        """
        if is_valid:
            self.prev_value = current_value
            self.miss_run = 0
            return current_value
        else:
            self.miss_run += 1
            return self.prev_value  # Always freeze at last valid value
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
# Position and size derivatives require temporal information.
# build_features() leaves x[10] and x[11] as 0.0 placeholders.
# dataset.py fills them in batch; evaluate.py/infer.py fill them frame-by-frame.

log_area = X[:, 9]    # index 9: log_area (always valid)
dt = 0.1              # Time step (0.1s @ 10 FPS)

# First derivative (velocity) — valid from frame 1
dlog_area_dt = zeros_like(log_area)
dlog_area_dt[1:] = (log_area[1:] - log_area[:-1]) / dt

# Second derivative (acceleration) — valid from frame 2
d2log_area_dt2 = zeros_like(log_area)
d2log_area_dt2[2:] = (dlog_area_dt[2:] - dlog_area_dt[1:-1]) / dt

# Update feature array
X[:, 10] = dlog_area_dt    # index 10: dlog_area_dt
X[:, 11] = d2log_area_dt2  # index 11: d2log_area_dt2
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

Feature importance is computed automatically at end of training via `src/feature_importance.py` (permutation importance over 5 repeats). Results are saved to `outputs/logs/feature_importance_*.json`.

The old importance results listed here were from the previous 30-feature system and no longer apply to the current 51-feature / 102-dim model. Re-run training to obtain updated importance scores.

**Design-intent expected ranking** (qualitative):
- Optical flow divergence features (indices 34-35, 38-39) — approach signal
- Bbox growth rate (index 10) — proximity signal
- Interaction features (indices 48-50) — combined approach + wrist signal
- Wrist-to-torso distances (indices 19-20) — extension signal
- Elbow angles (indices 23-24) — strike preparation posture

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

| Index | Feature Name | Description | Validity |
|-------|--------------|-------------|----------|
| 0 | det_conf | YOLOv8 detection confidence | Always |
| 1 | kp_conf_mean | Mean keypoint confidence | Always |
| 2 | kp_conf_min | Min keypoint confidence | Always |
| 3 | visible_kp_count | Number of visible keypoints | Always |
| 4 | anyc | Bbox touches any frame edge | Always |
| 5 | crop_left | Bbox touches left edge | Always |
| 6 | crop_right | Bbox touches right edge | Always |
| 7 | crop_top | Bbox touches top edge | Always |
| 8 | crop_bottom | Bbox touches bottom edge | Always |
| 9 | log_area | log(bbox area) | Always |
| 10 | dlog_area_dt | d(log_area)/dt — filled by dataset/evaluate/infer | Frame ≥1 |
| 11 | d2log_area_dt2 | d²(log_area)/dt² — filled by dataset/evaluate/infer | Frame ≥2 |
| 12 | bbox_center_x | Normalised bbox centre x | Always |
| 13 | bbox_center_y | Normalised bbox centre y | Always |
| 14 | bbox_center_vx | Bbox centre x velocity | Frame ≥1 |
| 15 | bbox_center_vy | Bbox centre y velocity | Frame ≥1 |
| 16 | bbox_aspect | Bbox width/height ratio | Always |
| 17 | dist_l_wrist_l_shoulder | Left wrist to left shoulder distance | Pose |
| 18 | dist_r_wrist_r_shoulder | Right wrist to right shoulder distance | Pose |
| 19 | dist_l_wrist_torso | Left wrist to torso centre distance | Pose |
| 20 | dist_r_wrist_torso | Right wrist to torso centre distance | Pose |
| 21 | shoulder_width | Distance between shoulders | Pose |
| 22 | hip_width | Distance between hips | Pose |
| 23 | angle_l_elbow | Left elbow angle (shoulder-elbow-wrist) | Pose |
| 24 | angle_r_elbow | Right elbow angle | Pose |
| 25 | angle_l_knee | Left knee angle (hip-knee-ankle) | Pose |
| 26 | angle_r_knee | Right knee angle | Pose |
| 27 | dist_l_ankle_l_hip | Left ankle to left hip distance | Pose |
| 28 | dist_r_ankle_r_hip | Right ankle to right hip distance | Pose |
| 29 | dist_l_ankle_torso | Left ankle to torso centre distance | Pose |
| 30 | dist_r_ankle_torso | Right ankle to torso centre distance | Pose |
| 31 | stance_width | Distance between ankles | Pose |
| 32 | translation_signed_torso | Signed net translation in torso ROI | Flow |
| 33 | translation_torso | Translation magnitude in torso ROI | Flow |
| 34 | divergence_torso | mean(max(radial,0)) in torso ROI | Flow |
| 35 | div_ratio_torso | Radial/(radial+tangential) in torso ROI | Flow |
| 36 | translation_signed_lower | Signed net translation in lower ROI | Flow |
| 37 | translation_lower | Translation magnitude in lower ROI | Flow |
| 38 | divergence_lower | mean(max(radial,0)) in lower ROI | Flow |
| 39 | div_ratio_lower | Radial/(radial+tangential) in lower ROI | Flow |
| 40 | bg_flow_coherence | Background flow coherence | Flow |
| 41 | flow_ok | Flow validity flag | Always |
| 42 | torso_compression_ratio | Torso height/width ratio | Pose |
| 43 | wrist_height_asymmetry | |left_wrist_y - right_wrist_y| / torso_height | Pose |
| 44 | face_visibility | Nose/eye keypoint confidence (face-on proxy) | Pose |
| 45 | max_wrist_extension_velocity | Max per-wrist extension speed — filled by dataset/evaluate/infer | Frame ≥1 |
| 46 | max_wrist_extension_accel | Max per-wrist extension acceleration — filled by dataset/evaluate/infer | Frame ≥2 |
| 47 | log_scale | log(torso_height_px) — pose-derived apparent size | Pose |
| 48 | approach_rate | max(dlog_scale_dt,0) × max(translation_signed_torso,0) | Computed |
| 49 | expansion_proximity | divergence_torso × proximity_weight | Computed |
| 50 | acceleration_proximity | max_wrist_extension_accel × proximity_weight | Computed |

**Note**: Indices 10–11 (dlog_area_dt, d2log_area_dt2), 45–46 (wrist velocity/acceleration), and 48–50 (interaction features) are 0.0 placeholders in `build_features()`; they are filled frame-by-frame in `dataset.py`, `evaluate.py`, and `infer.py`.

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
    frame_stride: int = 3         # process every 3rd frame → 10 FPS at 30 FPS input

    # Temporal window
    window_len: int = 5           # Frames per window
    step_dt: float = 0.1          # Time step (s)
    safe_window_stride: int = 3   # Safe video stride
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

    # Pose detection — PC (ultralytics) backend
    pose_backend: str = "ultralytics"   # "ultralytics" | "hailo"
    yolo_pt_path:  str = "models/yolov8m-pose.pt"   # PC .pt weights
    yolo_conf: float = 0.25       # Detection confidence threshold
    yolo_iou:  float = 0.5        # NMS IoU threshold

    # Pose detection — Hailo NPU backend (Pi 5 only)
    # yolo_hef_path is overridden in Config.for_pi():
    yolo_hef_path: str = "models/yolov8m_pose.hef"
    #   Full path on Pi:
    #   /home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef
```

#### Platform Presets

```python
Config.for_pc()   # PC / development: ultralytics backend, yolov8m-pose.pt
Config.for_pi()   # Pi 5 + Hailo-8:  hailo backend, yolov8m_pose.hef, early_persist=1
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
