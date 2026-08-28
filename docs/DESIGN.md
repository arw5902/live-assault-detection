# Real-Time Physical Threat Detection System - Design Specification

**Version**: 1.0.0
**Last Updated**: February 2026
**Author**: Real-Time Physical Threat Detection Team

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

- Process video streams in real-time (>=10 FPS)
- Provide early warning alerts (target: 0.5-1.0s before contact)
- Minimize false positives on normal behavior
- Run on edge devices (Raspberry Pi 5 with Hailo-8L)

### 1.2 Solution Approach

The system combines:
1. **Pose Estimation**: YOLOv8-Pose for body keypoint detection
2. **Optical Flow**: Lucas-Kanade for motion analysis
3. **Feature Engineering**: 59-dimensional feature vector (+ 59 validity masks = 118-dim model input)
4. **Temporal Modeling**: Sequence classification over the temporal window - GRU (deployment default), LSTM (best window-level accuracy), and Transformer are all supported
5. **Single THREAT Alert**: Binary threshold-based detection

### 1.3 System Requirements

#### Functional Requirements
- FR1: Detect assault behavior before physical contact
- FR2: Provide a single binary THREAT alert based on a tuned hazard threshold
- FR3: Process standard video formats (MP4, AVI)
- FR4: Support real-time inference (>=10 FPS)
- FR5: Log detections with timestamps and confidence scores

#### Non-Functional Requirements
- NFR1: Detection Rate >=90%
- NFR2: Pre-contact Warning Rate >=60%
- NFR3: False Positive Rate <=10%
- NFR4: Mean Lead Time >=0.2s
- NFR5: Model Size <=5 MB (for edge deployment)

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
│ VIDEO INPUT STREAM │
│ (30 FPS, BGR) │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESSING │
│ - Downsample to 10 FPS (frame_stride=3) │
│ - Convert to grayscale (for optical flow) │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ POSE DETECTION (YOLOv8) │
│ - Detect person bounding box │
│ - Extract 17 COCO keypoints (x, y, conf) │
│ - Select largest person in frame │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ TRACKING & SMOOTHING │
│ - SingleTargetTracker: maintain bbox consistency │
│ - Carry-forward imputation (missing keypoints) │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION (59-dim) │
│ - Reliability/metadata: det_conf, kp stats, crop (9) │
│ - Bbox/approach: log_area, derivatives, center, vel (8) │
│ - Upper body pose: wrist/elbow distances and angles (8) │
│ - Lower body pose: knee/ankle distances and stance (7) │
│ - Optical flow: torso/lower ROI + background (10) │
│ - Posture: torso compression, wrist asym, face vis (3) │
│ - Dynamics: wrist vel/accel, log_scale (3) │
│ - Body-shape extras: torso_height_px, wrist_y_rel, │
│ d_wrist_y_rel_dt, ankle_spread, d_ankle_spread_dt, │
│ nose_y_rel, d_nose_y_rel_dt, upper_lower_async (8) │
│ - Interaction: approach_rate, expansion, accel (3) │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ TEMPORAL WINDOWING │
│ - Sliding window: 5 frames (0.5s) │
│ - Buffer: deque with maxlen=5 │
│ - Input shape: [1, 5, 118] (59 features + 59 masks) │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ GRU MODEL INFERENCE │
│ - 1-layer GRU (64 hidden units) │
│ - Dropout (0.25) │
│ - FC layer + Sigmoid │
│ - Output: hazard score in [0, 1] │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ POST-PROCESSING & ALERTING │
│ - EMA smoothing (alpha=0.7) │
│ - Persistence logic (2 frames) │
│ - Single THREAT threshold (default 0.50, tuned per model) │
└──────────────────────────┬──────────────────────────────────┘
 │
 v
┌─────────────────────────────────────────────────────────────┐
│ ALERT OUTPUT │
│ - Alert state: NONE / THREAT │
│ - Hazard score: 0.0 - 1.0 │
│ - Timestamp: frame index / time │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### Training Data Flow
```
Video Files (MP4)

Extract Frames @ 10 FPS (frame_stride=3)

Pose Detection (YOLOv8)

Feature Extraction (59-dim per frame) + Validity Masks (59-dim)

Post-processing: fill dlog_area_dt, d2log_area_dt2, wrist derivatives

Sliding Windows (5 frames, stride varies)

Dataset: [N, 5, 118] windows (59 features + 59 masks concatenated)

Video-Level Train/Val Split (80/20)

DataLoader (batch_size=64)

GRU Model Training (Focal Loss)

Threshold Tuning (maximize F1)

Save Best Model (timestamped) + Threshold in meta.json
```

#### Inference Data Flow
```
Video Frame (BGR)

Pose Detection -> 17 keypoints

Tracking -> Consistent bbox

Feature Extraction -> 59-dim vector + 59-dim mask

Fill temporal derivatives frame-by-frame (dlog_area_dt, wrist vel/accel,
 d_wrist_y_rel_dt, d_ankle_spread_dt, d_nose_y_rel_dt)

Add interaction features -> 59-dim final vector

Add to Window Buffer (deque, maxlen=5)

If buffer full (5 frames):

GRU Inference -> hazard score [input: 1x5x118]

EMA Smoothing

Single Threshold Comparison (THREAT vs NONE)

Alert State Output
```

### 2.3 Module Diagram

```
┌────────────────────────────────────────────────────────┐
│ src/config.py │
│ Configuration dataclass with all hyperparameters │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/pose_detector.py │
│ YOLOv8-Pose wrapper for person detection │
│ - infer(frame) -> {bbox, kps, det_conf} │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/tracker.py │
│ Simple single-target tracker │
│ - update(bbox) -> bbox │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/flow.py │
│ Optical flow utilities │
│ - lk_flow(): Lucas-Kanade tracking │
│ - radial_tangential_stats(): Motion decomposition │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/features.py │
│ Feature extraction from pose + flow │
│ - build_features() -> 56-dim vector + mask (base) │
│ - add_interaction_features() -> 59-dim final │
│ - FeatureState: carry-forward imputation │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/dataset.py │
│ Dataset building and windowing │
│ - extract_sequences(): video -> frames -> features │
│ - windowize(): sliding window extraction │
│ - build_dataset_per_video(): video-level separation │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/model.py │
│ Hazard models (GRU for deployment, LSTM for accuracy) │
│ - forward(x) -> hazard score │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/train.py │
│ Training script with Focal Loss │
│ - Video-level stratified split │
│ - F1-based model saving │
│ - Threshold tuning │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/evaluate.py │
│ Holdout evaluation against single THREAT threshold │
│ - Lead time calculation │
│ - False positive detection │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ src/infer.py │
│ Real-time inference pipeline │
│ - EMA smoothing │
│ - Persistence logic │
│ - Single THREAT alerting │
└────────────────────────────────────────────────────────┘
```

---

## 2.4 Raspberry Pi 5 Deployment (Hailo-8 NPU)

### 2.4.1 Hardware Configuration

- Raspberry Pi 5 (8 GB RAM)
- Hailo-8 AI accelerator via M.2 HAT+
- USB or CSI camera, 640x480 @ 30 FPS input video

### 2.4.2 HEF Model

The system uses the pre-compiled `yolov8m_pose.hef` from the **hailo-rpi5-examples** resource bundle (no recompilation required):

```
/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef
```

This HEF targets 640x640 input and emits 9 raw convolutional output tensors - 3 per detection stride - with **no** post-processing fused inside the HEF. All decoding (DFL, NMS, keypoint formula) is performed on the host CPU.

| Tensor | Shape | dtype | Content |
|--------|-------|-------|---------|
| `conv59` | (1, 80, 80, 64) | uint8 | DFL box regression - stride 8 |
| `conv60` | (1, 80, 80, 1) | uint8 | Person confidence - stride 8 |
| `conv61` | (1, 80, 80, 51) | uint16 | Keypoints 17x(x,y,vis) - stride 8 |
| `conv75` | (1, 40, 40, 64) | uint8 | DFL box regression - stride 16 |
| `conv76` | (1, 40, 40, 1) | uint8 | Person confidence - stride 16 |
| `conv77` | (1, 40, 40, 51) | uint16 | Keypoints 17x(x,y,vis) - stride 16 |
| `conv90` | (1, 20, 20, 64) | uint8 | DFL box regression - stride 32 |
| `conv91` | (1, 20, 20, 1) | uint8 | Person confidence - stride 32 |
| `conv92` | (1, 20, 20, 51) | uint16 | Keypoints 17x(x,y,vis) - stride 32 |

### 2.4.3 Dequantisation

HailoRT returns **raw quantised integers** (uint8 for box/conf, uint16 for keypoints). Per-tensor quantisation parameters are read from the HEF at startup using the HailoRT API:

```python
for info in hef.get_output_vstream_infos():
 qi = info.quant_info
 self._out_quant[info.name] = (float(qi.qp_scale), float(qi.qp_zp))
```

Dequantisation: **float = (raw\_int - zero\_point) x scale**

| Tensor group | Typical scale | Typical zp | Dequantised range |
|---|---|---|---|
| Conf C=1 (uint8) | ~ 1/255 ~ 0.0039 | 0 | [0.0, 1.0] - post-sigmoid |
| Box DFL C=64 (uint8) | ~ 0.07-0.08 | ~ 73-87 | float logits |
| Keypoints C=51 (uint16) | ~ 0.0005 | ~ 17 000-19 600 | ~ [-7, +5] logits |

### 2.4.4 Decode Differences vs. Standard Ultralytics

Three decoding rules differ from the standard Ultralytics path:

**1. Confidence - sigmoid already fused by the Hailo compiler**

The Hailo compiler embeds sigmoid into the conf output: the dequantised scale ~ 1/255 means values are already in [0, 1]. Applying `sigmoid()` a second time would push every zero-valued background anchor from 0.0 -> 0.5, causing all ~8 000 background anchors to fire simultaneously and produce a wildly jumping bbox. The dequantised value is compared directly to `conf_thresh`:

```python
conf_flat = conf_raw.reshape(N).astype(np.float32) # already [0, 1]
mask = conf_flat > conf_thresh
```

**2. Keypoint x, y - no sigmoid; multiply-and-offset formula only**

The YOLOv8-pose `cv4` conv head outputs raw logits for x and y (no activation in the head). The correct host-side decode is:

```
kp_x = (kp_x_logit x 2.0 + grid_x) x stride
kp_y = (kp_y_logit x 2.0 + grid_y) x stride
```

Dequantised logits in [-7, +5] produce pixel coordinates spanning the full image.
Applying `sigmoid()` before the x2 multiply clamps all keypoints to a ~64 px band around the anchor centre (2 grid cells at stride 32), causing the **"crowded skeleton"** artefact where all 17 joints collapse to a tiny cluster at the anchor location.

**3. Keypoint visibility - sigmoid applied**

```python
kp_vis = sigmoid(vis_logit) # vis is a raw logit; sigmoid IS needed
```

### 2.4.5 Config Preset

`Config.for_pi()` sets all Pi-specific overrides; `Config.for_pc()` remains unchanged for training and evaluation:

```python
@classmethod
def for_pi(cls) -> "Config":
 """Raspberry Pi 5 + Hailo-8 HAT+ preset."""
 return cls(
 pose_backend = "hailo",
 yolo_hef_path = "/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef",
 early_persist = 2, # ~2 steps at the Pi's reduced inference rate
 )
```

Run inference on the Pi with:

```bash
python -m src.infer # live camera
python -m src.infer path/to/video.mp4 # recorded video
```

---

## 3. Feature Engineering

### 3.1 Feature Vector Overview

The system extracts a **59-dimensional feature vector** per frame. Each feature has a corresponding validity mask, giving a **118-dimensional model input** ([features || masks]).

```
Feature Vector (59 dimensions):
├── Reliability / Metadata (9 dims) indices 0-8
│ det_conf, kp_conf_mean, kp_conf_min, visible_kp_count,
│ anyc, crop_left, crop_right, crop_top, crop_bottom
│
├── Bbox / Approach (8 dims) indices 9-16
│ log_area, dlog_area_dt*, d2log_area_dt2*,
│ bbox_center_x, bbox_center_y, bbox_center_vx, bbox_center_vy, bbox_aspect
│
├── Upper Body Geometry (8 dims) indices 17-24
│ dist_l_wrist_l_shoulder, dist_r_wrist_r_shoulder,
│ dist_l_wrist_torso, dist_r_wrist_torso,
│ shoulder_width, hip_width, angle_l_elbow, angle_r_elbow
│
├── Lower Body Geometry (7 dims) indices 25-31
│ angle_l_knee, angle_r_knee,
│ dist_l_ankle_l_hip, dist_r_ankle_r_hip,
│ dist_l_ankle_torso, dist_r_ankle_torso, stance_width
│
├── Optical Flow - Torso ROI (4 dims) indices 32-35
│ translation_signed_torso, translation_torso,
│ divergence_torso, div_ratio_torso
│
├── Optical Flow - Lower ROI (4 dims) indices 36-39
│ translation_signed_lower, translation_lower,
│ divergence_lower, div_ratio_lower
│
├── Optical Flow - Background (2 dims) indices 40-41
│ bg_flow_coherence, flow_ok
│
├── Posture (3 dims) indices 42-44
│ torso_compression_ratio, wrist_height_asymmetry, face_visibility
│
├── Dynamics (3 dims) indices 45-47
│ max_wrist_extension_velocity, max_wrist_extension_accel, log_scale
│
├── Body-shape Extras (8 dims) indices 48-55
│ torso_height_px, wrist_y_rel, d_wrist_y_rel_dt,
│ ankle_spread, d_ankle_spread_dt,
│ nose_y_rel, d_nose_y_rel_dt, upper_lower_async
│
└── Interaction Features (3 dims) indices 56-58
 approach_rate (56), expansion_proximity (57), acceleration_proximity (58)

* dlog_area_dt (index 10) and d2log_area_dt2 (index 11) are 0.0 placeholders in
 build_features(); filled from frame history by dataset.py / evaluate.py / infer.py.
* max_wrist_extension_velocity (index 45) and max_wrist_extension_accel (index 46)
 are likewise 0.0 placeholders filled from frame history.
* d_wrist_y_rel_dt (50), d_ankle_spread_dt (52), and d_nose_y_rel_dt (54)
 are also 0.0 placeholders filled from frame history.

Mask Vector (59 dimensions): validity flags (1 = valid, 0 = invalid/imputed)
Model Input = [features || masks] -> 118 dimensions
```

### 3.2 Detailed Feature Descriptions

#### 3.2.1 Reliability / Metadata (9 dims, indices 0-8)

Detection quality indicators that tell the GRU how much to trust the other features in this frame.

```python
det_conf # YOLOv8 detection confidence [0,1]
kp_conf_mean # Mean keypoint confidence [0,1]
kp_conf_min # Min keypoint confidence [0,1]
visible_kp_count # Count of keypoints above confidence threshold [0,17]
anyc # 1 if bbox touches any frame edge (cropped)
crop_left # 1 if bbox touches left edge
crop_right # 1 if bbox touches right edge
crop_top # 1 if bbox touches top edge
crop_bottom # 1 if bbox touches bottom edge
```
*Rationale*: Reliability flags allow the GRU to down-weight detections with low confidence or heavy cropping, which frequently occur just before contact.

#### 3.2.2 Bbox / Approach (8 dims, indices 9-16)

The size, position, and motion of the person's bounding box - the primary proximity and approach signal that requires no pose keypoints.

```python
log_area # log(bbox_width x bbox_height) - more stable than raw area
dlog_area_dt # d(log_area)/dt - bbox growth rate (approach speed proxy)
d2log_area_dt2 # d^2(log_area)/dt^2 - bbox growth acceleration
bbox_center_x # Normalised bbox centre x [0,1]
bbox_center_y # Normalised bbox centre y [0,1]
bbox_center_vx # Bbox centre x velocity (frame-to-frame)
bbox_center_vy # Bbox centre y velocity (frame-to-frame)
bbox_aspect # bbox_width / bbox_height
```
*Note*: `dlog_area_dt` (index 10) and `d2log_area_dt2` (index 11) are 0.0 placeholders in `build_features()`; filled from frame history by `dataset.py` / `evaluate.py` / `infer.py`.

*Rationale*: Expanding bbox (`dlog_area_dt > 0`) is the simplest approach signal. Acceleration distinguishes sudden rushes from slow walks.

#### 3.2.3 Upper Body Geometry (8 dims, indices 17-24)

Normalised skeletal distances and joint angles describing arm configuration - captures punch, grab, and strike wind-up postures.

```python
dist_l_wrist_l_shoulder # Left wrist-shoulder distance (normalised)
dist_r_wrist_r_shoulder # Right wrist-shoulder distance
dist_l_wrist_torso # Left wrist-torso-centre distance
dist_r_wrist_torso # Right wrist-torso-centre distance
shoulder_width # Shoulder-shoulder distance
hip_width # Hip-hip distance
angle_l_elbow # Left elbow angle (shoulder-elbow-wrist, radians)
angle_r_elbow # Right elbow angle
```
*Rationale*: Wrist extension (large dist_wrist_torso) and bent elbows (small angle) are common pre-strike signatures. Shoulder width provides a body-size normaliser.

#### 3.2.4 Lower Body Geometry (7 dims, indices 25-31)

Normalised leg joint angles and ankle distances describing lower-body stance - captures charging, kicking, and lunging preparation.

```python
angle_l_knee # Left knee angle (hip-knee-ankle, radians)
angle_r_knee # Right knee angle
dist_l_ankle_l_hip # Left ankle-hip distance
dist_r_ankle_r_hip # Right ankle-hip distance
dist_l_ankle_torso # Left ankle-torso-centre distance
dist_r_ankle_torso # Right ankle-torso-centre distance
stance_width # Ankle-ankle distance
```
*Rationale*: Bent knees and wide stance indicate charging or lunging preparation. Ankle-torso distance captures leg extension toward the officer.

#### 3.2.5 Optical Flow - Torso ROI (4 dims, indices 32-35)

Translation-decomposed flow features for the upper-body region - separately quantifies how fast the torso is approaching the camera (translation) and how much the upper body is expanding within the frame (divergence).

```python
# See Section 3.3 for the full algorithm. Summary:
# 1. Background flow subtracted from ROI flow (camera-motion compensation)
# 2. Median of compensated ROI flow = translational component (v_med)
# 3. Residual = compensated flow - v_med = deformation / expansion
# 4. translation_signed: radial projection of v_med onto each point's outward dir
# 5. divergence: median positive radial of residual (pure expansion signal)
# 6. div_ratio: divergence / (divergence + mean_residual_mag + eps)

translation_signed_torso # Positive = approaching camera, negative = retreating
translation_torso # ||v_med|| - speed of translational motion
divergence_torso # median(radial_resid > 0) - body expanding in frame
div_ratio_torso # divergence / (divergence + resid_mag + eps)
 # ~1.0 -> residual is pure outward expansion
 # ~0.0 -> residual is tangential / random
```
*Rationale*: `translation_signed_torso` cleanly measures the approach component; `divergence_torso` measures local body expansion (arm extension, torso lean). Separating the two avoids the naive radial decomposition's conflation of translation with expansion. See Section 3.3 for full derivation.

#### 3.2.6 Optical Flow - Lower ROI (4 dims, indices 36-39)

The same translation-decomposed flow analysis applied to a separate lower-body region - detects leg-driven attacks (kicks, tackles) that are invisible in the torso ROI.

```python
# Same two-stage decomposition as torso ROI (see Section 3.3),
# but applied to a rectangle offset downward to cover hips/legs,
# and using the lower ROI's own centre for decomposition.

translation_signed_lower # Positive = legs approaching camera (kick/tackle run-up)
translation_lower # ||v_med|| - translational speed of lower body
divergence_lower # median(radial_resid > 0) - lower body expanding in frame
div_ratio_lower # Expansion fraction of lower-body residual flow
```
*Rationale*: Separate lower-body tracking captures kicks and tackles that produce distinct leg-forward flow patterns not visible in the torso ROI. Using the lower ROI's own centre is critical: a forward leg stride from the torso centre would appear mostly tangential, but from the lower-body centre it correctly appears as a strong outward (radial) motion. See Section 3.3.4.

#### 3.2.7 Optical Flow - Background (2 dims, indices 40-41)

Camera-motion characterisation and flow validity flag - used both for background subtraction (upstream) and as context features telling the model how much camera movement is present.

```python
bg_flow_coherence # mean(v_bg) / (||mean(v_bg)|| + eps) - coherence of background motion
flow_ok # 1 if LK tracking succeeded for this frame, 0 otherwise
```
*Note*: Raw background flow magnitude is intentionally excluded - it was a dataset confounder (body-worn cameras from different environments had different typical background motion levels). Background vectors are subtracted from ROI flow before decomposition; `bg_flow_coherence` captures whether camera motion is translational (coherent) vs shaky.

#### 3.2.8 Posture (3 dims, indices 42-44)

Body shape and orientation features that capture attack-preparation stances not expressed by raw skeletal distances alone - torso lean, arm asymmetry, and face-on orientation.

```python
torso_compression_ratio # Ratio of torso height to width; drops when person leans forward
wrist_height_asymmetry # |left_wrist_y - right_wrist_y| / torso_height
face_visibility # Keypoint confidence of nose/eyes as proxy for face-on orientation
```
*Rationale*: Forward lean, asymmetric wrist height, and face-on orientation are common attack-preparation postures not captured by raw distances.

#### 3.2.9 Dynamics (3 dims, indices 45-47)

Temporal derivatives of wrist extension and a pose-derived proximity estimate - captures the speed and acceleration of a strike motion, and provides a size-stable distance signal.

```python
max_wrist_extension_velocity # Max per-wrist extension speed (d(dist_wrist_torso)/dt)
max_wrist_extension_accel # Max per-wrist extension acceleration
log_scale # log(torso_height_px) - pose-derived apparent size;
 # scale-invariant to arm raises unlike log_area
```
*Note*: `max_wrist_extension_velocity` (index 45) and `max_wrist_extension_accel` (index 46) are 0.0 placeholders in `build_features()`; filled from frame history alongside the bbox derivatives.

*Rationale*: Wrist velocity and acceleration capture the dynamics of a strike wind-up. `log_scale` provides a proximity estimate invariant to arm raises (unlike bbox area which grows when arms extend).

#### 3.2.10 Body-shape Extras (8 dims, indices 48-55)

Pose-shape statistics that capture body posture and limb deployment in a way that survives missing keypoints (lenient single-side fallbacks where possible).

```python
torso_height_px # Lenient single-side fallback torso height in pixels
wrist_y_rel # Average wrist vertical position relative to torso
d_wrist_y_rel_dt # Time derivative of wrist_y_rel (placeholder; filled from history)
ankle_spread # Ankle separation, normalised
d_ankle_spread_dt # Time derivative of ankle_spread (placeholder; filled from history)
nose_y_rel # Nose vertical position relative to torso
d_nose_y_rel_dt # Time derivative of nose_y_rel (placeholder; filled from history)
upper_lower_async # Upper-vs-lower body asynchrony score
```
*Rationale*: Lenient `torso_height_px` (`Strict log_scale` lives at index 47) preserves a usable scale signal even when one shoulder/hip is missing. The wrist/nose/ankle vertical relatives capture postural cues (raised hands, ducked head, widened stance) that the prior shape descriptors missed. The asynchrony term flags wind-ups where the upper body and lower body move on different beats.

#### 3.2.11 Interaction Features (3 dims, indices 56-58)

Multiplicative combinations of motion and proximity signals - encodes the principle that the same movement is only threatening when the person is already close.

```python
approach_rate # max(dlog_scale_dt, 0) x max(translation_signed_torso, 0)
 # zero for distant persons and for retreating persons
expansion_proximity # divergence_torso x proximity_weight
 # where proximity_weight = exp(log_scale x 0.1)
acceleration_proximity # max_wrist_extension_accel x proximity_weight
```
*Rationale*: Motion features are only threatening when the person is close. These interaction terms gate the strongest approach/strike signals by proximity, reducing false positives from distant rapid motion.

### 3.3 Optical Flow Algorithm: Translation-Decomposed Radial Analysis

#### 3.3.1 Design Motivation

Body-worn cameras create a specific optical flow problem that naive radial/tangential decomposition cannot solve cleanly.

**The naive radial decomposition problem:**

When a person walks directly toward the camera, every tracked point inside their bounding box generates an outward flow vector (expansion). This is the desired signal. However, the person also has a global translational component - the entire bounding box shifts downward as the person approaches (the apparent size grows *and* the bounding box centroid moves). For points on the *left* and *right* sides of the torso, a purely downward translation contributes to their tangential component (perpendicular to the radial direction), inflating the tangential term and suppressing the radial ratio. The result: a direct approach looks *less* radial than it really is, because the translational component is partly misclassified as tangential.

**The fix: separate translation from expansion first.**

The person's body motion in the image can be decomposed into two components:

1. **Global translation** - the entire ROI moving as a rigid body (camera approaching or person walking toward camera, lateral sway, etc.)
2. **Local deformation (expansion)** - points spreading outward from the ROI centre as the person fills more of the frame

By separating these before computing the radial decomposition, each component can be measured cleanly and independently.

---

#### 3.3.2 Algorithm: Two-Stage Flow Decomposition

**Stage 1 - Background compensation:**

```
v_bg = LK flow of ~200 points sampled outside the person's bbox (+ margin)
v_bg_med = median(v_bg) # Robust estimate of camera translation
v_roi_compensated = v_roi - v_bg_med # Remove camera motion from ROI flow
```

`v_bg_med` is the median background flow vector: a robust estimate of rigid camera motion. Subtracting it from each ROI flow vector removes camera shake, panning, and walking-induced background shift before any further analysis.

---

**Stage 2 - Translation/expansion separation within the ROI:**

After background compensation, the remaining ROI flow `v` still contains both:
- The person's own translational motion relative to the camera (walking toward/away)
- Local body expansion (person growing larger in frame)

These are separated using the median of the compensated ROI flow:

```
v_med = median(v_roi_compensated) # = person's translational component
v_resid = v_roi_compensated - v_med # = residual deformation (expansion)
```

The **median is robust** to the non-uniform nature of body flow: limbs move differently from the torso, but the dominant rigid-body translation dominates the median.

---

**Stage 3 - Feature extraction from each component:**

From the **translational component** (`v_med`):

```
# Project median flow onto each point's outward radial direction:
d = p0 - center # vector from ROI centre to each tracked point
r = ||d|| + eps

radial_translation = (d - v_med) / r # per-point radial projection of translation

translation_signed = mean(radial_translation)
 # Positive -> ROI uniformly expanding via translation = person approaching camera
 # Negative -> ROI uniformly contracting = person retreating

translation_mag = ||v_med|| # Speed of translational motion
```

`translation_signed` is the cleanest approach signal: it is positive if and only if the entire ROI is moving outward (away from its centre), which occurs when the person is getting closer. It is negative when retreating.

From the **residual component** (`v_resid`):

```
radial_resid = (d - v_resid) / r # per-point radial of deformation flow

divergence = median(radial_resid[radial_resid > 0])
 # Median magnitude of outward-only residual radial components (robust)
 # Positive when body surface points are locally spreading outward
 # Zero when there is no net expansion in the residual

resid_mag = mean(||v_resid||)
div_ratio = divergence / (divergence + resid_mag + eps)
 # ~ 1.0 -> residual flow is predominantly outward expansion
 # ~ 0.0 -> residual is tangential (arm waving, rotation) or zero
```

`divergence` captures **intra-body expansion**: e.g., a punch extends one arm outward from the torso centre, generating a strong positive radial residual in the direction of the strike. `div_ratio` normalises by total residual magnitude to distinguish a genuine expansion from random limb movement.

---

#### 3.3.3 Why This Decomposition Is Better Than Alternatives

| Approach | Translation signal | Expansion signal | Camera-motion robust | Lateral rejection |
|----------|--------------------|-----------------|---------------------|-------------------|
| Raw flow magnitude | Conflated | Conflated |  No |  No |
| Bbox area growth | Indirect (1D) | Indirect (1D) | Partial |  No |
| Simple radial/tangential on raw flow | Polluted by translation | Polluted by translation | Partial | Partial |
| Dense divergence field (dfx/dx + dfy/dy) |  Absent | Noisy |  No |  No |
| **This approach** |  Clean (`translation_signed`) |  Clean (`divergence`) |  Yes (2-stage) |  Yes (`div_ratio`) |

Key advantages:
- **No confusion between translation and expansion**: a person walking toward the camera at constant size (distant, zoomed out) gives high `translation_signed` but low `divergence`. A person standing still while throwing a punch gives low `translation_signed` but high `divergence`. Both are correctly captured.
- **Two independent threat indicators**: the GRU can learn that high `translation_signed` *combined with* high `divergence` is the most dangerous pattern (fast approach + arm extension), while either alone may be benign.
- **Numerically stable**: using median (not mean) for the translation estimate is robust to outlier flow vectors from limb tips and clothing edges.

---

#### 3.3.4 Multi-ROI Strategy

Two separate ROIs are processed independently:

**Torso ROI** (`translation_signed_torso`, `translation_torso`, `divergence_torso`, `div_ratio_torso`):
- Square region centred on the torso centre (shoulder-hip midpoint), side = 1.2 x torso height
- Centre of decomposition = torso centre
- Captures: approach speed, upper-body expansion, punch/grab wind-up (arm extends outward from torso centre)

**Lower ROI** (`translation_signed_lower`, `translation_lower`, `divergence_lower`, `div_ratio_lower`):
- Rectangle **anchored at the actual hip midpoint**: `hip_mid_y = mean(l_hip.y, r_hip.y)` when both hips are visible; falls back to `torso_c[1] + 0.5 x torso_s` when hips are occluded
- Width = 1.5 x torso_s, centred horizontally on the torso; extends **2.0 x torso_s downward** from `hip_mid_y` (covers hips -> ankles)
- Centre of decomposition = lower ROI centre (not torso centre)
- Captures: leg/hip motion, kick preparation (leg extends outward from lower-body centre), tackle approach
- Note: the previous design anchored the lower ROI 0.4 x torso_s below torso centre, causing ~67% overlap with the torso ROI and failing to cover the legs; anchoring at `hip_mid_y` eliminates the overlap and ensures feet coverage

Using the **lower ROI's own centre** is important: if the torso centre were used instead, a forward leg stride (kick preparation) would appear mostly tangential (the leg moves downward, which is perpendicular to the radial direction from the torso), losing the approach signal. With the lower-body centre, the same leg stride correctly appears as a strong outward (radial) motion.

---

#### 3.3.5 Full Per-Frame Pipeline

```
Frame t-1, Frame t

Pose detection -> torso_center, lower_center, bbox

Sample ~200 background points (outside bbox + 20px margin)

Sample ~200 points inside torso ROI
Sample ~200 points inside lower ROI

Lucas-Kanade tracking: v_bg, v_torso, v_lower

Background estimation: v_bg_med = median(v_bg)
Background coherence: bg_coh = ||mean(v_bg)|| / (mean(||v_bg||) + eps)

Background subtraction:
 v_torso_comp = v_torso - v_bg_med
 v_lower_comp = v_lower - v_bg_med

radial_tangential_stats(p0_torso, v_torso_comp, torso_center)
 -> translation_signed_torso, translation_torso,
 divergence_torso, div_ratio_torso

radial_tangential_stats(p0_lower, v_lower_comp, lower_center)
 -> translation_signed_lower, translation_lower,
 divergence_lower, div_ratio_lower

Output: 10 flow features (indices 32-41)
```

### 3.4 Mask Vector

Each feature has a corresponding mask bit indicating validity:

```python
mask[i] = 1 # Feature is valid (computed from confident keypoints)
mask[i] = 0 # Feature is invalid (keypoint missing, imputed from previous frame)
```

The mask is concatenated with features before feeding to GRU:
```python
input = concatenate([features, mask], axis=-1) # [59] + [59] = [118]
```

This allows the model to learn the reliability of each feature dynamically.

### 3.5 Feature Importance (From Training)

Feature importance is computed via permutation importance after training (see `src/feature_importance.py`). The permutation importance measures the drop in F1 score when each feature is randomly shuffled.

**Top 5 features by permutation importance (measured):**

1. **`expansion_proximity`** (index 57) - 24.7%: `divergence_torso x proximity_weight`; body expansion from close range is the single strongest combined threat signal.
2. **`torso_height_px`** (index 48) - 20.8%: lenient pose-derived apparent size; a direct proximity signal that survives single-side keypoint loss.
3. **`divergence_torso`** (index 34) - 9.9%: median positive radial of torso ROI residual flow; captures upper-body expansion (arm extension, lean).
4. **`divergence_lower`** (index 38) - 7.3%: same expansion measure for the lower-body ROI; captures kick/tackle leg deployment.
5. **`acceleration_proximity`** (index 58) - 7.1%: `max_wrist_extension_accel x proximity_weight`; explosive wrist acceleration gated by proximity.

The top 5 account for ~69.8% of total importance. Proximity-related features (`expansion_proximity`, `torso_height_px`, `acceleration_proximity`) and torso/lower-body flow divergence dominate, confirming that **proximity-gated motion and body expansion** are the most discriminative patterns.

Run `python -m src.train` to regenerate the full ranked importance list for the current model.

---

## 4. Model Architecture

### 4.1 Network Structure

```
Input: [batch, window_len, input_dim]
 [B, 5, 118]



┌──────────────────────────────────────┐
│ GRU Layer (1 layer) │
│ - input_size: 118 │
│ - hidden_size: 64 │
│ - num_layers: 1 │
│ - batch_first: True │
│ - dropout: 0.0 (no effect, 1 layer) │
└──────────────────────────────────────┘


[B, 5, 64] -> Take last timestep -> [B, 64]



┌──────────────────────────────────────┐
│ Dropout (0.25) │
└──────────────────────────────────────┘


┌──────────────────────────────────────┐
│ Fully Connected (64 -> 1) │
└──────────────────────────────────────┘


┌──────────────────────────────────────┐
│ Sigmoid Activation │
└──────────────────────────────────────┘


Output: [batch] in [0, 1]
 Hazard score
```

### 4.2 Model Specification

```python
class HazardGRU(nn.Module):
 def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.25):
 super().__init__()

 # GRU layer
 self.gru = nn.GRU(
 input_size=input_dim, # 118 (59 features + 59 masks)
 hidden_size=hidden, # 64
 num_layers=1,
 batch_first=True
 )

 # Dropout (applied AFTER GRU for single layer)
 self.dropout = nn.Dropout(dropout)

 # Output layer
 self.fc = nn.Linear(hidden, 1)

 def forward(self, x):
 # x: [B, T, D] = [batch, 5, 118]
 out, _ = self.gru(x) # [B, T, H] = [B, 5, 64]
 h_last = out[:, -1, :] # [B, H] = [B, 64]
 h_last = self.dropout(h_last) # [B, 64]
 y = self.fc(h_last) # [B, 1]
 y = torch.sigmoid(y) # [B, 1] in [0, 1]
 return y.squeeze(1) # [B]
```

### 4.3 Parameter Count

```
GRU:
 - Input weights: 118 x (64 x 3) = 22,656
 - Hidden weights: 64 x (64 x 3) = 12,288
 - Biases: 64 x 3 x 2 = 384
 - Total GRU: ~ 35,328

Dropout: 0 parameters

FC:
 - Weights: 64 x 1 = 64
 - Bias: 1
 - Total FC: 65

Total Parameters: ~ 35K
Model Size: <1 MB
```

**Other available architectures** (same input/output contract, selected via Config):

- `HazardLSTM` - drop-in LSTM replacement for the GRU layer. Achieves higher window-level F1 than GRU on both within-domain and cross-domain test sets, but is more expensive at inference. Used as the accuracy reference; GRU is preferred for on-device deployment.
- `HazardTransformer` - small encoder stack with positional encoding. Included for window-level within-domain comparison only; discarded after that stage and not used in cross-domain evaluation or deployment.

All variants accept `[B, 5, 118]` and emit `[B] in [0, 1]`.

### 4.4 Why GRU for Deployment?

The codebase trains and evaluates **both GRU and LSTM**. On window-level F1, LSTM outperforms GRU on both the within-domain and cross-domain test sets. At the deployment (video-stream) level on the Raspberry Pi 5 + Hailo-8 platform, however, the GRU attains a higher detection rate, a lower false-positive rate, and a shorter detection delay, which is why it is the deployment default. Additional engineering benefits:

- Fewer parameters and lower per-step compute (simpler gating)
- Faster CPU-side inference, leaving more headroom for the pose/flow stages
- Smaller memory footprint, easier to keep within the streaming latency budget

**Transformer (discarded)**: A small Transformer encoder variant was implemented for window-level within-domain comparison. Its accuracy did not justify the added cost, and it was dropped after that stage - not carried into cross-domain evaluation or deployment.

**Advantages over Simple RNN** (applies to both GRU and LSTM):
- Better gradient flow (no vanishing gradient)
- Gated memory captures longer dependencies than vanilla RNN
- Stable training on small datasets

### 4.5 Input Representation

The model receives **concatenated features and masks**:

```python
# Features: [B, T, 59]
# Masks: [B, T, 59]
# Input: [B, T, 118] = concatenate([features, masks], axis=-1)
```

This design allows the GRU to:
1. Learn feature values from the first 59 dimensions
2. Learn feature reliability from the next 59 dimensions
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
BCE(p, y) = -y-log(p) - (1-y)-log(1-p)
```

Problems:
- **Class imbalance**: Safe examples outnumber attack examples
- **Easy examples dominate**: Model focuses on already-correct predictions
- **Hard examples neglected**: Subtle pre-contact patterns are ignored

#### 5.1.2 Focal Loss Definition

```
FL(p_t) = -alpha_t - (1 - p_t)^gamma - log(p_t)

where:
 p_t = p if y = 1 (attack)
 = 1-p if y = 0 (safe)

 alpha_t = alpha if y = 1
 = 1-alpha if y = 0
```

**Parameters**:
- **gamma (gamma)**: Focusing parameter
 - gamma = 0: Equivalent to BCE
 - gamma = 2: Standard (RetinaNet paper) - our setting
 - Higher gamma -> more focus on hard examples

- **alpha (alpha)**: Class balance weight
 - alpha = 0.5: No class weighting
 - alpha = 0.75: Emphasize positive class (our setting)
 - Higher alpha -> prioritize attack detection

#### 5.1.3 How Focal Loss Works

**Weighting by Difficulty**:
```
Easy example (p_t = 0.95): (1 - 0.95)^2 = 0.0025 -> Heavily down-weighted
Medium (p_t = 0.7): (1 - 0.7)^2 = 0.09 -> Some weight
Hard (p_t = 0.5): (1 - 0.5)^2 = 0.25 -> Full weight
Very hard (p_t = 0.3): (1 - 0.3)^2 = 0.49 -> Emphasized
```

With gamma=2, easy examples (model confident and correct) are down-weighted by ~100x compared to hard examples.

#### 5.1.4 Implementation

```python
def focal_loss(pred, target, gamma=2.0, alpha=0.75):
 """
 Focal Loss for binary classification.

 Args:
 pred: predicted probabilities [batch_size] in [0, 1]
 target: ground truth labels [batch_size] in {0, 1}
 gamma: focusing parameter (default: 2.0)
 alpha: positive class weight (default: 0.75)
 """
 # BCE loss without reduction
 bce = F.binary_cross_entropy(pred, target, reduction='none')

 # Compute p_t (probability of correct class)
 p_t = pred * target + (1 - pred) * (1 - target)

 # Compute alpha_t (weight for correct class)
 alpha_t = alpha * target + (1 - alpha) * (1 - target)

 # Focal loss: alpha_t * (1 - p_t)^gamma * BCE
 focal = alpha_t * ((1 - p_t) ** gamma) * bce

 return focal.mean()
```

#### 5.1.5 Why Focal Loss for This Problem

1. **Class Imbalance**: Safe windows outnumber attack windows (~1.3:1 after stride adjustment)
2. **Hard Example Mining**: Pre-contact behavior is subtle and easily missed
3. **Recall Priority**: Missing an attack is worse than false alarms (alpha=0.75 prioritizes attacks)
4. **Gradient Focus**: Forces model to learn from challenging pre-contact patterns

**Empirical Results** (historical sweep used to pick gamma/alpha):
- Weighted BCE (gamma=0, alpha=0.57): baseline pre-contact rate, near-zero lead
- Focal Loss (gamma=2, alpha=0.57): improved pre-contact rate and lead time
- Focal Loss (gamma=2, alpha=0.75): **chosen setting** - best F1 with positive-class emphasis

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
# Safe videos: stride = 3 (reduce redundancy while keeping enough samples)
safe_windows = windowize(safe_features, window_len=5, stride=3)

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
Issue: Adjacent windows share 4/5 frames -> validation sees training data!
```

**Solution**: Split at video level
```python
# Group windows by source video
video_data = [(video1_windows, video1_labels),
 (video2_windows, video2_labels), ...]

# Split videos (not windows)
train_videos, val_videos = split_videos(video_data, val_fraction=0.20)

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
Video A: 500 frames -> 496 windows (stride=1)
Video B: 50 frames -> 46 windows (stride=1)

Random 20% video split might give 10% or 30% window split!
```

**Solution**: Stratified split by window count
```python
def split_by_windows(video_list, val_fraction=0.20):
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
input_dim = 118 # 59 features + 59 masks
hidden = 64 # GRU hidden units
dropout = 0.25 # Dropout rate

# Optimization
lr = 1e-3 # Learning rate
batch_size = 64 # Mini-batch size
epochs = 40 # Training epochs
optimizer = Adam # Adaptive learning rate

# Loss
focal_gamma = 2.0 # Focusing parameter
focal_alpha = 0.75 # Attack class weight

# Early stopping
patience = 10 # Stop if no F1 improvement
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
├── hazard_gru_YYYYMMDD_HHMMSS.pt # Model weights (best F1, timestamped)
└── meta.json # {"input_dim": 118, "best_threshold": 0.65, "model_file": "hazard_gru_...pt"}
```

---

## 6. Inference Pipeline

### 6.1 Real-time Processing

```python
# Initialize
detector = PoseDetector()
tracker = SingleTargetTracker()
model = load_model("outputs/checkpoints/hazard_gru.pt")
buffer = deque(maxlen=5) # Sliding window
hazard_ema = 0.0 # EMA state
persist = 0 # Persistence counter

# Process video stream
for frame in video_stream:
 # 1. Pose detection
 det = detector.infer(frame)
 if det is None:
 continue

 # 2. Tracking
 bbox = tracker.update(det["bbox"])

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

 # 8. Single-threshold alerting
 if hazard_ema > early_thresh and persist >= early_persist:
 alert = "THREAT"
 else:
 alert = "NONE"

 # 9. Output
 print(f"t={timestamp} hazard={hazard_ema:.3f} state={alert}")
```

### 6.2 Post-processing

#### 6.2.1 EMA Smoothing

```python
# Exponential Moving Average
alpha = 0.7 # Responsiveness (higher = more responsive)
hazard_ema = (1 - alpha) * hazard_ema + alpha * hazard_current
```

**Purpose**:
- Reduce frame-to-frame jitter
- Smooth out brief spikes (noise)
- Preserve true trends

**Effect**:
```
Frame: 1 2 3 4 5 6 7 8
Raw: 0.2 0.8 0.3 0.7 0.9 0.8 0.7 0.6
EMA: 0.2 0.6 0.4 0.6 0.8 0.8 0.7 0.7
```

#### 6.2.2 Persistence Logic

```python
# Require threshold exceeded for multiple frames
if hazard_ema > early_thresh:
 persist += 1
else:
 persist = max(0, persist - 1) # Decay, but don't go negative

# Trigger alert only if persist >= threshold
if persist >= early_persist: # e.g., 2 frames
 alert = "PRE-CONTACT"
```

**Purpose**:
- Reduce false alarms from brief spikes
- Require sustained elevated hazard
- Allow brief dips without clearing alert

**Effect**:
```
Frame: 1 2 3 4 5 6 7 8
Hazard: 0.3 0.6 0.4 0.6 0.7 0.6 0.5 0.4
Exceed: No Yes No Yes Yes Yes Yes No
Persist: 0 1 0 1 2 3 4 3
Alert: No No No No YES YES YES YES
```

#### 6.2.3 Single THREAT Threshold

```python
# Single tuned threshold (default 0.50; the training run replaces this
# with the value chosen by the validation F1 sweep, stored in meta.json)
threshold = early_thresh # e.g. 0.50

if hazard_ema >= threshold and persist >= early_persist:
 state = "THREAT"
else:
 state = "NONE"
```

**Purpose**:
- One unambiguous binary alert state for downstream consumers
- Threshold is calibrated per trained model (saved in `meta.json`)
- Persistence requirement guards against single-frame spikes

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
buffer = deque(maxlen=5) # Automatically removes old frames

# Process frames at reduced resolution for pose detection
frame_resized = cv2.resize(frame, (640, 480))

# Use float16 for inference (on GPU)
model.half()
input = input.half()
```

---

## 7. Evaluation Metrics

### 7.1 Detection Metrics

Evaluation in the paper is window-level (each 5-frame window scored
independently), reported on a within-domain test set and a cross-domain
set (FALEBaction). The numbers below are from the paper.

#### Detection Rate (Recall)
```
Recall = TP / (TP + FN)

Within-domain @ selected threshold:  GRU 0.677   LSTM 0.672
Cross-domain, push phase:            GRU 98.8%   LSTM 97.7%
Cross-domain, preparatory phase:     GRU 77.4%   LSTM 76.6%
```

#### False Positive Rate
```
Within-domain false-positive %:      GRU 0.66%   LSTM 0.68%
Cross-domain tranquil-zone FP:       GRU 5.9%    LSTM 2.9%
```

### 7.2 Warning Metrics

#### Early vs. late detection
```
The paper does not report a single mean lead time. Instead it classifies
each cross-domain detection by the zone it fires in: a detection is "early"
if it lands in the preparatory zone or before the contact frame in the push
zone, and "late" at or after contact.

Cross-domain detection rates by zone:
 Preparatory (pre-contact):  GRU 77.4%   LSTM 76.6%
 Push (imminent contact):    GRU 98.8%   LSTM 97.7%
```

### 7.3 Window-level Results

Window-level performance at the selected threshold (from the paper):

| Metric | GRU | LSTM | Transformer |
|--------|-----|------|-------------|
| Threshold | 0.65 | 0.60 | 0.65 |
| Accuracy | 0.915 | 0.913 | 0.896 |
| Precision | 0.972 | 0.971 | 0.996 |
| Recall | 0.677 | 0.672 | 0.584 |
| F1 | 0.798 | 0.794 | 0.736 |
| False-positive % | 0.66% | 0.68% | 0.09% |
| Average Precision | 0.906 | 0.903 | 0.891 |
| ROC-AUC | 0.938 | 0.936 | 0.917 |

The Transformer reaches the highest precision but its lower recall/F1 make it
too conservative, so the recurrent models are preferred for deployment.

### 7.4 F1 Score
```
F1 = 2 * Precision * Recall / (Precision + Recall)

Precision = TP / (TP + FP)   # minimize false alarms
Recall = TP / (TP + FN)      # maximize attack detection

Within-domain F1: GRU 0.798, LSTM 0.794 (see the table above).
Cross-domain AUC-ROC: GRU 0.904, LSTM 0.931.
```

---

## 8. Implementation Details

### 8.1 Carry-forward Imputation

```python
class FeatureState:
 def __init__(self):
 self.prev = None # Last valid value per feature

 def init(self, dim):
 self.prev = np.zeros((dim,), dtype=np.float32)

 def impute(self, x, valid):
 """
 Freeze each feature at its last valid value when it is invalid.

 Returns:
 the input value where valid, the last valid value elsewhere
 """
 out = np.where(valid > 0.5, x, self.prev).astype(np.float32)
 self.prev = out
 return out
```

**Example**:
```
Frame: 1 2 3 4 5 6
Raw value: - 5.0 - - 8.0 -
Valid: No Yes No No Yes No
Imputed: 0.0 5.0 5.0 5.0 8.0 8.0

 init valid cf cf valid cf

cf = carry forward; there is no limit on the carry length
```

Carry-forward keeps the input sequence smooth across dropped keypoints, which
matters because most of the feature vector feeds temporal derivatives. Zero
filling would inject a spurious step change into every derivative on the frame
a keypoint is lost and again on the frame it returns. The validity mask is
passed to the model alongside the imputed value, so an imputed reading remains
distinguishable from a measured one.

Note that this imputation is applied in `dataset.py` during training only.
`evaluate.py` and `infer.py` do not construct a `FeatureState`, so at inference
an invalid feature reaches the model as its raw default rather than its last
valid value.

### 8.2 Bounding Box Derivatives

```python
# Position and size derivatives require temporal information.
# build_features() leaves x[10] and x[11] as 0.0 placeholders.
# dataset.py fills them in batch; evaluate.py/infer.py fill them frame-by-frame.

log_area = X[:, 9] # index 9: log_area (always valid)
dt = 0.1 # Time step (0.1s @ 10 FPS)

# First derivative (velocity) - valid from frame 1
dlog_area_dt = zeros_like(log_area)
dlog_area_dt[1:] = (log_area[1:] - log_area[:-1]) / dt

# Second derivative (acceleration) - valid from frame 2
d2log_area_dt2 = zeros_like(log_area)
d2log_area_dt2[2:] = (dlog_area_dt[2:] - dlog_area_dt[1:-1]) / dt

# Update feature array
X[:, 10] = dlog_area_dt # index 10: dlog_area_dt
X[:, 11] = d2log_area_dt2 # index 11: d2log_area_dt2
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
 return empty arrays # Video too short

 windows_X, windows_M, windows_y = [], [], []

 for i in range(0, T - window_len + 1, stride):
 # Extract window
 window_x = X[i : i + window_len] # [window_len, D]
 window_m = M[i : i + window_len] # [window_len, D]
 window_y = y[i + window_len - 1] # Last frame label

 windows_X.append(window_x)
 windows_M.append(window_m)
 windows_y.append(window_y)

 return stack(windows_X), stack(windows_M), array(windows_y)
```

**Why use last frame label?**
- Label represents hazard at END of window
- Model sees context (5 frames) and predicts current hazard
- Consistent with online inference (buffer of past frames -> current prediction)

### 8.4 Video-Level Split Implementation

```python
def split_by_videos(video_data, val_fraction=0.20):
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

### 9.1 Reported Results

Window-level metrics are in section 7.3. Headline figures from the paper:

```
Within-domain (GRU / LSTM):
 F1:        0.798 / 0.794
 Precision: 0.972 / 0.971
 Recall:    0.677 / 0.672
 ROC-AUC:   0.938 / 0.936

Cross-domain (FALEBaction, GRU / LSTM):
 Push-phase detection:        98.8% / 97.7%
 Preparatory-phase detection: 77.4% / 76.6%
 AUC-ROC:                     0.904 / 0.931

Deployment: ~10 Hz end-to-end on Raspberry Pi 5 + Hailo-8.
```

### 9.2 Error Analysis

Common failure modes observed during development:

- **Subtle approaches with no visible wind-up** - the hazard score stays
  below threshold when there is little pose or flow signal before contact.
- **Partial occlusion / cropped subjects** - when the person is mostly
  off-screen the bbox and keypoints degrade, weakening the features.
- **Label ambiguity near onset** - early wind-up can fire a frame or two
  before the labeled onset; some of these are arguably correct.
- **Rapid attacks** - when onset-to-contact is shorter than the 5-frame
  window there is little pre-contact signal, pushing detection late.

### 9.3 Feature Importance Analysis

Feature importance is computed automatically at end of training via `src/feature_importance.py` (permutation importance over 5 repeats). Results are saved to `outputs/logs/feature_importance_*.json`.

The current 59-feature / 118-dim model produces the Top 5 importances reported in Section 3.5. Re-run training to refresh.

**Design-intent expected ranking** (qualitative):
- Optical flow divergence features (indices 34, 38) - approach / expansion signal
- Bbox growth rate (index 10) - proximity signal
- Body-shape extras (indices 48-55) - torso scale and posture cues
- Interaction features (indices 56-58) - combined approach + wrist signal
- Wrist-to-torso distances (indices 19-20) - extension signal
- Elbow angles (indices 23-24) - strike preparation posture

### 9.4 Threshold Selection

The operating threshold is chosen per model by sweeping thresholds on the
validation set and picking the highest-F1 point (ties broken by recall, then
higher threshold). Lower thresholds raise recall and give earlier warnings at
the cost of precision; higher thresholds do the reverse.

Selected thresholds (from the paper): GRU 0.65, LSTM 0.60, Transformer 0.65.
The value is stored in `outputs/checkpoints/meta.json` and read at inference.

---

## 10. Design Decisions

### 10.1 Why 5-frame Windows?

**Considered**: 3, 5, 8, 10 frames

**Chosen**: 5 frames (0.5s @ 10 FPS)

**Rationale**:
- Captures short-term temporal patterns
- Small enough to localize attack onset
- Large enough for motion estimation
- Fast inference (small input size)
- Less prone to overfitting

**Trade-offs**:
- May miss longer-term context (pre-onset posture)
- Sensitive to rapid attacks

**Alternatives considered**:
- 3 frames: Too short for reliable motion patterns
- 8-10 frames: Better context, but slower inference and more overfitting risk

### 10.2 Why Stride Differs for Safe vs Attack?

**Safe videos**: stride = 3
**Attack videos**: stride = 1

**Rationale**:
- Safe videos: Mostly redundant frames (little change)
- Attack videos: Sequential behavior (each frame matters)
- Balances dataset (1.3:1 ratio without excessive safe redundancy)
- Preserves attack temporal coherence

**Alternative considered**:
- Random sampling: Breaks temporal coherence, loses sequential patterns
- Equal stride: Either too many safe samples or too few attack samples

### 10.3 GRU vs LSTM (and the Discarded Transformer)

The codebase supports both GRU and LSTM. They are not alternatives - both are kept and used at different stages:

- **LSTM**: higher window-level F1 on within-domain and cross-domain test sets; used as the accuracy reference.
- **GRU**: deployment default on Raspberry Pi 5 + Hailo-8. At the deployment (video-stream) level, the GRU attains a higher detection rate, a lower false-positive rate, and a shorter detection delay than the LSTM. It also has fewer parameters, lower per-step compute, and a smaller memory footprint.

**Trade-offs**:
- GRU has slightly lower window-level F1 than LSTM
- LSTM is more expensive per step, eating into the on-device latency budget

**Transformer** (discarded): A small Transformer encoder variant was trained and compared at the window-level within-domain stage only. It did not provide enough benefit to justify the added cost, so it was not carried into the cross-domain evaluation or deployment.

### 10.4 Why Concatenate Features and Masks?

**Alternatives considered**:
1. **Impute only, no mask**: Model doesn't know what's imputed
2. **Masked loss**: Doesn't inform model during inference
3. **Separate encoders**: More complex, more parameters

**Chosen**: Concatenate [features, masks]

**Rationale**:
- Simple and effective
- Model learns to weight reliable features
- No additional complexity
- Works well in practice

### 10.5 Why Focal Loss over Weighted BCE?

**Progression**:
1. Standard BCE: baseline
2. Weighted BCE (alpha=0.57): improved recall
3. Focal Loss (gamma=2, alpha=0.57): better hard-example weighting
4. Focal Loss (gamma=2, alpha=0.75): **chosen** - best F1 + lead time balance

**Rationale**:
- Focuses on hard examples (subtle pre-contact behavior)
- Down-weights easy examples (obvious attacks)
- Better than static class weighting
- Prioritizes recall (alpha=0.75)

### 10.6 Why Video-Level Split?

**Problem**: Window-based split causes data leakage
- Adjacent windows share 4/5 frames
- Validation set "sees" training data

**Solution**: Split at video level

**Rationale**:
- No data leakage
- True generalization test
- Realistic evaluation (unseen videos)

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
| 10 | dlog_area_dt | d(log_area)/dt - filled by dataset/evaluate/infer | Frame >=1 |
| 11 | d2log_area_dt2 | d^2(log_area)/dt^2 - filled by dataset/evaluate/infer | Frame >=2 |
| 12 | bbox_center_x | Normalised bbox centre x | Always |
| 13 | bbox_center_y | Normalised bbox centre y | Always |
| 14 | bbox_center_vx | Bbox centre x velocity | Frame >=1 |
| 15 | bbox_center_vy | Bbox centre y velocity | Frame >=1 |
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
| 34 | divergence_torso | median(max(radial,0)) in torso ROI | Flow |
| 35 | div_ratio_torso | Radial/(radial+tangential) in torso ROI | Flow |
| 36 | translation_signed_lower | Signed net translation in lower ROI | Flow |
| 37 | translation_lower | Translation magnitude in lower ROI | Flow |
| 38 | divergence_lower | median(max(radial,0)) in lower ROI | Flow |
| 39 | div_ratio_lower | Radial/(radial+tangential) in lower ROI | Flow |
| 40 | bg_flow_coherence | Background flow coherence | Flow |
| 41 | flow_ok | Flow validity flag | Always |
| 42 | torso_compression_ratio | Torso height/width ratio | Pose |
| 43 | wrist_height_asymmetry | |left_wrist_y - right_wrist_y| / torso_height | Pose |
| 44 | face_visibility | Nose/eye keypoint confidence (face-on proxy) | Pose |
| 45 | max_wrist_extension_velocity | Max per-wrist extension speed - filled by dataset/evaluate/infer | Frame >=1 |
| 46 | max_wrist_extension_accel | Max per-wrist extension acceleration - filled by dataset/evaluate/infer | Frame >=2 |
| 47 | log_scale | log(torso_height_px) - strict pose-derived apparent size | Pose |
| 48 | torso_height_px | Lenient single-side fallback torso height (px) | Pose |
| 49 | wrist_y_rel | Avg wrist vertical position relative to torso | Pose |
| 50 | d_wrist_y_rel_dt | Time derivative of wrist_y_rel - filled by dataset/evaluate/infer | Frame >=1 |
| 51 | ankle_spread | Ankle separation, normalised | Pose |
| 52 | d_ankle_spread_dt | Time derivative of ankle_spread - filled by dataset/evaluate/infer | Frame >=1 |
| 53 | nose_y_rel | Nose vertical position relative to torso | Pose |
| 54 | d_nose_y_rel_dt | Time derivative of nose_y_rel - filled by dataset/evaluate/infer | Frame >=1 |
| 55 | upper_lower_async | Upper-vs-lower body asynchrony score | Pose |
| 56 | approach_rate | max(dlog_scale_dt,0) x max(translation_signed_torso,0) | Computed |
| 57 | expansion_proximity | divergence_torso x proximity_weight | Computed |
| 58 | acceleration_proximity | max_wrist_extension_accel x proximity_weight | Computed |

**Note**: Indices 10-11 (dlog_area_dt, d2log_area_dt2), 45-46 (wrist velocity/acceleration), 50/52/54 (body-shape derivatives), and 56-58 (interaction features) are 0.0 placeholders in `build_features()`; they are filled frame-by-frame in `dataset.py`, `evaluate.py`, and `infer.py`.

### 11.2 COCO-17 Keypoint Layout

```
 0: nose
 / \
 1 2: l_eye, r_eye
 / \
 3 4: l_ear, r_ear
 \ /
 \ /
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
 input_fps: int = 30 # Source video FPS
 frame_stride: int = 3 # process every 3rd frame -> 10 FPS at 30 FPS input

 # Temporal window
 window_len: int = 5 # Frames per window
 step_dt: float = 0.1 # Time step (s)
 safe_window_stride: int = 2 # Safe video stride
 attack_window_stride: int = 1 # Attack video stride

 # Keypoint validity
 kp_conf_thresh: float = 0.35 # Minimum keypoint confidence

 # Cropping detection
 crop_eps: float = 0.03 # Edge margin (3%)

 # Optical flow (Lucas-Kanade)
 lk_win_size: int = 21 # LK window size
 lk_max_level: int = 3 # Pyramid levels
 lk_criteria_count: int = 20 # Max iterations
 lk_criteria_eps: float = 0.03 # Convergence epsilon
 max_flow_points: int = 200 # Sample points

 # Training
 gru_hidden: int = 64 # GRU hidden units
 dropout: float = 0.25 # Dropout rate
 lr: float = 1e-3 # Learning rate
 batch_size: int = 64 # Batch size
 epochs: int = 40 # Training epochs

 # Focal Loss
 focal_gamma: float = 2.0 # Focusing parameter
 focal_alpha: float = 0.75 # Attack class weight

 # Inference
 ema_alpha: float = 0.7 # EMA smoothing
 early_thresh: float = 0.2 # Initial THREAT threshold (replaced after training tuning)
 early_persist: int = 2 # Persistence frames

 # Pose detection - PC (ultralytics) backend
 pose_backend: str = "ultralytics" # "ultralytics" | "hailo"
 yolo_pt_path: str = "models/yolov8m-pose.pt" # PC .pt weights
 yolo_conf: float = 0.25 # Detection confidence threshold
 yolo_iou: float = 0.5 # NMS IoU threshold

 # Pose detection - Hailo NPU backend (Pi 5 only)
 # yolo_hef_path is overridden in Config.for_pi():
 yolo_hef_path: str = "models/yolov8m_pose.hef"
 # Full path on Pi:
 # /home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef
```

#### Platform Presets

```python
Config.for_pc() # PC / development: ultralytics backend, yolov8m-pose.pt
Config.for_pi() # Pi 5 + Hailo-8: hailo backend, yolov8m_pose.hef, early_persist=2
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
