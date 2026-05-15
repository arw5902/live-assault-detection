from dataclasses import dataclass

@dataclass
class Config:
    """
    All parameters shared between PC and Pi.
    The class-body defaults are PC values.
    Use Config.for_pc() or Config.for_pi() to get a platform-specific instance.
    """
    data_root: str = "data"
    safe_dir: str = "safe"
    attack_dir: str = "attack"

    # Video
    input_fps: int = 30
    frame_stride: int = 3  # process every 3rd frame → 10 FPS at 30 FPS input

    # Model window
    window_len: int = 5 # 0.5s at 10 FPS
    step_dt: float = 0.1 #10 FPS
    safe_window_stride: int = 2
    attack_window_stride: int = 1

    # Keypoint validity
    kp_conf_thresh: float = 0.35
    carry_forward_steps: int = 3

    # Cropping
    crop_eps: float = 0.03  # 3% border margin

    # Optical flow (LK)
    lk_win_size: int = 21
    lk_max_level: int = 3
    lk_criteria_count: int = 20
    lk_criteria_eps: float = 0.03
    max_flow_points: int = 600  # 600 needed on both platforms: uniform clothing gives few trackable pts

    # Background homography estimation (camera-motion compensation).
    # Replaces simple median subtraction with a full projective transform estimated
    # via RANSAC from background points — correctly handles camera yaw/pitch/roll.
    # Falls back to median when RANSAC inlier count is below bg_min_inliers.
    bg_ransac_thresh: float = 3.0   # reprojection error threshold in pixels
    bg_min_inliers:   int   = 20    # min RANSAC inliers to trust homography; else median

    # ROI sizing for optical flow decomposition (multiples of torso_scale).
    torso_roi_scale:  float = 1.2   # side length of square torso ROI
    lower_roi_scale:  float = 2.0   # height of lower-body ROI below hip midpoint

    # log_scale validity guards — protect dlog_scale_dt from foreshortening artefacts.
    bend_tilt_thresh: float = 35.0  # degrees from vertical; above → log_scale invalidated
    hip_bottom_margin_px: float = 8.0  # min px from frame bottom for hip to be "not cropped"

    # Proximity weight exponent for interaction features.
    # exp(log_scale * proximity_exponent) creates distance-dependent scaling.
    # 0.3 → ~57% more weight at close (ls≈5.5) vs far (ls≈4.0).
    proximity_exponent: float = 0.3

    # Focal Loss
    focal_gamma: float = 2.0 # was 4
    focal_alpha: float = 0.75 # was 0.25

    # GRU
    gru_hidden: int = 64
    dropout: float = 0.25   # slightly lower; short window already limits capacity/overfit
    lr: float = 1e-3
    batch_size: int = 64    # more windows per batch (typically feasible)
    epochs: int = 40        # short window = easier optimization; train a bit longer

    # Hazard smoothing / alerting — binary THREAT / NONE detection
    ema_alpha: float = 0.7  # more responsive (less lag) with short window
    early_thresh: float = 0.2 # starting value was 0.35; overridden by meta.json best_threshold
    early_persist: int = 2  # PC default: 0.2 s at 10 FPS; Pi may use 1 (see for_pi)

    # Audio fusion (PANNs-based threat detection)
    # Audio is optional — disabled automatically if panns_inference is not installed.
    audio_enabled: bool = True          # set False to skip audio entirely
    audio_window_sec: float = 1.0       # PANNs classification window length (seconds)
    audio_thresh: float = 0.3           # min audio_score to activate boost
    audio_boost_alpha: float = 0.25     # boost strength (0 = off, 1 = full)

    # Inference and recording resolution — must match the training data resolution.
    # PiCamera2 is already opened at this size natively.  Any other source
    # (USB webcam, arbitrary video file) is resized to (infer_w × infer_h) before
    # pose detection and optical flow, keeping pixel-magnitude features (log_scale,
    # log_area, flow magnitudes) on the same scale as the training data.
    infer_w: int = 640
    infer_h: int = 480

    # Pose backend — path used depends on backend
    pose_backend: str = "ultralytics"  # PC default; Pi uses "hailo" (see for_pi)
    yolo_pt_path:  str = "models/yolov8m-pose.pt"          # PC — ultralytics .pt weights
    yolo_hef_path: str = "/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef"  # Pi — Hailo HEF
    #yolo_hef_path: str = "models/yolov8n-pose.hef"
    #yolo_imgsz: int = 416   # The HEF on the Pi is for 416 x 416 images
    yolo_conf: float = 0.25
    yolo_iou:  float = 0.5

    # ── Named platform presets ─────────────────────────────────────────────────

    @classmethod
    def for_pc(cls) -> "Config":
        """
        Development PC preset (default values).
        - Pose: ultralytics YOLOv8m (CPU or CUDA), ~30-50 ms with GPU
        - Flow: 600 sample points, ~20 ms on a desktop CPU
        - Alert: persist=2 (0.2 s at 10 FPS inference)
        """
        return cls()

    @classmethod
    def for_pi(cls) -> "Config":
        """
        Raspberry Pi 5 + Hailo AI HAT+ (26 TOPs, Hailo-8) preset.
        - Pose: YOLOv8m HEF on Hailo-8 NPU, ~35 ms (observed); HEF compiled at 416×416
        - Flow: 600 sample points (same as PC) — needed because uniform clothing
          has very few trackable pixels; reducing to 300 causes lk_flow to find
          < 2 good points on the torso ROI and return None (flow_ok = 0)
        - Alert: persist=2
        Total: ~35 + 40 + 5 = ~80 ms typical → ~10-12 Hz → gap ~3 frames at 30 fps
        """
        return cls(
            pose_backend  = "hailo",
            early_persist = 2,      # was 1 Hz before
        )


# 56 base features (indices 0-55) from build_features()
# + 3 interaction features (indices 56-58) from add_interaction_features()
# Model input is 118-dim: [59 features, 59 validity_masks] concatenated.
FEATURE_NAMES = [
    # Reliability/Metadata (9 dims) - indices 0-8
    "det_conf",
    "kp_conf_mean",
    "kp_conf_min",
    "visible_kp_count",
    "anyc",
    "crop_left",
    "crop_right",
    "crop_top",
    "crop_bottom",

    # Bbox/Approach features (8 dims) - indices 9-16
    "log_area",
    "dlog_area_dt",
    "d2log_area_dt2",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_center_vx",
    "bbox_center_vy",
    "bbox_aspect",

    # Upper body geometry (8 dims) - indices 17-24
    "dist_l_wrist_l_shoulder",
    "dist_r_wrist_r_shoulder",
    "dist_l_wrist_torso",
    "dist_r_wrist_torso",
    "shoulder_width",
    "hip_width",
    "angle_l_elbow",
    "angle_r_elbow",

    # Lower body geometry (7 dims) - indices 25-31
    "angle_l_knee",
    "angle_r_knee",
    "dist_l_ankle_l_hip",
    "dist_r_ankle_r_hip",
    "dist_l_ankle_torso",
    "dist_r_ankle_torso",
    "stance_width",

    # Optical flow - torso ROI (4 dims) - indices 32-35
    "translation_signed_torso",
    "translation_torso",
    "divergence_torso",
    "div_ratio_torso",

    # Optical flow - lower ROI (4 dims) - indices 36-39
    "translation_signed_lower",
    "translation_lower",
    "divergence_lower",
    "div_ratio_lower",

    # Optical flow - background (2 dims) - indices 40-41
    "bg_flow_coherence",
    "flow_ok",

    # Posture features (3 dims) - indices 42-44
    "torso_compression_ratio",
    "wrist_height_asymmetry",
    "face_visibility",

    # Dynamics (4 dims) - indices 45-48
    "max_wrist_extension_velocity",
    "max_wrist_extension_accel",
    "log_scale",
    "torso_height_px",      # best-effort: single-side fallback, no hip_near_bottom/bend_like guards

    # Wrist direction (2 dims) - indices 49-50
    "wrist_y_rel",          # max wrist-hip vertical offset / bbox_h (raw position)
    "d_wrist_y_rel_dt",     # temporal derivative of wrist_y_rel (raise/strike rate)

    # Gait dynamics (2 dims) - indices 51-52
    "ankle_spread",         # |ankle_L_x − ankle_R_x| / bbox_w (stride width)
    "d_ankle_spread_dt",    # temporal derivative of ankle_spread (gait cadence)

    # Head motion (2 dims) - indices 53-54
    "nose_y_rel",           # (nose_y − shoulder_mid_y) / bbox_h (head position)
    "d_nose_y_rel_dt",      # temporal derivative of nose_y_rel (ducking rate)

    # Upper-lower body asynchrony (1 dim) - index 55
    "upper_lower_async",    # |translation_torso − translation_lower| (desync signal)

    # Interaction features (3 dims) - indices 56-58
    "approach_rate",
    "expansion_proximity",
    "acceleration_proximity",
]

# Invariant: FEATURE_NAMES must list every feature in the 59-dim vector
# produced by build_features() + add_interaction_features() in src/features.py.
# Drop or add a name here only when the feature pipeline changes accordingly.
assert len(FEATURE_NAMES) == 59, f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected 59"
