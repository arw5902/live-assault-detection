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
    step_dt: float = 0.1
    safe_window_stride: int = 3
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

    # Focal Loss
    focal_gamma: float = 4.0 # was 3
    focal_alpha: float = 0.25 # was 0.75

    # GRU
    gru_hidden: int = 64
    dropout: float = 0.25   # slightly lower; short window already limits capacity/overfit
    lr: float = 1e-3
    batch_size: int = 64    # more windows per batch (typically feasible)
    epochs: int = 40        # short window = easier optimization; train a bit longer

    # Hazard smoothing / alerting
    ema_alpha: float = 0.7  # more responsive (less lag) with short window
    early_thresh: float = 0.2 # starting value was 0.35
    early_persist: int = 2  # PC default: 0.2 s at 10 FPS; Pi uses 1 (see for_pi)
    high_thresh: float = 0.60
    critical_thresh: float = 0.80
    hysteresis: float = 0.05

    # Pose backend — path used depends on backend
    pose_backend: str = "ultralytics"  # PC default; Pi uses "hailo" (see for_pi)
    yolo_pt_path:  str = "models/yolov8m-pose.pt"          # PC — ultralytics .pt weights
    yolo_hef_path: str = "/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef"  # Pi — Hailo HEF
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
        Raspberry Pi 5 + Hailo AI HAT+ preset.
        - Pose: YOLOv8m HEF on Hailo NPU, ~174 ms
        - Flow: 600 sample points (same as PC) — needed because uniform clothing
          has very few trackable pixels; reducing to 300 causes lk_flow to find
          < 2 good points on the torso ROI and return None (flow_ok = 0)
        - Alert: persist=1 — 1 step is enough at ~1-4 Hz effective inference rate
        Total: ~174 + 106 + 15 = ~295 ms → ~3-4 Hz → gap ~9 frames at 30 fps
        """
        return cls(
            pose_backend  = "hailo",
            early_persist = 1,      # at ~1 Hz inference, 1 step ≈ 1 s of sustained attack
        )


# Feature names for 51-dimensional raw feature vector
# These correspond EXACTLY to the features extracted in src/features.py
# Model input is 102-dimensional: [51 features, 51 validity_masks] concatenated
# Changes vs previous version:
#   - track_age removed (spurious predictor correlated with video length)
#   - bg_flow_mag removed (dataset confounder; background already subtracted from flow vectors)
#   - flow_mag_p90_torso/lower replaced with pos_radial_mean and radial_energy_frac
#     (more direction-specific divergence signals, less correlated with raw speed)
#   - lower ROI decomposition now uses its own center (not torso_c) so forward
#     leg stride is correctly classified as radial approach, not tangential
#   - interaction features added (motion × proximity) to contextualize threat: motion only matters when close
#   - energy_ratio_upper_lower replaced by log_scale (pose-derived apparent size; invariant to arm raises)
#   - approach_proximity replaced by approach_rate = max(dlog_scale_dt,0) × max(trans_signed_torso,0)
#     (size-invariant Z-approach signal; zero for distant/retreating persons)
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

    # Dynamics (3 dims) - indices 45-47
    "max_wrist_extension_velocity",
    "max_wrist_extension_accel",
    "log_scale",

    # Interaction features (3 dims) - indices 48-50
    "approach_rate",
    "expansion_proximity",
    "acceleration_proximity",
]
