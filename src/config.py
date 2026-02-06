from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = "data"
    safe_dir: str = "safe"
    attack_dir: str = "attack"

    # Video
    input_fps: int = 30
    proc_fps: int = 10
    frame_stride: int = 3  # 30->10

    # Model window
    window_len: int = 5    # 0.5s at 10 FPS
    step_dt: float = 0.1
    safe_window_stride = 5
    attack_window_stride = 1

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
    max_flow_points: int = 200

    # GRU
    gru_hidden: int = 64
    dropout: float = 0.25   # slightly lower; short window already limits capacity/overfit
    lr: float = 1e-3
    batch_size: int = 64    # more windows per batch (typically feasible)
    epochs: int = 40        # short window = easier optimization; train a bit longer

    # Hazard smoothing / alerting
    ema_alpha: float = 0.7  # more responsive (less lag) with short window
    early_thresh: float = 0.2 # starting value was 0.35
    early_persist: int = 2  # 0.2s persistence at 10 FPS to reduce flicker
    high_thresh: float = 0.60
    critical_thresh: float = 0.80
    hysteresis: float = 0.05

    pose_backend: str = "ultralytics"  # "ultralytics" (PC) or "hailo" (Pi)
    yolo_pt_path: str = "models/yolov8m-pose.pt"
    yolo_hef_path: str = "models/yolov8m-pose.hef"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5
