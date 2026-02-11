import os, glob
import numpy as np
from typing import Tuple, Dict
from .video_io import iter_video_frames
from .tracker import SingleTargetTracker
from .features import build_features, FeatureState

def list_videos(root, subdir):
    return sorted(glob.glob(os.path.join(root, subdir, "*.mp4")))

def count_windows(T: int, win: int, stride: int) -> int:
    """Number of contiguous windows of length win with start step stride."""
    if T < win:
        return 0
    stride = max(int(stride), 1)
    return 1 + (T - win) // stride

def extract_sequences(video_path: str, label_mode: str, detector, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: [T, D] float32
      M: [T, D] validity mask (0/1)
      y: [T] hazard targets
    """
    tracker = SingleTargetTracker()
    prev_gray = None
    prev_bbox = None

    feats = []
    masks = []

    # Carry-forward imputation for missing features (keeps coherence; does NOT synthesize derivatives)
    fstate = None

    for _, frame in iter_video_frames(video_path, cfg.frame_stride):
        det = detector.infer(frame)
        if det is None:
            # no detection -> skip timestep (keeps windows contiguous over remaining timesteps)
            continue

        bbox, track_age, lost = tracker.update(det["bbox"])
        det["bbox"] = bbox

        x, m, _, prev_gray = build_features(frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
        prev_bbox = bbox

        if fstate is None:
            fstate = FeatureState(cfg.carry_forward_steps)
            fstate.init(len(x))

        # Impute values only; keep mask bits to let the GRU learn missingness
        x = fstate.impute(x, m)

        feats.append(x)
        masks.append(m)

    if len(feats) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    X = np.stack(feats).astype(np.float32)
    M = np.stack(masks).astype(np.float32)

    # Compute bbox derivatives for log_area at indices 9..11 placeholders:
    # x[9]=log_area, x[10]=dlog_area_dt, x[11]=d2log_area_dt2
    log_area = X[:, 9]
    d1 = np.zeros_like(log_area)
    d2 = np.zeros_like(log_area)
    dt = cfg.step_dt
    if len(log_area) >= 2:
        d1[1:] = (log_area[1:] - log_area[:-1]) / dt
    if len(log_area) >= 3:
        d2[2:] = (d1[2:] - d1[1:-1]) / dt
    X[:, 10] = d1
    X[:, 11] = d2
    # Update masks: frame 0 has no prior → dlog_area_dt invalid; frame 0-1 → d2 invalid
    if len(d1) >= 1:
        M[1:, 10] = 1.0   # valid from frame 1 onward
        M[0, 10] = 0.0    # frame 0 has no derivative
    if len(d2) >= 2:
        M[2:, 11] = 1.0   # valid from frame 2 onward
        M[:2, 11] = 0.0   # frames 0-1 have no second derivative

    # Compute wrist extension dynamics at indices 45-46:
    # x[19]=dist_l_wrist_torso, x[20]=dist_r_wrist_torso
    # x[45]=max_wrist_extension_velocity, x[46]=max_wrist_extension_accel
    dist_l_wrist = X[:, 19]
    dist_r_wrist = X[:, 20]

    vel_l = np.zeros_like(dist_l_wrist)
    vel_r = np.zeros_like(dist_r_wrist)
    if len(dist_l_wrist) >= 2:
        vel_l[1:] = (dist_l_wrist[1:] - dist_l_wrist[:-1]) / dt
        vel_r[1:] = (dist_r_wrist[1:] - dist_r_wrist[:-1]) / dt

    accel_l = np.zeros_like(vel_l)
    accel_r = np.zeros_like(vel_r)
    if len(vel_l) >= 2:
        accel_l[1:] = (vel_l[1:] - vel_l[:-1]) / dt
        accel_r[1:] = (vel_r[1:] - vel_r[:-1]) / dt

    X[:, 45] = np.maximum(vel_l, vel_r)
    X[:, 46] = np.maximum(accel_l, accel_r)
    # Update masks: velocity valid from frame 1, acceleration valid from frame 2
    if len(vel_l) >= 1:
        M[1:, 45] = 1.0
        M[0, 45] = 0.0
    if len(accel_l) >= 2:
        M[2:, 46] = 1.0
        M[:2, 46] = 0.0

    # Compute dlog_scale_dt from pose-derived log_scale at index 47
    # log_scale is the log of the minimum of {shoulder_width_px, hip_width_px, torso_height_px}
    # (lateral widths only included when both eyes visible; torso_height always included)
    # Its temporal derivative measures the rate of apparent size growth — invariant to person
    # physical size because it uses rate of change, not absolute size.
    log_scale = X[:, 47]
    dlog_scale_dt = np.zeros_like(log_scale)
    if len(log_scale) >= 2:
        dlog_scale_dt[1:] = (log_scale[1:] - log_scale[:-1]) / dt

    # Add interaction features: motion × proximity coupling
    # These encode "motion is only threatening when person is close AND approaching"
    # Append 3 new features after existing 48 → total 51 features
    trans_signed_torso = X[:, 32]   # signed: positive=approaching, negative=retreating
    divergence_torso = X[:, 34]     # torso divergence
    wrist_accel = X[:, 46]

    # approach_rate: only positive when BOTH scale is growing AND flow is toward camera.
    # A person lifting his leg 5m away barely changes apparent size → dlog_scale_dt ≈ 0 → approach_rate ≈ 0.
    # A retreating person has trans_signed_torso < 0 → approach_rate = 0.
    approach_rate = np.maximum(dlog_scale_dt, 0.0) * np.maximum(trans_signed_torso, 0.0)

    # expansion_proximity and acceleration_proximity still use log_scale as proximity weight
    # (larger apparent size = person is closer = stronger signal weight)
    proximity_weight = np.exp(log_scale * 0.1)

    expansion_proximity = divergence_torso * proximity_weight
    acceleration_proximity = wrist_accel * proximity_weight

    interaction_features = np.stack([approach_rate, expansion_proximity, acceleration_proximity], axis=1)
    X = np.concatenate([X, interaction_features], axis=1)

    # Extend masks for new features (all valid if flow_ok)
    flow_ok_mask = M[:, 41].copy()  # flow_ok is at index 41
    interaction_masks = np.tile(flow_ok_mask.reshape(-1, 1), (1, 3))
    M = np.concatenate([M, interaction_masks], axis=1)

    # Targets
    if label_mode == "safe":
        y = np.zeros((X.shape[0],), dtype=np.float32)
    else:
        # Attack videos are trimmed to start from onset - all frames are attack
        y = np.ones((X.shape[0],), dtype=np.float32)

    return X, M, y

def windowize(X: np.ndarray, M: np.ndarray, y: np.ndarray, win: int, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce contiguous windows (frame coherence preserved within each window):
      Xw: [N, win, D]
      Mw: [N, win, D]
      yw: [N] hazard at end of window
    """
    T, D = X.shape
    if T < win:
        return (
            np.zeros((0, win, D), dtype=np.float32),
            np.zeros((0, win, D), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    stride = max(int(stride), 1)

    Xw, Mw, yw = [], [], []
    for i in range(0, T - win + 1, stride):
        Xw.append(X[i : i + win])
        Mw.append(M[i : i + win])
        yw.append(y[i + win - 1])

    return np.stack(Xw), np.stack(Mw), np.array(yw, dtype=np.float32)

def build_dataset_per_video(detector, cfg):
    """
    Build dataset with video-level separation (prevents data leakage during train/val split).
    Returns list of (Xw, Mw, yw) tuples, one per video.
    """
    safe_videos = list_videos(cfg.data_root, cfg.safe_dir)
    attack_videos = list_videos(cfg.data_root, cfg.attack_dir)

    safe_stride = getattr(cfg, "safe_window_stride", 5)
    attack_stride = getattr(cfg, "attack_window_stride", 1)

    video_data = []  # List of (Xw, Mw, yw) tuples

    # For reporting
    counts: Dict[str, int] = {"safe_windows": 0, "attack_windows": 0, "safe_steps": 0, "attack_steps": 0}

    for vp in safe_videos:
        X, M, y = extract_sequences(vp, "safe", detector, cfg)
        if X.shape[0] == 0:
            continue
        counts["safe_steps"] += int(X.shape[0])

        counts["safe_windows"] += count_windows(X.shape[0], cfg.window_len, safe_stride)

        Xw, Mw, yw = windowize(X, M, y, cfg.window_len, safe_stride)
        if Xw.shape[0] == 0:
            continue
        video_data.append((Xw, Mw, yw))

    for vp in attack_videos:
        X, M, y = extract_sequences(vp, "attack", detector, cfg)
        if X.shape[0] == 0:
            continue
        counts["attack_steps"] += int(X.shape[0])

        counts["attack_windows"] += count_windows(X.shape[0], cfg.window_len, attack_stride)

        Xw, Mw, yw = windowize(X, M, y, cfg.window_len, attack_stride)
        if Xw.shape[0] == 0:
            continue
        video_data.append((Xw, Mw, yw))

    if not video_data:
        raise RuntimeError("No training windows built. Check detector output and dataset paths.")

    # Print imbalance report
    safe_w = counts["safe_windows"]
    attack_w = counts["attack_windows"]
    ratio = safe_w / max(1, attack_w)
    print(
        "Dataset window balance:\n"
        f"  safe_videos={len(safe_videos)} steps={counts['safe_steps']} windows={safe_w} stride={safe_stride}\n"
        f"  attack_videos={len(attack_videos)} steps={counts['attack_steps']} windows={attack_w} stride={attack_stride}\n"
        f"  safe/attack window ratio = {ratio:.2f}"
    )

    return video_data

def build_dataset(detector, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build dataset and concatenate all videos (for backward compatibility).
    WARNING: This concatenates across videos. Use build_dataset_per_video for train/val splitting.
    """
    video_data = build_dataset_per_video(detector, cfg)

    all_X = [xw for xw, _, _ in video_data]
    all_M = [mw for _, mw, _ in video_data]
    all_y = [yw for _, _, yw in video_data]

    X = np.concatenate(all_X, axis=0)
    M = np.concatenate(all_M, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, M, y