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

    # Compute bbox derivatives for log_area at indices 10..12 placeholders:
    # x[10]=log_area, x[11]=dlog_area_dt, x[12]=d2log_area_dt2
    log_area = X[:, 10]
    d1 = np.zeros_like(log_area)
    d2 = np.zeros_like(log_area)
    dt = cfg.step_dt
    if len(log_area) >= 2:
        d1[1:] = (log_area[1:] - log_area[:-1]) / dt
    if len(log_area) >= 3:
        d2[2:] = (d1[2:] - d1[1:-1]) / dt
    X[:, 11] = d1
    X[:, 12] = d2

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