import json
import numpy as np
import torch
import cv2
from collections import deque
from .config import Config
from .model import HazardGRU
from .pose_detector import PoseDetector
from .tracker import SingleTargetTracker
from .features import build_features, add_interaction_features
from .utils import set_seed

def main(video_path: str):
    # Set random seed for reproducibility (CRITICAL: must be before feature extraction)
    # This ensures deterministic optical flow point sampling
    set_seed(seed=42, deterministic=True)

    cfg = Config()
    detector = PoseDetector()
    tracker = SingleTargetTracker()

    # Load model metadata and checkpoint
    checkpoint = "outputs/checkpoints/hazard_gru.pt"
    with open("outputs/checkpoints/meta.json") as f:
        meta = json.load(f)
        input_dim = meta["input_dim"]
        early_thresh = meta.get("best_threshold", cfg.early_thresh)  # Use tuned threshold

    print(f"Using detection threshold: {early_thresh:.2f} (optimized from training)")
    print("Deterministic mode: ON (reproducible optical flow sampling)")

    model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    buf = deque(maxlen=cfg.window_len)
    prev_gray = None
    prev_bbox = None
    hazard_ema = 0.0
    persist = 0

    # Track wrist distances for velocity/acceleration computation
    prev_wrist_dist_l = None
    prev_wrist_dist_r = None
    prev_wrist_vel_l = 0.0
    prev_wrist_vel_r = 0.0
    dt = cfg.step_dt

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % cfg.frame_stride != 0:
            frame_idx += 1
            continue

        det = detector.infer(frame)
        if det is None:
            frame_idx += 1
            continue

        bbox, track_age, lost = tracker.update(det["bbox"])
        det["bbox"] = bbox

        x, m, dbg, prev_gray = build_features(frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
        prev_bbox = bbox

        # Compute wrist dynamics for interaction features
        dist_l = x[19]  # dist_l_wrist_torso
        dist_r = x[20]  # dist_r_wrist_torso

        wrist_vel = 0.0
        wrist_accel = 0.0

        if prev_wrist_dist_l is not None:
            vel_l = (dist_l - prev_wrist_dist_l) / dt
            vel_r = (dist_r - prev_wrist_dist_r) / dt
            wrist_vel = max(vel_l, vel_r)

            accel_l = (vel_l - prev_wrist_vel_l) / dt
            accel_r = (vel_r - prev_wrist_vel_r) / dt
            wrist_accel = max(accel_l, accel_r)

            prev_wrist_vel_l = vel_l
            prev_wrist_vel_r = vel_r

        prev_wrist_dist_l = dist_l
        prev_wrist_dist_r = dist_r

        # Add interaction features
        x, m = add_interaction_features(x, m, wrist_vel, wrist_accel)

        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        buf.append(xm)

        if len(buf) == cfg.window_len:
            inp = torch.from_numpy(np.stack(buf)[None,:,:])
            with torch.no_grad():
                hazard = float(model(inp).item())
            hazard_ema = (1-cfg.ema_alpha)*hazard_ema + cfg.ema_alpha*hazard

            # persistence for pre-contact warning
            if hazard_ema > early_thresh:
                persist += 1
            else:
                persist = max(0, persist-1)

            level = "NONE"
            if hazard_ema > cfg.critical_thresh:
                level = "CRITICAL"
            elif hazard_ema > cfg.high_thresh:
                level = "HIGH"
            elif persist >= cfg.early_persist:
                level = "PRE-CONTACT"

            print(f"t={frame_idx/cfg.input_fps:.2f}s hazard={hazard_ema:.3f} level={level} dbg={dbg}")

        frame_idx += 1

    cap.release()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
