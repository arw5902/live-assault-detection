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

    # Track log_area for dlog_area_dt / d2log_area_dt2 (indices 10, 11)
    # build_features() leaves these as 0.0 placeholders; dataset.py fills them from the full
    # sequence. We must replicate that here frame-by-frame to avoid a train/eval mismatch.
    prev_log_area = None
    prev_dlog_area_dt = 0.0

    # Track log_scale for dlog_scale_dt computation (approach_rate)
    prev_log_scale = None
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

        # Compute temporal derivatives — must match dataset.py post-processing exactly,
        # since build_features() leaves x[10], x[11], x[45], x[46] as 0.0 placeholders.

        log_area = x[9]
        dist_l = x[19]  # dist_l_wrist_torso
        dist_r = x[20]  # dist_r_wrist_torso
        log_scale = x[47]  # pose-derived log apparent size

        # --- log_area derivatives (indices 10, 11) ---
        dlog_area_dt = 0.0
        d2log_area_dt2 = 0.0
        have_log_area_deriv = prev_log_area is not None
        if have_log_area_deriv:
            dlog_area_dt = (log_area - prev_log_area) / dt
            d2log_area_dt2 = (dlog_area_dt - prev_dlog_area_dt) / dt
        prev_log_area = log_area
        prev_dlog_area_dt = dlog_area_dt
        x[10] = dlog_area_dt
        x[11] = d2log_area_dt2
        m[10] = 1.0 if have_log_area_deriv else 0.0
        m[11] = 1.0 if have_log_area_deriv else 0.0

        # --- wrist extension velocity / acceleration (indices 45, 46) ---
        wrist_vel = 0.0
        wrist_accel = 0.0
        have_wrist_deriv = prev_wrist_dist_l is not None
        if have_wrist_deriv:
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
        x[45] = wrist_vel
        x[46] = wrist_accel
        m[45] = 1.0 if have_wrist_deriv else 0.0
        m[46] = 1.0 if have_wrist_deriv else 0.0

        # --- log_scale derivative for approach_rate (index 48) ---
        dlog_scale_dt = 0.0
        if prev_log_scale is not None:
            dlog_scale_dt = (log_scale - prev_log_scale) / dt
        prev_log_scale = log_scale

        # Add interaction features (approach_rate uses dlog_scale_dt + trans_signed_torso)
        x, m = add_interaction_features(x, m, wrist_vel, wrist_accel, dlog_scale_dt)

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
