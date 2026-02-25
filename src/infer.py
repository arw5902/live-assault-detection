import sys
import json
import argparse
import os
from datetime import datetime
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

# BGR colours for each hazard level
LEVEL_COLOR = {
    "NONE":        (  0, 200,   0),  # green
    "PRE-CONTACT": (  0, 165, 255),  # orange
    "HIGH":        (  0,  69, 255),  # red-orange
    "CRITICAL":    (  0,   0, 255),  # red
}


def draw_overlay(frame: np.ndarray, bbox, level: str, hazard: float) -> np.ndarray:
    """Return an annotated copy of frame (original is not modified)."""
    vis   = frame.copy()
    h, w  = vis.shape[:2]
    color = LEVEL_COLOR.get(level, (0, 200, 0))

    # Person bounding box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Hazard bar along the top edge (width ∝ score)
    bar_w = int(w * min(max(hazard, 0.0), 1.0))
    cv2.rectangle(vis, (0, 0), (bar_w, 8), color, -1)

    # Status banner at the bottom
    cv2.rectangle(vis, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, f"{level}  {hazard:.2f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return vis


def _open_picamera2(width: int, height: int, fps: int):
    """Open Pi Camera Module via picamera2 library."""
    from picamera2 import Picamera2
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "BGR888"},
        controls={"FrameDurationLimits": (int(1e6 // fps), int(1e6 // fps))})
    picam2.configure(cfg)
    picam2.start()
    return picam2


def run(video_source,
        display: bool      = False,
        record: bool       = False,
        record_dir: str    = "outputs/recordings",
        use_picamera2: bool = False):
    """
    Core inference loop — video file or live camera.

    Parameters
    ----------
    video_source  : int  → camera device index (cv2.VideoCapture)
                    str  → video file path
    display       : show an annotated OpenCV window
                    (automatically enabled when video_source is int)
    record        : write raw (un-annotated) frames to .mp4 and save a
                    per-frame hazard JSON sidecar.  The saved video is fully
                    compatible with evaluate.py — add a labels.json and run
                    evaluate.py on it to measure accuracy offline.
    record_dir    : directory for saved recordings
    use_picamera2 : use picamera2 for Pi Camera Module (RPi5 + AI HAT+).
                    When False and source is int, cv2.VideoCapture is used
                    (suitable for USB webcam or libcamera V4L2 bridge).
    """
    set_seed(seed=42, deterministic=True)

    cfg      = Config()
    detector = PoseDetector(cfg)
    tracker  = SingleTargetTracker()

    # ── load GRU model ────────────────────────────────────────────────────────
    with open("outputs/checkpoints/meta.json") as f:
        meta         = json.load(f)
        input_dim    = meta["input_dim"]
        early_thresh = meta.get("best_threshold", cfg.early_thresh)
        model_file   = meta.get("model_file", "hazard_gru.pt")

    # GRU runs on CPU on Pi — too small to benefit from NPU, and PyTorch
    # is not available on the Hailo NPU without DFC compilation.
    # On a PC with GPU this will use CUDA automatically.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(
        torch.load(f"outputs/checkpoints/{model_file}", map_location=device))
    model.eval()
    model.to(device)

    print(f"Threshold   : {early_thresh:.2f} (from training optimisation)")
    print(f"GRU device  : {device}")
    print(f"Pose backend: {cfg.pose_backend}")
    print("Deterministic: ON (reproducible optical flow sampling)")

    # ── temporal state (mirrors evaluate.py exactly) ──────────────────────────
    buf               = deque(maxlen=cfg.window_len)
    prev_gray         = None
    prev_bbox         = None
    hazard_ema        = 0.0
    persist           = 0
    level             = "NONE"
    bbox              = None

    prev_wrist_dist_l = None;  prev_wrist_dist_r = None
    prev_wrist_vel_l  = 0.0;   prev_wrist_vel_r  = 0.0
    prev_log_area     = None;  prev_dlog_area_dt = None
    prev_log_scale    = None
    dt                = cfg.step_dt

    # ── open video source ─────────────────────────────────────────────────────
    is_camera = isinstance(video_source, int)
    if is_camera:
        display = True          # always show window for live camera

    cap    = None
    picam2 = None

    if is_camera and use_picamera2:
        cam_w, cam_h, cam_fps = 640, 480, 30
        picam2   = _open_picamera2(cam_w, cam_h, cam_fps)
        frame_w  = cam_w
        frame_h  = cam_h
        fps_src  = float(cam_fps)
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: cannot open video source: {video_source}")
            detector.release()
            return
        fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.input_fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── optional recording ────────────────────────────────────────────────────
    writer      = None
    record_path = None
    hazard_log  = []

    if record:
        os.makedirs(record_dir, exist_ok=True)
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = os.path.join(record_dir, f"cam_{ts}.mp4")
        fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
        writer      = cv2.VideoWriter(record_path, fourcc, fps_src,
                                      (frame_w, frame_h))
        print(f"Recording raw frames to : {record_path}")
        print("(run evaluate.py on this file + labels.json to measure accuracy)")

    # ── main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    try:
        while True:
            # ── capture ───────────────────────────────────────────────────────
            if picam2 is not None:
                frame = picam2.capture_array()   # BGR numpy array
                ok    = True
            else:
                ok, frame = cap.read()
            if not ok:
                break

            # Save raw (un-annotated) frame for evaluate.py replay
            if writer is not None:
                writer.write(frame)

            # Skip non-strided frames but keep display responsive
            if frame_idx % cfg.frame_stride != 0:
                if display:
                    cv2.imshow("Hazard Detection",
                               draw_overlay(frame, bbox, level, hazard_ema))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue

            # ── pose detection ────────────────────────────────────────────────
            det = detector.infer(frame)
            if det is None:
                if display:
                    cv2.imshow("Hazard Detection",
                               draw_overlay(frame, bbox, level, hazard_ema))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue

            bbox, track_age, lost = tracker.update(det["bbox"])
            det["bbox"] = bbox

            x, m, dbg, prev_gray = build_features(
                frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
            prev_bbox = bbox

            log_area  = x[9]
            dist_l    = x[19]   # dist_l_wrist_torso
            dist_r    = x[20]   # dist_r_wrist_torso
            log_scale = x[47]

            # ── log_area derivatives (indices 10, 11) — mirrors evaluate.py ──
            dlog_area_dt   = 0.0
            d2log_area_dt2 = 0.0
            have_log_area_deriv = prev_log_area is not None

            if have_log_area_deriv:
                dlog_area_dt = (log_area - prev_log_area) / dt
                if prev_dlog_area_dt is not None:
                    d2log_area_dt2 = (dlog_area_dt - prev_dlog_area_dt) / dt

            prev_log_area     = log_area
            prev_dlog_area_dt = dlog_area_dt

            x[10] = dlog_area_dt;    m[10] = 1.0 if have_log_area_deriv else 0.0
            x[11] = d2log_area_dt2;  m[11] = 1.0 if have_log_area_deriv else 0.0

            # ── wrist velocity / acceleration (indices 45, 46) ────────────────
            wrist_vel   = 0.0
            wrist_accel = 0.0
            have_wrist_deriv = prev_wrist_dist_l is not None

            if have_wrist_deriv:
                vel_l = (dist_l - prev_wrist_dist_l) / dt
                vel_r = (dist_r - prev_wrist_dist_r) / dt
                wrist_vel   = max(vel_l, vel_r)
                accel_l     = (vel_l - prev_wrist_vel_l) / dt
                accel_r     = (vel_r - prev_wrist_vel_r) / dt
                wrist_accel = max(accel_l, accel_r)
                prev_wrist_vel_l = vel_l
                prev_wrist_vel_r = vel_r
            else:
                prev_wrist_vel_l = 0.0
                prev_wrist_vel_r = 0.0

            prev_wrist_dist_l = dist_l
            prev_wrist_dist_r = dist_r

            x[45] = wrist_vel;    m[45] = 1.0 if have_wrist_deriv else 0.0
            x[46] = wrist_accel;  m[46] = 1.0 if have_wrist_deriv else 0.0

            # ── log_scale derivative for approach_rate (index 48) ─────────────
            dlog_scale_dt = 0.0
            if prev_log_scale is not None:
                dlog_scale_dt = (log_scale - prev_log_scale) / dt
            prev_log_scale = log_scale

            x, m = add_interaction_features(x, m, wrist_vel, wrist_accel,
                                             dlog_scale_dt)

            xm = np.concatenate([x, m], axis=0).astype(np.float32)
            buf.append(xm)

            # ── GRU inference ─────────────────────────────────────────────────
            if len(buf) == cfg.window_len:
                inp_t = torch.from_numpy(
                    np.stack(buf)[None, :, :]).to(device)
                with torch.no_grad():
                    hazard_raw = float(model(inp_t).item())

                hazard_ema = ((1 - cfg.ema_alpha) * hazard_ema
                              + cfg.ema_alpha * hazard_raw)

                if hazard_ema > early_thresh:
                    persist += 1
                else:
                    persist = max(0, persist - 1)

                level = "NONE"
                if   hazard_ema > cfg.critical_thresh: level = "CRITICAL"
                elif hazard_ema > cfg.high_thresh:      level = "HIGH"
                elif persist    >= cfg.early_persist:   level = "PRE-CONTACT"

                print(f"t={frame_idx / cfg.input_fps:.2f}s  "
                      f"raw={hazard_raw:.3f}  ema={hazard_ema:.3f}  "
                      f"level={level}  dbg={dbg}")

                if record_path is not None:
                    hazard_log.append({
                        "frame_idx":  frame_idx,
                        "hazard_raw": round(hazard_raw, 4),
                        "hazard_ema": round(hazard_ema, 4),
                        "level":      level,
                    })

            # ── display ───────────────────────────────────────────────────────
            if display:
                cv2.imshow("Hazard Detection",
                           draw_overlay(frame, bbox, level, hazard_ema))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    finally:
        # ── cleanup (always runs, even on exception / KeyboardInterrupt) ──────
        detector.release()
        if picam2 is not None:
            picam2.stop()
        if cap is not None and cap.isOpened():
            cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        # Save hazard log alongside the recording
        if record_path is not None and hazard_log:
            log_path = record_path.replace(".mp4", "_hazard.json")
            with open(log_path, "w") as f:
                json.dump(hazard_log, f, indent=2)
            print(f"Hazard log saved : {log_path}")
            print("Next: add ground-truth labels and run evaluate.py for "
                  "accuracy measurement.")


# ── backwards-compatible entry point ─────────────────────────────────────────

def main(video_path: str):
    """
    Kept for existing callers that do:
        from src.infer import main; main("myvideo.mp4")
    """
    run(video_path, display=False, record=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hazard detection — video file or live camera")

    parser.add_argument(
        "source", nargs="?", default="0",
        help="Video file path, or camera device index as integer "
             "(default: 0 = first camera device)")

    parser.add_argument(
        "--display", action="store_true",
        help="Show annotated video window "
             "(automatically enabled when source is a camera index)")

    parser.add_argument(
        "--picamera2", action="store_true",
        help="Use picamera2 for Pi Camera Module (RPi5 + AI HAT+). "
             "Without this flag an integer source uses cv2.VideoCapture "
             "(suitable for USB webcam).")

    parser.add_argument(
        "--record", action="store_true",
        help="Record raw camera frames to .mp4 and save per-frame hazard "
             "scores to a JSON sidecar.  The recording can be replayed "
             "through evaluate.py with a labels.json to measure accuracy.")

    parser.add_argument(
        "--record-dir", default="outputs/recordings",
        help="Directory for saved recordings (default: outputs/recordings)")

    args = parser.parse_args()

    # "0", "1", … → int (camera device index); anything else → file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(source,
        display=args.display,
        record=args.record,
        record_dir=args.record_dir,
        use_picamera2=args.picamera2)
