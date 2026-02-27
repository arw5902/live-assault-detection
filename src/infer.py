import sys
import json
import argparse
import os
import threading
import time
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

# COCO-17 skeleton: pairs of keypoint indices to connect with a line
# Index → joint:  0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear
#                 5=l_shoulder 6=r_shoulder 7=l_elbow 8=r_elbow
#                 9=l_wrist 10=r_wrist 11=l_hip 12=r_hip
#                 13=l_knee 14=r_knee 15=l_ankle 16=r_ankle
COCO17_SKELETON = [
    (0,  1), (0,  2),           # nose → eyes
    (1,  3), (2,  4),           # eyes → ears
    (5,  6),                    # left shoulder → right shoulder
    (5,  7), (7,  9),           # left arm
    (6,  8), (8, 10),           # right arm
    (5, 11), (6, 12),           # shoulders → hips
    (11, 12),                   # left hip → right hip
    (11, 13), (13, 15),         # left leg
    (12, 14), (14, 16),         # right leg
]

# Per-joint colour (BGR): face=yellow, arms=cyan, torso=white, legs=magenta
_KP_COLOR = [
    (0, 255, 255),  #  0 nose
    (0, 255, 255),  #  1 l_eye
    (0, 255, 255),  #  2 r_eye
    (0, 255, 255),  #  3 l_ear
    (0, 255, 255),  #  4 r_ear
    (255, 255, 0),  #  5 l_shoulder   cyan
    (255, 255, 0),  #  6 r_shoulder
    (255, 255, 0),  #  7 l_elbow
    (255, 255, 0),  #  8 r_elbow
    (255, 255, 0),  #  9 l_wrist
    (255, 255, 0),  # 10 r_wrist
    (255, 0, 255),  # 11 l_hip        magenta
    (255, 0, 255),  # 12 r_hip
    (255, 0, 255),  # 13 l_knee
    (255, 0, 255),  # 14 r_knee
    (255, 0, 255),  # 15 l_ankle
    (255, 0, 255),  # 16 r_ankle
]

_KP_CONF_THRESH = 0.3   # minimum keypoint confidence to draw

class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
def draw_skeleton(vis: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """
    Draw COCO-17 keypoints and skeleton lines onto vis (in-place).
    kps : (17, 3)  columns = [x, y, confidence]
    Only draws joints / limbs whose endpoint confidences both exceed
    _KP_CONF_THRESH.  Returns vis for convenience.
    """
    # Skeleton lines first (drawn under the joint dots)
    for i, j in COCO17_SKELETON:
        if kps[i, 2] >= _KP_CONF_THRESH and kps[j, 2] >= _KP_CONF_THRESH:
            pt1 = (int(kps[i, 0]), int(kps[i, 1]))
            pt2 = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.line(vis, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

    # Joint dots on top
    for idx, (x, y, c) in enumerate(kps):
        if c >= _KP_CONF_THRESH:
            cv2.circle(vis, (int(x), int(y)), 4, _KP_COLOR[idx], -1, cv2.LINE_AA)

    return vis


def draw_overlay(frame: np.ndarray, bbox, level: str, hazard: float,
                 kps: np.ndarray = None) -> np.ndarray:
    """
    Return an annotated copy of frame (original is not modified).
    kps : (17, 3) COCO-17 keypoints [x, y, conf], or None to skip skeleton.
    """
    vis   = frame.copy()
    h, w  = vis.shape[:2]
    color = LEVEL_COLOR.get(level, (0, 200, 0))

    # COCO-17 skeleton (drawn before the bbox so bbox sits on top)
    if kps is not None:
        draw_skeleton(vis, kps)

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


class _BackgroundCapture:
    """
    Captures frames from a live camera (Picamera2 or cv2.VideoCapture) in a
    background thread at the camera's native FPS, writing every frame to the
    VideoWriter.  The inference loop reads the *latest* frame at its own pace
    (governed by cfg.step_dt) without ever stalling the writer.

    This decouples recording FPS (= camera hardware FPS, e.g. 30 Hz) from
    inference FPS (~10 Hz), fixing the 'video plays too fast' bug that
    occurred when VideoWriter was told fps=30 but the inference loop only
    delivered ~10 frames per second to it.
    """

    def __init__(self, source, writer):
        """
        source : Picamera2 object  OR  cv2.VideoCapture object
        writer : cv2.VideoWriter or None
        """
        self._src    = source
        self._writer = writer
        self._frame  = None
        self._count  = 0          # total frames captured so far
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        self._thread = t

    def _run(self):
        while not self._stop.is_set():
            if hasattr(self._src, 'capture_array'):   # picamera2
                frame = self._src.capture_array()
            else:                                       # cv2.VideoCapture
                ok, frame = self._src.read()
                if not ok:
                    self._stop.set()
                    break
            if self._writer is not None:
                self._writer.write(frame)
            with self._lock:
                self._frame = frame
                self._count += 1

    def read(self):
        """Return (frame, count).  frame is None until the first capture."""
        with self._lock:
            return self._frame, self._count

    @property
    def stopped(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def run(video_source,
        display: bool       = False,
        record: bool        = False,
        record_dir: str     = "outputs/recordings",
        use_picamera2: bool = False,
        show_skeleton: bool = True,
        enable_pi: bool     = True):
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
    show_skeleton : overlay COCO-17 keypoints and limb lines on the display
                    window.  Has no effect on the raw recording.
    """
    set_seed(seed=42, deterministic=True)

    cfg      = Config.for_pi() if enable_pi else Config.for_pc()
    detector = PoseDetector(cfg)
    tracker  = SingleTargetTracker()

    print(f"Pose backend: {cfg.pose_backend}")


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
    kps               = None    # last known COCO-17 keypoints (17, 3)

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
        # Try codecs in order; mp4v works on most desktops, avc1/XVID on Pi
        writer = None
        for codec in ("mp4v", "avc1", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(record_path, fourcc, fps_src,
                                     (frame_w, frame_h))
            if writer.isOpened():
                print(f"Recording raw frames to : {record_path}  (codec={codec})")
                break
            writer.release()
            writer = None
        if writer is None:
            print(f"WARNING: could not open VideoWriter — recording disabled. "
                  f"Check OpenCV codec support on this platform.")
            record_path = None
        else:
            print("(run evaluate.py on this file + labels.json to measure accuracy)")

    # ── background capture thread (camera sources only) ───────────────────────
    # The background thread captures at the camera's native FPS and writes
    # every frame to the VideoWriter, completely independently of the inference
    # loop.  The inference loop grabs the *latest* frame at cfg.step_dt
    # intervals (~10 Hz) without blocking the recording.
    # File sources are read sequentially in the main loop (no thread needed).
    bg_cap = None
    if is_camera:
        source = picam2 if picam2 is not None else cap
        bg_cap = _BackgroundCapture(source, writer)
        # Block until at least one frame has been captured
        while bg_cap.read()[0] is None and not bg_cap.stopped:
            time.sleep(0.01)
        next_infer_t = time.monotonic()

    # ── main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    bg_count  = 0   # actual frames written to VideoWriter by background thread
    try:
        while True:
            # ── capture ───────────────────────────────────────────────────────
            if bg_cap is not None:
                # Camera path: sleep until the next inference slot, then
                # grab whatever frame the background thread captured most
                # recently.  Recording runs at full camera FPS independently.
                sleep_s = next_infer_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_infer_t += cfg.step_dt

                frame, bg_count = bg_cap.read()
                if frame is None:
                    continue
                if bg_cap.stopped:
                    break
            else:
                # File path: read frames sequentially
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
                                   draw_overlay(frame, bbox, level, hazard_ema,
                                                kps if show_skeleton else None))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    frame_idx += 1
                    continue

            # ── pose detection ────────────────────────────────────────────────
            _t0 = time.perf_counter()
            det = detector.infer(frame)
            _t_pose = time.perf_counter() - _t0
            if det is None:
                if display:
                    cv2.imshow("Hazard Detection",
                               draw_overlay(frame, bbox, level, hazard_ema,
                                            kps if show_skeleton else None))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue

            bbox, track_age, lost = tracker.update(det["bbox"])
            det["bbox"] = bbox
            kps = det["kps"]            # (17, 3) — carry forward for display

            _t1 = time.perf_counter()
            x, m, dbg, prev_gray = build_features(
                frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
            _t_flow = time.perf_counter() - _t1
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
                _t2 = time.perf_counter()
                with torch.no_grad():
                    hazard_raw = float(model(inp_t).item())
                _t_gru = time.perf_counter() - _t2

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

                # Elapsed-time label: camera uses actual frames written ÷ fps
                # (accurate regardless of inference speed on the Pi);
                # file uses recorded frame number ÷ input FPS.
                t_s = (bg_count / fps_src if is_camera
                       else frame_idx / cfg.input_fps)
                print(f"t={t_s:.2f}s  "
                      f"raw={hazard_raw:.3f}  ema={hazard_ema:.3f}  "
                      f"level={level}  "
                      f"pose={_t_pose*1000:.0f}ms  "
                      f"flow={_t_flow*1000:.0f}ms  "
                      f"gru={_t_gru*1000:.0f}ms  "
                      f"total={(_t_pose+_t_flow+_t_gru)*1000:.0f}ms  "
                      f"dbg={dbg}")

                if record_path is not None:
                    # For camera: use the background thread's actual write count
                    # so frame_idx in the JSON matches the real MP4 frame number
                    # regardless of how fast/slow inference runs on the Pi.
                    rec_frame_idx = (bg_count if is_camera else frame_idx)
                    hazard_log.append({
                        "frame_idx":  rec_frame_idx,
                        "hazard_raw": round(hazard_raw, 4),
                        "hazard_ema": round(hazard_ema, 4),
                        "level":      level,
                    })

            # ── display ───────────────────────────────────────────────────────
            if display:
                cv2.imshow("Hazard Detection",
                           draw_overlay(frame, bbox, level, hazard_ema,
                                        kps if show_skeleton else None))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    finally:
        # ── cleanup (always runs, even on exception / KeyboardInterrupt) ──────
        if bg_cap is not None:
            bg_cap.stop()
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

    parser.add_argument(
        "--no-skeleton", action="store_true",
        help="Disable COCO-17 keypoint and limb overlay on the display window. "
             "Has no effect on the raw recording.")

    parser.add_argument(
        "--pi", action="store_true", help="Enable Pi configurations.")

    args = parser.parse_args()

    # "0", "1", … → int (camera device index); anything else → file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Create logs directory
    os.makedirs("outputs/log", exist_ok=True)

    # Timestamped log filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("outputs/log", f"infer_{ts}.log")

    # Redirect stdout and stderr
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout

    print(f"Logging to {log_path}")

    run(source,
        display=args.display,
        record=args.record,
        record_dir=args.record_dir,
        use_picamera2=args.picamera2,
        show_skeleton=not args.no_skeleton,
        enable_pi=args.pi)
