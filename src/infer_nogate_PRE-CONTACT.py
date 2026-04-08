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
from .features import build_features, add_interaction_features, compute_torso_height_frac
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

    def __init__(self, source, writer, resize_hw=None):
        """
        source    : Picamera2 object  OR  cv2.VideoCapture object
        writer    : cv2.VideoWriter or None
        resize_hw : (H, W) to resize every frame before writing / storing,
                    or None to keep the native resolution.  Pass
                    (cfg.infer_h, cfg.infer_w) to match training resolution.
        """
        self._src       = source
        self._writer    = writer
        self._resize_hw = resize_hw   # (H, W) or None
        self._frame     = None
        self._count     = 0           # total frames captured so far
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
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
            # Resize to inference resolution when the camera native size differs
            # (no-op for PiCamera2, which is already opened at 640 × 480).
            if self._resize_hw is not None:
                th, tw = self._resize_hw
                if frame.shape[0] != th or frame.shape[1] != tw:
                    frame = cv2.resize(frame, (tw, th))
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
        enable_pi: bool     = True,
        verbose: bool       = False,
        debug: bool         = False,
        simulate_live: bool = False):
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
    simulate_live : pace a video-file source to match the file's native FPS
                    and use actual wall-clock dt for feature derivatives —
                    making a pre-recorded video behave identically to a live
                    camera feed.  Ignored when video_source is a camera int.
    """
    set_seed(seed=42, deterministic=True)

    # Note Config.for_pi() will disable skeleton being displayed on screen, 
    # probably due to hailo backend instead of ultralytics
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
    dt                = cfg.step_dt  # file-path default (= frame_stride/input_fps = 0.1 s)
    t_last_infer      = None         # wall-clock time of last processed frame (camera path)

    # ── open video source ─────────────────────────────────────────────────────
    is_camera = isinstance(video_source, int)
    if is_camera:
        display      = True     # always show window for live camera
        simulate_live = False   # simulate_live is only for file sources

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
                                     (cfg.infer_w, cfg.infer_h))
            if writer.isOpened():
                print(f"Recording raw frames to : {record_path}  "
                      f"(codec={codec}, {cfg.infer_w}×{cfg.infer_h})")
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
        bg_cap = _BackgroundCapture(source, writer,
                                    resize_hw=(cfg.infer_h, cfg.infer_w))
        # Block until at least one frame has been captured
        while bg_cap.read()[0] is None and not bg_cap.stopped:
            time.sleep(0.01)
        next_infer_t = time.monotonic()

    # ── simulate_live: pacing state for file sources ──────────────────────────
    # frame_period = time between consecutive video frames (1/fps_src).
    # next_frame_t = monotonic clock target for the *next* frame read.
    # Both are only used when simulate_live=True.
    if simulate_live:
        frame_period = 1.0 / fps_src
        next_frame_t = time.monotonic()
        print(f"Live simulation: ON  "
              f"(pacing to {fps_src:.1f} FPS, wall-clock dt for features)")
    else:
        frame_period = None
        next_frame_t = None

    # ── main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    bg_count  = 0   # actual frames written to VideoWriter by background thread
    try:
        while True:
            # ── capture ───────────────────────────────────────────────────────
            t_frame = None   # set in camera path; stays None for file path
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
                t_frame = time.monotonic()   # timestamp of this grabbed frame
            else:
                # File path: read frames sequentially
                ok, frame = cap.read()
                if not ok:
                    break

                # Resize to training resolution so pixel-magnitude features
                # (log_scale, log_area, flow magnitudes) match the scale the
                # GRU was trained on, and so that the recording is consistent
                # with what evaluate.py will replay.
                if frame.shape[1] != cfg.infer_w or frame.shape[0] != cfg.infer_h:
                    frame = cv2.resize(frame, (cfg.infer_w, cfg.infer_h))

                # Save raw (un-annotated) frame for evaluate.py replay
                if writer is not None:
                    writer.write(frame)

                # simulate_live: pace frame delivery to the video's native FPS
                # so that a pre-recorded file behaves like a live camera feed.
                # Sleeping here means each iteration takes ~1/fps_src seconds,
                # exactly as if frames were arriving from a real camera.
                # t_frame is then set to the actual wall-clock time at which
                # this frame was "received", so frame_dt (used for all velocity /
                # acceleration features) reflects real elapsed time rather than
                # the fixed cfg.step_dt constant.  This keeps features on the
                # same physical scale as training (dt ≈ step_dt = 0.1 s).
                if simulate_live:
                    sleep_s = next_frame_t - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    next_frame_t += frame_period  # advance target for next frame

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

                # Mark the wall-clock timestamp of this processed frame so the
                # downstream dt computation (t_frame - t_last_infer) mirrors the
                # camera path exactly.  Only set for simulate_live; plain file
                # mode keeps t_frame=None and uses fixed cfg.step_dt instead.
                if simulate_live:
                    t_frame = time.monotonic()

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

            # Camera path: use actual wall-clock Δt between consecutively
            # processed frames so that velocity/acceleration features stay on
            # the same physical scale as the training data (10 FPS, dt=0.1 s).
            # File path: always use cfg.step_dt (= frame_stride/input_fps = 0.1 s).
            if t_frame is not None:
                frame_dt = (t_frame - t_last_infer) if t_last_infer is not None else dt
                t_last_infer = t_frame   # ready for next iteration
            else:
                frame_dt = dt            # file path — already correct

            _t1 = time.perf_counter()
            x, m, dbg, prev_gray = build_features(
                frame, prev_gray, prev_bbox, det, track_age, lost, cfg,
                dt=frame_dt)
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
                dlog_area_dt = (log_area - prev_log_area) / frame_dt
                if prev_dlog_area_dt is not None:
                    d2log_area_dt2 = (dlog_area_dt - prev_dlog_area_dt) / frame_dt

            prev_log_area     = log_area
            prev_dlog_area_dt = dlog_area_dt

            x[10] = dlog_area_dt;    m[10] = 1.0 if have_log_area_deriv else 0.0
            x[11] = d2log_area_dt2;  m[11] = 1.0 if have_log_area_deriv else 0.0

            # ── wrist velocity / acceleration (indices 45, 46) ────────────────
            wrist_vel   = 0.0
            wrist_accel = 0.0
            have_wrist_deriv = prev_wrist_dist_l is not None

            if have_wrist_deriv:
                vel_l = (dist_l - prev_wrist_dist_l) / frame_dt
                vel_r = (dist_r - prev_wrist_dist_r) / frame_dt
                wrist_vel   = max(vel_l, vel_r)
                accel_l     = (vel_l - prev_wrist_vel_l) / frame_dt
                accel_r     = (vel_r - prev_wrist_vel_r) / frame_dt
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
            # Guard: only update prev_log_scale when keypoints are reliable
            # (m[47] > 0.5).  Invalid frames (ok=0) produce junk log_scale from
            # the bbox-area fallback; including them causes spurious spikes.
            log_scale_valid = (float(m[47]) > 0.5)
            dlog_scale_dt = 0.0
            if log_scale_valid and prev_log_scale is not None:
                dlog_scale_dt = (log_scale - prev_log_scale) / frame_dt
            if log_scale_valid:
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

                # Torso-height-fraction proximity gate.
                # Suppress alerts when the torso appears small in the frame
                # (torso_height_frac < far_height_fraction), meaning the person
                # is far away and unlikely to be an immediate threat.
                #
                #   torso_height_frac: ||hip_mid − shoulder_mid|| / frame_h.
                #     2-D Euclidean, robust to camera tilt.
                #     Returns 0.0 when keypoints are unavailable.
                #
                # Distance gate removed — all alerts pass regardless of subject distance.
                log_scale_now     = float(x[47])           # kept for diagnostics
                log_scale_ok      = (float(m[47]) > 0.5)   # kept for diagnostics
                torso_height_frac = compute_torso_height_frac(
                    kps, frame.shape[0], cfg.kp_conf_thresh, bbox)

                # Elapsed-time label: camera uses actual frames written ÷ fps
                # (accurate regardless of inference speed on the Pi);
                # file uses recorded frame number ÷ input FPS.
                t_s = (bg_count / fps_src if is_camera
                       else frame_idx / cfg.input_fps)
                # Diagnostics:
                #   thf → torso_height_frac = ||hip_mid−shoulder_mid|| / frame_h
                #   ls  → log_scale (ok=0 → keypoints invalid, using bbox fallback)
                if debug:
                    print(f"t={t_s:.2f}s  frame={frame_idx}  "
                          f"raw={hazard_raw:.3f}  ema={hazard_ema:.3f}  "
                          f"level={level}  "
                          f"thf={torso_height_frac:.3f}  "
                          f"ls={log_scale_now:.2f}(ok={int(log_scale_ok)})  "
                          f"pose={_t_pose*1000:.0f}ms  "
                          f"flow={_t_flow*1000:.0f}ms  "
                          f"gru={_t_gru*1000:.0f}ms  "
                          f"total={(_t_pose+_t_flow+_t_gru)*1000:.0f}ms  "
                          f"dbg={dbg}")
                elif verbose:
                    # Compact one-line summary per frame — less noisy than --debug.
                    print(f"t={t_s:.2f}s  frame={frame_idx}  "
                          f"level={level}  ema={hazard_ema:.3f}  "
                          f"thf={torso_height_frac:.3f}  persist={persist}")

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


# ── eval-dir: run full live pipeline on a pre-recorded dataset ───────────────

def run_on_file(video_path, cfg, model, device, detector, early_thresh,
                verbose=False, debug=False, simulate_live=False):
    """
    Run the full live detection state machine on a pre-recorded video file.

    Mirrors the file-path branch of run() exactly — same temporal state:
    hazard_ema, persist counter, log_area derivatives,
    wrist derivatives, and the proximity+approach gate.

    Unlike evaluate.py (which scores raw hazard per frame with no EMA/persist),
    this function replicates real-world deployment faithfully, so the timing
    differences compared to evaluate.py reflect genuine pipeline latency.

    Parameters
    ----------
    video_path   : str         — path to .mp4 video file
    cfg          : Config      — platform config (for_pc() or for_pi())
    model        : HazardGRU   — loaded model in eval mode
    device       : torch.device
    detector     : PoseDetector
    early_thresh : float       — PRE-CONTACT threshold (from meta.json)
    verbose      : bool        — collect per-GRU-frame timeline when True
    debug        : bool        — print per-frame gate diagnostics (mirrors
                                 run() console output; captured by TeeLogger)

    Returns
    -------
    dict with keys:
        first_precontact_frame : int   — first frame where level != NONE (−1 if never)
        first_high_frame       : int   — first frame where level ∈ {HIGH, CRITICAL}
        first_critical_frame   : int   — first frame where level == CRITICAL
        max_hazard_raw         : float — max raw GRU output seen
        max_hazard_ema         : float — max hazard_ema seen
        timeline               : list  — per-frame dicts (empty when verbose=False)
    None if the video file cannot be opened.
    """
    tracker = SingleTargetTracker()
    buf     = deque(maxlen=cfg.window_len)

    # ── temporal state — mirrors run() file-path branch exactly ───────────────
    prev_gray         = None
    prev_bbox         = None
    hazard_ema        = 0.0
    persist           = 0
    level             = "NONE"

    prev_wrist_dist_l = None;  prev_wrist_dist_r = None
    prev_wrist_vel_l  = 0.0;   prev_wrist_vel_r  = 0.0
    prev_log_area     = None;  prev_dlog_area_dt = None
    prev_log_scale    = None
    dt                = cfg.step_dt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # simulate_live: pace every frame to the video's native FPS and use
    # actual wall-clock dt for feature derivatives — identical to run() camera path.
    if simulate_live:
        fps_src      = cap.get(cv2.CAP_PROP_FPS) or cfg.input_fps
        frame_period = 1.0 / fps_src
        next_frame_t = time.monotonic()
        t_last_infer = None   # wall-clock time of last processed frame
    else:
        frame_period = None
        next_frame_t = None
        t_last_infer = None   # unused in non-live path

    first_precontact_frame = -1
    first_high_frame       = -1
    first_critical_frame   = -1
    max_hazard_raw         = 0.0
    max_hazard_ema         = 0.0
    timeline               = []
    frame_idx              = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # simulate_live: pace every frame (strided or not) to the video's native
        # FPS, exactly as frames would arrive from a live camera.
        if simulate_live:
            sleep_s = next_frame_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_frame_t += frame_period  # advance target for next frame

        # Apply frame stride (non-strided frames advance the index but skip inference)
        if frame_idx % cfg.frame_stride != 0:
            frame_idx += 1
            continue

        # Compute the time delta used for all velocity / acceleration features.
        # simulate_live: wall-clock Δt between processed frames (mirrors run()
        #   camera path; dt ≈ step_dt when inference is fast enough, >step_dt
        #   when a frame takes longer than step_dt — same catch-up behaviour).
        # Normal batch mode: fixed cfg.step_dt (matches training, deterministic).
        if simulate_live:
            t_now    = time.monotonic()
            frame_dt = (t_now - t_last_infer) if t_last_infer is not None else dt
            t_last_infer = t_now
        else:
            frame_dt = dt

        # Resize to training resolution — keeps pixel-magnitude features on
        # the same scale as the training data (log_scale, log_area, flow mags).
        if frame.shape[1] != cfg.infer_w or frame.shape[0] != cfg.infer_h:
            frame = cv2.resize(frame, (cfg.infer_w, cfg.infer_h))

        _t0 = time.perf_counter()
        det = detector.infer(frame)
        _t_pose = time.perf_counter() - _t0
        if det is None:
            if debug:
                print(f"  frame={frame_idx:5d}  det=MISS")
            frame_idx += 1
            continue

        bbox, track_age, lost = tracker.update(det["bbox"])
        det["bbox"] = bbox

        _t1 = time.perf_counter()
        x, m, dbg, prev_gray = build_features(
            frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
        _t_flow = time.perf_counter() - _t1
        prev_bbox = bbox

        log_area  = x[9]
        dist_l    = x[19]   # dist_l_wrist_torso
        dist_r    = x[20]   # dist_r_wrist_torso
        log_scale = x[47]

        # ── log_area derivatives (indices 10, 11) ─────────────────────────────
        dlog_area_dt   = 0.0
        d2log_area_dt2 = 0.0
        have_log_area_deriv = prev_log_area is not None
        if have_log_area_deriv:
            dlog_area_dt = (log_area - prev_log_area) / frame_dt
            if prev_dlog_area_dt is not None:
                d2log_area_dt2 = (dlog_area_dt - prev_dlog_area_dt) / frame_dt
        prev_log_area     = log_area
        prev_dlog_area_dt = dlog_area_dt
        x[10] = dlog_area_dt;    m[10] = 1.0 if have_log_area_deriv else 0.0
        x[11] = d2log_area_dt2;  m[11] = 1.0 if have_log_area_deriv else 0.0

        # ── wrist velocity / acceleration (indices 45, 46) ────────────────────
        wrist_vel   = 0.0
        wrist_accel = 0.0
        have_wrist_deriv = prev_wrist_dist_l is not None
        if have_wrist_deriv:
            vel_l = (dist_l - prev_wrist_dist_l) / frame_dt
            vel_r = (dist_r - prev_wrist_dist_r) / frame_dt
            wrist_vel   = max(vel_l, vel_r)
            accel_l     = (vel_l - prev_wrist_vel_l) / frame_dt
            accel_r     = (vel_r - prev_wrist_vel_r) / frame_dt
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

        # ── log_scale derivative for approach_rate (index 48) ────────────────
        # Guard: only update prev_log_scale when keypoints are valid (mirrors run()).
        log_scale_valid = (float(m[47]) > 0.5)
        dlog_scale_dt = 0.0
        if log_scale_valid and prev_log_scale is not None:
            dlog_scale_dt = (log_scale - prev_log_scale) / frame_dt
        if log_scale_valid:
            prev_log_scale = log_scale

        x, m = add_interaction_features(x, m, wrist_vel, wrist_accel,
                                        dlog_scale_dt)

        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        buf.append(xm)

        # ── GRU inference (only when window is full) ──────────────────────────
        if len(buf) == cfg.window_len:
            inp_t = torch.from_numpy(
                np.stack(buf)[None, :, :]).to(device)
            _t2 = time.perf_counter()
            with torch.no_grad():
                hazard_raw = float(model(inp_t).item())
            _t_gru = time.perf_counter() - _t2

            max_hazard_raw = max(max_hazard_raw, hazard_raw)

            hazard_ema = ((1 - cfg.ema_alpha) * hazard_ema
                          + cfg.ema_alpha * hazard_raw)
            max_hazard_ema = max(max_hazard_ema, hazard_ema)

            if hazard_ema > early_thresh:
                persist += 1
            else:
                persist = max(0, persist - 1)

            level = "NONE"
            if   hazard_ema > cfg.critical_thresh: level = "CRITICAL"
            elif hazard_ema > cfg.high_thresh:      level = "HIGH"
            elif persist    >= cfg.early_persist:   level = "PRE-CONTACT"

            # Distance gate removed — all alerts pass regardless of subject distance.
            log_scale_now     = float(x[47])           # kept for diagnostics
            log_scale_ok      = (float(m[47]) > 0.5)   # kept for diagnostics
            torso_height_frac = compute_torso_height_frac(
                det["kps"].astype(np.float32), frame.shape[0], cfg.kp_conf_thresh, bbox)

            # Per-frame diagnostics — captured by TeeLogger → eval_dir_<ts>.log.
            #   thf → torso_height_frac = ||hip_mid−shoulder_mid|| / frame_h
            #   ls  → log_scale (ok=0 → keypoints invalid)
            if debug:
                print(f"  frame={frame_idx:5d}  "
                      f"raw={hazard_raw:.3f}  ema={hazard_ema:.3f}  "
                      f"level={level:<11s}  "
                      f"thf={torso_height_frac:.3f}  "
                      f"ls={log_scale_now:.2f}(ok={int(log_scale_ok)})  "
                      f"persist={persist}  "
                      f"pose={_t_pose*1000:.0f}ms  "
                      f"flow={_t_flow*1000:.0f}ms  "
                      f"gru={_t_gru*1000:.0f}ms  "
                      f"total={(_t_pose+_t_flow+_t_gru)*1000:.0f}ms  "
                      f"dbg={dbg}")

            # Record first detection per alert level (after gate suppression).
            if level != "NONE":
                if first_precontact_frame == -1:
                    first_precontact_frame = frame_idx
                if level in ("HIGH", "CRITICAL") and first_high_frame == -1:
                    first_high_frame = frame_idx
                if level == "CRITICAL" and first_critical_frame == -1:
                    first_critical_frame = frame_idx

            if verbose:
                timeline.append({
                    "frame":            frame_idx,
                    "level":            level,
                    "hazard_raw":       round(hazard_raw, 4),
                    "hazard_ema":       round(hazard_ema, 4),
                    "torso_height_frac": round(torso_height_frac, 4),
                    "log_scale":         round(log_scale_now, 3),
                    "persist":          persist,
                })

        frame_idx += 1

    cap.release()

    return {
        "first_precontact_frame": first_precontact_frame,
        "first_high_frame":       first_high_frame,
        "first_critical_frame":   first_critical_frame,
        "max_hazard_raw":         max_hazard_raw,
        "max_hazard_ema":         max_hazard_ema,
        "timeline":               timeline,
    }


def evaluate_dir(eval_dir, verbose=False, debug=False, enable_pi=False,
                 simulate_live=False):
    """
    Evaluate the FULL live detection pipeline on a pre-recorded dataset.

    Mirrors the summary format of evaluate.py but uses run_on_file() (which
    replicates the hazard_ema + persist + gate state machine of run()) instead
    of evaluate.py's simpler raw-hazard scoring.  Results therefore match
    real-world deployment more accurately than evaluate.py.

    Dataset layout (same as evaluate.py / data/ directory):
        eval_dir/
            labels.json          — attack video labels (same schema as data/)
            attack/              — attack .mp4 files
            safe/                — compliance/safe .mp4 files

    labels.json schema (per entry):
        {
            "video.mp4": {
                "category":    "attack",
                "onset_frame": <int>,   // compliance → attack transition
                "attack_frame": <int>   // moment of first physical contact
            }, ...
        }

    Parameters
    ----------
    eval_dir  : str  — directory containing attack/, safe/, labels.json
    verbose   : bool — dump compact frame-by-frame timeline table per video
    debug     : bool — print full per-frame gate diagnostics (mirrors run()
                       console output); captured by TeeLogger → log file
    enable_pi : bool — use Config.for_pi() instead of Config.for_pc()
    """
    # ── logging ───────────────────────────────────────────────────────────────
    os.makedirs("outputs/logs", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("outputs/logs", f"eval_dir_{ts}.log")
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout
    print(f"Logging to {log_path}")

    set_seed(seed=42, deterministic=True)

    cfg = Config.for_pi() if enable_pi else Config.for_pc()

    with open("outputs/checkpoints/meta.json") as f:
        meta         = json.load(f)
        input_dim    = meta["input_dim"]
        early_thresh = meta.get("best_threshold", cfg.early_thresh)
        model_file   = meta.get("model_file", "hazard_gru.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(
        torch.load(f"outputs/checkpoints/{model_file}", map_location=device))
    model.eval()
    model.to(device)

    detector = PoseDetector(cfg)

    labels_file = os.path.join(eval_dir, "labels.json")
    with open(labels_file) as f:
        labels = json.load(f)

    print("=" * 80)
    print("EVAL-DIR EVALUATION (Full Live Pipeline) — Pre-contact Detection")
    print(f"Directory  : {eval_dir}")
    print(f"Threshold  : {early_thresh:.2f}  (from training optimisation)")
    print(f"Config     : {'Pi' if enable_pi else 'PC'}")
    print(f"GRU device : {device}")
    print(f"Live sim   : {'ON  (wall-clock dt, native-FPS pacing per video)' if simulate_live else 'OFF (fixed dt=step_dt, batch speed)'}")
    print("=" * 80)

    attack_results = []
    safe_results   = []

    # ── attack videos ─────────────────────────────────────────────────────────
    print("\n--- Processing Attack Videos ---")
    attack_dir_path = os.path.join(eval_dir, "attack")
    if not os.path.exists(attack_dir_path):
        print(f"Warning: {attack_dir_path} not found")
    else:
        for video_name, gt in labels.items():
            video_path = os.path.join(attack_dir_path, video_name)
            if not os.path.exists(video_path):
                print(f"Warning: {video_name} not found in attack/, skipping")
                continue

            print(f"Processing {video_name}...", end=" ", flush=True)
            if debug:
                print(f"\n  [debug] {video_name}")
            result = run_on_file(video_path, cfg, model, device, detector,
                                 early_thresh, verbose=verbose, debug=debug,
                                 simulate_live=simulate_live)

            if result is None:
                print("FAILED (cannot open)")
                continue

            result["video_name"]   = video_name
            result["ground_truth"] = gt
            attack_results.append(result)

            detected = result["first_precontact_frame"] >= 0
            print(f"{'DETECTED' if detected else 'MISSED'}  "
                  f"(max_ema={result['max_hazard_ema']:.3f}  "
                  f"max_raw={result['max_hazard_raw']:.3f})")

            if verbose and result["timeline"]:
                _print_verbose_timeline(video_name, result["timeline"])

    # ── safe videos ───────────────────────────────────────────────────────────
    print("\n--- Processing Safe Videos ---")
    safe_dir_path = os.path.join(eval_dir, "safe")
    if not os.path.exists(safe_dir_path):
        print(f"Warning: {safe_dir_path} not found")
    else:
        for video_file in sorted(os.listdir(safe_dir_path)):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(safe_dir_path, video_file)

            print(f"Processing {video_file}...", end=" ", flush=True)
            if debug:
                print(f"\n  [debug] {video_file}")
            result = run_on_file(video_path, cfg, model, device, detector,
                                 early_thresh, verbose=verbose, debug=debug,
                                 simulate_live=simulate_live)

            if result is None:
                print("FAILED (cannot open)")
                continue

            result["video_name"]   = video_file
            result["ground_truth"] = {"category": "safe"}
            safe_results.append(result)

            detected = result["first_precontact_frame"] >= 0
            print(f"{'FP' if detected else 'OK'}  "
                  f"(max_ema={result['max_hazard_ema']:.3f}  "
                  f"max_raw={result['max_hazard_raw']:.3f})")

            if verbose and result["timeline"]:
                _print_verbose_timeline(video_file, result["timeline"])

    # ── attack summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ATTACK VIDEOS — Multi-Level Warning Analysis")
    print("=" * 80)

    detected_attacks        = []
    missed_attacks          = []
    false_positives_attacks = []   # any level fires before onset_frame
    lead_times              = []
    lead_times_high         = []
    lead_times_critical     = []

    for r in attack_results:
        gt           = r["ground_truth"]
        attack_frame = gt["attack_frame"]
        onset_frame  = gt.get("onset_frame", 0)
        first_det    = r["first_precontact_frame"]

        if first_det >= 0:
            if first_det < onset_frame:
                # Alert fired during compliance phase → false positive
                false_positives_attacks.append(r)
                print(f"{r['video_name']:20s} | FP @ {first_det:4d} "
                      f"(before onset@{onset_frame:4d})  "
                      f"max_ema={r['max_hazard_ema']:.3f}")
            else:
                detected_attacks.append(r)
                lead_time = attack_frame - first_det
                lead_times.append(lead_time)

                # HIGH / CRITICAL lead times (only if first fire ≥ onset)
                if r["first_high_frame"] >= onset_frame:
                    lead_times_high.append(
                        attack_frame - r["first_high_frame"])
                if r["first_critical_frame"] >= onset_frame:
                    lead_times_critical.append(
                        attack_frame - r["first_critical_frame"])

                if first_det < attack_frame:    status = "PRE-CONTACT"
                elif first_det == attack_frame: status = "ON-TIME"
                else:                           status = "LATE"

                warning_info = f"PC@{r['first_precontact_frame']:4d}"
                if r["first_high_frame"] >= 0:
                    warning_info += f" H@{r['first_high_frame']:4d}"
                if r["first_critical_frame"] >= 0:
                    warning_info += f" C@{r['first_critical_frame']:4d}"

                print(f"{r['video_name']:20s} | "
                      f"Onset@{onset_frame:4d} Attack@{attack_frame:4d} "
                      f"Lead={lead_time:+4d} [{status}] | "
                      f"Warnings: {warning_info} | "
                      f"max_ema={r['max_hazard_ema']:.3f}")
        else:
            missed_attacks.append(r)
            print(f"{r['video_name']:20s} | MISSED "
                  f"(max_ema={r['max_hazard_ema']:.3f})")

    detection_rate   = len(detected_attacks)        / max(1, len(attack_results))
    miss_rate        = len(missed_attacks)           / max(1, len(attack_results))
    fp_attack_rate   = len(false_positives_attacks)  / max(1, len(attack_results))

    print(f"\nDetection Rate : {detection_rate:.1%} "
          f"({len(detected_attacks)}/{len(attack_results)})")
    print(f"Missed Rate    : {miss_rate:.1%} "
          f"({len(missed_attacks)}/{len(attack_results)})")
    print(f"FP (pre-onset) : {fp_attack_rate:.1%} "
          f"({len(false_positives_attacks)}/{len(attack_results)})")

    if lead_times:
        pc_warnings     = [lt for lt in lead_times if lt > 0]
        pc_warning_rate = len(pc_warnings) / max(1, len(lead_times))

        print(f"\n--- PRE-CONTACT Level (threshold={early_thresh:.2f}) ---")
        print(f"Pre-contact Warning Rate : {pc_warning_rate:.1%} "
              f"({len(pc_warnings)}/{len(lead_times)} detected attacks)")
        print(f"  Mean Lead Time   : {np.mean(lead_times):.1f} frames "
              f"({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Median Lead Time : {np.median(lead_times):.1f} frames "
              f"({np.median(lead_times)/cfg.input_fps:.2f}s)")
        if pc_warnings:
            print(f"  Pre-contact only : {np.mean(pc_warnings):.1f} frames "
                  f"({np.mean(pc_warnings)/cfg.input_fps:.2f}s)")

        if lead_times_high:
            high_pc = [lt for lt in lead_times_high if lt > 0]
            high_rate = len(lead_times_high) / max(1, len(detected_attacks))
            print(f"\n--- HIGH Level (threshold={cfg.high_thresh:.2f}) ---")
            print(f"HIGH Warning Rate  : {high_rate:.1%} "
                  f"({len(lead_times_high)}/{len(detected_attacks)} detected attacks)")
            print(f"  Mean Lead Time   : {np.mean(lead_times_high):.1f} frames "
                  f"({np.mean(lead_times_high)/cfg.input_fps:.2f}s)")
            print(f"  Median Lead Time : {np.median(lead_times_high):.1f} frames "
                  f"({np.median(lead_times_high)/cfg.input_fps:.2f}s)")
            if high_pc:
                print(f"  Pre-contact HIGH : {len(high_pc)/max(1,len(lead_times_high)):.1%} "
                      f"({len(high_pc)}/{len(lead_times_high)})")

        if lead_times_critical:
            crit_pc = [lt for lt in lead_times_critical if lt > 0]
            crit_rate = len(lead_times_critical) / max(1, len(detected_attacks))
            print(f"\n--- CRITICAL Level (threshold={cfg.critical_thresh:.2f}) ---")
            print(f"CRITICAL Warn Rate : {crit_rate:.1%} "
                  f"({len(lead_times_critical)}/{len(detected_attacks)} detected attacks)")
            print(f"  Mean Lead Time   : {np.mean(lead_times_critical):.1f} frames "
                  f"({np.mean(lead_times_critical)/cfg.input_fps:.2f}s)")
            print(f"  Median Lead Time : {np.median(lead_times_critical):.1f} frames "
                  f"({np.median(lead_times_critical)/cfg.input_fps:.2f}s)")
            if crit_pc:
                print(f"  Pre-contact CRIT : {len(crit_pc)/max(1,len(lead_times_critical)):.1%} "
                      f"({len(crit_pc)}/{len(lead_times_critical)})")

    # ── safe video summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAFE VIDEOS")
    print("=" * 80)

    false_positives = []
    true_negatives  = []

    for r in safe_results:
        first_det = r["first_precontact_frame"]
        if first_det >= 0:
            false_positives.append(r)
            print(f"{r['video_name']:20s} | FP @ frame {first_det}  "
                  f"(max_ema={r['max_hazard_ema']:.3f})")
        else:
            true_negatives.append(r)

    fp_rate = len(false_positives) / max(1, len(safe_results))
    tn_rate = len(true_negatives)  / max(1, len(safe_results))

    print(f"\nFalse Positive Rate (safe) : {fp_rate:.1%} "
          f"({len(false_positives)}/{len(safe_results)})")
    print(f"True Negative Rate         : {tn_rate:.1%} "
          f"({len(true_negatives)}/{len(safe_results)})")

    # ── overall summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total Videos : {len(attack_results) + len(safe_results)}  "
          f"(attacks={len(attack_results)}  safe={len(safe_results)})")
    print(f"\nThresholds:")
    print(f"  PRE-CONTACT : {early_thresh:.2f}")
    print(f"  HIGH        : {cfg.high_thresh:.2f}")
    print(f"  CRITICAL    : {cfg.critical_thresh:.2f}")
    print(f"\nAttack Detection  : {detection_rate:.1%}")
    print(f"FP (safe)         : {fp_rate:.1%}")
    print(f"FP (pre-onset)    : {fp_attack_rate:.1%}")
    if lead_times:
        print(f"\nWarning Performance:")
        print(f"  PRE-CONTACT : {pc_warning_rate:.1%} pre-contact | "
              f"Mean Lead: {np.mean(lead_times):.1f} frames "
              f"({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        if lead_times_high:
            h_pc_rate = len([lt for lt in lead_times_high if lt > 0]) / max(1, len(lead_times_high))
            print(f"  HIGH        : {len(lead_times_high)}/{len(detected_attacks)} attacks | "
                  f"{h_pc_rate:.1%} pre-contact | "
                  f"Mean Lead: {np.mean(lead_times_high):.1f} frames "
                  f"({np.mean(lead_times_high)/cfg.input_fps:.2f}s)")
        if lead_times_critical:
            c_pc_rate = len([lt for lt in lead_times_critical if lt > 0]) / max(1, len(lead_times_critical))
            print(f"  CRITICAL    : {len(lead_times_critical)}/{len(detected_attacks)} attacks | "
                  f"{c_pc_rate:.1%} pre-contact | "
                  f"Mean Lead: {np.mean(lead_times_critical):.1f} frames "
                  f"({np.mean(lead_times_critical)/cfg.input_fps:.2f}s)")
    print("=" * 80)

    print(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to : {log_path}")

    detector.release()


def _print_verbose_timeline(video_name, timeline):
    """Print a formatted frame-by-frame timeline table for verbose mode."""
    print(f"\n  --- Verbose timeline: {video_name} ---")
    hdr = (f"  {'frame':>6}  {'level':>10}  {'raw':>6}  {'ema':>6}  "
           f"{'thf':>7}  {'ls':>6}  {'persist':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in timeline:
        print(f"  {row['frame']:>6}  {row['level']:>10}  "
              f"{row['hazard_raw']:>6.3f}  {row['hazard_ema']:>6.3f}  "
              f"{row['torso_height_frac']:>7.3f}  "
              f"{row['log_scale']:>6.3f}  "
              f"{row['persist']:>7}")
    print()


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

    parser.add_argument(
        "--eval-dir", metavar="DIR", default=None,
        help="Evaluate the full live pipeline on a pre-recorded dataset "
             "directory (must contain attack/, safe/, labels.json).  "
             "Mirrors evaluate.py but uses the complete state machine "
             "(hazard_ema, persist, gate) so results match real deployment.  "
             "Cannot be combined with a video source or --record.")

    parser.add_argument(
        "--verbose", action="store_true",
        help="With --eval-dir: dump the compact frame-by-frame timeline table "
             "(level, hazard_raw, hazard_ema, gate, thf, log_scale, "
             "persist) for every video processed.")

    parser.add_argument(
        "--debug", action="store_true",
        help="With --eval-dir: print full per-frame diagnostics for every "
             "video (mirrors the live infer.py console output — raw, ema, level, "
             "thf, ls, ok, persist, dbg). "
             "Also flags det=MISS frames where pose detection yields no bbox. "
             "Output is captured by the TeeLogger → eval_dir_<timestamp>.log.")

    parser.add_argument(
        "--simulate-live", action="store_true",
        help="Pace frame delivery to each video's native FPS and use actual "
             "wall-clock dt for feature derivatives, making pre-recorded videos "
             "behave identically to a live camera feed.  Works for both single-video "
             "inference and --eval-dir batch evaluation — use on the Pi5 to obtain "
             "a detection-rate / FP-rate report that reflects real-time performance "
             "without needing a live camera.  Has no effect on camera (int) sources.")

    args = parser.parse_args()

    # ── eval-dir mode: evaluate a pre-recorded dataset ────────────────────────
    if args.eval_dir is not None:
        # evaluate_dir() sets up its own TeeLogger internally
        evaluate_dir(args.eval_dir,
                     verbose=args.verbose,
                     debug=args.debug,
                     enable_pi=args.pi,
                     simulate_live=args.simulate_live)
        sys.exit(0)

    # ── live / file inference mode ────────────────────────────────────────────
    # "0", "1", … → int (camera device index); anything else → file path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Create logs directory
    os.makedirs("outputs/logs", exist_ok=True)

    # Timestamped log filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("outputs/logs", f"infer_{ts}.log")

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
        enable_pi=args.pi,
        verbose=args.verbose,
        debug=args.debug,
        simulate_live=args.simulate_live)
