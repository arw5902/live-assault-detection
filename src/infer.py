import sys
import json
import math
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
from .model import HazardGRU, HazardLSTM, HazardTransformer
from .pose_detector import PoseDetector
from .tracker import SingleTargetTracker
from .features import build_features, add_interaction_features, compute_torso_height_frac, TemporalDerivatives
from .dataset import extract_sequences, windowize
from .feature_importance import compute_permutation_importance, print_importance_ranking, save_importance_results
from .utils import set_seed

# BGR colours for each hazard level (binary: THREAT / NONE)
LEVEL_COLOR = {
    "NONE":        (  0, 200,   0),  # green
    "THREAT":      (  0,   0, 255),  # red
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


def _dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 color, thickness: int = 2,
                 dash: int = 8, gap: int = 4) -> None:
    """Draw a dashed rectangle on img (in-place). OpenCV has no native dash support."""
    def _seg(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max(int((dx**2 + dy**2) ** 0.5), 1)
        step   = dash + gap
        for s in range(0, length, step):
            e  = min(s + dash, length)
            t0 = s / length;  t1 = e / length
            pt0 = (int(p1[0] + dx * t0), int(p1[1] + dy * t0))
            pt1 = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
            cv2.line(img, pt0, pt1, color, thickness, cv2.LINE_AA)

    _seg((x1, y1), (x2, y1))   # top
    _seg((x2, y1), (x2, y2))   # right
    _seg((x2, y2), (x1, y2))   # bottom
    _seg((x1, y2), (x1, y1))   # left


def draw_overlay(frame: np.ndarray, bbox, level: str, hazard: float,
                 kps: np.ndarray = None,
                 roi_torso=None, roi_lower=None) -> np.ndarray:
    """
    Return an annotated copy of frame (original is not modified).
    kps       : (17, 3) COCO-17 keypoints [x, y, conf], or None to skip skeleton.
    roi_torso : (x1, y1, x2, y2) torso optical-flow ROI, or None to skip.
    roi_lower : (x1, y1, x2, y2) lower-body optical-flow ROI, or None to skip.
    """
    vis   = frame.copy()
    h, w  = vis.shape[:2]
    color = LEVEL_COLOR.get(level, (0, 200, 0))

    # COCO-17 skeleton (drawn before the bbox so bbox sits on top)
    if kps is not None:
        draw_skeleton(vis, kps)

    # Torso ROI — cyan dashed rectangle
    if roi_torso is not None:
        tx1, ty1, tx2, ty2 = [int(v) for v in roi_torso]
        _dashed_rect(vis, tx1, ty1, tx2, ty2, (255, 210, 0))
        cv2.putText(vis, "T", (tx1 + 2, ty1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 210, 0), 1, cv2.LINE_AA)

    # Lower-body ROI — magenta dashed rectangle
    if roi_lower is not None:
        lx1, ly1, lx2, ly2 = [int(v) for v in roi_lower]
        _dashed_rect(vis, lx1, ly1, lx2, ly2, (255, 0, 200))
        cv2.putText(vis, "L", (lx1 + 2, ly1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 200), 1, cv2.LINE_AA)

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
    """Core inference loop — video file or live camera.

    Args:
        video_source: int → camera device index (cv2.VideoCapture);
            str → video file path.
        display: show an annotated OpenCV window. Automatically enabled
            when video_source is int.
        record: write raw (un-annotated) frames to .mp4 and save a
            per-frame hazard JSON sidecar. The saved video is fully
            compatible with evaluate.py — add a labels.json and run
            evaluate.py on it to measure accuracy offline.
        record_dir: directory for saved recordings.
        use_picamera2: use picamera2 for Pi Camera Module (RPi5 + AI HAT+).
            When False and source is int, cv2.VideoCapture is used
            (suitable for USB webcam or libcamera V4L2 bridge).
        show_skeleton: overlay COCO-17 keypoints and limb lines on the
            display window. Has no effect on the raw recording.
        simulate_live: pace a video-file source to match the file's native
            FPS and use actual wall-clock dt for feature derivatives —
            making a pre-recorded video behave identically to a live camera
            feed. Ignored when video_source is a camera int.
    """
    set_seed(seed=42, deterministic=True)

    # Note Config.for_pi() will disable skeleton being displayed on screen, 
    # probably due to hailo backend instead of ultralytics
    cfg      = Config.for_pi() if enable_pi else Config.for_pc()
    detector = PoseDetector(cfg)
    tracker  = SingleTargetTracker()

    print(f"Pose backend: {cfg.pose_backend}")


    # ── load model ────────────────────────────────────────────────────────────
    with open("outputs/checkpoints/meta.json") as f:
        meta         = json.load(f)
        input_dim    = meta["input_dim"]
        early_thresh = meta.get("best_threshold", cfg.early_thresh)
        model_file   = meta.get("model_file", "hazard_gru.pt")
        model_type   = meta.get("model_type", "gru")

    # Model runs on CPU on Pi — too small to benefit from NPU, and PyTorch
    # is not available on the Hailo NPU without DFC compilation.
    # On a PC with GPU this will use CUDA automatically.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "lstm":
        model = HazardLSTM(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    elif model_type == "transformer":
        model = HazardTransformer(input_dim=input_dim, d_model=cfg.gru_hidden, dropout=0.0)
    else:
        model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(
        torch.load(f"outputs/checkpoints/{model_file}", map_location=device))
    model.eval()
    model.to(device)

    print(f"Threshold   : {early_thresh:.2f} (from training optimisation)")
    print(f"{model_type.upper()} device : {device}")
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
    roi_torso         = None    # last known torso optical-flow ROI (x1,y1,x2,y2)
    roi_lower         = None    # last known lower-body optical-flow ROI (x1,y1,x2,y2)

    deriv             = TemporalDerivatives()
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
        fps_src  = float(cam_fps)
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: cannot open video source: {video_source}")
            detector.release()
            return
        fps_src = cap.get(cv2.CAP_PROP_FPS) or cfg.input_fps

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
                # model was trained on, and so that the recording is consistent
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
                                                kps if show_skeleton else None,
                                                roi_torso=roi_torso,
                                                roi_lower=roi_lower))
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
                                            kps if show_skeleton else None,
                                            roi_torso=roi_torso,
                                            roi_lower=roi_lower))
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
            roi_torso = dbg.get("roi_torso")   # carry forward for display
            roi_lower = dbg.get("roi_lower")   # carry forward for display

            # Temporal derivatives + interaction features — shared implementation.
            dlog_scale_dt = deriv.update(x, m, frame_dt)
            x, m = add_interaction_features(x, m, x[45], x[46], dlog_scale_dt, cfg=cfg)

            xm = np.concatenate([x, m], axis=0).astype(np.float32)
            buf.append(xm)

            # ── model inference ───────────────────────────────────────────────
            if len(buf) == cfg.window_len:
                inp_t = torch.from_numpy(
                    np.stack(buf)[None, :, :]).to(device)
                _t2 = time.perf_counter()
                with torch.no_grad():
                    hazard_raw = float(model(inp_t).item())
                _t_model = time.perf_counter() - _t2

                hazard_ema = ((1 - cfg.ema_alpha) * hazard_ema
                              + cfg.ema_alpha * hazard_raw)

                if hazard_ema > early_thresh:
                    persist += 1
                else:
                    persist = max(0, persist - 1)

                level = "NONE"
                if hazard_ema > early_thresh and persist >= cfg.early_persist:
                    level = "THREAT"

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
                          f"{model_type}={_t_model*1000:.0f}ms  "
                          f"total={(_t_pose+_t_flow+_t_model)*1000:.0f}ms  "
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
                                        kps if show_skeleton else None,
                                        roi_torso=roi_torso,
                                        roi_lower=roi_lower))
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
                verbose=False, debug=False, simulate_live=False,
                ignore_start_sec=0.0,
                ignore_start_frame=0, model_type="gru"):
    """
    Run the full live detection state machine on a pre-recorded video file.

    Mirrors the file-path branch of run() exactly — same temporal state:
    hazard_ema, persist counter, log_area derivatives,
    wrist derivatives, and the proximity+approach gate.

    Unlike evaluate.py (which scores raw hazard per frame with no EMA/persist),
    this function replicates real-world deployment faithfully, so the timing
    differences compared to evaluate.py reflect genuine pipeline latency.

    Args:
        video_path: path to .mp4 video file.
        cfg: platform config (for_pc() or for_pi()).
        model: loaded HazardGRU/HazardLSTM in eval mode.
        device: torch.device.
        detector: PoseDetector instance.
        early_thresh: THREAT threshold (from meta.json best_threshold).
        verbose: collect per-frame timeline when True.
        debug: print per-frame gate diagnostics (mirrors run() console
            output; captured by TeeLogger).
        ignore_start_sec: ignore any THREAT detections in the first N
            seconds of the video (default 0 = no skip).
        ignore_start_frame: ignore THREAT detections before this raw-video
            frame index (30 fps). When > 0, takes precedence over
            ignore_start_sec. Used to align the skip region with the
            labelled 'first_stand' frame from labels.json.

    Returns:
        Dict with keys:
            first_precontact_frame: first frame where level == THREAT (−1 if never).
            all_threat_frames: every frame where level == THREAT.
            max_hazard_raw: max raw model output seen.
            max_hazard_ema: max hazard_ema seen.
            timeline: per-frame dicts (empty when verbose=False).
        None if the video file cannot be opened.
    """
    # Seed RNG so each video gets identical point sampling regardless of
    # processing order within evaluate_dir().
    set_seed(seed=42, deterministic=True)

    tracker = SingleTargetTracker()
    buf     = deque(maxlen=cfg.window_len)

    # ── temporal state — mirrors run() exactly ───────────────────────────────
    prev_gray         = None
    prev_bbox         = None
    hazard_ema        = 0.0
    persist           = 0
    level             = "NONE"
    deriv             = TemporalDerivatives()
    dt                = cfg.step_dt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Compute the frame index threshold for ignoring early THREAT detections.
    # ignore_start_frame (explicit frame number) takes precedence when set;
    # otherwise fall back to ignore_start_sec * video_fps.  Any THREAT
    # detection at frame_idx < skip_before_frame is suppressed.
    _vid_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.input_fps
    if ignore_start_frame > 0:
        skip_before_frame = int(ignore_start_frame)
    elif ignore_start_sec > 0:
        skip_before_frame = int(ignore_start_sec * _vid_fps)
    else:
        skip_before_frame = 0

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
    all_threat_frames      = []   # every frame where level == "THREAT"
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

        # Temporal derivatives + interaction features — shared implementation.
        dlog_scale_dt = deriv.update(x, m, frame_dt)
        x, m = add_interaction_features(x, m, x[45], x[46], dlog_scale_dt, cfg=cfg)

        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        buf.append(xm)

        # ── model inference (only when window is full) ────────────────────────
        if len(buf) == cfg.window_len:
            inp_t = torch.from_numpy(
                np.stack(buf)[None, :, :]).to(device)
            _t2 = time.perf_counter()
            with torch.no_grad():
                hazard_raw = float(model(inp_t).item())
            _t_model = time.perf_counter() - _t2

            max_hazard_raw = max(max_hazard_raw, hazard_raw)

            hazard_ema = ((1 - cfg.ema_alpha) * hazard_ema
                          + cfg.ema_alpha * hazard_raw)
            max_hazard_ema = max(max_hazard_ema, hazard_ema)

            if hazard_ema > early_thresh:
                persist += 1
            else:
                persist = max(0, persist - 1)

            level = "NONE"
            if hazard_ema > early_thresh and persist >= cfg.early_persist:
                level = "THREAT"

            # Distance gate removed — all alerts pass regardless of subject distance.
            log_scale_now     = float(x[47])           # kept for diagnostics
            log_scale_ok      = (float(m[47]) > 0.5)   # kept for diagnostics
            torso_height_frac = compute_torso_height_frac(
                det["kps"].astype(np.float32), frame.shape[0], cfg.kp_conf_thresh, bbox)

            # Per-frame diagnostics — captured by TeeLogger → inference_<ts>.log.
            #   thf → torso_height_frac = ||hip_mid−shoulder_mid|| / frame_h
            #   ls  → log_scale (ok=0 → keypoints invalid)
            if debug:
                print(f"  frame={frame_idx:5d}  "
                      f"raw={hazard_raw:.3f}  ema={hazard_ema:.3f}  "
                      f"level={level:<8s}  "
                      f"thf={torso_height_frac:.3f}  "
                      f"ls={log_scale_now:.2f}(ok={int(log_scale_ok)})  "
                      f"persist={persist}  "
                      f"pose={_t_pose*1000:.0f}ms  "
                      f"flow={_t_flow*1000:.0f}ms  "
                      f"{model_type}={_t_model*1000:.0f}ms  "
                      f"total={(_t_pose+_t_flow+_t_model)*1000:.0f}ms  "
                      f"dbg={dbg}")

            # Record detection frames (skip frames before first_stand).
            if level == "THREAT" and frame_idx >= skip_before_frame:
                if first_precontact_frame == -1:
                    first_precontact_frame = frame_idx
                all_threat_frames.append(frame_idx)

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
        "all_threat_frames":      all_threat_frames,
        "max_hazard_raw":         max_hazard_raw,
        "max_hazard_ema":         max_hazard_ema,
        "timeline":               timeline,
    }


def _run_window_analysis(eval_dir, cfg, model, device, detector,
                         early_thresh, labels, attack_dir_path,
                         safe_dir_path, ts, verbose=False,
                         split_silence=False, model_type="gru"):
    """
    Window-level analysis (no EMA): F1, PR curve, ROC curve, feature importance.

    Extracts windows from eval-dir videos using the same segment logic as
    the live-simulation pipeline (excluding frames before 'first_stand',
    the silence zone, and frames after push_end_frame), then evaluates raw
    model scores.

    When verbose=True, prints per-window FP and FN locations (video name,
    segment, frame range) to help localize where mistakes occur.

    When split_silence=True, the two excluded zones are re-included:
      - silence zone  [first_stand_after_sit, last_backward_frame] → safe
        (labelled "silence-safe", extends the safe seg1 region)
      - seg2-approach (last_backward_frame, push_start_frame]      → attack
        (labelled "silence-attack", the approach phase before the push)
    """
    print("\n" + "=" * 80)
    print("WINDOW-LEVEL ANALYSIS (Inference, no EMA)")
    print("=" * 80)
    if split_silence:
        print("Mode: --split-silence (silence zone → safe, seg2-approach → attack)")
        print("Evaluated segments (attack videos):")
        print("  seg1-pre        = [first_stand, first_stand_after_sit)         → safe")
        print("  silence-safe    = [first_stand_after_sit, last_backward_frame] → safe")
        print("  silence-attack  = (last_backward_frame, push_start_frame]     → attack")
        print("  seg3-push       = (push_start_frame, push_end_frame]           → attack")
        print("Excluded segments:")
        print("  [0, first_stand)                                               → skip start")
        print("  (push_end_frame, end]                                          → post-attack")
    else:
        print("Mode: default (silence zone + seg2-approach excluded)")
        print("Evaluated segments (attack videos):")
        print("  seg1-pre   = [first_stand, first_stand_after_sit)     → safe")
        print("  seg3-push  = (push_start_frame, push_end_frame]       → attack")
        print("Excluded segments:")
        print("  [0, first_stand)                                      → skip start")
        print("  [first_stand_after_sit, last_backward_frame]           → silence zone")
        print("  (last_backward_frame, push_start_frame] (seg2-approach)→ label-noisy")
        print("  (push_end_frame, end]                                  → post-attack")
    print(f"Re-extracting windows from videos for raw {model_type.upper()} analysis...")

    set_seed(seed=42, deterministic=True)

    all_Xw, all_Mw, all_yw = [], [], []
    # Parallel metadata list aligned with all_Xw / all_Mw / all_yw after concat.
    # Each entry: (video_name, segment_label, end_raw_frame, start_raw_frame)
    # end_raw_frame = raw-video frame index (30 fps) where the window ENDS
    #                 — this is the frame at which the model makes its decision.
    # start_raw_frame = raw-video frame index where the window begins.
    all_meta = []

    # Fallback ignore duration when a video has no 'first_stand' label.
    _fallback_ignore_step = int(12.0 * cfg.input_fps) // cfg.frame_stride

    def _append_segment(Xw, Mw, yw, video_name, seg_label, seg_start_step,
                        seg_stride):
        """Append a windowized segment plus per-window metadata."""
        if Xw.shape[0] == 0:
            return
        all_Xw.append(Xw)
        all_Mw.append(Mw)
        all_yw.append(yw)
        for i in range(Xw.shape[0]):
            start_step = seg_start_step + i * seg_stride
            end_step   = start_step + cfg.window_len - 1
            start_raw  = start_step * cfg.frame_stride
            end_raw    = end_step   * cfg.frame_stride
            all_meta.append((video_name, seg_label, end_raw, start_raw))

    # Attack videos — split into valid segments, excluding frames before
    # first_stand, the silence zone, seg2-approach, and frames after
    # push_end_frame.  Frame indices in labels.json are raw 30fps; after
    # frame_stride they map to step = raw_frame // frame_stride.
    #
    # Segment 1: [first_stand_step, first_stand_after_sit)               → all safe (0)
    # Silence zone: [first_stand_after_sit, last_backward_frame]         → EXCLUDED
    # Segment 2: (last_backward_frame, push_start_frame]                 → EXCLUDED (label-noisy:
    #            subject often raises hands during approach, producing
    #            genuine threat cues that the "safe" label does not
    #            capture; keeping seg2 would inflate FP on windows that
    #            the model is arguably correctly flagging).
    # Segment 3: (push_start_frame, push_end_frame]                      → attack (1)
    # After push_end_frame                                                → EXCLUDED
    #
    # The silence zone and seg2-approach are adjacent and together form
    # one contiguous excluded region [first_stand_after_sit,
    # push_start_frame]; they are tracked separately for future work
    # (e.g. relabelling seg2-approach as a third "pre-attack" class).
    if os.path.exists(attack_dir_path):
        for video_name, gt in labels.items():
            video_path = os.path.join(attack_dir_path, video_name)
            if not os.path.exists(video_path):
                continue
            X, M, y = extract_sequences(video_path, "safe", detector, cfg)
            if X.shape[0] == 0:
                continue

            # Per-video skip: use labelled first_stand; fall back to 12s.
            first_stand_raw = gt.get("first_stand", None)
            if first_stand_raw is not None:
                ignore_start_step = first_stand_raw // cfg.frame_stride
            else:
                ignore_start_step = _fallback_ignore_step

            stand_step  = gt.get("first_stand_after_sit", 0) // cfg.frame_stride
            back_step   = gt.get("last_backward_frame", 0)   // cfg.frame_stride
            push_step   = gt.get("push_start_frame", 0)      // cfg.frame_stride
            end_step    = gt.get("push_end_frame", X.shape[0] * cfg.frame_stride) // cfg.frame_stride

            # Segment 1: from first_stand to before silence zone (all safe)
            seg1_start = ignore_start_step
            seg1_end   = stand_step
            if seg1_end > seg1_start and (seg1_end - seg1_start) >= cfg.window_len:
                seg1_X = X[seg1_start:seg1_end]
                seg1_M = M[seg1_start:seg1_end]
                seg1_y = np.zeros(seg1_end - seg1_start, dtype=np.float32)
                Xw, Mw, yw = windowize(seg1_X, seg1_M, seg1_y,
                                       cfg.window_len, stride=1)
                _append_segment(Xw, Mw, yw, video_name, "seg1-pre",
                                seg1_start, 1)

            # --split-silence: re-include the two excluded zones.
            #   silence zone  [stand_step, back_step+1)  → safe  ("silence-safe")
            #   seg2-approach [back_step+1, push_step+1) → attack ("silence-attack")
            if split_silence:
                # Silence-safe: [first_stand_after_sit, last_backward_frame]
                ss_start = stand_step
                ss_end   = back_step + 1
                if ss_end > ss_start and (ss_end - ss_start) >= cfg.window_len:
                    ss_X = X[ss_start:ss_end]
                    ss_M = M[ss_start:ss_end]
                    ss_y = np.zeros(ss_end - ss_start, dtype=np.float32)
                    Xw, Mw, yw = windowize(ss_X, ss_M, ss_y,
                                           cfg.window_len, stride=1)
                    _append_segment(Xw, Mw, yw, video_name, "silence-safe",
                                    ss_start, 1)

                # Silence-attack: (last_backward_frame, push_start_frame]
                sa_start = back_step + 1
                sa_end   = min(push_step + 1, X.shape[0])
                if sa_end > sa_start and (sa_end - sa_start) >= cfg.window_len:
                    sa_X = X[sa_start:sa_end]
                    sa_M = M[sa_start:sa_end]
                    sa_y = np.ones(sa_end - sa_start, dtype=np.float32)
                    Xw, Mw, yw = windowize(sa_X, sa_M, sa_y,
                                           cfg.window_len, stride=1)
                    _append_segment(Xw, Mw, yw, video_name, "silence-attack",
                                    sa_start, 1)

            # Segment 2 (approach): EXCLUDED when split_silence=False
            #     (label-noisy); otherwise included in silence-attack above.

            # Segment 3: push_start to push_end (all attack)
            seg3_start = push_step + 1
            seg3_end   = min(end_step + 1, X.shape[0])
            if seg3_end > seg3_start and (seg3_end - seg3_start) >= cfg.window_len:
                seg3_X = X[seg3_start:seg3_end]
                seg3_M = M[seg3_start:seg3_end]
                seg3_y = np.ones(seg3_end - seg3_start, dtype=np.float32)
                Xw, Mw, yw = windowize(seg3_X, seg3_M, seg3_y,
                                       cfg.window_len, stride=1)
                _append_segment(Xw, Mw, yw, video_name, "seg3-push",
                                seg3_start, 1)

    # Safe videos (if any)
    if os.path.exists(safe_dir_path):
        for video_file in sorted(os.listdir(safe_dir_path)):
            if not video_file.lower().endswith(".mp4"):
                continue
            video_path = os.path.join(safe_dir_path, video_file)
            X, M, y = extract_sequences(video_path, "safe", detector, cfg)
            if X.shape[0] == 0:
                continue
            Xw, Mw, yw = windowize(X, M, y, cfg.window_len, stride=1)
            _append_segment(Xw, Mw, yw, f"safe/{video_file}", "full",
                            0, 1)

    if not all_Xw:
        print("No windows extracted — skipping window-level analysis.")
        return

    Xw_all = np.concatenate(all_Xw)
    Mw_all = np.concatenate(all_Mw)
    yw_all = np.concatenate(all_yw)

    holdout_input = np.concatenate([Xw_all, Mw_all], axis=2).astype(np.float32)

    n_attack_w = int((yw_all >= 0.5).sum())
    n_safe_w = int((yw_all < 0.5).sum())
    print(f"  Total windows: {len(yw_all)} (attack={n_attack_w}, safe={n_safe_w})")

    # ── Collect raw model predictions (no EMA) ──────────────────────────────
    model.eval()
    all_scores = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(holdout_input), batch_size):
            batch = torch.from_numpy(holdout_input[i:i+batch_size]).to(device)
            scores = model(batch).cpu().numpy()
            all_scores.extend(scores)

    y_true = (yw_all >= 0.5).astype(int)
    y_score = np.array(all_scores)

    # ── Window-level metrics at selected threshold ───────────────────────────
    from sklearn.metrics import (precision_recall_fscore_support,
                                 accuracy_score,
                                 precision_recall_curve,
                                 average_precision_score,
                                 roc_curve, auc)

    y_pred = (y_score >= early_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    n_neg = max(1, int((y_true == 0).sum()))
    fp_count = int(((y_pred == 1) & (y_true == 0)).sum())
    fpr_val = fp_count / n_neg

    print(f"\n--- Window-Level Metrics @ threshold={early_thresh:.2f} (no EMA) ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  FPR       : {fpr_val:.2%}")

    # ── Verbose per-video timeline (per-window decisions) ───────────────────
    # Mirrors the --eval-dir --verbose timeline but adapted for window-level
    # analysis: one row per window, columns drop EMA / persist (no smoothing
    # applied here) and instead show ground-truth and a result tag so FP and
    # MISS rows are easy to eyeball.
    if verbose and len(all_meta) == len(y_true):
        from collections import defaultdict

        # Group window indices by video, preserving per-window metadata.
        vid_to_indices = defaultdict(list)
        for k in range(len(all_meta)):
            vid = all_meta[k][0]
            vid_to_indices[vid].append(k)

        for vid in sorted(vid_to_indices.keys()):
            idx_list = vid_to_indices[vid]
            # Sort by end-frame so the timeline reads chronologically.
            idx_list.sort(key=lambda k: all_meta[k][2])

            print(f"\n  --- Verbose timeline: {vid} ---")
            hdr = (f"  {'frame':>6}  {'start':>6}  {'level':>10}  "
                   f"{'raw':>6}  {'gt':>7}  {'result':>7}  "
                   f"{'segment':<14}")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

            for k in idx_list:
                _, seg, end_raw, start_raw = all_meta[k]
                raw_score = float(y_score[k])
                pred      = int(y_pred[k])
                truth     = int(y_true[k])
                level     = "THREAT" if pred == 1 else "NONE"
                gt_str    = "ATTACK" if truth == 1 else "SAFE"
                # Result tag: highlight mistakes (FP / MISS) so they stand out.
                if pred == 1 and truth == 0:
                    result = "FP"
                elif pred == 0 and truth == 1:
                    result = "MISS"
                elif pred == 1 and truth == 1:
                    result = "TP"
                else:
                    result = "TN"
                print(f"  {end_raw:>6}  {start_raw:>6}  {level:>10}  "
                      f"{raw_score:>6.3f}  {gt_str:>7}  {result:>7}  "
                      f"{seg:<14}")
            print()

    # ── Verbose per-video rate summary ───────────────────────────────────────
    # Per-video table of detection rates:
    #   seg3-push / silence-attack → true-positive rate (recall on the
    #                                 attack-labelled windows)
    #   seg1-pre / seg2-approach / safe → false-positive rate (safe windows
    #                                     that crossed threshold)
    # Percentages are more actionable than raw FP/FN counts for identifying
    # which videos drive the aggregate metric.
    if verbose and len(all_meta) == len(y_true):
        from collections import defaultdict

        # per-video per-segment tally: {video: {segment: [n_windows, n_flagged,
        #                                                 n_correct_label]}}
        #   n_windows       — total windows in (video, segment)
        #   n_flagged       — y_pred == 1 count
        #   n_attack_label  — y_true == 1 count (for sanity / expected label)
        per_vid = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        for k in range(len(all_meta)):
            vid, seg, _, _ = all_meta[k]
            per_vid[vid][seg][0] += 1
            if y_pred[k] == 1:
                per_vid[vid][seg][1] += 1
            if y_true[k] == 1:
                per_vid[vid][seg][2] += 1

        def _rate(n, total):
            return f"{n}/{total} ({100.0 * n / max(1, total):5.1f}%)"

        # Show silence-safe / silence-attack columns when the flag produced
        # windows.
        has_silence = any(
            ("silence-attack" in per_vid[v] or "silence-safe" in per_vid[v])
            for v in per_vid)

        print(f"\n--- Per-Video Window Rates @ threshold={early_thresh:.2f} ---")
        print(f"  seg3-push:       detection rate  (attack windows flagged)")
        if has_silence:
            print(f"  silence-attack:  detection rate  (seg2-approach → attack)")
            print(f"  silence-safe:    FP rate         (silence zone → safe)")
        print(f"  seg1-pre:        FP rate         (pre-stand safe windows flagged)")
        if not has_silence:
            print(f"  seg2-approach:   FP rate         (post-silence safe windows flagged)")
        print(f"  safe/*:          FP rate         (full safe video)")

        # Header — include silence columns when relevant.
        if has_silence:
            print(f"\n  {'video':<46}  {'seg1-pre (FP)':>18}  "
                  f"{'silence-safe (FP)':>22}  "
                  f"{'silence-attack (Det)':>22}  "
                  f"{'seg3-push (Det)':>22}")
            print(f"  {'-' * 46}  {'-' * 18}  {'-' * 22}  "
                  f"{'-' * 22}  {'-' * 22}")
        else:
            print(f"\n  {'video':<46}  {'seg1-pre (FP)':>18}  "
                  f"{'seg2-approach (FP)':>22}  {'seg3-push (Det)':>22}")
            print(f"  {'-' * 46}  {'-' * 18}  {'-' * 22}  {'-' * 22}")

        # Aggregates for summary line
        agg = defaultdict(lambda: [0, 0])  # segment → [total, flagged]

        for vid in sorted(per_vid.keys()):
            s1 = per_vid[vid].get("seg1-pre", [0, 0, 0])
            s2 = per_vid[vid].get("seg2-approach", [0, 0, 0])
            s3 = per_vid[vid].get("seg3-push", [0, 0, 0])
            sa = per_vid[vid].get("silence-attack", [0, 0, 0])
            ss = per_vid[vid].get("silence-safe", [0, 0, 0])
            sf = per_vid[vid].get("full", [0, 0, 0])

            agg["seg1-pre"][0]       += s1[0]; agg["seg1-pre"][1]       += s1[1]
            agg["seg2-approach"][0]  += s2[0]; agg["seg2-approach"][1]  += s2[1]
            agg["seg3-push"][0]      += s3[0]; agg["seg3-push"][1]      += s3[1]
            agg["silence-attack"][0] += sa[0]; agg["silence-attack"][1] += sa[1]
            agg["silence-safe"][0]   += ss[0]; agg["silence-safe"][1]   += ss[1]
            agg["full"][0]           += sf[0]; agg["full"][1]           += sf[1]

            # Safe videos go on their own row — use "full" column info in place
            # of seg3.  For attack videos, seg1/seg2/seg3 columns are populated.
            if sf[0] > 0:
                # safe video — print FP rate under a dedicated "full" column
                if has_silence:
                    print(f"  {vid:<46}  {'—':>18}  {'—':>22}  "
                          f"{'—':>22}  {_rate(sf[1], sf[0]):>22}  (safe)")
                else:
                    print(f"  {vid:<46}  {'—':>18}  {'—':>22}  "
                          f"{_rate(sf[1], sf[0]):>22}  (safe)")
            else:
                s1_str = _rate(s1[1], s1[0]) if s1[0] else "(no windows)"
                s2_str = _rate(s2[1], s2[0]) if s2[0] else "(no windows)"
                s3_str = _rate(s3[1], s3[0]) if s3[0] else "(no windows)"
                sa_str = _rate(sa[1], sa[0]) if sa[0] else "(no windows)"
                ss_str = _rate(ss[1], ss[0]) if ss[0] else "(no windows)"
                if has_silence:
                    print(f"  {vid:<46}  {s1_str:>18}  {ss_str:>22}  "
                          f"{sa_str:>22}  {s3_str:>22}")
                else:
                    print(f"  {vid:<46}  {s1_str:>18}  {s2_str:>22}  "
                          f"{s3_str:>22}")

        # Aggregate summary row
        if has_silence:
            print(f"  {'-' * 46}  {'-' * 18}  {'-' * 22}  "
                  f"{'-' * 22}  {'-' * 22}")
        else:
            print(f"  {'-' * 46}  {'-' * 18}  {'-' * 22}  {'-' * 22}")

        total_s1 = _rate(agg['seg1-pre'][1], agg['seg1-pre'][0]) \
                   if agg['seg1-pre'][0] else "—"
        total_s2 = _rate(agg['seg2-approach'][1], agg['seg2-approach'][0]) \
                   if agg['seg2-approach'][0] else "—"
        total_s3 = _rate(agg['seg3-push'][1], agg['seg3-push'][0]) \
                   if agg['seg3-push'][0] else "—"
        total_sa = _rate(agg['silence-attack'][1], agg['silence-attack'][0]) \
                   if agg['silence-attack'][0] else "—"
        total_ss = _rate(agg['silence-safe'][1], agg['silence-safe'][0]) \
                   if agg['silence-safe'][0] else "—"
        if has_silence:
            print(f"  {'TOTAL (all attack videos)':<46}  "
                  f"{total_s1:>18}  {total_ss:>22}  {total_sa:>22}  "
                  f"{total_s3:>22}")
        else:
            print(f"  {'TOTAL (all attack videos)':<46}  "
                  f"{total_s1:>18}  {total_s2:>22}  {total_s3:>22}")
        if agg['full'][0]:
            total_sf = _rate(agg['full'][1], agg['full'][0])
            if has_silence:
                print(f"  {'TOTAL (all safe videos)':<46}  "
                      f"{'—':>18}  {'—':>22}  {'—':>22}  "
                      f"{total_sf:>22}")
            else:
                print(f"  {'TOTAL (all safe videos)':<46}  "
                      f"{'—':>18}  {'—':>22}  {total_sf:>22}")

        # Call out the videos with the worst push-phase detection (lowest recall)
        push_recalls = []
        for vid, segs in per_vid.items():
            s3 = segs.get("seg3-push", [0, 0, 0])
            if s3[0] > 0:
                push_recalls.append((vid, s3[1] / s3[0], s3[1], s3[0]))
        push_recalls.sort(key=lambda x: x[1])

        if push_recalls:
            print(f"\n  Lowest push-phase detection rates (bottom 10):")
            for vid, rate, flagged, total in push_recalls[:10]:
                print(f"    {vid:<48}  {flagged:>3d}/{total:<3d}  "
                      f"({100.0 * rate:5.1f}%)")

    # ── PR Curve ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("outputs/plots", exist_ok=True)

        # Per-threshold metrics for markers
        marker_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        marker_metrics = {}
        for t in marker_thresholds:
            yp = (y_score >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y_true, yp, average='binary', zero_division=0)
            fp_c = int(((yp == 1) & (y_true == 0)).sum())
            marker_metrics[t] = (p, r, f, fp_c / n_neg)

        all_markers = list(marker_thresholds)
        if early_thresh not in all_markers:
            all_markers.append(early_thresh)
            all_markers.sort()
            yp = (y_score >= early_thresh).astype(int)
            p_s, r_s, f_s, _ = precision_recall_fscore_support(
                y_true, yp, average='binary', zero_division=0)
            fp_s = int(((yp == 1) & (y_true == 0)).sum())
            marker_metrics[early_thresh] = (p_s, r_s, f_s, fp_s / n_neg)

        # --- Precision-Recall curve ---
        prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(8, 6))

        for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
            r_iso = np.linspace(0.01, 1.0, 200)
            p_iso = (f1_val * r_iso) / (2 * r_iso - f1_val)
            valid = (p_iso > 0) & (p_iso <= 1)
            ax.plot(r_iso[valid], p_iso[valid], '--', color='gray',
                    alpha=0.35, linewidth=0.8)
            label_idx = np.where(valid)[0]
            if len(label_idx) > 0:
                li = label_idx[-1]
                ax.annotate(f"F1={f1_val}", (r_iso[li], p_iso[li]),
                            fontsize=7, color='gray', alpha=0.7,
                            ha='left', va='bottom')

        ax.plot(rec_arr, prec_arr, linewidth=2, color='#1f77b4',
                label=f"PR curve (AP={ap:.3f})")

        table_lines = []
        marker_items = []
        for t in all_markers:
            p_m, r_m, f_m, fpr_m = marker_metrics[t]
            if r_m == 0 and p_m == 0:
                continue
            is_selected = abs(t - early_thresh) < 0.005
            color = 'red' if is_selected else '#1f77b4'
            size = 10 if is_selected else 6
            zorder = 10 if is_selected else 5
            ax.plot(r_m, p_m, 'o', color=color, markersize=size,
                    zorder=zorder)
            marker_items.append((t, r_m, p_m, f_m, fpr_m, is_selected, color))

        # Fan-out labels if clustered
        angles = []
        n_items = len(marker_items)
        if n_items > 0:
            r_vals = [mi[1] for mi in marker_items]
            p_vals = [mi[2] for mi in marker_items]
            clustered = (max(r_vals) - min(r_vals) < 0.05
                         and max(p_vals) - min(p_vals) < 0.05)
            if clustered and n_items > 1:
                start_angle = 200
                sweep = min(200, 30 * n_items)
                for i in range(n_items):
                    angle_deg = start_angle - (sweep * i / max(1, n_items - 1))
                    angle_rad = math.radians(angle_deg)
                    dist = 18
                    dx = dist * math.cos(angle_rad)
                    dy = dist * math.sin(angle_rad)
                    angles.append((dx, dy))
            else:
                angles = [(6, 4)] * n_items

        for idx, (t, r_m, p_m, f_m, fpr_m, is_selected, color) in enumerate(marker_items):
            dx, dy = angles[idx]
            ax.annotate(f"{t:.2f}", (r_m, p_m),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=5.5, color=color,
                        fontweight='bold' if is_selected else 'normal',
                        arrowprops=dict(arrowstyle='-', color=color,
                                        lw=0.5, alpha=0.4)
                        if (abs(dx) > 10 or abs(dy) > 10) else None)

            tag = "★" if is_selected else " "
            table_lines.append(
                f"{tag} t={t:.2f}  P={p_m:.2f}  R={r_m:.2f}  "
                f"F1={f_m:.2f}  FPR={fpr_m:>6.2%}")

        table_text = "\n".join(table_lines)
        txt = ax.text(0.02, 0.02, table_text, transform=ax.transAxes,
                      fontsize=7, fontfamily='monospace', verticalalignment='bottom',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='#cccccc', alpha=0.9))

        fig.canvas.draw()
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        bb_axes = bb.transformed(ax.transAxes.inverted())
        ax.legend(loc="lower left", fontsize=8,
                  bbox_to_anchor=(0.01, bb_axes.y1 + 0.01))

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        _fname_suffix = "_split_silence" if split_silence else ""
        ax.set_title(f"Precision-Recall Curve (Cross-Domain Test Set) - {model_type.upper()}", fontsize=12)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        pr_path = f"outputs/plots/pr_curve_inference{_fname_suffix}_{ts}.png"
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  PR curve saved to  : {pr_path}  (AP={ap:.3f})")

        # --- ROC curve ---
        fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr_arr, tpr_arr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_arr, tpr_arr, linewidth=2, color='#1f77b4',
                label=f"ROC curve (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

        roc_table_lines = []
        roc_marker_items = []
        for t in all_markers:
            p_m, r_m, f_m, fpr_m = marker_metrics[t]
            best_dist = float('inf')
            tpr_m = r_m
            fpr_plot = fpr_m
            for rt, rfp, rtp in zip(roc_thresholds, fpr_arr, tpr_arr):
                d = abs(rt - t)
                if d < best_dist:
                    best_dist = d
                    fpr_plot, tpr_m = rfp, rtp
            is_selected = abs(t - early_thresh) < 0.005
            color = 'red' if is_selected else '#1f77b4'
            size = 6 if is_selected else 3
            zorder = 10 if is_selected else 5
            ax.plot(fpr_plot, tpr_m, 'o', color=color, markersize=size,
                    zorder=zorder)
            roc_marker_items.append((t, fpr_plot, tpr_m, is_selected, color,
                                     r_m, f_m, fpr_m))

        roc_angles = []
        n_roc = len(roc_marker_items)
        if n_roc > 0:
            fpr_vals = [mi[1] for mi in roc_marker_items]
            tpr_vals = [mi[2] for mi in roc_marker_items]
            clustered = (max(fpr_vals) - min(fpr_vals) < 0.05
                         and max(tpr_vals) - min(tpr_vals) < 0.05)
            if clustered and n_roc > 1:
                start_angle = -30
                sweep = min(200, 30 * n_roc)
                for i in range(n_roc):
                    angle_deg = start_angle - (sweep * i / max(1, n_roc - 1))
                    angle_rad = math.radians(angle_deg)
                    dist = 18
                    dx = dist * math.cos(angle_rad)
                    dy = dist * math.sin(angle_rad)
                    roc_angles.append((dx, dy))
            else:
                roc_angles = [(10, 0)] * n_roc

        for idx, (t, fpr_plot, tpr_m, is_selected, color, r_m, f_m, fpr_m) in enumerate(roc_marker_items):
            dx, dy = roc_angles[idx]
            ax.annotate(f"{t:.2f}", (fpr_plot, tpr_m),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=5.5, color=color,
                        fontweight='bold' if is_selected else 'normal',
                        va='center',
                        arrowprops=dict(arrowstyle='-', color=color,
                                        lw=0.5, alpha=0.4)
                        if (abs(dx) > 10 or abs(dy) > 10) else None)

            tag = "★" if is_selected else " "
            roc_table_lines.append(
                f"{tag} t={t:.2f}  TPR={r_m:.2f}  FPR={fpr_m:>6.2%}  "
                f"F1={f_m:.2f}")

        roc_table_text = "\n".join(roc_table_lines)
        txt2 = ax.text(0.98, 0.02, roc_table_text, transform=ax.transAxes,
                        fontsize=7, fontfamily='monospace', verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='#cccccc', alpha=0.9))

        fig.canvas.draw()
        bb2 = txt2.get_window_extent(renderer=fig.canvas.get_renderer())
        bb2_axes = bb2.transformed(ax.transAxes.inverted())
        ax.legend(loc="lower right", fontsize=8,
                  bbox_to_anchor=(0.99, bb2_axes.y1 + 0.01))

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax.set_title(f"ROC Curve (Cross-Domain Test Set) - {model_type.upper()}", fontsize=12)
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        roc_path = f"outputs/plots/roc_curve_inference{_fname_suffix}_{ts}.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ROC curve saved to : {roc_path}  (AUC={roc_auc:.3f})")

        # Threshold comparison table
        print(f"\n  Threshold comparison (inference, n_neg={n_neg}):")
        print(f"  {'thresh':>6s}   {'Prec':>5s}   {'Rec':>5s}   {'F1':>5s}   {'FP%':>6s}  note")
        print(f"  {'-'*46}")
        for t in all_markers:
            p_m, r_m, f_m, fpr_m = marker_metrics[t]
            note = " ★ selected" if abs(t - early_thresh) < 0.005 else ""
            print(f"  {t:>6.2f}  {p_m:.3f}  {r_m:.3f}  {f_m:.3f}  {fpr_m:>6.2%}{note}")

    except ImportError as e:
        print(f"\n  Skipping PR/ROC curves: {e}")

    # ── Feature Importance ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPUTING FEATURE IMPORTANCE (Inference, no EMA)")
    print("=" * 80)

    from torch.utils.data import TensorDataset, DataLoader

    eval_dataset = TensorDataset(
        torch.from_numpy(holdout_input).float(),
        torch.from_numpy(yw_all).float())
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    importances, baseline_f1 = compute_permutation_importance(
        model=model,
        val_loader=eval_loader,
        threshold=early_thresh,
        device=device,
        n_repeats=5,
    )

    print_importance_ranking(importances, baseline_f1)

    _fi_suffix = "_split_silence" if split_silence else ""
    importance_output = f"outputs/logs/feature_importance_inference{_fi_suffix}_{ts}.json"
    save_importance_results(importances, baseline_f1, importance_output)


def evaluate_dir(eval_dir, verbose=False, debug=False, enable_pi=False,
                 simulate_live=False, analyze=False,
                 split_silence=False):
    """
    Evaluate the FULL live detection pipeline on a pre-recorded dataset.

    Mirrors the summary format of evaluate.py but uses run_on_file() (which
    replicates the hazard_ema + persist + gate state machine of run()) instead
    of evaluate.py's simpler raw-hazard scoring.  Results therefore match
    real-world deployment more accurately than evaluate.py.

    Dataset layout:
        eval_dir/
            labels.json          — attack video labels
            attack/              — attack .mp4 files
            safe/                — (optional) safe .mp4 files

    labels.json schema (per entry):
        {
            "video.mp4": {
                "category":            "attack",
                "first_stand_after_sit": <int>,  // person stands from chair
                "last_backward_frame":   <int>,  // last frame of backward motion
                "push_start_frame":      <int>,  // push/attack begins
                "contact_frame":        <int>   // push/attack ends (contact)
            }, ...
        }

    Detection zone classification (default):
        before first_stand_after_sit                         → False Positive
        between first_stand_after_sit and last_backward_frame → Silenced
        between last_backward_frame and push_start_frame     → False Positive
        between push_start_frame and contact_frame          → Early Detection
            lead_time = contact_frame − detection_frame
        after contact_frame                                 → Late Detection

    Detection zone classification (--split-silence):
        before first_stand_after_sit                         → False Positive
        between first_stand_after_sit and last_backward_frame → False Positive
        between last_backward_frame and push_start_frame     → Early Detection
        between push_start_frame and contact_frame          → Early Detection
        after contact_frame                                 → Late Detection

    Args:
        eval_dir: directory containing attack/, labels.json, optionally safe/.
        verbose: dump compact frame-by-frame timeline table per video.
        debug: print full per-frame gate diagnostics (mirrors run() console
            output); captured by TeeLogger → log file.
        enable_pi: use Config.for_pi() instead of Config.for_pc().
    """
    # ── logging ───────────────────────────────────────────────────────────────
    os.makedirs("outputs/logs", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("outputs/logs", f"inference_{ts}.log")
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
        model_type   = meta.get("model_type", "gru")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "lstm":
        model = HazardLSTM(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    elif model_type == "transformer":
        model = HazardTransformer(input_dim=input_dim, d_model=cfg.gru_hidden, dropout=0.0)
    else:
        model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(
        torch.load(f"outputs/checkpoints/{model_file}", map_location=device))
    model.eval()
    model.to(device)

    detector = PoseDetector(cfg)

    labels_file = os.path.join(eval_dir, "labels.json")
    with open(labels_file) as f:
        labels = json.load(f)

    mode_label = "Window-Level Analysis" if analyze else "Live Detection"
    print("=" * 80)
    print(f"INFERENCE ({mode_label})")
    print(f"Directory  : {eval_dir}")
    print(f"Threshold  : {early_thresh:.2f}  (from training optimisation)")
    print(f"Config     : {'Pi' if enable_pi else 'PC'}")
    print(f"{model_type.upper()} device : {device}")
    print(f"Live sim   : {'ON  (wall-clock dt, native-FPS pacing per video)' if simulate_live else 'OFF (fixed dt=step_dt, batch speed)'}")
    print(f"Ignore     : frames before 'first_stand' per video (fallback: 12.0s)")
    print("=" * 80)

    attack_results = []
    safe_results   = []
    attack_dir_path = os.path.join(eval_dir, "attack")

    safe_dir_path = os.path.join(eval_dir, "safe")

    if analyze:
        _run_window_analysis(eval_dir, cfg, model, device, detector,
                             early_thresh, labels, attack_dir_path,
                             safe_dir_path, ts, verbose=verbose,
                             split_silence=split_silence,
                             model_type=model_type)
        print(f"\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log saved to : {log_path}")
        detector.release()
        return

    # ── Live detection mode (default) ──────────────────────────────────────

    # ── attack videos ─────────────────────────────────────────────────────────
    print("\n--- Processing Attack Videos ---")
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

            # Use labelled first_stand as the skip region; fall back to 12s.
            _fs_frame = gt.get("first_stand", 0)
            result = run_on_file(video_path, cfg, model, device, detector,
                                 early_thresh, verbose=verbose, debug=debug,
                                 simulate_live=simulate_live,
                                 ignore_start_frame=_fs_frame,
                                 model_type=model_type)

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

    # ── safe videos (optional) ────────────────────────────────────────────────
    if not os.path.exists(safe_dir_path):
        print("\n(no safe/ directory found — skipping safe video evaluation)")
    else:
        for video_file in sorted(os.listdir(safe_dir_path)):
            if not video_file.lower().endswith(".mp4"):
                continue

            video_path = os.path.join(safe_dir_path, video_file)

            print(f"Processing {video_file}...", end=" ", flush=True)
            if debug:
                print(f"\n  [debug] {video_file}")

            # Safe videos have no first_stand label; fall back to 12s.
            result = run_on_file(video_path, cfg, model, device, detector,
                                 early_thresh, verbose=verbose, debug=debug,
                                 simulate_live=simulate_live,
                                 ignore_start_sec=12.0,
                                 model_type=model_type)

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

    # ── attack summary (zone-based classification) ──────────────────────────
    print("\n" + "=" * 80)
    print("ATTACK VIDEOS — Zone-Based Detection Analysis")
    print("=" * 80)
    if split_silence:
        print("Zones (--split-silence):")
        print("  FP       = [first_stand, first_stand_after_sit)")
        print("           OR [first_stand_after_sit, last_backward_frame]  (silence zone → safe)")
        print("  EARLY    = (last_backward_frame, contact_frame]  (seg2-approach + push → attack)")
        print("  LATE     = after contact_frame")
    else:
        print("Zones:  FP = before first_stand_after_sit OR between last_backward_frame")
        print("             and push_start_frame")
        print("        SILENCED = between first_stand_after_sit and last_backward_frame")
        print("        EARLY    = between push_start_frame and contact_frame")
        print("        LATE     = after contact_frame")
    print("-" * 80)

    early_detections   = []   # detected between push_start and push_end
    late_detections    = []   # detected after push_end
    silenced_only      = []   # all detections fell in the silenced zone
    false_positives_attacks = []   # first non-silenced detection is FP
    missed_attacks     = []   # no detection at all
    lead_times         = []   # contact_frame − detection_frame (early only)
    late_delays        = []   # detection_frame − contact_frame (late only)
    videos_with_silenced = []  # videos that had any silenced detections

    for r in attack_results:
        gt                   = r["ground_truth"]
        first_stand_after_sit = gt["first_stand_after_sit"]
        last_backward_frame  = gt["last_backward_frame"]
        push_start_frame     = gt["push_start_frame"]
        contact_frame       = gt["contact_frame"]
        all_threats          = r.get("all_threat_frames", [])

        if not all_threats:
            # No detection at all
            missed_attacks.append(r)
            print(f"{r['video_name']:50s} | MISSED "
                  f"(max_ema={r['max_hazard_ema']:.3f})")
            continue

        if split_silence:
            # --split-silence: 1st silence zone → FP, 2nd silence zone → EARLY.
            # No silenced zone to skip; take the very first threat frame.
            first_actionable = all_threats[0] if all_threats else -1
            sil_suffix = ""

            det = first_actionable
            if det < 0:
                missed_attacks.append(r)
                print(f"{r['video_name']:50s} | MISSED "
                      f"(max_ema={r['max_hazard_ema']:.3f})")
                continue

            if det < first_stand_after_sit:
                # seg1-pre → FP
                false_positives_attacks.append(r)
                print(f"{r['video_name']:50s} | FP @ {det:5d} "
                      f"(seg1-pre, before stand@{first_stand_after_sit})  "
                      f"max_ema={r['max_hazard_ema']:.3f}")

            elif det <= last_backward_frame:
                # 1st silence zone → safe → FP
                false_positives_attacks.append(r)
                print(f"{r['video_name']:50s} | FP @ {det:5d} "
                      f"(silence-safe [{first_stand_after_sit}–{last_backward_frame}])  "
                      f"max_ema={r['max_hazard_ema']:.3f}")

            elif det <= contact_frame:
                # 2nd silence zone (seg2-approach) or push phase → EARLY
                lead_time = contact_frame - det
                lead_times.append(lead_time)
                early_detections.append(r)
                zone = ("seg2-approach" if det <= push_start_frame
                        else "push")
                print(f"{r['video_name']:50s} | EARLY  "
                      f"Detect@{det:5d}  [{zone}]  "
                      f"push_begins@{push_start_frame} contact@{contact_frame}  "
                      f"Lead={lead_time:+5d} frames "
                      f"({lead_time/cfg.input_fps:.2f}s)  "
                      f"max_ema={r['max_hazard_ema']:.3f}")

            else:
                # After contact_frame → LATE
                late_detections.append(r)
                delay = det - contact_frame
                late_delays.append(delay)
                print(f"{r['video_name']:50s} | LATE   "
                      f"Detect@{det:5d}  "
                      f"push_begins@{push_start_frame} contact@{contact_frame}  "
                      f"Delay={delay:+5d} frames "
                      f"({delay/cfg.input_fps:.2f}s)  "
                      f"max_ema={r['max_hazard_ema']:.3f}")

        else:
            # Default: 1st silence zone is silenced; 2nd silence zone is FP.
            # Count silenced detections and find first actionable detection.
            silenced_frames = [tf for tf in all_threats
                               if first_stand_after_sit <= tf <= last_backward_frame]
            n_silenced = len(silenced_frames)
            sil_suffix = ""
            if n_silenced > 0:
                sil_tags = " ".join(f"Silenced@{sf}" for sf in silenced_frames[:5])
                if n_silenced > 5:
                    sil_tags += f" ...+{n_silenced - 5} more"
                sil_suffix = f"  [{sil_tags}]"
                videos_with_silenced.append((r['video_name'], n_silenced,
                                             silenced_frames))

            first_actionable = -1
            for tf in all_threats:
                if tf < first_stand_after_sit:
                    first_actionable = tf
                    break
                elif tf <= last_backward_frame:
                    continue   # silenced zone — skip
                else:
                    first_actionable = tf
                    break

            if first_actionable < 0:
                silenced_only.append(r)
                print(f"{r['video_name']:50s} | SILENCED "
                      f"(all {n_silenced} detections in silenced zone "
                      f"[{first_stand_after_sit}–{last_backward_frame}])  "
                      f"max_ema={r['max_hazard_ema']:.3f}"
                      f"{sil_suffix}")
                continue

            det = first_actionable

            if det < first_stand_after_sit:
                false_positives_attacks.append(r)
                print(f"{r['video_name']:50s} | FP @ {det:5d} "
                      f"(before stand@{first_stand_after_sit})  "
                      f"max_ema={r['max_hazard_ema']:.3f}"
                      f"{sil_suffix}")

            elif det <= push_start_frame:
                false_positives_attacks.append(r)
                print(f"{r['video_name']:50s} | FP @ {det:5d} "
                      f"(between backward@{last_backward_frame} and "
                      f"push_start@{push_start_frame})  "
                      f"max_ema={r['max_hazard_ema']:.3f}"
                      f"{sil_suffix}")

            elif det <= contact_frame:
                lead_time = contact_frame - det
                lead_times.append(lead_time)
                early_detections.append(r)
                print(f"{r['video_name']:50s} | EARLY  "
                      f"Detect@{det:5d}  "
                      f"push_begins@{push_start_frame} contact@{contact_frame}  "
                      f"Lead={lead_time:+5d} frames "
                      f"({lead_time/cfg.input_fps:.2f}s)  "
                      f"max_ema={r['max_hazard_ema']:.3f}"
                      f"{sil_suffix}")

            else:
                late_detections.append(r)
                delay = det - contact_frame
                late_delays.append(delay)
                print(f"{r['video_name']:50s} | LATE   "
                      f"Detect@{det:5d}  "
                      f"push_begins@{push_start_frame} contact@{contact_frame}  "
                      f"Delay={delay:+5d} frames "
                      f"({delay/cfg.input_fps:.2f}s)  "
                      f"max_ema={r['max_hazard_ema']:.3f}"
                      f"{sil_suffix}")

    # ── attack statistics ──────────────────────────────────────────────────
    n_total = len(attack_results)
    n_early = len(early_detections)
    n_late  = len(late_detections)
    n_fp    = len(false_positives_attacks)
    n_sil   = len(silenced_only)
    n_miss  = len(missed_attacks)

    print("\n" + "-" * 80)
    print(f"Early Detections : {n_early:3d}/{n_total}  ({n_early/max(1,n_total):.1%})")
    print(f"Late  Detections : {n_late:3d}/{n_total}  ({n_late/max(1,n_total):.1%})")
    print(f"False Positives  : {n_fp:3d}/{n_total}  ({n_fp/max(1,n_total):.1%})")
    if not split_silence:
        print(f"Silenced (only)  : {n_sil:3d}/{n_total}  ({n_sil/max(1,n_total):.1%})")
    print(f"Missed           : {n_miss:3d}/{n_total}  ({n_miss/max(1,n_total):.1%})")

    # Silenced detections summary (per-video) — only in default mode
    if not split_silence:
        n_vids_with_sil = len(videos_with_silenced)
        total_sil_frames = sum(cnt for _, cnt, _ in videos_with_silenced)
        print(f"\n--- Silenced Zone Detections ---")
        print(f"  Videos with silenced detections: {n_vids_with_sil}/{n_total}")
        print(f"  Total silenced frames          : {total_sil_frames}")
        if n_vids_with_sil > 0:
            sil_counts = [cnt for _, cnt, _ in videos_with_silenced]
            print(f"  Per-video: mean={np.mean(sil_counts):.1f}  "
                  f"median={np.median(sil_counts):.0f}  "
                  f"min={np.min(sil_counts)}  max={np.max(sil_counts)}")

    if lead_times:
        print(f"\n--- Early Detection Lead Times (threshold={early_thresh:.2f}) ---")
        print(f"  Mean Lead Time   : {np.mean(lead_times):.1f} frames "
              f"({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Median Lead Time : {np.median(lead_times):.1f} frames "
              f"({np.median(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Min  Lead Time   : {np.min(lead_times):.0f} frames "
              f"({np.min(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Max  Lead Time   : {np.max(lead_times):.0f} frames "
              f"({np.max(lead_times)/cfg.input_fps:.2f}s)")

    # Overall detection timing (early + late), relative to contact_frame
    # Positive = before contact (early), negative = after contact (late)
    if lead_times or late_delays:
        all_offsets = [t for t in lead_times] + [-d for d in late_delays]
        print(f"\n--- Overall Detection Timing (early + late, n={len(all_offsets)}) ---")
        print(f"  Mean  offset     : {np.mean(all_offsets):+.1f} frames "
              f"({np.mean(all_offsets)/cfg.input_fps:+.2f}s)")
        print(f"  Median offset    : {np.median(all_offsets):+.1f} frames "
              f"({np.median(all_offsets)/cfg.input_fps:+.2f}s)")
        print(f"  Best  (earliest) : {np.max(all_offsets):+.0f} frames "
              f"({np.max(all_offsets)/cfg.input_fps:+.2f}s)")
        print(f"  Worst (latest)   : {np.min(all_offsets):+.0f} frames "
              f"({np.min(all_offsets)/cfg.input_fps:+.2f}s)")
        print(f"  (positive = before contact, negative = after contact)")

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
            print(f"{r['video_name']:50s} | FP @ frame {first_det}  "
                  f"(max_ema={r['max_hazard_ema']:.3f})")
        else:
            true_negatives.append(r)

    fp_rate = len(false_positives) / max(1, len(safe_results))
    tn_rate = len(true_negatives)  / max(1, len(safe_results))

    if safe_results:
        print(f"\nFalse Positive Rate (safe) : {fp_rate:.1%} "
              f"({len(false_positives)}/{len(safe_results)})")
        print(f"True Negative Rate         : {tn_rate:.1%} "
              f"({len(true_negatives)}/{len(safe_results)})")
    else:
        print("(no safe videos found)")

    # ── overall summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total Videos : {n_total + len(safe_results)}  "
          f"(attacks={n_total}  safe={len(safe_results)})")
    print(f"\nThreshold (THREAT) : {early_thresh:.2f}")
    print(f"\nAttack Videos:")
    print(f"  Early Detection  : {n_early/max(1,n_total):.1%}  ({n_early}/{n_total})")
    print(f"  Late  Detection  : {n_late/max(1,n_total):.1%}  ({n_late}/{n_total})")
    print(f"  False Positive   : {n_fp/max(1,n_total):.1%}  ({n_fp}/{n_total})")
    if not split_silence:
        print(f"  Silenced (only)  : {n_sil/max(1,n_total):.1%}  ({n_sil}/{n_total})")
        print(f"  Silenced (any)   : {n_vids_with_sil/max(1,n_total):.1%}  ({n_vids_with_sil}/{n_total})")
    print(f"  Missed           : {n_miss/max(1,n_total):.1%}  ({n_miss}/{n_total})")
    if lead_times:
        print(f"\nEarly Detection Lead Time:")
        print(f"  Mean  : {np.mean(lead_times):.1f} frames "
              f"({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Median: {np.median(lead_times):.1f} frames "
              f"({np.median(lead_times)/cfg.input_fps:.2f}s)")
    if safe_results:
        print(f"\nSafe FP Rate      : {fp_rate:.1%}")
    print("=" * 80)

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        help="With --eval-dir (default mode): dump the compact frame-by-frame "
             "timeline table (level, hazard_raw, hazard_ema, gate, thf, "
             "log_scale, persist) for every video processed.  "
             "With --eval-dir --analyze: list every false-positive and "
             "false-negative window with its source video, segment, and "
             "frame range — useful for locating failure modes in the footage.")

    parser.add_argument(
        "--debug", action="store_true",
        help="With --eval-dir: print full per-frame diagnostics for every "
             "video (mirrors the live infer.py console output — raw, ema, level, "
             "thf, ls, ok, persist, dbg). "
             "Also flags det=MISS frames where pose detection yields no bbox. "
             "Output is captured by the TeeLogger → inference_<timestamp>.log.")

    parser.add_argument(
        "--simulate-live", action="store_true",
        help="Pace frame delivery to each video's native FPS and use actual "
             "wall-clock dt for feature derivatives, making pre-recorded videos "
             "behave identically to a live camera feed.  Works for both single-video "
             "inference and --eval-dir batch evaluation — use on the Pi5 to obtain "
             "a detection-rate / FP-rate report that reflects real-time performance "
             "without needing a live camera.  Has no effect on camera (int) sources.")

    parser.add_argument(
        "--analyze", action="store_true",
        help="With --eval-dir: run window-level analysis (F1, PR curve, "
             "ROC curve, feature importance) instead of Live Detection.  "
             "This mode re-extracts windows from the dataset and evaluates "
             "raw model scores (no EMA), producing metrics and plots for the paper.")

    parser.add_argument(
        "--split-silence", action="store_true",
        help="With --eval-dir --analyze: re-include the two excluded "
             "zones.  The silence zone [first_stand_after_sit, "
             "last_backward_frame] is labelled safe ('silence-safe'); "
             "seg2-approach (last_backward_frame, push_start_frame] is "
             "labelled attack ('silence-attack').  Tests whether the "
             "approach phase carries genuine pre-attack signal.")

    args = parser.parse_args()

    # ── eval-dir mode: evaluate a pre-recorded dataset ────────────────────────
    if args.eval_dir is not None:
        # evaluate_dir() sets up its own TeeLogger internally
        evaluate_dir(args.eval_dir,
                     verbose=args.verbose,
                     debug=args.debug,
                     enable_pi=args.pi,
                     simulate_live=args.simulate_live,
                     analyze=args.analyze,
                     split_silence=args.split_silence)
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
