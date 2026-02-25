import numpy as np
import cv2
from .config import Config

# ── Hailo host-side post-processing helpers ───────────────────────────────────

_REG_MAX = 16                                          # YOLOv8 DFL bins
_PROJ    = np.arange(_REG_MAX, dtype=np.float32)      # [0, 1, ..., 15]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _dfl_decode(box_raw: np.ndarray) -> np.ndarray:
    """
    Decode DFL (Distribution Focal Loss) box regression.
    box_raw : (N, 4 * REG_MAX)  raw logits from the network
    Returns : (N, 4)            ltrb distances in grid-cell units
    """
    n = box_raw.shape[0]
    b = box_raw.reshape(n, 4, _REG_MAX).astype(np.float32)
    b -= b.max(axis=-1, keepdims=True)          # numerical stability
    e  = np.exp(b)
    s  = e / e.sum(axis=-1, keepdims=True)      # softmax over REG_MAX bins
    return (s * _PROJ).sum(axis=-1)             # weighted sum → (N, 4)


def _decode_scale(box_raw, conf_raw, kps_raw, stride, conf_thresh):
    """
    Decode one output scale into candidate detections.

    box_raw  : (H, W, 64)   DFL box regression logits
    conf_raw : (H, W,  1)   person class logits
    kps_raw  : (H, W, 51)   raw keypoint predictions (x, y, vis) × 17
    stride   : int          feature-map stride (8, 16, or 32)

    Returns (boxes, confs, kps) for anchors above conf_thresh,
    all coordinates in MODEL INPUT pixel space (e.g. 640×640).

    NOTE: sigmoid is applied here to conf and keypoint visibility.
    If the HEF already embeds sigmoid (depends on ONNX export options)
    detections may be over-confident; remove _sigmoid() calls in that case.
    """
    H, W = box_raw.shape[:2]
    N    = H * W

    conf_flat = _sigmoid(conf_raw.reshape(N).astype(np.float32))
    mask = conf_flat > conf_thresh
    if not mask.any():
        return (np.empty((0, 4),     np.float32),
                np.empty((0,),       np.float32),
                np.empty((0, 17, 3), np.float32))

    idx    = np.where(mask)[0]
    conf_f = conf_flat[idx]

    # Anchor centres in grid-cell units
    gy = (idx // W).astype(np.float32)
    gx = (idx  % W).astype(np.float32)
    ax = gx + 0.5
    ay = gy + 0.5

    # DFL decode → ltrb (grid-cell units) → pixel xyxy
    ltrb = _dfl_decode(box_raw.reshape(N, 64).astype(np.float32)[idx])
    x1 = (ax - ltrb[:, 0]) * stride
    y1 = (ay - ltrb[:, 1]) * stride
    x2 = (ax + ltrb[:, 2]) * stride
    y2 = (ay + ltrb[:, 3]) * stride
    boxes = np.stack([x1, y1, x2, y2], axis=1)   # (m, 4)

    # Keypoint decode — Ultralytics formula:
    #   kx = (kx_raw * 2.0 + (anchor_x - 0.5)) * stride
    #      = (kx_raw * 2.0 + gx) * stride   [since ax - 0.5 == gx]
    #   vis = sigmoid(vis_raw)
    kps_f = kps_raw.reshape(N, 17, 3).astype(np.float32)[idx]
    kp_x  = (kps_f[:, :, 0] * 2.0 + gx[:, None]) * stride
    kp_y  = (kps_f[:, :, 1] * 2.0 + gy[:, None]) * stride
    kp_v  = _sigmoid(kps_f[:, :, 2])
    kps_out = np.stack([kp_x, kp_y, kp_v], axis=2)  # (m, 17, 3)

    return boxes, conf_f, kps_out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """Standard greedy NMS. Returns list of kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def _postprocess(outputs, conf_thresh, iou_thresh, input_wh, orig_shape):
    """
    Full host-side post-process of the 9 raw Hailo output tensors.

    outputs   : dict  {stream_name: np.ndarray shape (1, H, W, C)}
    input_wh  : (W, H)  model input size, e.g. (640, 640)
    orig_shape: (H, W, C)  original BGR frame shape

    Stream grouping is detected automatically from channel count:
      C == 64  → DFL box regression
      C ==  1  → person confidence
      C == 51  → keypoints (17 × 3)

    Returns dict {bbox, det_conf, kps} in ORIGINAL frame coordinates,
    or None if no person detected.
    """
    box_map = {}; conf_map = {}; kps_map = {}

    for name, arr in outputs.items():
        a = np.array(arr)[0]        # drop batch dim → (H, W, C)
        H, W, C = a.shape
        stride = input_wh[0] // W   # 640//80=8, 640//40=16, 640//20=32
        if   C == 64: box_map[stride]  = a
        elif C ==  1: conf_map[stride] = a
        elif C == 51: kps_map[stride]  = a

    if not box_map:
        return None

    all_boxes, all_confs, all_kps = [], [], []
    for stride in sorted(box_map):
        if stride not in conf_map or stride not in kps_map:
            continue
        b, c, k = _decode_scale(
            box_map[stride], conf_map[stride], kps_map[stride],
            stride, conf_thresh)
        if len(b):
            all_boxes.append(b); all_confs.append(c); all_kps.append(k)

    if not all_boxes:
        return None

    boxes = np.concatenate(all_boxes)
    confs = np.concatenate(all_confs)
    kps   = np.concatenate(all_kps)

    boxes = np.clip(boxes, 0, max(input_wh))
    keep  = _nms(boxes, confs, iou_thresh)
    if not keep:
        return None

    kb = boxes[keep]; kc = confs[keep]; kk = kps[keep]
    areas = (kb[:, 2] - kb[:, 0]) * (kb[:, 3] - kb[:, 1])
    best  = areas.argmax()

    bbox     = kb[best].tolist()
    det_conf = float(kc[best])
    kps17    = kk[best].astype(np.float32)   # (17, 3)

    # Rescale from model input space → original frame space
    iw, ih = input_wh
    oh, ow = orig_shape[:2]
    if iw != ow or ih != oh:
        sx, sy = ow / iw, oh / ih
        bbox[0] *= sx; bbox[2] *= sx
        bbox[1] *= sy; bbox[3] *= sy
        kps17[:, 0] *= sx
        kps17[:, 1] *= sy

    return {"bbox": bbox, "det_conf": det_conf, "kps": kps17}


# ── PoseDetector ──────────────────────────────────────────────────────────────

class PoseDetector:
    """
    Unified pose detector supporting two backends:
      "ultralytics" — YOLOv8 .pt via Ultralytics (PC / development)
      "hailo"       — YOLOv8 .hef via HailoRT NPU (Raspberry Pi 5 + AI HAT+)

    Backend is selected by cfg.pose_backend.
    Both backends return the same dict format:
      {"bbox": [x1,y1,x2,y2], "det_conf": float, "kps": np.ndarray (17,3)}
    """

    def __init__(self, cfg: Config = None):
        if cfg is None:
            cfg = Config()
        self.cfg = cfg
        if cfg.pose_backend == "hailo":
            self._init_hailo()
        else:
            self._init_ultralytics()

    # ── Ultralytics backend ───────────────────────────────────────────────────

    def _init_ultralytics(self):
        from ultralytics import YOLO
        self._model   = YOLO(self.cfg.yolo_pt_path)
        self._backend = "ultralytics"

    def _infer_ultralytics(self, frame: np.ndarray):
        r = self._model(frame,
                        conf=self.cfg.yolo_conf,
                        iou=self.cfg.yolo_iou,
                        verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        boxes   = r.boxes.xyxy.cpu().numpy()
        scores  = r.boxes.conf.cpu().numpy()
        kps     = r.keypoints.xy.cpu().numpy()
        kp_conf = r.keypoints.conf.cpu().numpy()
        i       = ((boxes[:, 2] - boxes[:, 0]) *
                   (boxes[:, 3] - boxes[:, 1])).argmax()
        kps17 = np.concatenate([kps[i], kp_conf[i, :, None]], axis=1)
        return {"bbox": boxes[i].tolist(),
                "det_conf": float(scores[i]),
                "kps": kps17}

    # ── Hailo backend ─────────────────────────────────────────────────────────

    def _init_hailo(self):
        from hailo_platform import (VDevice, HEF, ConfigureParams,
                                    InputVStreamParams, OutputVStreamParams,
                                    InferVStreams, HailoStreamInterface)

        target = VDevice()
        hef    = HEF(self.cfg.yolo_hef_path)

        # PCIe interface for AI HAT+; fall back to default if API differs
        try:
            params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe)
        except TypeError:
            params = ConfigureParams.create_from_hef(hef)

        ng  = target.configure(hef, params)[0]
        ngp = ng.create_params()

        inp       = hef.get_input_vstream_infos()[0]
        iv_params = InputVStreamParams.make(ng)
        ov_params = OutputVStreamParams.make(ng)

        # Keep inference context alive across frames to avoid per-call overhead
        self._pipeline  = InferVStreams(ng, iv_params, ov_params)
        self._activated = ng.activate(ngp)
        self._pipeline.__enter__()
        self._activated.__enter__()

        self._input_name = inp.name
        self._input_wh   = (inp.shape[1], inp.shape[0])  # (W, H) e.g. (640,640)
        self._target     = target
        self._backend    = "hailo"

    def _infer_hailo(self, frame: np.ndarray):
        iw, ih     = self._input_wh
        orig_shape = frame.shape

        img = cv2.resize(frame, (iw, ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(img, axis=0)          # (1, H, W, 3) uint8

        outputs = self._pipeline.infer({self._input_name: inp})
        return _postprocess(outputs,
                            self.cfg.yolo_conf,
                            self.cfg.yolo_iou,
                            self._input_wh,
                            orig_shape)

    # ── public API ────────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray):
        """
        Run pose detection on a single BGR frame.
        Returns dict {bbox, det_conf, kps} or None if no person detected.
        Output format is identical for both backends.
        """
        if self._backend == "hailo":
            return self._infer_hailo(frame)
        return self._infer_ultralytics(frame)

    def release(self):
        """
        Release Hailo device context.  Call once when inference is complete.
        No-op for the Ultralytics backend.
        """
        if getattr(self, "_backend", None) != "hailo":
            return
        for attr in ("_activated", "_pipeline"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.__exit__(None, None, None)
                except Exception:
                    pass
        tgt = getattr(self, "_target", None)
        if tgt is not None:
            try:
                tgt.release()
            except Exception:
                pass

    def __del__(self):
        """Safety-net cleanup if release() was not called explicitly."""
        self.release()
