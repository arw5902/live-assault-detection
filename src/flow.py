from typing import Tuple, Optional
import numpy as np
import cv2

def sample_points_in_box(x1,y1,x2,y2, max_points=200, margin=0):
    x1 = int(x1+margin); y1 = int(y1+margin)
    x2 = int(x2-margin); y2 = int(y2-margin)
    if x2 <= x1+5 or y2 <= y1+5:
        return None
    xs = np.random.randint(x1, x2, size=(max_points,))
    ys = np.random.randint(y1, y2, size=(max_points,))
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    return pts.reshape(-1,1,2)

def sample_points_background(h, w, bbox, max_points=200, margin=20):
    if bbox is None:
        # sample anywhere
        xs = np.random.randint(0, w, size=(max_points,))
        ys = np.random.randint(0, h, size=(max_points,))
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        return pts.reshape(-1,1,2)

    x1,y1,x2,y2 = map(int, bbox)
    mask = np.ones((h,w), dtype=np.uint8)
    x1m = max(0, x1-margin); y1m = max(0, y1-margin)
    x2m = min(w-1, x2+margin); y2m = min(h-1, y2+margin)
    mask[y1m:y2m, x1m:x2m] = 0

    ys, xs = np.where(mask > 0)
    if len(xs) < max_points:
        max_points = len(xs)
    if max_points <= 0:
        return None
    idx = np.random.choice(len(xs), size=(max_points,), replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts.reshape(-1,1,2)

def lk_flow(prev_gray, curr_gray, pts, win_size, max_level, criteria):
    if pts is None or len(pts) == 0:
        return None, None, None
    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=criteria
    )
    good = st.reshape(-1) == 1
    if good.sum() < 2:          # lowered from 5: uniform clothing gives few trackable pts
        return None, None, None
    p0 = pts.reshape(-1,2)[good]
    p1 = nxt.reshape(-1,2)[good]
    v = p1 - p0
    return p0, p1, v

def median_flow(v):
    if v is None or len(v) == 0:
        return np.array([0.0,0.0], dtype=np.float32), False
    return np.median(v, axis=0).astype(np.float32), True

def radial_tangential_stats(p0, v, center):
    """
    Decompose optical flow into translation, divergence and residual components.

    For a body-worn camera the dominant motion is global translation (body moving
    toward camera = uniform downward shift). Decomposing this directly into
    radial/tangential conflates translation with expansion: points to the left/right
    of center get their translational motion split into tangential, inflating t.

    Fix: subtract the median ROI flow (local translation) before computing
    divergence, so the remaining signal is pure expansion/contraction.

    Returns 4 values per ROI:
      translation_signed - signed approach speed: positive = moving toward camera,
                           negative = moving away. Computed as mean radial component
                           of the median (translational) flow relative to ROI center.
      translation_mag    - magnitude of median (translational) flow in ROI
      divergence         - mean positive radial of RESIDUAL flow (pure expansion)
      div_ratio          - divergence / (divergence + residual_mag + eps)
    """
    # p0: Nx2, v: Nx2, center: (2,)

    # Step 1: separate global translation from local deformation
    v_med = np.median(v, axis=0)                  # median = translational component
    translation_mag = float(np.linalg.norm(v_med))
    v_resid = v - v_med.reshape(1, 2)             # residual = deformation / expansion

    # Signed translation: project median flow onto outward radial direction per point
    # Positive = ROI points moving away from center on average = approaching camera
    # Negative = ROI points moving toward center = retreating from camera
    d = p0 - center.reshape(1, 2)
    r = np.linalg.norm(d, axis=1) + 1e-6
    radial_translation = np.sum(d * v_med.reshape(1, 2), axis=1) / r
    translation_signed = float(np.mean(radial_translation))

    # Step 2: compute radial on RESIDUAL flow only
    radial = np.sum(d * v_resid, axis=1) / r      # positive = expanding (approaching)

    # Divergence: mean magnitude of positive-only radial residuals
    pos_mask = radial > 0
    divergence = float(np.mean(radial[pos_mask])) if pos_mask.any() else 0.0

    # Divergence ratio (bounded 0-1)
    resid_mag = float(np.mean(np.linalg.norm(v_resid, axis=1)))
    div_ratio = divergence / (divergence + resid_mag + 1e-6)

    return translation_signed, translation_mag, divergence, div_ratio
