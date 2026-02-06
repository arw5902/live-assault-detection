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
    if good.sum() < 5:
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
    # p0: Nx2, v: Nx2, center: (2,)
    d = p0 - center.reshape(1,2)
    r = np.linalg.norm(d, axis=1) + 1e-6
    # radial component
    radial = np.sum(d * v, axis=1) / r
    # tangential magnitude in 2D via cross product z-component
    tang = np.abs(d[:,0]*v[:,1] - d[:,1]*v[:,0]) / r
    e = float(np.median(radial))
    t = float(np.median(tang))
    R = float(max(e,0.0) / (max(e,0.0) + t + 1e-6))
    mag_p90 = float(np.percentile(np.linalg.norm(v, axis=1), 90))
    return e, t, R, mag_p90
