from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from .pose_utils import COCO17, kp_valid, get_point, torso_center_and_scale, angle
from .flow import (sample_points_in_box, sample_points_background, lk_flow,
                   median_flow, radial_tangential_stats,
                   estimate_background_homography, predict_flow_homography)

class FeatureState:
    def __init__(self, carry_steps=3):
        self.prev = None
        self.prev_valid = None
        self.miss_run = None
        self.carry_steps = carry_steps

    def init(self, dim):
        self.prev = np.zeros((dim,), dtype=np.float32)
        self.prev_valid = np.zeros((dim,), dtype=np.float32)
        self.miss_run = np.zeros((dim,), dtype=np.int32)

    def impute(self, x, valid):
        # Always freeze at last known value when a feature is missing.
        # miss_run tracks consecutive missing frames (informational; not used to change behaviour).
        out = x.copy()
        for i in range(len(out)):
            if valid[i] > 0.5:
                self.miss_run[i] = 0
            else:
                self.miss_run[i] += 1
                out[i] = self.prev[i]
        self.prev = out
        self.prev_valid = valid
        return out

class TemporalDerivatives:
    """Frame-by-frame temporal derivative tracker.

    Computes dlog_area_dt, d2log_area_dt2, wrist vel/accel, and
    dlog_scale_dt from successive feature vectors.  Single source of
    truth — used by evaluate.py, infer.py run(), and infer.py run_on_file().
    """

    def __init__(self):
        self.prev_log_area       = None   # type: float | None
        self.prev_dlog_area_dt   = None   # type: float | None
        self.prev_log_scale      = None   # type: float | None
        self.prev_wrist_dist_l   = None   # type: float | None
        self.prev_wrist_dist_r   = None   # type: float | None
        self.prev_wrist_vel_l    = 0.0
        self.prev_wrist_vel_r    = 0.0
        self.prev_wrist_y_rel    = None   # type: float | None
        self.prev_ankle_spread   = None   # type: float | None
        self.prev_nose_y_rel     = None   # type: float | None

    def update(self, x: np.ndarray, m: np.ndarray, dt: float) -> float:
        """Compute temporal derivatives and write them into *x* and *m*.

        Reads:
            x[9]  log_area,  x[19] dist_l_wrist_torso,  x[20] dist_r_wrist_torso,
            x[47] log_scale, m[47] log_scale validity,
            x[49] wrist_y_rel, m[49],  x[51] ankle_spread, m[51],
            x[53] nose_y_rel, m[53]
        Writes:
            x[10] dlog_area_dt,      x[11] d2log_area_dt2,   m[10], m[11]
            x[45] max_wrist_vel,     x[46] max_wrist_accel,  m[45], m[46]
            x[50] d_wrist_y_rel_dt,  m[50]
            x[52] d_ankle_spread_dt, m[52]
            x[54] d_nose_y_rel_dt,   m[54]

        Returns
        -------
        dlog_scale_dt : float
            Needed by add_interaction_features().
        """
        log_area  = float(x[9])
        dist_l    = float(x[19])
        dist_r    = float(x[20])
        log_scale = float(x[47])
        log_scale_valid = float(m[47]) > 0.5

        # ── log_area derivatives (indices 10, 11) ─────────────────────────
        dlog_area_dt   = 0.0
        d2log_area_dt2 = 0.0
        have_log_area  = self.prev_log_area is not None
        if have_log_area:
            dlog_area_dt = (log_area - self.prev_log_area) / dt
            if self.prev_dlog_area_dt is not None:
                d2log_area_dt2 = (dlog_area_dt - self.prev_dlog_area_dt) / dt
        self.prev_log_area     = log_area
        self.prev_dlog_area_dt = dlog_area_dt
        x[10] = dlog_area_dt;    m[10] = 1.0 if have_log_area else 0.0
        x[11] = d2log_area_dt2;  m[11] = 1.0 if have_log_area else 0.0

        # ── wrist velocity / acceleration (indices 45, 46) ───────────────
        wrist_vel   = 0.0
        wrist_accel = 0.0
        have_wrist  = self.prev_wrist_dist_l is not None
        if have_wrist:
            vel_l = (dist_l - self.prev_wrist_dist_l) / dt
            vel_r = (dist_r - self.prev_wrist_dist_r) / dt
            wrist_vel   = max(vel_l, vel_r)
            accel_l     = (vel_l - self.prev_wrist_vel_l) / dt
            accel_r     = (vel_r - self.prev_wrist_vel_r) / dt
            wrist_accel = max(accel_l, accel_r)
            self.prev_wrist_vel_l = vel_l
            self.prev_wrist_vel_r = vel_r
        else:
            self.prev_wrist_vel_l = 0.0
            self.prev_wrist_vel_r = 0.0
        self.prev_wrist_dist_l = dist_l
        self.prev_wrist_dist_r = dist_r
        x[45] = wrist_vel;    m[45] = 1.0 if have_wrist else 0.0
        x[46] = wrist_accel;  m[46] = 1.0 if have_wrist else 0.0

        # ── log_scale derivative ──────────────────────────────────────────
        # Guard: only compute when current keypoints are valid (m[47]).
        # prev_log_scale is only updated on valid frames, so the derivative
        # is always between two valid readings.
        dlog_scale_dt = 0.0
        if log_scale_valid and self.prev_log_scale is not None:
            dlog_scale_dt = (log_scale - self.prev_log_scale) / dt
        if log_scale_valid:
            self.prev_log_scale = log_scale

        # ── wrist_y_rel derivative (index 50) ────────────────────────────
        wrist_y_rel       = float(x[49])
        wrist_y_rel_valid = float(m[49]) > 0.5
        d_wrist_y_rel_dt  = 0.0
        have_wrist_y      = wrist_y_rel_valid and self.prev_wrist_y_rel is not None
        if have_wrist_y:
            d_wrist_y_rel_dt = (wrist_y_rel - self.prev_wrist_y_rel) / dt
        if wrist_y_rel_valid:
            self.prev_wrist_y_rel = wrist_y_rel
        x[50] = d_wrist_y_rel_dt;  m[50] = 1.0 if have_wrist_y else 0.0

        # ── ankle_spread derivative (index 52) ───────────────────────────
        ankle_spread       = float(x[51])
        ankle_spread_valid = float(m[51]) > 0.5
        d_ankle_spread_dt  = 0.0
        have_ankle         = ankle_spread_valid and self.prev_ankle_spread is not None
        if have_ankle:
            d_ankle_spread_dt = (ankle_spread - self.prev_ankle_spread) / dt
        if ankle_spread_valid:
            self.prev_ankle_spread = ankle_spread
        x[52] = d_ankle_spread_dt;  m[52] = 1.0 if have_ankle else 0.0

        # ── nose_y_rel derivative (index 54) ─────────────────────────────
        nose_y_rel       = float(x[53])
        nose_y_rel_valid = float(m[53]) > 0.5
        d_nose_y_rel_dt  = 0.0
        have_nose        = nose_y_rel_valid and self.prev_nose_y_rel is not None
        if have_nose:
            d_nose_y_rel_dt = (nose_y_rel - self.prev_nose_y_rel) / dt
        if nose_y_rel_valid:
            self.prev_nose_y_rel = nose_y_rel
        x[54] = d_nose_y_rel_dt;  m[54] = 1.0 if have_nose else 0.0

        return dlog_scale_dt


def crop_flags(bbox, w, h, eps):
    if bbox is None:
        return 1, (1,1,1,1)
    x1,y1,x2,y2 = bbox
    left = 1 if x1 <= eps*w else 0
    right = 1 if x2 >= (1-eps)*w else 0
    top = 1 if y1 <= eps*h else 0
    bottom = 1 if y2 >= (1-eps)*h else 0
    anyc = 1 if (left or right or top or bottom) else 0
    return anyc, (left,right,top,bottom)

def bbox_features(bbox, prev_bbox, dt):
    # returns: log_area, 0.0 (dlog_area_dt placeholder — filled by dataset.py/evaluate.py/infer.py),
    # cx, cy, dcx_dt, dcy_dt, aspect
    x1,y1,x2,y2 = bbox
    w = max(x2-x1, 1e-6)
    h = max(y2-y1, 1e-6)
    area = w*h
    log_area = np.log(area)
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    if prev_bbox is None:
        return log_area, 0.0, cx, cy, 0.0, 0.0, (w/h)
    px1,py1,px2,py2 = prev_bbox
    pcx = (px1+px2)/2.0
    pcy = (py1+py2)/2.0
    dcx = (cx-pcx)/dt
    dcy = (cy-pcy)/dt
    return log_area, 0.0, cx, cy, dcx, dcy, (w/h)

# Pixel tolerance for frame-edge crop detection in compute_torso_height_frac.
# A bbox edge within this many pixels of the frame boundary is considered
# cropped (person too close to fit in frame).
_CROP_PX = 2


def compute_torso_height_frac(
    kps: np.ndarray,
    frame_h: float,
    kp_conf_thresh: float,
    bbox: Optional[list] = None,
) -> float:
    """Torso height fraction = ||hip_mid − shoulder_mid|| / frame_h.

    Uses the 2-D Euclidean distance between the midpoint of the shoulders
    (COCO-17 kps[5], kps[6]) and the midpoint of the hips (kps[11], kps[12]).
    When one side is occluded the visible side is used as a fallback.

    Keypoints below kp_conf_thresh are treated as unavailable.  When ALL
    keypoints for a group (shoulders or hips) are unavailable, the function
    checks whether the bbox is cropped at the corresponding frame edge:

      • Hips missing + bbox bottom at/near frame bottom  → hips are below the
        frame (person is VERY close).  Hip midpoint is estimated at the bottom
        frame edge so the torso fraction reflects actual proximity rather than
        returning 0.0 and falsely triggering suppression.
      • Shoulders missing + bbox top at/near frame top  → symmetric treatment.
      • Keypoints missing with NO corresponding crop  → genuinely occluded or
        undetectable; returns 0.0 so the gate suppresses (conservative fallback).

    Args:
        kps           : (17, 3) COCO-17 array — (x, y, conf) per joint.
        frame_h       : frame height in pixels.
        kp_conf_thresh: minimum confidence to treat a keypoint as valid.
        bbox          : [x1, y1, x2, y2] bounding box in pixels.  Required for
                        crop-aware fallback; pass None to disable (returns 0.0
                        for all fully-missing groups, original behaviour).
    """
    L_SH, R_SH, L_HP, R_HP = 5, 6, 11, 12  # COCO-17 keypoint indices
    has_l_sh = kps[L_SH, 2] >= kp_conf_thresh
    has_r_sh = kps[R_SH, 2] >= kp_conf_thresh
    has_l_hp = kps[L_HP, 2] >= kp_conf_thresh
    has_r_hp = kps[R_HP, 2] >= kp_conf_thresh

    bbox_cx = ((float(bbox[0]) + float(bbox[2])) * 0.5) if bbox is not None else 0.0

    # ── shoulder midpoint ────────────────────────────────────────────────────
    if has_l_sh and has_r_sh:
        sh = (kps[L_SH, :2] + kps[R_SH, :2]) * 0.5
    elif has_l_sh:
        sh = kps[L_SH, :2]
    elif has_r_sh:
        sh = kps[R_SH, :2]
    else:
        # All shoulder keypoints missing — check for top-edge crop.
        # If the bbox top is at the frame top, the shoulders are above the
        # frame (person filling or exceeding the frame height) → very close.
        if bbox is not None and float(bbox[1]) <= _CROP_PX:
            sh = np.array([bbox_cx, 0.0], dtype=np.float32)
        else:
            return 0.0  # genuinely missing, no crop evidence → gate suppresses

    # ── hip midpoint ─────────────────────────────────────────────────────────
    if has_l_hp and has_r_hp:
        hp = (kps[L_HP, :2] + kps[R_HP, :2]) * 0.5
    elif has_l_hp:
        hp = kps[L_HP, :2]
    elif has_r_hp:
        hp = kps[R_HP, :2]
    else:
        # All hip keypoints missing — check for bottom-edge crop.
        # If the bbox bottom is at the frame bottom, the hips are below the
        # frame (person too close to fit) → very close, do not suppress.
        if bbox is not None and float(bbox[3]) >= float(frame_h) - _CROP_PX:
            hp = np.array([bbox_cx, float(frame_h)], dtype=np.float32)
        else:
            return 0.0  # genuinely missing, no crop evidence → gate suppresses

    torso_px = float(np.linalg.norm(hp - sh))
    return torso_px / max(float(frame_h), 1.0)


def build_features(
    frame_bgr: np.ndarray,
    prev_gray: Optional[np.ndarray],
    prev_bbox: Optional[list],
    det: Dict[str, Any],
    track_age: int,
    lost: int,
    cfg,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    """
    Returns:
      x: feature values (float32)
      m: validity mask bits aligned to x (float32 0/1) for pose/flow-derived parts
      debug: dict for logging
      curr_gray: for next step

    dt: actual elapsed seconds since the previous processed frame.  When None
        (default) cfg.step_dt is used, which is correct for the file path
        (frame_stride / input_fps = 3/30 = 0.1 s).  The live camera path in
        infer.py passes the wall-clock Δt so that bbox velocity features
        (bbox_center_vx, bbox_center_vy) stay on the same physical scale as
        the training data regardless of the actual inference rate.
    """
    _dt = dt if dt is not None else cfg.step_dt
    h, w = frame_bgr.shape[:2]
    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    bbox = det["bbox"]
    kps = det["kps"].astype(np.float32)
    det_conf = float(det["det_conf"])
    kp_confs = kps[:,2]
    kp_conf_mean = float(np.mean(kp_confs))
    kp_conf_min = float(np.min(kp_confs))
    visible_kp_count = float(np.sum(kp_confs >= cfg.kp_conf_thresh))

    anyc, (cl,cr,ct,cb) = crop_flags(bbox, w, h, cfg.crop_eps)

    # torso center + scale
    torso_c, torso_s, torso_ok = torso_center_and_scale(kps, cfg.kp_conf_thresh)
    if not torso_ok:
        # fallback to bbox center and bbox height
        x1,y1,x2,y2 = bbox
        torso_c = np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)
        torso_s = float(max(y2-y1, 1e-6))

    norm = float(torso_s + 1e-6)

    # --- bbox approach features (invalid if cropped)
    log_area, _, cx, cy, dcx, dcy, aspect = bbox_features(bbox, prev_bbox, _dt)
    # x[10] (dlog_area_dt) and x[11] (d2log_area_dt2) are 0.0 placeholders here;
    # filled from frame history by dataset.py / evaluate.py / infer.py.

    # Normalize center by image dims
    cx_n = cx / w
    cy_n = cy / h
    dcx_n = (dcx / w)
    dcy_n = (dcy / h)

    # --- pose features (values + validity)
    def dist(a_idx, b_idx):
        va = kp_valid(kps, a_idx, cfg.kp_conf_thresh)
        vb = kp_valid(kps, b_idx, cfg.kp_conf_thresh)
        if not (va and vb):
            return 0.0, 0.0
        da = get_point(kps, a_idx)
        db = get_point(kps, b_idx)
        return float(np.linalg.norm(da-db) / norm), 1.0

    def ang(a_idx, b_idx, c_idx):
        va = kp_valid(kps, a_idx, cfg.kp_conf_thresh)
        vb = kp_valid(kps, b_idx, cfg.kp_conf_thresh)
        vc = kp_valid(kps, c_idx, cfg.kp_conf_thresh)
        if not (va and vb and vc):
            return 0.0, 0.0
        a = get_point(kps, a_idx); b = get_point(kps, b_idx); c = get_point(kps, c_idx)
        return float(angle(a,b,c)), 1.0

    # Upper distances
    d_wl_sl, v_wl_sl = dist(COCO17["l_wrist"], COCO17["l_shoulder"])
    d_wr_sr, v_wr_sr = dist(COCO17["r_wrist"], COCO17["r_shoulder"])
    # wrist to torso center
    def dist_to_torso(idx):
        v = kp_valid(kps, idx, cfg.kp_conf_thresh)
        if not v:
            return 0.0, 0.0
        p = get_point(kps, idx)
        return float(np.linalg.norm(p - torso_c) / norm), 1.0
    d_wl_tc, v_wl_tc = dist_to_torso(COCO17["l_wrist"])
    d_wr_tc, v_wr_tc = dist_to_torso(COCO17["r_wrist"])

    # widths
    d_sh_w, v_sh_w = dist(COCO17["l_shoulder"], COCO17["r_shoulder"])
    d_hip_w, v_hip_w = dist(COCO17["l_hip"], COCO17["r_hip"])

    # angles
    a_el_l, v_el_l = ang(COCO17["l_shoulder"], COCO17["l_elbow"], COCO17["l_wrist"])
    a_el_r, v_el_r = ang(COCO17["r_shoulder"], COCO17["r_elbow"], COCO17["r_wrist"])
    a_kn_l, v_kn_l = ang(COCO17["l_hip"], COCO17["l_knee"], COCO17["l_ankle"])
    a_kn_r, v_kn_r = ang(COCO17["r_hip"], COCO17["r_knee"], COCO17["r_ankle"])

    # ankle to hip / torso
    d_al_hl, v_al_hl = dist(COCO17["l_ankle"], COCO17["l_hip"])
    d_ar_hr, v_ar_hr = dist(COCO17["r_ankle"], COCO17["r_hip"])
    d_al_tc, v_al_tc = dist_to_torso(COCO17["l_ankle"])
    d_ar_tc, v_ar_tc = dist_to_torso(COCO17["r_ankle"])

    # stance width
    d_st, v_st = dist(COCO17["l_ankle"], COCO17["r_ankle"])

    # --- optical flow features (if prev_gray exists)
    # ROI definitions
    x1,y1,x2,y2 = bbox
    roi_person = (x1,y1,x2,y2)
    # torso ROI ~ square around torso center
    side = cfg.torso_roi_scale * torso_s
    tx1 = max(0, int(torso_c[0] - side/2)); tx2 = min(w-1, int(torso_c[0] + side/2))
    ty1 = max(0, int(torso_c[1] - side/2)); ty2 = min(h-1, int(torso_c[1] + side/2))
    roi_torso = (tx1,ty1,tx2,ty2)

    # lower ROI: hip-to-ankle region, below (not overlapping) the torso ROI.
    # Anchored at mid-hip y: prefer actual hip keypoints; fall back to
    # torso_c + torso_s/2 (geometric hip from torso centre/scale).
    # Width is 1.5×torso_s to capture stance changes and kick/lunge leg spread.
    lh_side_w = 1.5 * torso_s
    if (kp_valid(kps, COCO17["l_hip"], cfg.kp_conf_thresh) and
            kp_valid(kps, COCO17["r_hip"], cfg.kp_conf_thresh)):
        hip_mid_y = float((kps[COCO17["l_hip"], 1] + kps[COCO17["r_hip"], 1]) / 2.0)
    else:
        hip_mid_y = float(torso_c[1] + 0.5 * torso_s)
    lx1 = max(0, int(torso_c[0] - lh_side_w / 2))
    lx2 = min(w - 1, int(torso_c[0] + lh_side_w / 2))
    ly1 = max(0, int(hip_mid_y))
    ly2 = min(h - 1, int(hip_mid_y + cfg.lower_roi_scale * torso_s))
    roi_lower = (lx1, ly1, lx2, ly2)

    # Defaults
    trans_signed_torso=trans_torso=div_torso=div_ratio_torso=0.0
    trans_signed_low=trans_low=div_low=div_ratio_low=0.0
    bg_coh=0.0
    flow_ok = 0.0

    if prev_gray is not None:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.lk_criteria_count, cfg.lk_criteria_eps)

        pts_bg = sample_points_background(h,w,roi_person,max_points=cfg.max_flow_points)
        p0_bg, p1_bg, v_bg = lk_flow(prev_gray, curr_gray, pts_bg, cfg.lk_win_size, cfg.lk_max_level, crit)
        vbg_med, vbg_ok = median_flow(v_bg)

        # Homography-based background model: handles camera yaw/pitch/roll
        # (dominant artefacts on a body-worn camera).  Falls back to median
        # when RANSAC finds too few inliers (e.g. featureless background).
        bg_H, bg_n_inliers = estimate_background_homography(
            p0_bg, p1_bg, cfg.bg_ransac_thresh)
        use_homography = (bg_H is not None) and (bg_n_inliers >= cfg.bg_min_inliers)

        if vbg_ok:
            mean_v = np.mean(v_bg, axis=0)
            bg_coh = float(np.linalg.norm(mean_v) / (np.mean(np.linalg.norm(v_bg, axis=1)) + 1e-6))

        # torso ROI — try tight square first; fall back to full person bbox when
        # uniform clothing gives < 2 trackable points (plain shirt, etc.).
        pts_t = sample_points_in_box(*roi_torso, max_points=cfg.max_flow_points, margin=2)
        p0_t, p1_t, v_t = lk_flow(prev_gray, curr_gray, pts_t, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_t is None:
            pts_t = sample_points_in_box(*roi_person, max_points=cfg.max_flow_points, margin=2)
            p0_t, p1_t, v_t = lk_flow(prev_gray, curr_gray, pts_t, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_t is not None and vbg_ok:
            if use_homography:
                v_t_bg = predict_flow_homography(bg_H, p0_t)
            else:
                v_t_bg = np.tile(vbg_med, (len(v_t), 1))
            v_t2 = v_t - v_t_bg
            trans_signed_torso, trans_torso, div_torso, div_ratio_torso = radial_tangential_stats(p0_t, v_t2, torso_c)
            flow_ok = 1.0

        # lower ROI — use its own center for decomposition so forward leg motion
        # is correctly classified as radial (approaching), not tangential.
        # Fall back to full person bbox when legs are off-frame or heavily occluded.
        lower_c = np.array([(lx1+lx2)/2.0, (ly1+ly2)/2.0], dtype=np.float32)
        pts_l = sample_points_in_box(*roi_lower, max_points=cfg.max_flow_points, margin=2)
        p0_l, p1_l, v_l = lk_flow(prev_gray, curr_gray, pts_l, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_l is None:
            pts_l = sample_points_in_box(*roi_person, max_points=cfg.max_flow_points, margin=2)
            p0_l, p1_l, v_l = lk_flow(prev_gray, curr_gray, pts_l, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_l is not None and vbg_ok:
            if use_homography:
                v_l_bg = predict_flow_homography(bg_H, p0_l)
            else:
                v_l_bg = np.tile(vbg_med, (len(v_l), 1))
            v_l2 = v_l - v_l_bg
            trans_signed_low, trans_low, div_low, div_ratio_low = radial_tangential_stats(p0_l, v_l2, lower_c)
            flow_ok = 1.0

    # Posture features
    torso_compression = 0.0
    wrist_height_asym = 0.0
    face_visibility = 0.0
    v_torso_compression = 0.0
    v_wrist_asym = 0.0
    v_face_vis = 1.0  # always valid (uses confidence scores)

    # Torso compression: torso_height / torso_width — aspect ratio of the torso.
    #   ~2.0 upright, <1.5 crouching/lunging.
    # Strict: both shoulders + both hips + shoulder width valid.
    # Fallback: any shoulder + any hip; width from shoulder, hip, or bbox.
    has_both_sh = kp_valid(kps, COCO17["l_shoulder"], cfg.kp_conf_thresh) and kp_valid(kps, COCO17["r_shoulder"], cfg.kp_conf_thresh)
    has_both_hp = kp_valid(kps, COCO17["l_hip"], cfg.kp_conf_thresh) and kp_valid(kps, COCO17["r_hip"], cfg.kp_conf_thresh)
    if has_both_sh and has_both_hp and v_sh_w > 0.5:
        # Strict path: midpoints of both sides
        sh_mid = (get_point(kps, COCO17["l_shoulder"]) + get_point(kps, COCO17["r_shoulder"])) / 2.0
        hp_mid = (get_point(kps, COCO17["l_hip"]) + get_point(kps, COCO17["r_hip"])) / 2.0
        sh_hp_dist = float(np.linalg.norm(sh_mid - hp_mid) / norm)
        torso_compression = sh_hp_dist / (d_sh_w + 1e-6)
        v_torso_compression = 1.0
    else:
        # Single-side fallback: any shoulder + any hip
        _sh_pt = None
        if kp_valid(kps, COCO17["l_shoulder"], cfg.kp_conf_thresh):
            _sh_pt = get_point(kps, COCO17["l_shoulder"])
        elif kp_valid(kps, COCO17["r_shoulder"], cfg.kp_conf_thresh):
            _sh_pt = get_point(kps, COCO17["r_shoulder"])
        _hp_pt = None
        if kp_valid(kps, COCO17["l_hip"], cfg.kp_conf_thresh):
            _hp_pt = get_point(kps, COCO17["l_hip"])
        elif kp_valid(kps, COCO17["r_hip"], cfg.kp_conf_thresh):
            _hp_pt = get_point(kps, COCO17["r_hip"])
        # Width: prefer shoulder width, then hip width, then bbox width
        if v_sh_w > 0.5:
            _width = d_sh_w
        elif v_hip_w > 0.5:
            _width = d_hip_w
        else:
            _x1b, _y1b, _x2b, _y2b = bbox
            _width = float(max(_x2b - _x1b, 1.0)) / norm
        if _sh_pt is not None and _hp_pt is not None and _width > 1e-6:
            _sh_hp_dist = float(np.linalg.norm(_hp_pt - _sh_pt) / norm)
            if _sh_hp_dist > 0.05:  # guard against overlapping keypoints
                torso_compression = _sh_hp_dist / (_width + 1e-6)
                v_torso_compression = 1.0

    # abs(wrist_L_y - wrist_R_y) / torso_scale
    has_wl = kp_valid(kps, COCO17["l_wrist"], cfg.kp_conf_thresh)
    has_wr = kp_valid(kps, COCO17["r_wrist"], cfg.kp_conf_thresh)
    if has_wl and has_wr:
        wl_y = kps[COCO17["l_wrist"], 1]
        wr_y = kps[COCO17["r_wrist"], 1]
        wrist_height_asym = float(abs(wl_y - wr_y) / norm)
        v_wrist_asym = 1.0

    # Face visibility score: average confidence of facial keypoints
    # High = facing camera (potential threat), Low = facing away (walking away, low threat)
    nose_conf = kps[COCO17["nose"], 2]
    l_eye_conf = kps[COCO17["l_eye"], 2]
    r_eye_conf = kps[COCO17["r_eye"], 2]
    l_ear_conf = kps[COCO17["l_ear"], 2]
    r_ear_conf = kps[COCO17["r_ear"], 2]
    face_visibility = float((nose_conf + l_eye_conf + r_eye_conf + l_ear_conf + r_ear_conf) / 5.0)

    # Wrist extension dynamics (velocity & acceleration computed in dataset.py from temporal derivatives)
    max_wrist_vel = 0.0
    max_wrist_accel = 0.0

    # Robust scale (distance proxy): use torso height only (mid-shoulder to mid-hip) when available.
    #
    # Rationale:
    #   Using lateral widths (shoulder/hip width) inside the distance proxy can create false "approach"
    #   spikes when a person turns from sideways to front-facing (width increases without true distance change).
    #   Torso height is far less sensitive to yaw and arm pose, so it is a safer scale cue for software-only
    #   proximity estimation on body-cam video.
    #
    # Notes:
    #   Torso height can still change with strong pitch / bending / crouching. Downstream, treat dlog_scale_dt
    #   as a soft cue and rely on visibility + articulation features to avoid false positives.
    # Torso-scale reliability gating:
    #   Neck/shoulder->hip scale can become unreliable when the person bends forward (foreshortening),
    #   when hips are truncated/cropped near the bottom border, or when hip keypoints jitter/shift.
    #   In those cases we mark log_scale invalid (v_log_scale=0) so FeatureState will carry-forward
    #   the last reliable scale instead of letting proximity features spike.

    # Default
    log_scale = 0.0
    v_log_scale = 0.0

    if has_both_sh and has_both_hp:
        sh_pt = (get_point(kps, COCO17["l_shoulder"]) + get_point(kps, COCO17["r_shoulder"])) / 2.0
        hp_pt = (get_point(kps, COCO17["l_hip"]) + get_point(kps, COCO17["r_hip"])) / 2.0
        torso_vec = hp_pt - sh_pt
        torso_height_px = float(np.linalg.norm(torso_vec))

        # Reliability tests
        # 1) Hips too close to bottom border (likely truncated / unstable).
        bottom_margin_px = float(max(cfg.hip_bottom_margin_px, 2.0 * cfg.crop_eps * h))
        hip_near_bottom = bool(hp_pt[1] >= (h - bottom_margin_px))

        # 2) Strong forward bend / pitch causes foreshortening: torso segment becomes far from vertical.
        #    Compute tilt from vertical axis (0=vertical).
        if torso_height_px > 1e-3:
            v_unit = torso_vec / torso_height_px
            # vertical axis is (0,1); clamp dot for numerical stability
            dot = float(np.clip(v_unit[1], -1.0, 1.0))
            tilt_rad = float(np.arccos(abs(dot)))
            tilt_deg = tilt_rad * (180.0 / np.pi)
        else:
            tilt_deg = 90.0

        bend_like = bool(tilt_deg > cfg.bend_tilt_thresh)

        if (torso_height_px > 5.0) and (not hip_near_bottom) and (not bend_like):
            log_scale = float(np.log(max(torso_height_px, 1.0)))
            v_log_scale = 1.0
        else:
            # Keep a value (so debug shows it), but mark invalid so the imputer carries forward.
            log_scale = float(np.log(max(torso_height_px, 1.0))) if torso_height_px > 1.0 else 0.0
            v_log_scale = 0.0
    else:
        # Fallback to sqrt(bbox_area) when torso keypoints are not available
        x1b, y1b, x2b, y2b = bbox
        log_scale = float(np.log(max(np.sqrt((x2b - x1b) * (y2b - y1b)), 1.0)))
        v_log_scale = 0.0  # mark as unreliable fallback

    # Best-effort torso height (px): single-side shoulder/hip fallback.
    # Unlike log_scale: no hip_near_bottom or bend_like invalidation — those guards
    # exist on log_scale to protect its derivative (approach_rate) from jitter, but
    # torso_ht_px is not differentiated so the looser condition is safe.
    # v=1 when at least one shoulder + one hip keypoint exceeds kp_conf_thresh (real geometry).
    # v=0 only when no usable keypoints at all → falls back to bbox height.
    has_l_sh_be = kp_valid(kps, COCO17["l_shoulder"], cfg.kp_conf_thresh)
    has_r_sh_be = kp_valid(kps, COCO17["r_shoulder"], cfg.kp_conf_thresh)
    has_l_hp_be = kp_valid(kps, COCO17["l_hip"],      cfg.kp_conf_thresh)
    has_r_hp_be = kp_valid(kps, COCO17["r_hip"],      cfg.kp_conf_thresh)

    if has_l_sh_be and has_r_sh_be:
        sh_be = (get_point(kps, COCO17["l_shoulder"]) + get_point(kps, COCO17["r_shoulder"])) * 0.5
    elif has_l_sh_be:
        sh_be = get_point(kps, COCO17["l_shoulder"])
    elif has_r_sh_be:
        sh_be = get_point(kps, COCO17["r_shoulder"])
    else:
        sh_be = None

    if has_l_hp_be and has_r_hp_be:
        hp_be = (get_point(kps, COCO17["l_hip"]) + get_point(kps, COCO17["r_hip"])) * 0.5
    elif has_l_hp_be:
        hp_be = get_point(kps, COCO17["l_hip"])
    elif has_r_hp_be:
        hp_be = get_point(kps, COCO17["r_hip"])
    else:
        hp_be = None

    if sh_be is not None and hp_be is not None:
        torso_ht_px = float(np.linalg.norm(np.asarray(hp_be) - np.asarray(sh_be)))
        v_torso_ht_px = 1.0 if torso_ht_px > 5.0 else 0.0
    else:
        # Pure bbox fallback: bbox height as coarse body-size proxy.
        _x1b, _y1b, _x2b, _y2b = bbox
        torso_ht_px = float(max(_y2b - _y1b, 1.0))
        v_torso_ht_px = 0.0

    # ── Phase E features (computed after torso_ht_px so sh_be/hp_be are available) ──

    # #12: Wrist vertical position relative to hip — captures raised fist/weapon.
    #   max of (hip_y − wrist_y) / bbox_h for L/R wrist.
    #   Positive = wrist above hip (raised), negative = wrist below hip.
    #   Uses single-side hip fallback (same as torso_ht_px).
    wrist_y_rel   = 0.0
    v_wrist_y_rel = 0.0
    _bbox_h = max(bbox[3] - bbox[1], 1.0)
    if hp_be is not None:
        hp_y = float(hp_be[1])
        best_wrist_y_rel = -999.0
        if has_wl:
            wl_y_val = float(kps[COCO17["l_wrist"], 1])
            best_wrist_y_rel = max(best_wrist_y_rel, (hp_y - wl_y_val) / _bbox_h)
        if has_wr:
            wr_y_val = float(kps[COCO17["r_wrist"], 1])
            best_wrist_y_rel = max(best_wrist_y_rel, (hp_y - wr_y_val) / _bbox_h)
        if best_wrist_y_rel > -999.0:
            wrist_y_rel   = best_wrist_y_rel
            v_wrist_y_rel = 1.0

    # #13: Ankle spread — horizontal stride width, captures running/charging.
    #   |ankle_L_x − ankle_R_x| / bbox_w
    ankle_spread   = 0.0
    v_ankle_spread = 0.0
    _bbox_w = max(bbox[2] - bbox[0], 1.0)
    has_l_ankle = kp_valid(kps, COCO17["l_ankle"], cfg.kp_conf_thresh)
    has_r_ankle = kp_valid(kps, COCO17["r_ankle"], cfg.kp_conf_thresh)
    if has_l_ankle and has_r_ankle:
        la_x = float(kps[COCO17["l_ankle"], 0])
        ra_x = float(kps[COCO17["r_ankle"], 0])
        ankle_spread   = abs(la_x - ra_x) / _bbox_w
        v_ankle_spread = 1.0

    # #14: Nose vertical position relative to shoulder midpoint — captures ducking/lunging.
    #   (shoulder_mid_y − nose_y) / bbox_h.  Positive = head above shoulders (normal),
    #   decreasing = ducking.
    nose_y_rel   = 0.0
    v_nose_y_rel = 0.0
    has_nose_kp = kp_valid(kps, COCO17["nose"], cfg.kp_conf_thresh)
    if has_nose_kp and sh_be is not None:
        sh_y  = float(sh_be[1])
        ns_y  = float(kps[COCO17["nose"], 1])
        nose_y_rel   = (sh_y - ns_y) / _bbox_h
        v_nose_y_rel = 1.0

    # #16: Upper-lower body asynchrony — |trans_torso − trans_lower|.
    #   Large value = arms moving but legs still (or vice versa).
    upper_lower_async = abs(trans_torso - trans_low)

    # Assemble feature vector (values) + validity mask for pose/flow parts
    # Note: bbox features validity depends on cropping.
    bbox_valid = 0.0 if anyc else 1.0

    x = np.array([
        # reliability (0-8)
        det_conf, kp_conf_mean, kp_conf_min, visible_kp_count,
        float(anyc), float(cl), float(cr), float(ct), float(cb),

        # approach / bbox (9-16)
        log_area, 0.0, 0.0,
        cx_n, cy_n, dcx_n, dcy_n, aspect,

        # upper geometry (17-24)
        d_wl_sl, d_wr_sr, d_wl_tc, d_wr_tc, d_sh_w, d_hip_w,
        a_el_l, a_el_r,

        # lower geometry (25-31)
        a_kn_l, a_kn_r, d_al_hl, d_ar_hr, d_al_tc, d_ar_tc, d_st,

        # flow torso/lower + bg (32-41)
        trans_signed_torso, trans_torso, div_torso, div_ratio_torso,
        trans_signed_low, trans_low, div_low, div_ratio_low,
        bg_coh, flow_ok,

        # posture features (42-44)
        torso_compression, wrist_height_asym, face_visibility,

        # dynamics (45-47), best-effort torso height px (48)
        max_wrist_vel, max_wrist_accel, log_scale,
        torso_ht_px,

        # wrist direction (49-50)
        wrist_y_rel, 0.0,           # d_wrist_y_rel_dt placeholder

        # gait dynamics (51-52)
        ankle_spread, 0.0,          # d_ankle_spread_dt placeholder

        # head motion (53-54)
        nose_y_rel, 0.0,            # d_nose_y_rel_dt placeholder

        # upper-lower async (55)
        upper_lower_async
    ], dtype=np.float32)

    # validity mask aligned to x for the parts that can be missing:
    m = np.ones_like(x, dtype=np.float32)
    m[9:17] = bbox_valid
    pose_valids = [v_wl_sl,v_wr_sr,v_wl_tc,v_wr_tc,v_sh_w,v_hip_w,v_el_l,v_el_r]
    m[17:25] = np.array(pose_valids, dtype=np.float32)
    lower_valids = [v_kn_l,v_kn_r,v_al_hl,v_ar_hr,v_al_tc,v_ar_tc,v_st]
    m[25:32] = np.array(lower_valids, dtype=np.float32)
    m[32:42] = flow_ok
    m[42] = v_torso_compression
    m[43] = v_wrist_asym
    m[44] = v_face_vis
    # Indices 10, 11 (dlog_area_dt, d2log_area_dt2) and 45, 46 (wrist vel/accel) are
    # 0.0 placeholders filled by dataset.py / evaluate.py / infer.py from frame history.
    # Similarly 50, 52, 54 are derivative placeholders filled by TemporalDerivatives.
    # Mark invalid here so the GRU knows they are not yet computed.
    m[10] = 0.0
    m[11] = 0.0
    m[45] = 0.0
    m[46] = 0.0
    m[47] = v_log_scale
    m[48] = v_torso_ht_px
    m[49] = v_wrist_y_rel
    m[50] = 0.0    # d_wrist_y_rel_dt — placeholder
    m[51] = v_ankle_spread
    m[52] = 0.0    # d_ankle_spread_dt — placeholder
    m[53] = v_nose_y_rel
    m[54] = 0.0    # d_nose_y_rel_dt — placeholder
    m[55] = flow_ok  # upper_lower_async valid when flow is ok

    debug = {
        "bbox_valid": bbox_valid,
        "torso_ok": float(torso_ok),
        "flow_ok": flow_ok,
        "div_torso": div_torso,
        "div_low": div_low,
        "bg_coh": bg_coh,
        "torso_compression": torso_compression,
        "wrist_height_asym": wrist_height_asym,
        "torso_ht_px": torso_ht_px,
        "roi_torso": roi_torso,
        "roi_lower": roi_lower,
    }
    return x, m, debug, curr_gray

def add_interaction_features(
    x: np.ndarray,
    m: np.ndarray,
    wrist_vel: float = 0.0,
    wrist_accel: float = 0.0,
    dlog_scale_dt: float = 0.0,
    cfg = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add interaction features (motion × proximity) to a single feature vector.
    Used during frame-by-frame inference and evaluation when temporal derivatives
    are available from previous frames.

    Args:
        x: feature vector [56] from build_features
        m: mask vector [56]
        wrist_vel: max wrist extension velocity (computed externally from frame history)
        wrist_accel: max wrist extension acceleration (computed externally from frame history)
        dlog_scale_dt: rate of change of log_scale (log_scale[t] - log_scale[t-1]) / dt.
                       Positive = person is growing in apparent size (approaching).
                       Computed externally from successive build_features() calls.

    Returns:
        x_aug: augmented feature vector [59]
        m_aug: augmented mask vector [59]
    """
    # Extract base features
    log_scale = x[47]            # pose-derived log apparent size (index 47)
    trans_signed_torso = x[32]  # signed flow: positive=approaching, negative=retreating
    divergence_torso = x[34]    # torso divergence

    # approach_rate: positive only when BOTH apparent size is growing AND flow is toward camera.
    # Distant-person leg lift: dlog_scale_dt ≈ 0 → approach_rate ≈ 0.
    # Retreating person: trans_signed_torso < 0 → approach_rate = 0.
    approach_rate = float(max(dlog_scale_dt, 0.0) * max(trans_signed_torso, 0.0))

    # Proximity weight: prefer log_scale (strict) when valid; fall back to
    # log(torso_ht_px) (lenient, index 48) when log_scale is invalid (ok=0)
    # so that a stale carry-forward value doesn't inflate expansion_proximity
    # for a far person whose keypoints temporarily failed strict validity checks.
    # Exponent 0.3 creates ~57% more weight for close (ls≈5.5) vs far (ls≈4.0).
    ls_ok = m[47] > 0.5
    th_ok = m[48] > 0.5
    if ls_ok:
        scale_for_weight = float(x[47])                     # log_scale  — strict, reliable
    elif th_ok:
        scale_for_weight = float(np.log(max(x[48], 1.0)))  # log(torso_ht_px) — lenient fallback
    else:
        scale_for_weight = float(x[47])                     # stale carry-forward — last resort
    prox_exp = cfg.proximity_exponent if cfg is not None else 0.3
    proximity_weight = float(np.exp(scale_for_weight * prox_exp))

    expansion_proximity = divergence_torso * proximity_weight
    acceleration_proximity = wrist_accel * proximity_weight

    # Augment feature vector
    flow_ok_val = m[41]  # flow_ok at index 41
    x_aug = np.concatenate([x, [approach_rate, expansion_proximity, acceleration_proximity]])
    m_aug = np.concatenate([m, [flow_ok_val, flow_ok_val, flow_ok_val]])

    return x_aug, m_aug