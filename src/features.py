from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from .pose_utils import COCO17, kp_valid, get_point, torso_center_and_scale, angle
from .flow import (sample_points_in_box, sample_points_background, lk_flow,
                   median_flow, radial_tangential_stats)

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
        # carry-forward for <= carry_steps
        out = x.copy()
        for i in range(len(out)):
            if valid[i] > 0.5:
                self.miss_run[i] = 0
            else:
                self.miss_run[i] += 1
                if self.miss_run[i] <= self.carry_steps:
                    out[i] = self.prev[i]
                else:
                    out[i] = self.prev[i]  # freeze
        self.prev = out
        self.prev_valid = valid
        return out

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
    # returns: log_area, dlog_area_dt, d2log_area_dt2 placeholder (computed outside),
    # cx,cy, dcx_dt,dcy_dt, aspect
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

def build_features(
    frame_bgr: np.ndarray,
    prev_gray: Optional[np.ndarray],
    prev_bbox: Optional[list],
    det: Dict[str, Any],
    track_age: int,
    lost: int,
    cfg,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    """
    Returns:
      x: feature values (float32)
      m: validity mask bits aligned to x (float32 0/1) for pose/flow-derived parts
      debug: dict for logging
      curr_gray: for next step
    """
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
        torso_ok = False

    norm = float(torso_s + 1e-6)

    # --- bbox approach features (invalid if cropped)
    log_area, _, cx, cy, dcx, dcy, aspect = bbox_features(bbox, prev_bbox, cfg.step_dt)
    # dlog_area_dt computed from stored prev log_area (outside), but we include as feature here as 0 placeholder.
    # We'll compute properly in dataset builder (simpler) OR keep a rolling prev inside inference.

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
    side = 1.2 * torso_s
    tx1 = max(0, int(torso_c[0] - side/2)); tx2 = min(w-1, int(torso_c[0] + side/2))
    ty1 = max(0, int(torso_c[1] - side/2)); ty2 = min(h-1, int(torso_c[1] + side/2))
    roi_torso = (tx1,ty1,tx2,ty2)

    # lower ROI centered at mid-hip if possible else torso
    # (use torso center for simplicity; can refine)
    lh_side_w = 1.5 * torso_s
    lh_side_h = 1.2 * torso_s
    lx1 = max(0, int(torso_c[0] - lh_side_w/2)); lx2 = min(w-1, int(torso_c[0] + lh_side_w/2))
    ly1 = max(0, int(torso_c[1] - lh_side_h/2 + 0.4*torso_s)); ly2 = min(h-1, int(torso_c[1] + lh_side_h/2 + 0.4*torso_s))
    roi_lower = (lx1,ly1,lx2,ly2)

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

        if vbg_ok:
            mean_v = np.mean(v_bg, axis=0)
            bg_coh = float(np.linalg.norm(mean_v) / (np.mean(np.linalg.norm(v_bg, axis=1)) + 1e-6))

        # torso ROI
        pts_t = sample_points_in_box(*roi_torso, max_points=cfg.max_flow_points, margin=2)
        p0_t, p1_t, v_t = lk_flow(prev_gray, curr_gray, pts_t, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_t is not None and vbg_ok:
            v_t2 = v_t - vbg_med.reshape(1,2)
            trans_signed_torso, trans_torso, div_torso, div_ratio_torso = radial_tangential_stats(p0_t, v_t2, torso_c)
            flow_ok = 1.0

        # lower ROI — use its own center for decomposition so forward leg motion
        # is correctly classified as radial (approaching), not tangential
        lower_c = np.array([(lx1+lx2)/2.0, (ly1+ly2)/2.0], dtype=np.float32)
        pts_l = sample_points_in_box(*roi_lower, max_points=cfg.max_flow_points, margin=2)
        p0_l, p1_l, v_l = lk_flow(prev_gray, curr_gray, pts_l, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_l is not None and vbg_ok:
            v_l2 = v_l - vbg_med.reshape(1,2)
            trans_signed_low, trans_low, div_low, div_ratio_low = radial_tangential_stats(p0_l, v_l2, lower_c)
            flow_ok = 1.0

    # New posture features
    torso_compression = 0.0
    wrist_height_asym = 0.0
    face_visibility = 0.0
    v_torso_compression = 0.0
    v_wrist_asym = 0.0
    v_face_vis = 1.0  # always valid (uses confidence scores)

    # shoulder-to-hip distance / shoulder_width
    has_sh = kp_valid(kps, COCO17["l_shoulder"], cfg.kp_conf_thresh) and kp_valid(kps, COCO17["r_shoulder"], cfg.kp_conf_thresh)
    has_hp = kp_valid(kps, COCO17["l_hip"], cfg.kp_conf_thresh) and kp_valid(kps, COCO17["r_hip"], cfg.kp_conf_thresh)
    if has_sh and has_hp and v_sh_w > 0.5:
        sh_mid = (get_point(kps, COCO17["l_shoulder"]) + get_point(kps, COCO17["r_shoulder"])) / 2.0
        hp_mid = (get_point(kps, COCO17["l_hip"]) + get_point(kps, COCO17["r_hip"])) / 2.0
        sh_hp_dist = float(np.linalg.norm(sh_mid - hp_mid) / norm)
        torso_compression = sh_hp_dist / (d_sh_w + 1e-6)
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

    # Energy ratio (computed in dataset.py from keypoint velocities)
    energy_ratio = 0.0

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

        # dynamics placeholders (45-47)
        max_wrist_vel, max_wrist_accel, energy_ratio
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

    debug = {
        "bbox_valid": bbox_valid,
        "torso_ok": float(torso_ok),
        "flow_ok": flow_ok,
        "div_torso": div_torso,
        "div_low": div_low,
        "bg_coh": bg_coh,
        "torso_compression": torso_compression,
        "wrist_height_asym": wrist_height_asym
    }
    return x, m, debug, curr_gray

def add_interaction_features(x: np.ndarray, m: np.ndarray, wrist_vel: float = 0.0, wrist_accel: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add interaction features (motion × proximity) to a single feature vector.
    Used during online inference when temporal derivatives are available from previous frames.

    Args:
        x: feature vector [48] from build_features
        m: mask vector [48]
        wrist_vel: max wrist extension velocity (computed externally from frame history)
        wrist_accel: max wrist extension acceleration (computed externally from frame history)

    Returns:
        x_aug: augmented feature vector [51]
        m_aug: augmented mask vector [51]
    """
    # Extract base features
    log_area = x[9]
    trans_signed_torso = x[32]   # signed: positive=approaching, negative=retreating
    divergence_torso = x[34]     # torso divergence

    # Proximity weight
    proximity_weight = np.exp(log_area * 0.1)

    # Use signed translation so retreating person gets NEGATIVE approach_proximity
    approach_proximity = trans_signed_torso * proximity_weight
    expansion_proximity = divergence_torso * proximity_weight
    acceleration_proximity = wrist_accel * proximity_weight

    # Augment feature vector
    flow_ok_val = m[41]  # flow_ok at index 41
    x_aug = np.concatenate([x, [approach_proximity, expansion_proximity, acceleration_proximity]])
    m_aug = np.concatenate([m, [flow_ok_val, flow_ok_val, flow_ok_val]])

    return x_aug, m_aug
