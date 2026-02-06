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
    e_torso=t_torso=R_torso=magp_torso=0.0
    e_low=t_low=R_low=magp_low=0.0
    bg_mag=0.0
    bg_coh=0.0
    flow_ok = 0.0

    if prev_gray is not None:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.lk_criteria_count, cfg.lk_criteria_eps)

        pts_bg = sample_points_background(h,w,roi_person,max_points=cfg.max_flow_points)
        p0_bg, p1_bg, v_bg = lk_flow(prev_gray, curr_gray, pts_bg, cfg.lk_win_size, cfg.lk_max_level, crit)
        vbg_med, vbg_ok = median_flow(v_bg)

        if vbg_ok:
            bg_mag = float(np.median(np.linalg.norm(v_bg, axis=1)))
            mean_v = np.mean(v_bg, axis=0)
            bg_coh = float(np.linalg.norm(mean_v) / (np.mean(np.linalg.norm(v_bg, axis=1)) + 1e-6))

        # torso ROI
        pts_t = sample_points_in_box(*roi_torso, max_points=cfg.max_flow_points, margin=2)
        p0_t, p1_t, v_t = lk_flow(prev_gray, curr_gray, pts_t, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_t is not None and vbg_ok:
            v_t2 = v_t - vbg_med.reshape(1,2)
            e_torso, t_torso, R_torso, magp_torso = radial_tangential_stats(p0_t, v_t2, torso_c)
            flow_ok = 1.0

        # lower ROI
        pts_l = sample_points_in_box(*roi_lower, max_points=cfg.max_flow_points, margin=2)
        p0_l, p1_l, v_l = lk_flow(prev_gray, curr_gray, pts_l, cfg.lk_win_size, cfg.lk_max_level, crit)
        if v_l is not None and vbg_ok:
            v_l2 = v_l - vbg_med.reshape(1,2)
            e_low, t_low, R_low, magp_low = radial_tangential_stats(p0_l, v_l2, torso_c)
            flow_ok = 1.0

    # motion energies from pose (computed later in dataset with derivatives preferred), placeholder 0 here
    upper_energy = 0.0
    lower_energy = 0.0
    ratio_ul = 0.0

    # Assemble feature vector (values) + validity mask for pose/flow parts
    # Note: bbox features validity depends on cropping.
    bbox_valid = 0.0 if anyc else 1.0

    x = np.array([
        # reliability
        det_conf, kp_conf_mean, kp_conf_min, visible_kp_count,
        float(anyc), float(cl), float(cr), float(ct), float(cb),
        float(track_age),

        # approach / bbox
        log_area, 0.0, 0.0,  # dlog_area_dt, d2log_area_dt2 computed later
        cx_n, cy_n, dcx_n, dcy_n, aspect,

        # upper geometry
        d_wl_sl, d_wr_sr, d_wl_tc, d_wr_tc, d_sh_w, d_hip_w,
        a_el_l, a_el_r,

        # lower geometry
        a_kn_l, a_kn_r, d_al_hl, d_ar_hr, d_al_tc, d_ar_tc, d_st,

        # flow torso/lower + bg
        e_torso, t_torso, R_torso, magp_torso,
        e_low, t_low, R_low, magp_low,
        bg_mag, bg_coh, flow_ok,

        # placeholders for energies (optional)
        upper_energy, lower_energy, ratio_ul
    ], dtype=np.float32)

    # validity mask aligned to x for the parts that can be missing:
    # For simplicity: mark pose-derived scalars with their computed valid flags; bbox_valid; flow_ok.
    m = np.ones_like(x, dtype=np.float32)
    # bbox derivatives placeholders will be valid only if bbox_valid
    m[10:18] = bbox_valid  # log_area..aspect
    # Upper pose valids
    pose_valids = [v_wl_sl,v_wr_sr,v_wl_tc,v_wr_tc,v_sh_w,v_hip_w,v_el_l,v_el_r]
    m[18:26] = np.array(pose_valids, dtype=np.float32)
    # Lower pose valids
    lower_valids = [v_kn_l,v_kn_r,v_al_hl,v_ar_hr,v_al_tc,v_ar_tc,v_st]
    m[26:33] = np.array(lower_valids, dtype=np.float32)
    # Flow
    m[33:44] = flow_ok

    debug = {
        "bbox_valid": bbox_valid,
        "torso_ok": float(torso_ok),
        "flow_ok": flow_ok,
        "R_torso": R_torso,
        "R_low": R_low,
        "bg_mag": bg_mag
    }
    return x, m, debug, curr_gray
