import numpy as np

COCO17 = {
    "nose":0,"l_eye":1,"r_eye":2,"l_ear":3,"r_ear":4,
    "l_shoulder":5,"r_shoulder":6,"l_elbow":7,"r_elbow":8,
    "l_wrist":9,"r_wrist":10,"l_hip":11,"r_hip":12,
    "l_knee":13,"r_knee":14,"l_ankle":15,"r_ankle":16
}

def kp_valid(kps, idx, thr):
    return kps[idx,2] >= thr

def get_point(kps, idx):
    return kps[idx,:2].astype(np.float32)

def midpoint(a, b):
    return (a + b) * 0.5

def torso_center_and_scale(kps, thr):
    has_ls = kp_valid(kps, COCO17["l_shoulder"], thr)
    has_rs = kp_valid(kps, COCO17["r_shoulder"], thr)
    has_lh = kp_valid(kps, COCO17["l_hip"], thr)
    has_rh = kp_valid(kps, COCO17["r_hip"], thr)

    if has_ls and has_rs:
        ms = midpoint(get_point(kps, COCO17["l_shoulder"]), get_point(kps, COCO17["r_shoulder"]))
        ms_ok = True
    elif has_ls:
        ms = get_point(kps, COCO17["l_shoulder"]); ms_ok = True
    elif has_rs:
        ms = get_point(kps, COCO17["r_shoulder"]); ms_ok = True
    else:
        ms_ok = False

    if has_lh and has_rh:
        mh = midpoint(get_point(kps, COCO17["l_hip"]), get_point(kps, COCO17["r_hip"]))
        mh_ok = True
    elif has_lh:
        mh = get_point(kps, COCO17["l_hip"]); mh_ok = True
    elif has_rh:
        mh = get_point(kps, COCO17["r_hip"]); mh_ok = True
    else:
        mh_ok = False

    if ms_ok and mh_ok:
        c = midpoint(ms, mh)
        scale = float(np.linalg.norm(ms - mh))
        return c, max(scale, 1e-6), True
    return None, None, False

def angle(a, b, c):
    # angle ABC in radians
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + 1e-6
    nbc = np.linalg.norm(bc) + 1e-6
    cosang = float(np.clip(np.dot(ba, bc) / (nba*nbc), -1.0, 1.0))
    return float(np.arccos(cosang))
