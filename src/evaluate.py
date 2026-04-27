import os
import json
import math
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
import cv2
from collections import deque
from .config import Config, FEATURE_NAMES
from .model import HazardGRU, HazardLSTM, HazardTransformer, HazardCNN
from .pose_detector import PoseDetector
from .tracker import SingleTargetTracker
from .features import build_features, add_interaction_features, compute_torso_height_frac, TemporalDerivatives
from .dataset import extract_sequences, windowize
from .feature_importance import compute_permutation_importance, print_importance_ranking, save_importance_results
from .utils import set_seed, print_seed_info

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def evaluate_video(video_path, ground_truth, model, detector, cfg, device, threshold, debug=False, debug_before_frame=None):
    """
    Evaluate single video.

    Returns:
        dict with detection info including multi-level warning thresholds
    """
    # Seed RNG so each video gets identical point sampling regardless of
    # processing order.
    set_seed(seed=42, deterministic=True)

    tracker = SingleTargetTracker()
    buf = deque(maxlen=cfg.window_len)
    prev_gray = None
    prev_bbox = None

    deriv = TemporalDerivatives()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_idx = 0
    detections = []  # List of (frame, hazard_score)
    dt = cfg.step_dt

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % cfg.frame_stride != 0:
            frame_idx += 1
            continue

        # Resize to training resolution so pixel-magnitude features (log_scale,
        # log_area, flow magnitudes) match the scale the GRU was trained on.
        # Mirrors the resize added to infer.py — must be kept in sync.
        if frame.shape[1] != cfg.infer_w or frame.shape[0] != cfg.infer_h:
            frame = cv2.resize(frame, (cfg.infer_w, cfg.infer_h))

        det = detector.infer(frame)
        if det is None:
            frame_idx += 1
            continue

        bbox, track_age, lost = tracker.update(det["bbox"])
        det["bbox"] = bbox

        x, m, dbg, prev_gray = build_features(frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
        prev_bbox = bbox

        # Compute temporal derivatives (dlog_area_dt, d2log_area_dt2, wrist
        # vel/accel, dlog_scale_dt) — shared implementation prevents desync.
        dlog_scale_dt = deriv.update(x, m, dt)

        # Add interaction features (approach_rate uses dlog_scale_dt + trans_signed_torso)
        x, m = add_interaction_features(x, m, x[45], x[46], dlog_scale_dt, cfg=cfg)

        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        buf.append(xm)

        if len(buf) == cfg.window_len:
            inp = torch.from_numpy(np.stack(buf)[None,:,:]).to(device)
            with torch.no_grad():
                hazard = float(model(inp).item())

            # Distance gate removed — mirrors infer.py (torso_height_frac kept for diagnostics only).
            torso_height_frac = compute_torso_height_frac(
                det["kps"].astype(np.float32), frame.shape[0], cfg.kp_conf_thresh, bbox)
            detections.append((frame_idx, hazard))

            in_debug_window = debug_before_frame is None or frame_idx < debug_before_frame
            if debug and hazard >= threshold and in_debug_window:
                kps = det["kps"].astype(np.float32)
                log_scale_val = float(x[47])
                log_scale_ok  = int(float(m[47]) > 0.5)
                print(f"  frame={frame_idx:4d}  hazard={hazard:.3f}  "
                      f"thf={torso_height_frac:.3f}  "
                      f"ls={log_scale_val:.2f}(ok={log_scale_ok})  "
                      f"bbox_valid={dbg['bbox_valid']:.0f}  "
                      f"torso_ok={dbg['torso_ok']:.0f}  "
                      f"flow_ok={dbg['flow_ok']:.0f}")
                print(f"         kps: c_lsh={kps[5,2]:.2f}  c_rsh={kps[6,2]:.2f}  "
                      f"c_lhp={kps[11,2]:.2f}  c_rhp={kps[12,2]:.2f}  "
                      f"(kp_thr={cfg.kp_conf_thresh})")
                print(f"         flow: div_torso={dbg['div_torso']:.3f}  "
                      f"div_low={dbg['div_low']:.3f}  "
                      f"bg_coh={dbg['bg_coh']:.3f}  "
                      f"expansion_prox={x[57]:.3f}  "
                      f"approach_rate={x[56]:.3f}  "
                      f"torso_ht_px={x[48]:.1f}(ok={int(float(m[48])>0.5)})")

        frame_idx += 1

    cap.release()

    if not detections:
        return {
            'first_detection_frame': -1,
            'first_hazard': 0.0,
            'max_hazard': 0.0,
            'detections': [],
            'first_precontact_frame': -1,
            'first_high_frame': -1,
            'first_critical_frame': -1
        }

    # Find first detection above primary threshold.
    first_detection_frame = -1
    first_hazard = 0.0
    for frame, hazard in detections:
        if hazard >= threshold:
            first_detection_frame = frame
            first_hazard = hazard
            break

    # Find first THREAT detection frame.
    first_precontact_frame = -1
    for frame, hazard in detections:
        if first_precontact_frame == -1 and hazard >= threshold:
            first_precontact_frame = frame

    max_hazard = max(h for _, h in detections)

    return {
        'first_detection_frame': first_detection_frame,
        'first_hazard': first_hazard,
        'max_hazard': max_hazard,
        'detections': detections,
        'first_precontact_frame': first_precontact_frame,
    }

def main(holdout_dir: str, debug: bool = False):
    # Setup logging
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/logs/evaluate_{timestamp}.log"
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")

    # Set random seed for reproducibility (CRITICAL: must be before feature extraction)
    set_seed(seed=42, deterministic=True)
    print_seed_info(seed=42, deterministic=True)
    print()

    cfg = Config.for_pc()
    detector = PoseDetector(cfg)

    # Load model and best threshold
    with open("outputs/checkpoints/meta.json") as f:
        meta = json.load(f)
        input_dim = meta["input_dim"]
        threshold = meta.get("best_threshold", cfg.early_thresh)
        model_file = meta.get("model_file", "hazard_gru.pt")
        model_type = meta.get("model_type", "gru")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "lstm":
        model = HazardLSTM(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    elif model_type == "transformer":
        model = HazardTransformer(input_dim=input_dim, d_model=cfg.gru_hidden, dropout=0.0)
    elif model_type == "cnn":
        model = HazardCNN(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    else:
        model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(torch.load(f"outputs/checkpoints/{model_file}", map_location=device))
    model.eval()

    # Load labels for attack videos from holdout/labels.json
    labels_file = os.path.join(holdout_dir, "labels.json")
    with open(labels_file) as f:
        labels = json.load(f)

    print("=" * 80)
    print("HOLDOUT EVALUATION - Pre-contact Detection")
    print(f"Holdout directory: {holdout_dir}")
    print(f"Using threshold: {threshold:.2f} (from training optimization)")
    print("=" * 80)

    # Evaluate all videos
    attack_results = []
    safe_results = []

    # Process attack videos from labels.json (in 'attack' subdirectory only)
    print("\n--- Processing Attack Videos ---")
    attack_dir = os.path.join(holdout_dir, 'attack')
    if not os.path.exists(attack_dir):
        print(f"Warning: {attack_dir} not found")
    else:
        for video_name, gt in labels.items():
            video_path = os.path.join(attack_dir, video_name)

            if not os.path.exists(video_path):
                print(f"Warning: {video_name} not found in attack/, skipping")
                continue

            onset_frame = gt.get('onset_frame', 0)
            print(f"Processing {video_name}...", end=" ")
            result = evaluate_video(video_path, gt, model, detector, cfg, device, threshold,
                                    debug=debug, debug_before_frame=onset_frame)

            if result is None:
                print("FAILED (cannot open)")
                continue

            result['video_name'] = video_name
            result['ground_truth'] = gt
            attack_results.append(result)

            detected = result['first_detection_frame'] >= 0
            status = "DETECTED" if detected else "MISSED"
            print(f"{status} (max_hazard={result['max_hazard']:.3f})")

            # For MISSED cases with --debug: the first pass only showed pre-onset
            # frames (debug_before_frame=onset_frame), so max_hazard may come from
            # post-onset frames with no debug output.  Re-run without the frame
            # restriction so the user can see when/why the model eventually fired
            # and which frames were gated.  Result is discarded (diagnostics only).
            if debug and not detected:
                print(f"  [MISSED debug: onset@{onset_frame} — re-running, "
                      f"showing ALL frames ≥ threshold]")
                evaluate_video(video_path, gt, model, detector, cfg, device,
                               threshold, debug=True, debug_before_frame=None)

    # Process safe videos from 'safe' subdirectory only
    print("\n--- Processing Safe Videos ---")
    safe_dir = os.path.join(holdout_dir, 'safe')
    if not os.path.exists(safe_dir):
        print(f"Warning: {safe_dir} not found")
    else:
        for video_file in os.listdir(safe_dir):
            if not video_file.endswith('.mp4'):
                continue

            video_path = os.path.join(safe_dir, video_file)
            gt = {'category': 'safe'}

            print(f"Processing {video_file}...", end=" ")
            result = evaluate_video(video_path, gt, model, detector, cfg, device, threshold, debug=debug)

            if result is None:
                print("FAILED (cannot open)")
                continue

            result['video_name'] = video_file
            result['ground_truth'] = gt
            safe_results.append(result)
            detected = result['first_detection_frame'] >= 0
            print(f"{'FP' if detected else 'OK'} (max_hazard={result['max_hazard']:.3f})")

    print("\n" + "=" * 80)
    print("ATTACK VIDEOS — Threat Detection Analysis")
    print("=" * 80)

    detected_attacks = []
    missed_attacks = []
    false_positives_attacks = []
    lead_times = []

    for r in attack_results:
        gt = r['ground_truth']
        attack_frame = gt['attack_frame']
        onset_frame = gt.get('onset_frame', 0)
        first_det = r['first_detection_frame']

        if first_det >= 0:
            # Check if detection is before onset_frame (False Positive)
            if first_det < onset_frame:
                false_positives_attacks.append(r)
                print(f"{r['video_name']:20s} | FP @ {first_det:4d} (before onset@{onset_frame:4d}) first={r['first_hazard']:.3f} max={r['max_hazard']:.3f}")
            else:
                detected_attacks.append(r)
                lead_time = attack_frame - first_det
                lead_times.append(lead_time)

                if first_det < attack_frame:
                    status = "LEAD"
                elif first_det == attack_frame:
                    status = "ON-TIME"
                else:
                    status = "LATE"

                print(f"{r['video_name']:20s} | Onset@{onset_frame:4d} Attack@{attack_frame:4d} "
                      f"Detect@{first_det:4d} Lead={lead_time:+4d} [{status}] | "
                      f"first={r['first_hazard']:.3f} max={r['max_hazard']:.3f}")
        else:
            missed_attacks.append(r)
            print(f"{r['video_name']:20s} | MISSED (max_hazard={r['max_hazard']:.3f})")

    detection_rate = len(detected_attacks) / max(1, len(attack_results))
    miss_rate = len(missed_attacks) / max(1, len(attack_results))
    fp_attack_rate = len(false_positives_attacks) / max(1, len(attack_results))

    print(f"\nDetection Rate : {detection_rate:.1%} ({len(detected_attacks)}/{len(attack_results)})")
    print(f"Missed Rate    : {miss_rate:.1%} ({len(missed_attacks)}/{len(attack_results)})")
    print(f"FP (pre-onset) : {fp_attack_rate:.1%} ({len(false_positives_attacks)}/{len(attack_results)})")

    if lead_times:
        pre_contact_warnings = [lt for lt in lead_times if lt > 0]
        pre_contact_warning_rate = len(pre_contact_warnings) / max(1, len(lead_times))

        print(f"\n--- THREAT Detection (threshold={threshold:.2f}) ---")
        print(f"Pre-contact Rate   : {pre_contact_warning_rate:.1%} ({len(pre_contact_warnings)}/{len(lead_times)} detected attacks)")
        print(f"  Mean Lead Time   : {np.mean(lead_times):.1f} frames ({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Median Lead Time : {np.median(lead_times):.1f} frames ({np.median(lead_times)/cfg.input_fps:.2f}s)")
        if pre_contact_warnings:
            print(f"  Pre-contact only : {np.mean(pre_contact_warnings):.1f} frames ({np.mean(pre_contact_warnings)/cfg.input_fps:.2f}s)")

    print("\n" + "=" * 80)
    print("SAFE VIDEOS")
    print("=" * 80)

    false_positives = []
    true_negatives = []

    for r in safe_results:
        first_det = r['first_detection_frame']
        if first_det >= 0:
            false_positives.append(r)
            print(f"{r['video_name']:20s} | FP @ frame {first_det} (max_hazard={r['max_hazard']:.3f})")
        else:
            true_negatives.append(r)

    fp_rate = len(false_positives) / max(1, len(safe_results))
    tn_rate = len(true_negatives) / max(1, len(safe_results))

    print(f"\nFalse Positive Rate (safe) : {fp_rate:.1%} ({len(false_positives)}/{len(safe_results)})")
    print(f"True Negative Rate         : {tn_rate:.1%} ({len(true_negatives)}/{len(safe_results)})")

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total Videos : {len(attack_results) + len(safe_results)}  "
          f"(attacks={len(attack_results)}  safe={len(safe_results)})")
    print(f"\nThreshold (THREAT) : {threshold:.2f}")
    print(f"\nAttack Detection  : {detection_rate:.1%}")
    print(f"FP (safe)         : {fp_rate:.1%}")
    if fp_attack_rate > 0:
        print(f"FP (pre-onset)    : {fp_attack_rate:.1%}")
    if lead_times:
        print(f"\nDetection Performance:")
        print(f"  Pre-contact : {pre_contact_warning_rate:.1%} | "
              f"Mean Lead: {np.mean(lead_times):.1f} frames "
              f"({np.mean(lead_times)/cfg.input_fps:.2f}s)")
    print("=" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    # WINDOW-LEVEL ANALYSIS: PR curve + Feature Importance on holdout data
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("WINDOW-LEVEL ANALYSIS (holdout set)")
    print("=" * 80)
    print("Building window-level dataset from holdout videos...")

    # Re-seed for deterministic feature extraction
    set_seed(seed=42, deterministic=True)

    all_Xw, all_Mw, all_yw = [], [], []

    # Attack videos — holdout videos are NOT trimmed; they contain normal
    # frames before onset.  Extract all frames as "safe" (y=0) first, then
    # relabel frames from onset_frame onward as attack (y=1).
    # Frame indices in labels.json refer to raw 30fps frames; after
    # frame_stride downsampling they map to step = raw_frame // frame_stride.
    if os.path.exists(attack_dir):
        for video_name, gt in labels.items():
            video_path = os.path.join(attack_dir, video_name)
            if not os.path.exists(video_path):
                continue
            X, M, y = extract_sequences(video_path, "safe", detector, cfg)
            if X.shape[0] == 0:
                continue
            # Relabel: frames from onset onward are attack
            onset_frame = gt.get('onset_frame', 0)
            onset_step = onset_frame // cfg.frame_stride
            y[onset_step:] = 1.0
            Xw, Mw, yw = windowize(X, M, y, cfg.window_len, stride=1)
            if Xw.shape[0] > 0:
                all_Xw.append(Xw)
                all_Mw.append(Mw)
                all_yw.append(yw)

    # Safe videos
    if os.path.exists(safe_dir):
        for video_file in os.listdir(safe_dir):
            if not video_file.lower().endswith('.mp4'):
                continue
            video_path = os.path.join(safe_dir, video_file)
            X, M, y = extract_sequences(video_path, "safe", detector, cfg)
            if X.shape[0] == 0:
                continue
            Xw, Mw, yw = windowize(X, M, y, cfg.window_len,
                                   stride=cfg.safe_window_stride)
            if Xw.shape[0] > 0:
                all_Xw.append(Xw)
                all_Mw.append(Mw)
                all_yw.append(yw)

    if not all_Xw:
        print("No windows extracted — skipping window-level analysis.")
    else:
        Xw_all = np.concatenate(all_Xw)
        Mw_all = np.concatenate(all_Mw)
        yw_all = np.concatenate(all_yw)

        # Concatenate features + masks → model input
        holdout_input = np.concatenate([Xw_all, Mw_all], axis=2).astype(np.float32)

        n_attack_w = int((yw_all >= 0.5).sum())
        n_safe_w = int((yw_all < 0.5).sum())
        print(f"  Total windows: {len(yw_all)} (attack={n_attack_w}, safe={n_safe_w})")

        # ── Collect predictions ──────────────────────────────────────────────
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

        # ── Window-level metrics at selected threshold ───────────────────────
        from sklearn.metrics import (precision_recall_fscore_support,
                                     accuracy_score,
                                     precision_recall_curve,
                                     average_precision_score,
                                     roc_curve, auc)

        y_pred = (y_score >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        n_neg = max(1, int((y_true == 0).sum()))
        fp_count = int(((y_pred == 1) & (y_true == 0)).sum())
        fpr_val = fp_count / n_neg

        print(f"\n--- Window-Level Metrics @ threshold={threshold:.2f} ---")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  FPR       : {fpr_val:.2%}")

        # ── PR Curve ─────────────────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs("outputs/plots", exist_ok=True)

            # Pre-compute per-threshold metrics
            n_neg = max(1, int((y_true == 0).sum()))
            marker_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
            marker_metrics = {}
            for t in marker_thresholds:
                y_pred = (y_score >= t).astype(int)
                p, r, f, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0)
                fp_count = int(((y_pred == 1) & (y_true == 0)).sum())
                marker_metrics[t] = (p, r, f, fp_count / n_neg)

            # --- Precision-Recall curve ---
            prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)

            fig, ax = plt.subplots(figsize=(8, 6))

            # F1 iso-curves
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

            # Threshold markers
            table_lines = []
            all_markers = list(marker_thresholds)
            if threshold not in all_markers:
                all_markers.append(threshold)
                all_markers.sort()
                y_sel = (y_score >= threshold).astype(int)
                p_s, r_s, f_s, _ = precision_recall_fscore_support(
                    y_true, y_sel, average='binary', zero_division=0)
                fp_s = int(((y_sel == 1) & (y_true == 0)).sum())
                marker_metrics[threshold] = (p_s, r_s, f_s, fp_s / n_neg)

            marker_items = []
            for t in all_markers:
                p_m, r_m, f_m, fpr_m = marker_metrics[t]
                if r_m == 0 and p_m == 0:
                    continue
                is_selected = abs(t - threshold) < 0.005
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
            ax.set_title(f"Precision-Recall Curve (Within-Domain Test Set) - {model_type.upper()}", fontsize=12)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            pr_path = f"outputs/plots/pr_curve_holdout_{timestamp}.png"
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
                    dist = abs(rt - t)
                    if dist < best_dist:
                        best_dist = dist
                        fpr_plot, tpr_m = rfp, rtp
                is_selected = abs(t - threshold) < 0.005
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
            ax.set_title(f"ROC Curve (Within-Domain Test Set) - {model_type.upper()}", fontsize=12)
            ax.set_xlim([0, 1.02])
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            roc_path = f"outputs/plots/roc_curve_holdout_{timestamp}.png"
            fig.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ROC curve saved to : {roc_path}  (AUC={roc_auc:.3f})")

            # Print threshold comparison table
            print(f"\n  Threshold comparison (holdout set, n_neg={n_neg}):")
            print(f"  {'thresh':>6s}   {'Prec':>5s}   {'Rec':>5s}   {'F1':>5s}   {'FP%':>6s}  note")
            print(f"  {'-'*46}")
            for t in all_markers:
                p_m, r_m, f_m, fpr_m = marker_metrics[t]
                note = " ★ selected" if abs(t - threshold) < 0.005 else ""
                print(f"  {t:>6.2f}  {p_m:.3f}  {r_m:.3f}  {f_m:.3f}  {fpr_m:>6.2%}{note}")

        except ImportError as e:
            print(f"\n  Skipping PR/ROC curves: {e}")

        # ── Feature Importance ───────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("COMPUTING FEATURE IMPORTANCE (holdout set)")
        print("=" * 80)

        from torch.utils.data import TensorDataset, DataLoader

        holdout_dataset = TensorDataset(
            torch.from_numpy(holdout_input).float(),
            torch.from_numpy(yw_all).float())
        holdout_loader = DataLoader(holdout_dataset, batch_size=64, shuffle=False)

        importances, baseline_f1 = compute_permutation_importance(
            model=model,
            val_loader=holdout_loader,
            threshold=threshold,
            device=device,
            n_repeats=5,
        )

        print_importance_ranking(importances, baseline_f1)

        importance_output = f"outputs/logs/feature_importance_holdout_{timestamp}.json"
        save_importance_results(importances, baseline_f1, importance_output)

    print(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")

    logger.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-contact detection model on holdout set.")
    parser.add_argument("holdout_dir", help="Path to holdout directory (must contain attack/ and safe/ subdirs)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame gate diagnostics for FP frames (safe videos: all above threshold; "
                             "attack videos: above threshold before onset_frame)")
    args = parser.parse_args()
    main(args.holdout_dir, debug=args.debug)
