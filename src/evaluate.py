import os
import json
import sys
import argparse
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
from .utils import set_seed, print_seed_info, TeeLogger

def evaluate_video(video_path, ground_truth, model, detector, cfg, device, threshold, debug=False, debug_before_frame=None):
    """
    Evaluate a single video.

    Returns:
        dict with first_detection_frame, first_hazard, max_hazard, detections
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
        # log_area, flow magnitudes) match the scale the model was trained on.
        # Mirrors the resize added to infer.py - must be kept in sync.
        if frame.shape[1] != cfg.infer_w or frame.shape[0] != cfg.infer_h:
            frame = cv2.resize(frame, (cfg.infer_w, cfg.infer_h))

        det = detector.infer(frame)
        if det is None:
            frame_idx += 1
            continue

        bbox = tracker.update(det["bbox"])
        det["bbox"] = bbox

        x, m, dbg, prev_gray = build_features(frame, prev_gray, prev_bbox, det, cfg)
        prev_bbox = bbox

        # Compute temporal derivatives (dlog_area_dt, d2log_area_dt2, wrist
        # vel/accel, dlog_scale_dt) - shared implementation prevents desync.
        dlog_scale_dt = deriv.update(x, m, dt)

        # Add interaction features (approach_rate uses dlog_scale_dt + trans_signed_torso)
        x, m = add_interaction_features(x, m, x[45], x[46], dlog_scale_dt, cfg=cfg)

        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        buf.append(xm)

        if len(buf) == cfg.window_len:
            inp = torch.from_numpy(np.stack(buf)[None,:,:]).to(device)
            with torch.no_grad():
                hazard = float(model(inp).item())

            # Distance gate removed - mirrors infer.py (torso_height_frac kept for diagnostics only).
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
        }

    # Find first detection above primary threshold.
    first_detection_frame = -1
    first_hazard = 0.0
    for frame, hazard in detections:
        if hazard >= threshold:
            first_detection_frame = frame
            first_hazard = hazard
            break

    max_hazard = max(h for _, h in detections)

    return {
        'first_detection_frame': first_detection_frame,
        'first_hazard': first_hazard,
        'max_hazard': max_hazard,
        'detections': detections,
    }

def main(holdout_dir: str, debug: bool = False):
    # Setup logging
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/logs/evaluate_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger

    print(f"Evaluation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  log -> {log_file}")

    # Seed before feature extraction: the optical-flow point sampling in
    # flow.py draws from the global NumPy RNG.
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
    else:
        model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(torch.load(f"outputs/checkpoints/{model_file}", map_location=device))
    model.eval()

    # Load labels for attack videos from holdout/labels.json
    labels_file = os.path.join(holdout_dir, "labels.json")
    with open(labels_file) as f:
        labels = json.load(f)

    print(f"\nHoldout evaluation: {holdout_dir}  threshold={threshold:.2f}")

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
                print(f"  [MISSED debug: onset@{onset_frame} - re-running, "
                      f"showing ALL frames >= threshold]")
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

    print("\n-- Attack videos - threat detection --")

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

    print("\n-- Safe videos --")

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

    print("\n-- Overall summary --")
    print(f"Total Videos : {len(attack_results) + len(safe_results)}  "
          f"(attacks={len(attack_results)}  safe={len(safe_results)})")
    print(f"Threshold (THREAT) : {threshold:.2f}")
    print(f"Attack Detection  : {detection_rate:.1%}")
    print(f"FP (safe)         : {fp_rate:.1%}")
    if fp_attack_rate > 0:
        print(f"FP (pre-onset)    : {fp_attack_rate:.1%}")
    if lead_times:
        print(f"Detection Performance:")
        print(f"  Pre-contact : {pre_contact_warning_rate:.1%} | "
              f"Mean Lead: {np.mean(lead_times):.1f} frames "
              f"({np.mean(lead_times)/cfg.input_fps:.2f}s)")

    # Window-level analysis: PR curve + feature importance on holdout
    print("\n-- Window-level analysis (holdout set) --")
    print("Building window-level dataset from holdout videos...")

    # Re-seed for deterministic feature extraction
    set_seed(seed=42, deterministic=True)

    all_Xw, all_Mw, all_yw = [], [], []

    # Attack videos - holdout videos are NOT trimmed; they contain normal
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
        print("No windows extracted - skipping window-level analysis.")
    else:
        Xw_all = np.concatenate(all_Xw)
        Mw_all = np.concatenate(all_Mw)
        yw_all = np.concatenate(all_yw)

        # Concatenate features + masks -> model input
        holdout_input = np.concatenate([Xw_all, Mw_all], axis=2).astype(np.float32)

        n_attack_w = int((yw_all >= 0.5).sum())
        n_safe_w = int((yw_all < 0.5).sum())
        print(f"  Total windows: {len(yw_all)} (attack={n_attack_w}, safe={n_safe_w})")

        # Collect predictions
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

        # Window-level metrics at selected threshold
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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

        try:
            from . import _plot
            _plot.plot_pr_and_roc(
                y_true, y_score,
                selected_threshold=threshold,
                model_type=model_type,
                title_dataset="Within-Domain Test Set",
                pr_path=f"outputs/plots/pr_curve_holdout_{timestamp}.png",
                roc_path=f"outputs/plots/roc_curve_holdout_{timestamp}.png",
            )
        except ImportError as e:
            print(f"\n  Skipping PR/ROC curves: {e}")

        # Feature Importance
        print("\nComputing feature importance (holdout set)...")
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

    print(f"\nEvaluation done {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  log -> {log_file}")

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
