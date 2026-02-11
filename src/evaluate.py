import os
import json
import sys
from datetime import datetime
import numpy as np
import torch
import cv2
from collections import deque
from .config import Config
from .model import HazardGRU
from .pose_detector import PoseDetector
from .tracker import SingleTargetTracker
from .features import build_features, add_interaction_features
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

def evaluate_video(video_path, ground_truth, model, detector, cfg, device, threshold):
    """
    Evaluate single video.

    Returns:
        dict with detection info including multi-level warning thresholds
    """
    tracker = SingleTargetTracker()
    buf = deque(maxlen=cfg.window_len)
    prev_gray = None
    prev_bbox = None

    # Track wrist distances for velocity/acceleration computation
    prev_wrist_dist_l = None
    prev_wrist_dist_r = None
    prev_wrist_vel_l = 0.0
    prev_wrist_vel_r = 0.0

    # Track log_area for dlog_area_dt / d2log_area_dt2 (indices 10, 11)
    # build_features() leaves these as 0.0 placeholders; dataset.py fills them from the full
    # sequence. We must replicate that here frame-by-frame to avoid a train/eval mismatch.
    prev_log_area = None
    prev_dlog_area_dt = 0.0

    # Track log_scale for dlog_scale_dt computation (approach_rate)
    prev_log_scale = None

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

        det = detector.infer(frame)
        if det is None:
            frame_idx += 1
            continue

        bbox, track_age, lost = tracker.update(det["bbox"])
        det["bbox"] = bbox

        x, m, _, prev_gray = build_features(frame, prev_gray, prev_bbox, det, track_age, lost, cfg)
        prev_bbox = bbox

        # Compute temporal derivatives — must match dataset.py post-processing exactly,
        # since build_features() leaves x[10], x[11], x[45], x[46] as 0.0 placeholders.

        log_area = x[9]
        dist_l = x[19]  # dist_l_wrist_torso
        dist_r = x[20]  # dist_r_wrist_torso
        log_scale = x[47]  # pose-derived log apparent size

        # --- log_area derivatives (indices 10, 11) ---
        dlog_area_dt = 0.0
        d2log_area_dt2 = 0.0
        if prev_log_area is not None:
            dlog_area_dt = (log_area - prev_log_area) / dt
            d2log_area_dt2 = (dlog_area_dt - prev_dlog_area_dt) / dt
        prev_log_area = log_area
        prev_dlog_area_dt = dlog_area_dt
        x[10] = dlog_area_dt
        x[11] = d2log_area_dt2

        # --- wrist extension velocity / acceleration (indices 45, 46) ---
        wrist_vel = 0.0
        wrist_accel = 0.0
        if prev_wrist_dist_l is not None:
            vel_l = (dist_l - prev_wrist_dist_l) / dt
            vel_r = (dist_r - prev_wrist_dist_r) / dt
            wrist_vel = max(vel_l, vel_r)

            accel_l = (vel_l - prev_wrist_vel_l) / dt
            accel_r = (vel_r - prev_wrist_vel_r) / dt
            wrist_accel = max(accel_l, accel_r)

            prev_wrist_vel_l = vel_l
            prev_wrist_vel_r = vel_r
        prev_wrist_dist_l = dist_l
        prev_wrist_dist_r = dist_r
        x[45] = wrist_vel
        x[46] = wrist_accel

        # --- log_scale derivative for approach_rate (index 48) ---
        dlog_scale_dt = 0.0
        if prev_log_scale is not None:
            dlog_scale_dt = (log_scale - prev_log_scale) / dt
        prev_log_scale = log_scale

        # Add interaction features (approach_rate uses dlog_scale_dt + trans_signed_torso)
        x, m = add_interaction_features(x, m, wrist_vel, wrist_accel, dlog_scale_dt)

        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        buf.append(xm)

        if len(buf) == cfg.window_len:
            inp = torch.from_numpy(np.stack(buf)[None,:,:]).to(device)
            with torch.no_grad():
                hazard = float(model(inp).item())

            detections.append((frame_idx, hazard))

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

    # Find first detection above primary threshold
    first_detection_frame = -1
    first_hazard = 0.0
    for frame, hazard in detections:
        if hazard >= threshold:
            first_detection_frame = frame
            first_hazard = hazard
            break

    # Find first detection at each warning level
    first_precontact_frame = -1
    first_high_frame = -1
    first_critical_frame = -1

    for frame, hazard in detections:
        if first_precontact_frame == -1 and hazard >= threshold:
            first_precontact_frame = frame
        if first_high_frame == -1 and hazard >= cfg.high_thresh:
            first_high_frame = frame
        if first_critical_frame == -1 and hazard >= cfg.critical_thresh:
            first_critical_frame = frame

    max_hazard = max(h for _, h in detections)

    return {
        'first_detection_frame': first_detection_frame,
        'first_hazard': first_hazard,
        'max_hazard': max_hazard,
        'detections': detections,
        'first_precontact_frame': first_precontact_frame,
        'first_high_frame': first_high_frame,
        'first_critical_frame': first_critical_frame
    }

def main(holdout_dir: str):
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

    cfg = Config()
    detector = PoseDetector()

    # Load model and best threshold
    with open("outputs/checkpoints/meta.json") as f:
        meta = json.load(f)
        input_dim = meta["input_dim"]
        threshold = meta.get("best_threshold", cfg.early_thresh)  # Use tuned threshold or fallback to config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=0.0)
    model.load_state_dict(torch.load("outputs/checkpoints/hazard_gru.pt", map_location=device))
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

            print(f"Processing {video_name}...", end=" ")
            result = evaluate_video(video_path, gt, model, detector, cfg, device, threshold)

            if result is None:
                print("FAILED (cannot open)")
                continue

            result['video_name'] = video_name
            result['ground_truth'] = gt
            attack_results.append(result)

            detected = result['first_detection_frame'] >= 0
            status = "DETECTED" if detected else "MISSED"
            print(f"{status} (max_hazard={result['max_hazard']:.3f})")

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
            result = evaluate_video(video_path, gt, model, detector, cfg, device, threshold)

            if result is None:
                print("FAILED (cannot open)")
                continue

            result['video_name'] = video_file
            result['ground_truth'] = gt
            safe_results.append(result)
            detected = result['first_detection_frame'] >= 0
            print(f"{'FP' if detected else 'OK'} (max_hazard={result['max_hazard']:.3f})")
    
    print("\n" + "=" * 80)
    print("ATTACK VIDEOS - Multi-Level Warning Analysis")
    print("=" * 80)

    detected_attacks = []
    missed_attacks = []
    false_positives_attacks = []
    lead_times = []
    lead_times_high = []
    lead_times_critical = []

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

                # Track lead times for higher warning levels
                if r['first_high_frame'] >= onset_frame:
                    lead_times_high.append(attack_frame - r['first_high_frame'])
                if r['first_critical_frame'] >= onset_frame:
                    lead_times_critical.append(attack_frame - r['first_critical_frame'])

                if first_det < attack_frame:
                    status = "PRE-CONTACT"
                elif first_det == attack_frame:
                    status = "ON-TIME"
                else:
                    status = "LATE"

                # Build warning level string
                warning_info = f"PC@{r['first_precontact_frame']:4d}"
                if r['first_high_frame'] >= 0:
                    warning_info += f" H@{r['first_high_frame']:4d}"
                if r['first_critical_frame'] >= 0:
                    warning_info += f" C@{r['first_critical_frame']:4d}"

                print(f"{r['video_name']:20s} | Onset@{onset_frame:4d} Attack@{attack_frame:4d} Lead={lead_time:+4d} [{status}] | Warnings: {warning_info} | first={r['first_hazard']:.3f} max={r['max_hazard']:.3f}")
        else:
            missed_attacks.append(r)
            print(f"{r['video_name']:20s} | MISSED (max_hazard={r['max_hazard']:.3f})")

    detection_rate = len(detected_attacks) / max(1, len(attack_results))
    miss_rate = len(missed_attacks) / max(1, len(attack_results))
    fp_attack_rate = len(false_positives_attacks) / max(1, len(attack_results))

    print(f"\nDetection Rate: {detection_rate:.1%} ({len(detected_attacks)}/{len(attack_results)})")
    print(f"Missed Rate: {miss_rate:.1%} ({len(missed_attacks)}/{len(attack_results)})")
    print(f"False Positives (pre-onset detections): {fp_attack_rate:.1%} ({len(false_positives_attacks)}/{len(attack_results)})")

    if lead_times:
        pre_contact_warnings = [lt for lt in lead_times if lt > 0]
        pre_contact_warning_rate = len(pre_contact_warnings) / max(1, len(lead_times))

        print(f"\n--- PRE-CONTACT Level (threshold={threshold:.2f}) ---")
        print(f"Pre-contact Warning Rate: {pre_contact_warning_rate:.1%} ({len(pre_contact_warnings)}/{len(lead_times)} detected attacks)")
        print(f"  Mean Lead Time: {np.mean(lead_times):.1f} frames ({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        print(f"  Median Lead Time: {np.median(lead_times):.1f} frames ({np.median(lead_times)/cfg.input_fps:.2f}s)")
        if pre_contact_warnings:
            print(f"  Pre-contact warnings only: {np.mean(pre_contact_warnings):.1f} frames ({np.mean(pre_contact_warnings)/cfg.input_fps:.2f}s)")

        if lead_times_high:
            pre_contact_high = [lt for lt in lead_times_high if lt > 0]
            high_rate = len(lead_times_high) / max(1, len(detected_attacks))
            print(f"\n--- HIGH Level (threshold={cfg.high_thresh:.2f}) ---")
            print(f"HIGH Warning Rate: {high_rate:.1%} ({len(lead_times_high)}/{len(detected_attacks)} detected attacks)")
            print(f"  Mean Lead Time: {np.mean(lead_times_high):.1f} frames ({np.mean(lead_times_high)/cfg.input_fps:.2f}s)")
            print(f"  Median Lead Time: {np.median(lead_times_high):.1f} frames ({np.median(lead_times_high)/cfg.input_fps:.2f}s)")
            if pre_contact_high:
                high_warning_rate = len(pre_contact_high) / max(1, len(lead_times_high))
                print(f"  Pre-contact HIGH warnings: {high_warning_rate:.1%} ({len(pre_contact_high)}/{len(lead_times_high)})")

        if lead_times_critical:
            pre_contact_critical = [lt for lt in lead_times_critical if lt > 0]
            critical_rate = len(lead_times_critical) / max(1, len(detected_attacks))
            print(f"\n--- CRITICAL Level (threshold={cfg.critical_thresh:.2f}) ---")
            print(f"CRITICAL Warning Rate: {critical_rate:.1%} ({len(lead_times_critical)}/{len(detected_attacks)} detected attacks)")
            print(f"  Mean Lead Time: {np.mean(lead_times_critical):.1f} frames ({np.mean(lead_times_critical)/cfg.input_fps:.2f}s)")
            print(f"  Median Lead Time: {np.median(lead_times_critical):.1f} frames ({np.median(lead_times_critical)/cfg.input_fps:.2f}s)")
            if pre_contact_critical:
                critical_warning_rate = len(pre_contact_critical) / max(1, len(lead_times_critical))
                print(f"  Pre-contact CRITICAL warnings: {critical_warning_rate:.1%} ({len(pre_contact_critical)}/{len(lead_times_critical)})")
    
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

    print(f"\nFalse Positive Rate (safe videos): {fp_rate:.1%} ({len(false_positives)}/{len(safe_results)})")
    print(f"True Negative Rate: {tn_rate:.1%} ({len(true_negatives)}/{len(safe_results)})")
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total Videos: {len(attack_results) + len(safe_results)}")
    print(f"  Attacks: {len(attack_results)}")
    print(f"  Safe: {len(safe_results)}")
    print(f"\nThresholds:")
    print(f"  PRE-CONTACT: {threshold:.2f}")
    print(f"  HIGH: {cfg.high_thresh:.2f}")
    print(f"  CRITICAL: {cfg.critical_thresh:.2f}")
    print(f"\nAttack Detection: {detection_rate:.1%}")
    print(f"False Positive Rate (safe videos): {fp_rate:.1%}")
    if fp_attack_rate > 0:
        print(f"False Positive Rate (pre-onset): {fp_attack_rate:.1%}")
    if lead_times:
        print(f"\nWarning Performance:")
        print(f"  PRE-CONTACT: {pre_contact_warning_rate:.1%} pre-contact | Mean Lead: {np.mean(lead_times):.1f} frames ({np.mean(lead_times)/cfg.input_fps:.2f}s)")
        if lead_times_high:
            high_warning_rate = len([lt for lt in lead_times_high if lt > 0]) / max(1, len(lead_times_high))
            print(f"  HIGH: {len(lead_times_high)}/{len(detected_attacks)} attacks | {high_warning_rate:.1%} pre-contact | Mean Lead: {np.mean(lead_times_high):.1f} frames ({np.mean(lead_times_high)/cfg.input_fps:.2f}s)")
        if lead_times_critical:
            critical_warning_rate = len([lt for lt in lead_times_critical if lt > 0]) / max(1, len(lead_times_critical))
            print(f"  CRITICAL: {len(lead_times_critical)}/{len(detected_attacks)} attacks | {critical_warning_rate:.1%} pre-contact | Mean Lead: {np.mean(lead_times_critical):.1f} frames ({np.mean(lead_times_critical)/cfg.input_fps:.2f}s)")
    print("=" * 80)

    print(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log saved to: {log_file}")

    logger.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate <holdout_dir>")
        sys.exit(1)
    main(sys.argv[1])
