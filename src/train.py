import os
import json
import math
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             precision_recall_curve, average_precision_score,
                             roc_curve, auc)
from .config import Config, FEATURE_NAMES
from .dataset import build_dataset_per_video  # Video-level data for proper train/val split
from .model import HazardGRU, HazardLSTM, HazardTransformer, HazardCNN
from .pose_detector import PoseDetector
from .utils import set_seed, print_seed_info
from .feature_importance import compute_permutation_importance, print_importance_ranking, save_importance_results

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

class WindowDataset(Dataset):
    def __init__(self, X, M, y):
        # Apply mask by concatenating mask bits to inputs (simple and effective):
        # input = [features, mask] so model knows what is missing.
        self.X = np.concatenate([X, M], axis=2).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def main(model_type: str = "gru"):
    # Setup logging
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/logs/train_{timestamp}.log"
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print("=" * 80)

    # Set random seed for reproducibility (CRITICAL: must be before dataset building)
    set_seed(seed=42, deterministic=True)
    print_seed_info(seed=42, deterministic=True)
    print()

    cfg = Config.for_pc()
    detector = PoseDetector(cfg)

    # Build dataset per video (prevents data leakage)
    video_data = build_dataset_per_video(detector, cfg)

    # Separate safe and attack videos for class-balanced splitting
    safe_videos = []
    attack_videos = []

    for xw, mw, yw in video_data:
        # Check if video is safe or attack based on majority label
        if np.mean(yw) < 0.5:
            safe_videos.append((xw, mw, yw, xw.shape[0]))
        else:
            attack_videos.append((xw, mw, yw, xw.shape[0]))

    # Shuffle each class independently
    np.random.shuffle(safe_videos)
    np.random.shuffle(attack_videos)

    # Helper function for window-based split
    def split_by_windows(video_list, val_fraction=0.15):
        """Split videos to achieve target window fraction in validation set."""
        if len(video_list) == 0:
            return [], []

        total_windows = sum(size for _, _, _, size in video_list)
        target_val = int(val_fraction * total_windows)

        val_set = []
        train_set = []
        val_count = 0

        for xw, mw, yw, size in video_list:
            # Greedy assignment: add to val if closer to target
            if val_count < target_val and len(train_set) > 0:
                # Check if adding this video would overshoot less than not adding it
                overshoot_if_add = abs((val_count + size) - target_val)
                undershoot_if_skip = abs(val_count - target_val)

                if overshoot_if_add < undershoot_if_skip or len(val_set) == 0:
                    val_set.append((xw, mw, yw))
                    val_count += size
                else:
                    train_set.append((xw, mw, yw))
            elif len(val_set) == 0 or val_count < target_val:
                val_set.append((xw, mw, yw))
                val_count += size
            else:
                train_set.append((xw, mw, yw))

        return train_set, val_set

    # Split each class independently to maintain class balance
    safe_train, safe_val = split_by_windows(safe_videos, val_fraction=0.15)
    attack_train, attack_val = split_by_windows(attack_videos, val_fraction=0.15)

    # Combine train and val sets
    train_videos = safe_train + attack_train
    val_videos = safe_val + attack_val

    # Ensure both sets have at least one video
    if len(train_videos) == 0 or len(val_videos) == 0:
        raise RuntimeError("Train/val split resulted in empty set. Need more videos.")

    # Shuffle combined sets (maintains video-level separation, no data leakage)
    np.random.shuffle(train_videos)
    np.random.shuffle(val_videos)

    print(f"\nStratified video-level split (class-balanced):")
    print(f"  Total videos: {len(video_data)}")
    print(f"  Train videos: {len(train_videos)} (safe={len(safe_train)}, attack={len(attack_train)})")
    print(f"  Val videos: {len(val_videos)} (safe={len(safe_val)}, attack={len(attack_val)})")

    # Concatenate windows within train and val sets
    Xtr = np.concatenate([xw for xw, _, _ in train_videos], axis=0)
    Mtr = np.concatenate([mw for _, mw, _ in train_videos], axis=0)
    ytr = np.concatenate([yw for _, _, yw in train_videos], axis=0)

    Xva = np.concatenate([xw for xw, _, _ in val_videos], axis=0)
    Mva = np.concatenate([mw for _, mw, _ in val_videos], axis=0)
    yva = np.concatenate([yw for _, _, yw in val_videos], axis=0)

    # Dataset balance analysis (window-level after video split)
    n_safe_tr = np.sum(ytr == 0)
    n_attack_tr = np.sum(ytr == 1)
    n_safe_va = np.sum(yva == 0)
    n_attack_va = np.sum(yva == 1)

    total_tr = len(ytr)
    total_va = len(yva)
    total_all = total_tr + total_va
    imbalance_ratio_tr = max(n_safe_tr, n_attack_tr) / max(1, min(n_safe_tr, n_attack_tr))

    print(f"\nTrain set balance:")
    print(f"  Total windows: {total_tr} ({total_tr/total_all*100:.1f}% of all windows)")
    print(f"  Safe windows: {n_safe_tr} ({n_safe_tr/total_tr*100:.1f}%)")
    print(f"  Attack windows: {n_attack_tr} ({n_attack_tr/total_tr*100:.1f}%)")
    print(f"  Imbalance ratio: {imbalance_ratio_tr:.2f}:1")

    print(f"\nVal set balance:")
    print(f"  Total windows: {total_va} ({total_va/total_all*100:.1f}% of all windows)")
    print(f"  Safe windows: {n_safe_va} ({n_safe_va/total_va*100:.1f}%)")
    print(f"  Attack windows: {n_attack_va} ({n_attack_va/total_va*100:.1f}%)")

    # Verify split quality
    val_fraction = total_va / total_all
    if val_fraction < 0.10 or val_fraction > 0.20:
        print(f"\n  WARNING: Val fraction {val_fraction:.1%} is outside target range [10%-20%]")

    # Compute class imbalance ratio for reference
    attack_weight = n_safe_tr / max(1, n_attack_tr)
    print(f"\nClass imbalance ratio (safe/attack windows): {attack_weight:.2f}")
    print()

    ds_tr = WindowDataset(Xtr, Mtr, ytr)
    ds_va = WindowDataset(Xva, Mva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    input_dim = ds_tr.X.shape[2]
    num_raw_features = input_dim // 2  # Features are concatenated with masks

    # Verify FEATURE_NAMES matches actual features
    print("\n" + "=" * 80)
    print("FEATURE CONFIGURATION VERIFICATION")
    print("=" * 80)
    print(f"Model input dimension: {input_dim} (features + masks)")
    print(f"Number of raw features: {num_raw_features}")
    print(f"Number of FEATURE_NAMES: {len(FEATURE_NAMES)}")

    if num_raw_features != len(FEATURE_NAMES):
        print("\n⚠️  WARNING: Feature count mismatch!")
        print(f"   Expected: {num_raw_features} features")
        print(f"   FEATURE_NAMES has: {len(FEATURE_NAMES)} names")
        print("   Feature importance evaluation may be incorrect!")
    else:
        print("✓ Feature names match actual features")
    print("=" * 80)
    print()

    # Save metadata for inference
    os.makedirs("outputs/checkpoints", exist_ok=True)
    model_filename = f"hazard_{model_type}_{timestamp}.pt"
    model_path = f"outputs/checkpoints/{model_filename}"
    with open("outputs/checkpoints/meta.json", "w") as f:
        json.dump({"input_dim": int(input_dim), "best_threshold": 0.50,
                   "model_file": model_filename, "model_type": model_type}, f)

    if model_type == "lstm":
        model = HazardLSTM(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=cfg.dropout)
    elif model_type == "transformer":
        model = HazardTransformer(input_dim=input_dim, d_model=cfg.gru_hidden, dropout=cfg.dropout)
    elif model_type == "cnn":
        model = HazardCNN(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=cfg.dropout)
    else:
        model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=cfg.dropout)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type.upper()}  |  Parameters: {n_params:,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Focal Loss — down-weights easy (well-classified) examples so the model
    # focuses on hard, ambiguous samples.  Alpha balances positive/negative
    # classes; gamma controls the rate at which easy examples are down-weighted.
    focal_gamma = cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0
    focal_alpha = cfg.focal_alpha if hasattr(cfg, 'focal_alpha') else 0.75

    print(f"Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}")
    print(f"  (class imbalance ratio: {attack_weight:.2f}:1)")
    print()

    def focal_loss(pred, target):
        """
        Focal loss for binary classification.

        Args:
            pred: predicted probabilities [batch_size]
            target: ground truth labels [batch_size] (0 or 1)
        """
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # p_t = predicted probability of the true class
        p_t = pred * target + (1.0 - pred) * (1.0 - target)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** focal_gamma

        # Alpha weighting: alpha for positive class, (1-alpha) for negative
        alpha_weight = target * focal_alpha + (1.0 - target) * (1.0 - focal_alpha)

        return (alpha_weight * focal_weight * bce).mean()

    best_val_loss = 1e9
    global_best_f1 = 0.0
    global_best_rec = 0.0
    global_best_thresh = 0.50
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = focal_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(dl_tr))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = focal_loss(pred, yb)
                va_loss += float(loss.item())
        va_loss /= max(1, len(dl_va))

        # Track best validation loss for monitoring
        if va_loss < best_val_loss:
            best_val_loss = va_loss

        # Compute classification metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in dl_va:
                    pred = model(xb.to(device)).cpu().numpy()
                    all_preds.extend(pred)
                    all_labels.extend(yb.numpy())

            # Tune threshold: pick the best F1 score across all thresholds;
            # break ties by highest recall, then highest threshold.
            thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

            y_true = (np.array(all_labels) >= 0.5).astype(int)
            n_neg = max(1, int((y_true == 0).sum()))

            # --- pass 1: collect metrics at every threshold ----------------------
            thresh_table = []              # (thresh, acc, prec, rec, f1, fpr)
            for thresh in thresholds:
                y_pred = (np.array(all_preds) >= thresh).astype(int)
                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0)
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                fpr = fp / n_neg
                thresh_table.append((thresh, acc, prec, rec, f1, fpr))

            # --- pass 2: best F1, ties broken by recall then higher threshold ----
            best_row = max(thresh_table, key=lambda x: (x[4], x[3], x[0]))
            selection = "best-F1"

            best_thresh = best_row[0]
            acc, prec, rec, f1, fpr_at_thresh = (
                best_row[1], best_row[2], best_row[3], best_row[4], best_row[5])
            best_f1 = f1

            # Track global best and save model when the selected metric improves.
            # Primary comparison: F1 (at acceptable FP rate);
            # break ties by recall, then higher threshold.
            improved = (f1 > global_best_f1) or (
                f1 == global_best_f1 and rec > global_best_rec)
            if improved:
                global_best_f1 = best_f1
                global_best_rec = rec
                global_best_thresh = best_thresh
                best_epoch = epoch + 1

                # Save model checkpoint
                torch.save(model.state_dict(), model_path)

                # Update meta.json with best threshold
                with open("outputs/checkpoints/meta.json", "r") as f:
                    meta = json.load(f)
                meta["best_threshold"] = float(global_best_thresh)
                with open("outputs/checkpoints/meta.json", "w") as f:
                    json.dump(meta, f)

                print(f"  ★ New best — F1={f1:.3f} Rec={rec:.3f} FP%={fpr_at_thresh:.1%} "
                      f"@ thresh={best_thresh:.2f} [{selection}] — Model saved!")

            metrics_str = (f"  Val @ thresh={best_thresh:.2f} [{selection}]: "
                           f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} "
                           f"F1={f1:.3f} FP%={fpr_at_thresh:.1%}")
        else:
            metrics_str = ""

        print(f"epoch {epoch+1}/{cfg.epochs} train={tr_loss:.4f} val={va_loss:.4f} best_val={best_val_loss:.4f}")
        if metrics_str:
            print(metrics_str)

    print("\n" + "=" * 80)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best F1: {global_best_f1:.3f}  Recall: {global_best_rec:.3f}  at threshold {global_best_thresh:.2f} (epoch {best_epoch})")
    print(f"Model saved to: {model_path} (selected by best F1)")
    print("=" * 80)

    # ── Precision-Recall and ROC curves ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATING PR AND ROC CURVES")
    print("=" * 80)

    # Load best model and collect predictions on validation set
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            scores = model(xb.to(device)).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(yb.numpy())
    y_true  = (np.array(all_labels) >= 0.5).astype(int)
    y_score = np.array(all_scores)

    # Pre-compute per-threshold metrics for marker annotations
    n_neg = max(1, int((y_true == 0).sum()))
    marker_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    marker_metrics = {}   # thresh → (prec, rec, f1, fpr)
    for t in marker_thresholds:
        y_pred = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        fp_count = int(((y_pred == 1) & (y_true == 0)).sum())
        marker_metrics[t] = (p, r, f, fp_count / n_neg)

    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend
        import matplotlib.pyplot as plt

        os.makedirs("outputs/plots", exist_ok=True)

        # --- Precision-Recall curve with F1 iso-curves ---
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
            # Label at the right end of each curve
            label_idx = np.where(valid)[0]
            if len(label_idx) > 0:
                li = label_idx[-1]
                ax.annotate(f"F1={f1_val}", (r_iso[li], p_iso[li]),
                            fontsize=7, color='gray', alpha=0.7,
                            ha='left', va='bottom')

        # Main PR curve
        ax.plot(rec_arr, prec_arr, linewidth=2, color='#1f77b4',
                label=f"PR curve (AP={ap:.3f})")

        # Threshold markers — dots on the curve, labels in a table box
        table_lines = []  # collect text lines for the summary table
        # Ensure selected threshold is included in the marker list
        all_markers = list(marker_thresholds)
        sel_in_markers = any(abs(t - global_best_thresh) < 0.005
                             for t in marker_thresholds)
        if not sel_in_markers:
            all_markers.append(global_best_thresh)
            all_markers.sort()
            # Compute metrics for the extra threshold
            y_sel = (y_score >= global_best_thresh).astype(int)
            p_s, r_s, f_s, _ = precision_recall_fscore_support(
                y_true, y_sel, average='binary', zero_division=0)
            fp_s = int(((y_sel == 1) & (y_true == 0)).sum())
            marker_metrics[global_best_thresh] = (p_s, r_s, f_s, fp_s / n_neg)

        # Collect marker positions for label offset computation
        marker_items = []  # (t, r_m, p_m, f_m, fpr_m, is_selected, color)
        for t in all_markers:
            p_m, r_m, f_m, fpr_m = marker_metrics[t]
            if r_m == 0 and p_m == 0:
                continue
            is_selected = abs(t - global_best_thresh) < 0.005
            color = 'red' if is_selected else '#1f77b4'
            size = 10 if is_selected else 6
            zorder = 10 if is_selected else 5
            ax.plot(r_m, p_m, 'o', color=color, markersize=size,
                    zorder=zorder)
            marker_items.append((t, r_m, p_m, f_m, fpr_m, is_selected, color))

        # Compute spread-out label offsets to avoid overlap
        # Sort by (recall, precision) so adjacent labels get different offsets
        angles = []
        n_items = len(marker_items)
        if n_items > 0:
            # Check if points are clustered (span < 0.05 in both axes)
            r_vals = [mi[1] for mi in marker_items]
            p_vals = [mi[2] for mi in marker_items]
            clustered = (max(r_vals) - min(r_vals) < 0.05
                         and max(p_vals) - min(p_vals) < 0.05)
            if clustered and n_items > 1:
                # Fan labels outward in a radial pattern
                start_angle = 200  # degrees, starting from lower-left
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

            # Build table line
            tag = "★" if is_selected else " "
            table_lines.append(
                f"{tag} t={t:.2f}  P={p_m:.2f}  R={r_m:.2f}  "
                f"F1={f_m:.2f}  FPR={fpr_m:>6.2%}")

        # Draw summary table as a text box in the lower-left
        table_text = "\n".join(table_lines)
        txt = ax.text(0.02, 0.02, table_text, transform=ax.transAxes,
                      fontsize=7, fontfamily='monospace', verticalalignment='bottom',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='#cccccc', alpha=0.9))

        # Render once so we can measure the table's actual height, then
        # position the legend snugly above it with minimal gap.
        fig.canvas.draw()
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        bb_axes = bb.transformed(ax.transAxes.inverted())
        ax.legend(loc="lower left", fontsize=8,
                  bbox_to_anchor=(0.01, bb_axes.y1 + 0.01))

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"Precision-Recall Curve (Validation Set) - {model_type.upper()}", fontsize=12)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        pr_path = f"outputs/plots/pr_curve_{timestamp}.png"
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  PR curve saved to  : {pr_path}  (AP={ap:.3f})")

        # --- ROC curve with threshold markers and FP≤5% region ---
        fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr_arr, tpr_arr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_arr, tpr_arr, linewidth=2, color='#1f77b4',
                label=f"ROC curve (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

        # Threshold markers — dots on the curve, labels in a table box
        roc_table_lines = []
        roc_marker_items = []
        for t in all_markers:
            p_m, r_m, f_m, fpr_m = marker_metrics[t]
            # Find closest point on the ROC curve for this threshold
            best_dist = float('inf')
            tpr_m = r_m       # fallback to recall from marker_metrics
            fpr_plot = fpr_m  # fallback to computed FPR
            for rt, rfp, rtp in zip(roc_thresholds, fpr_arr, tpr_arr):
                dist = abs(rt - t)
                if dist < best_dist:
                    best_dist = dist
                    fpr_plot, tpr_m = rfp, rtp
            is_selected = abs(t - global_best_thresh) < 0.005
            color = 'red' if is_selected else '#1f77b4'
            size = 6 if is_selected else 3
            zorder = 10 if is_selected else 5
            ax.plot(fpr_plot, tpr_m, 'o', color=color, markersize=size,
                    zorder=zorder)
            roc_marker_items.append((t, fpr_plot, tpr_m, is_selected, color,
                                     r_m, f_m, fpr_m))

        # Compute spread-out label offsets for ROC markers
        roc_angles = []
        n_roc = len(roc_marker_items)
        if n_roc > 0:
            fpr_vals = [mi[1] for mi in roc_marker_items]
            tpr_vals = [mi[2] for mi in roc_marker_items]
            clustered = (max(fpr_vals) - min(fpr_vals) < 0.05
                         and max(tpr_vals) - min(tpr_vals) < 0.05)
            if clustered and n_roc > 1:
                start_angle = -30  # degrees, starting from lower-right
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

        # Draw summary table as a text box in the lower-right
        roc_table_text = "\n".join(roc_table_lines)
        txt2 = ax.text(0.98, 0.02, roc_table_text, transform=ax.transAxes,
                        fontsize=7, fontfamily='monospace', verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='#cccccc', alpha=0.9))

        # Render once so we can measure the table's actual height, then
        # position the legend snugly above it with minimal gap.
        fig.canvas.draw()
        bb2 = txt2.get_window_extent(renderer=fig.canvas.get_renderer())
        bb2_axes = bb2.transformed(ax.transAxes.inverted())
        ax.legend(loc="lower right", fontsize=8,
                  bbox_to_anchor=(0.99, bb2_axes.y1 + 0.01))

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax.set_title(f"ROC Curve (Validation Set) - {model_type.upper()}", fontsize=12)
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        roc_path = f"outputs/plots/roc_curve_{timestamp}.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ROC curve saved to : {roc_path}  (AUC={roc_auc:.3f})")

        # --- Print threshold comparison table ---
        print(f"\n  Threshold comparison (val set, n_neg={n_neg}):")
        print(f"  {'thresh':>6}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  "
              f"{'FP%':>6}  {'note'}")
        print(f"  {'-'*46}")
        for t in sorted(marker_metrics.keys()):
            p_m, r_m, f_m, fpr_m = marker_metrics[t]
            if abs(t - global_best_thresh) < 0.005:
                note = "★ selected (best F1)"
            else:
                note = ""
            print(f"  {t:>6.2f}  {p_m:>5.3f}  {r_m:>5.3f}  {f_m:>5.3f}  "
                  f"{fpr_m:>6.2%}  {note}")

    except ImportError:
        print("  matplotlib not installed — skipping curve plots")
        print("  Install with: pip install matplotlib")

    # Compute feature importance using permutation importance
    print("\n" + "=" * 80)
    print("COMPUTING FEATURE IMPORTANCE")
    print("=" * 80)

    # Compute permutation importance
    importances, baseline_f1 = compute_permutation_importance(
        model=model,
        val_loader=dl_va,
        threshold=global_best_thresh,
        device=device,
        n_repeats=5  # Repeat permutation 5 times for stability
    )

    # Print ranking
    print_importance_ranking(importances, baseline_f1)

    # Save results to JSON
    importance_output = f"outputs/logs/feature_importance_{timestamp}.json"
    save_importance_results(importances, baseline_f1, importance_output)

    print(f"\nLog saved to: {log_file}")
    print("=" * 80)

    logger.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train hazard detection model")
    parser.add_argument("--model", choices=["gru", "lstm", "transformer", "cnn"], default="gru",
                        help="Model architecture: gru (default), lstm, transformer, or cnn")
    args = parser.parse_args()
    main(model_type=args.model)
