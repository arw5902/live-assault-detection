import os
import json
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .config import Config, FEATURE_NAMES
from .dataset import build_dataset_per_video  # Video-level data for proper train/val split
from .model import HazardGRU, HazardLSTM, HazardTransformer
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

    print(f"Training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  log → {log_file}")

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

    print(f"\nFeatures: input_dim={input_dim} (features+masks)  raw={num_raw_features}  "
          f"FEATURE_NAMES={len(FEATURE_NAMES)}")
    if num_raw_features != len(FEATURE_NAMES):
        print(f"  WARNING: feature count mismatch — importance evaluation will be wrong")

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

    print(f"\nTraining done {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best F1: {global_best_f1:.3f}  Recall: {global_best_rec:.3f}  at threshold {global_best_thresh:.2f} (epoch {best_epoch})")
    print(f"Model saved to: {model_path} (selected by best F1)")

    # ── Precision-Recall and ROC curves ─────────────────────────────────────
    print("\nGenerating PR and ROC curves...")

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

    try:
        from . import _plot
        _plot.plot_pr_and_roc(
            y_true, y_score,
            selected_threshold=global_best_thresh,
            model_type=model_type,
            title_dataset="Validation Set",
            pr_path=f"outputs/plots/pr_curve_{timestamp}.png",
            roc_path=f"outputs/plots/roc_curve_{timestamp}.png",
        )
    except ImportError:
        print("  matplotlib not installed — skipping curve plots")
        print("  Install with: pip install matplotlib")

    print("\nComputing feature importance...")
    importances, baseline_f1 = compute_permutation_importance(
        model=model,
        val_loader=dl_va,
        threshold=global_best_thresh,
        device=device,
        n_repeats=5
    )
    print_importance_ranking(importances, baseline_f1)
    importance_output = f"outputs/logs/feature_importance_{timestamp}.json"
    save_importance_results(importances, baseline_f1, importance_output)

    print(f"\nLog saved to: {log_file}")

    logger.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train hazard detection model")
    parser.add_argument("--model", choices=["gru", "lstm", "transformer"], default="gru",
                        help="Model architecture: gru (default), lstm, or transformer")
    args = parser.parse_args()
    main(model_type=args.model)
