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
from .model import HazardGRU
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

def main():
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

    cfg = Config()
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

    # Compute class imbalance ratio for reference (focal loss handles this via alpha)
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
    model_filename = f"hazard_gru_{timestamp}.pt"
    model_path = f"outputs/checkpoints/{model_filename}"
    with open("outputs/checkpoints/meta.json", "w") as f:
        json.dump({"input_dim": int(input_dim), "best_threshold": 0.50, "model_file": model_filename}, f)  # best_threshold updated whenever a new best F1 is found

    model = HazardGRU(input_dim=input_dim, hidden=cfg.gru_hidden, dropout=cfg.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Focal Loss parameters from config
    focal_gamma = cfg.focal_gamma
    focal_alpha = cfg.focal_alpha

    print(f"Focal Loss parameters: gamma={focal_gamma:.2f}, alpha={focal_alpha:.3f}")
    print(f"  (gamma: higher=more focus on hard examples)")
    print(f"  (alpha: higher=prioritize attack class/recall)")
    print(f"  (class imbalance ratio: {attack_weight:.2f}:1)")
    print()

    def focal_loss(pred, target, gamma=2.0, alpha=0.25):
        """
        Focal Loss for binary classification.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Args:
            pred: predicted probabilities [batch_size]
            target: ground truth labels [batch_size] (0 or 1)
            gamma: focusing parameter (default: 2.0)
            alpha: weight for positive class (default: 0.25)
        """
        # Compute BCE loss without reduction
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # Compute p_t (probability of correct class)
        p_t = pred * target + (1 - pred) * (1 - target)

        # Compute alpha_t (weight for correct class)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)

        # Focal loss: alpha_t * (1 - p_t)^gamma * BCE
        focal = alpha_t * ((1 - p_t) ** gamma) * bce

        return focal.mean()

    best_val_loss = 1e9
    global_best_f1 = 0.0
    global_best_thresh = 0.50
    best_f1_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            # Focal Loss: emphasizes hard examples
            loss = focal_loss(pred, yb, gamma=focal_gamma, alpha=focal_alpha)
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
                # Use same focal loss for validation
                loss = focal_loss(pred, yb, gamma=focal_gamma, alpha=focal_alpha)
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

            # Tune threshold for best F1 score (extended range to include higher thresholds)
            thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            best_f1 = 0.0
            best_thresh = 0.30
            best_metrics = (0.0, 0.0, 0.0, 0.0)  # (acc, prec, rec, f1) fallback

            y_true = (np.array(all_labels) >= 0.5).astype(int)
            for thresh in thresholds:
                y_pred = (np.array(all_preds) >= thresh).astype(int)

                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_metrics = (acc, prec, rec, f1)

            acc, prec, rec, f1 = best_metrics

            # Track global best F1 and save model when F1 improves
            if best_f1 > global_best_f1:
                global_best_f1 = best_f1
                global_best_thresh = best_thresh
                best_f1_epoch = epoch + 1

                # Save model checkpoint based on best F1
                torch.save(model.state_dict(), model_path)

                # Update meta.json with best threshold
                with open("outputs/checkpoints/meta.json", "r") as f:
                    meta = json.load(f)
                meta["best_threshold"] = float(global_best_thresh)
                with open("outputs/checkpoints/meta.json", "w") as f:
                    json.dump(meta, f)

                print(f"  New best F1: {best_f1:.3f} at threshold {best_thresh:.2f} - Model saved!")

            metrics_str = f"  Val Metrics @ thresh={best_thresh:.2f}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}"
        else:
            metrics_str = ""

        print(f"epoch {epoch+1}/{cfg.epochs} train={tr_loss:.4f} val={va_loss:.4f} best_val={best_val_loss:.4f}")
        if metrics_str:
            print(metrics_str)

    print("\n" + "=" * 80)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best F1 score: {global_best_f1:.3f} at threshold {global_best_thresh:.2f} (epoch {best_f1_epoch})")
    print(f"Model saved to: {model_path} (based on best F1)")
    print("=" * 80)

    # Compute feature importance using permutation importance
    print("\n" + "=" * 80)
    print("COMPUTING FEATURE IMPORTANCE")
    print("=" * 80)

    # Load best model for feature importance computation
    model.load_state_dict(torch.load(model_path))

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
    main()
