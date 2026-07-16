"""
Permutation importance: shuffle each feature and measure the F1 drop.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score
from .config import FEATURE_NAMES


def compute_permutation_importance(model, val_loader, threshold, device, n_repeats=5):
    """
    Compute permutation importance for each feature.

    Args:
        model: trained hazard model
        val_loader: validation data loader
        threshold: classification threshold for binary prediction
        device: torch device
        n_repeats: number of times to permute each feature

    Returns:
        importances: dict mapping feature name to {mean_drop, std_drop, index}
        baseline_f1: baseline F1 without permutation
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_features = []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb.to(device)).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(yb.numpy())
            all_features.append(xb.numpy())

    all_features = np.concatenate(all_features, axis=0)  # [N, T, D]
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nFeature tensor: {all_features.shape}  "
          f"(N={all_features.shape[0]}, T={all_features.shape[1]}, D={all_features.shape[2]})")

    y_true = (all_labels >= 0.5).astype(int)
    y_pred_baseline = (all_preds >= threshold).astype(int)
    baseline_f1 = f1_score(y_true, y_pred_baseline, zero_division=0)

    print(f"\nBaseline F1 score: {baseline_f1:.4f}")
    print(f"Computing permutation importance (n_repeats={n_repeats})...")

    # Half the channels are features, the other half are validity masks.
    num_features = all_features.shape[2] // 2
    print(f"Number of features to evaluate: {num_features}")
    print(f"Number of feature names available: {len(FEATURE_NAMES)}")

    if num_features != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature count mismatch!\n"
            f"  Model has {num_features} features\n"
            f"  FEATURE_NAMES has {len(FEATURE_NAMES)} names\n"
            f"  Please update FEATURE_NAMES in src/config.py to match src/features.py"
        )

    importances = {}
    batch_size = val_loader.batch_size

    for feat_idx in range(num_features):
        feature_name = FEATURE_NAMES[feat_idx]
        f1_drops = []

        for repeat in range(n_repeats):
            features_permuted = all_features.copy()

            # Permute across windows but keep the same window-internal ordering,
            # so the within-window temporal structure of this feature is preserved.
            # Mask channel is intentionally NOT permuted.
            perm_idx = np.random.permutation(all_features.shape[0])
            features_permuted[:, :, feat_idx] = all_features[perm_idx, :, feat_idx]

            model.eval()
            permuted_preds = []
            with torch.no_grad():
                for i in range(0, len(features_permuted), batch_size):
                    batch = features_permuted[i:i+batch_size]
                    batch_tensor = torch.from_numpy(batch).float().to(device)
                    pred = model(batch_tensor).cpu().numpy()
                    permuted_preds.extend(pred)

            permuted_preds = np.array(permuted_preds)
            y_pred_permuted = (permuted_preds >= threshold).astype(int)
            permuted_f1 = f1_score(y_true, y_pred_permuted, zero_division=0)
            f1_drops.append(baseline_f1 - permuted_f1)

        mean_f1_drop = np.mean(f1_drops)
        std_f1_drop = np.std(f1_drops)

        importances[feature_name] = {
            'mean_drop': mean_f1_drop,
            'std_drop': std_f1_drop,
            'index': feat_idx
        }

        print(f"  [{feat_idx:2d}] {feature_name:30s}: F1 drop = {mean_f1_drop:+.4f} +/- {std_f1_drop:.4f}")

    return importances, baseline_f1


def print_importance_ranking(importances, baseline_f1):
    print(f"\nFeature importance ranking (baseline F1 = {baseline_f1:.4f}):")

    sorted_features = sorted(importances.items(),
                            key=lambda x: x[1]['mean_drop'],
                            reverse=True)
    total_importance = sum(max(0, imp['mean_drop']) for _, imp in sorted_features)

    print(f"{'Rank':<6} {'Feature':<30} {'F1 Drop':<12} {'Std':<10} {'Importance %':<12} {'Index':<6}")
    for rank, (name, imp) in enumerate(sorted_features, start=1):
        mean_drop = imp['mean_drop']
        std_drop = imp['std_drop']
        feat_idx = imp['index']

        if total_importance > 0:
            importance_pct = max(0, mean_drop) / total_importance * 100
        else:
            importance_pct = 0.0

        marker = "*" if rank <= 5 else "  "
        print(f"{rank:<4} {marker} {name:<30} {mean_drop:+.6f}  +/-{std_drop:.6f}   {importance_pct:6.2f}%      {feat_idx:<6}")

    print("\nTop 5 most important features:")
    for rank, (name, imp) in enumerate(sorted_features[:5], start=1):
        importance_pct = max(0, imp['mean_drop']) / total_importance * 100 if total_importance > 0 else 0.0
        print(f"  {rank}. {name}: {importance_pct:.1f}% (F1 drop: {imp['mean_drop']:+.4f})")
    print()


def save_importance_results(importances, baseline_f1, output_path):
    import json

    sorted_features = sorted(importances.items(),
                            key=lambda x: x[1]['mean_drop'],
                            reverse=True)
    total_importance = sum(max(0, imp['mean_drop']) for _, imp in sorted_features)

    results = {
        'baseline_f1': float(baseline_f1),
        'total_importance': float(total_importance),
        'features': []
    }

    for rank, (name, imp) in enumerate(sorted_features, start=1):
        importance_pct = max(0, imp['mean_drop']) / total_importance * 100 if total_importance > 0 else 0.0
        results['features'].append({
            'rank': rank,
            'name': name,
            'index': imp['index'],
            'f1_drop_mean': float(imp['mean_drop']),
            'f1_drop_std': float(imp['std_drop']),
            'importance_percent': float(importance_pct)
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Feature importance results saved to: {output_path}")
