# Feature Importance Evaluation Guide

## Overview

Feature importance computation has been added to the training pipeline. After training completes, the system automatically computes and reports which features contribute most to model performance using **permutation importance**.

## What is Permutation Importance?

Permutation importance measures how much model performance drops when a feature's values are randomly shuffled (permuted). Features that cause larger performance drops when permuted are more important.

**Algorithm:**
1. Compute baseline F1 score on validation set
2. For each feature:
 - Randomly shuffle (permute) that feature's values across all samples
 - Re-compute F1 score with permuted feature
 - F1 drop = baseline F1 - permuted F1
3. Repeat each permutation 5 times and average for stability
4. Rank features by average F1 drop

## Files Changed

### 1. **src/feature_importance.py** (NEW)

Contains three main functions:

- `compute_permutation_importance(model, val_loader, threshold, device, n_repeats=5)`
 - Computes permutation importance for all 59 features
 - Returns importance scores and baseline F1

- `print_importance_ranking(importances, baseline_f1)`
 - Prints features ranked by importance
 - Shows F1 drop, standard deviation, and percentage contribution

- `save_importance_results(importances, baseline_f1, output_path)`
 - Saves results to JSON file for further analysis

### 2. **src/train.py** (MODIFIED)

Added feature importance computation at the end of training:

```python
# After training completes, compute feature importance
importances, baseline_f1 = compute_permutation_importance(
 model=model,
 val_loader=dl_va,
 threshold=global_best_thresh,
 device=device,
 n_repeats=5
)

# Print ranking
print_importance_ranking(importances, baseline_f1)

# Save to JSON
save_importance_results(importances, baseline_f1, output_path)
```

### 3. **src/config.py** (MODIFIED)

Added `FEATURE_NAMES` list defining all 59 feature names (see `src/config.py` for the complete list). Key groups:

```python
FEATURE_NAMES = [
 # Reliability/Metadata indices 0-8
 # Bbox/Approach indices 9-16
 # Upper Body Geometry indices 17-24
 # Lower Body Geometry indices 25-31
 # Optical Flow - Torso ROI indices 32-35
 # Optical Flow - Lower ROI indices 36-39
 # Optical Flow - Background indices 40-41
 # Posture indices 42-44
 # Dynamics indices 45-47
 # Body-shape Extras indices 48-55
 # Interaction Features indices 56-58
]
```

## Usage

### Automatic Computation

Feature importance is now computed automatically after every training run:

```bash
python -m src.train
```

**Output:**
1. **Console output** - Printed ranking with F1 drops and importance percentages
2. **JSON file** - `outputs/logs/feature_importance_YYYYMMDD_HHMMSS.json`

### Example Output

```
================================================================================
FEATURE IMPORTANCE RANKING (sorted by F1 drop)
================================================================================
Baseline F1: 0.8330

Rank Feature F1 Drop Std Importance % Index
--------------------------------------------------------------------------------
1 * expansion_proximity +0.xxxxx +/-0.xxxxxx 24.70% 57
2 * torso_height_px +0.xxxxx +/-0.xxxxxx 20.80% 48
3 * divergence_torso +0.xxxxx +/-0.xxxxxx 9.90% 34
4 * divergence_lower +0.xxxxx +/-0.xxxxxx 7.30% 38
5 * acceleration_proximity +0.xxxxx +/-0.xxxxxx 7.10% 58
...
```

### JSON Output Structure

```json
{
 "baseline_f1": 0.8330,
 "total_importance": 0.xxxx,
 "features": [
 {
 "rank": 1,
 "name": "expansion_proximity",
 "index": 57,
 "f1_drop_mean": 0.xxxxxx,
 "f1_drop_std": 0.xxxxxx,
 "importance_percent": 24.70
 },
 {
 "rank": 2,
 "name": "torso_height_px",
 "index": 48,
 "f1_drop_mean": 0.xxxxxx,
 "f1_drop_std": 0.xxxxxx,
 "importance_percent": 20.80
 },
 ...
 ]
}
```

## Interpreting Results

### Importance Score Meaning

- **Positive F1 drop**: Feature is important (removing it hurts performance)
- **Negative F1 drop**: Feature may be harmful or redundant (removing it helps)
- **Zero F1 drop**: Feature is not used by the model

### Importance Percentage

Normalized percentage showing relative contribution:

```
importance_percent = (positive_f1_drop / sum_of_all_positive_drops) * 100
```

### Current Findings

Based on the current 59-feature model with `window_len=5`:

| Rank | Feature | Importance | Index | Category |
|------|---------|-----------|-------|----------|
| 1 | expansion_proximity | 24.7% | 57 | Interaction |
| 2 | torso_height_px | 20.8% | 48 | Body-shape Extras |
| 3 | divergence_torso | 9.9% | 34 | Optical Flow (Torso) |
| 4 | divergence_lower | 7.3% | 38 | Optical Flow (Lower) |
| 5 | acceleration_proximity | 7.1% | 58 | Interaction |

The top 5 features account for roughly 69.8% of total importance. The
interaction feature `expansion_proximity` and the apparent body size
`torso_height_px` together account for about 45%, which indicates that scale
and proximity cues carry most of the decision: the same motion scores higher
when the person is large in frame and getting larger. Flow divergence in the
torso and lower-body ROIs follows at ranks 3 and 4, and `acceleration_proximity`
at rank 5 contributes the proximity-weighted wrist-acceleration signal.

Two caveats apply when reading these numbers. The permutation shuffles a
feature channel while leaving its validity mask in place, which produces
value and mask combinations the model never saw during training, so part of
each F1 drop reflects that mismatch rather than genuine reliance. Several of
these features are also algebraically related: `expansion_proximity` and
`acceleration_proximity` are both products of a raw signal and a function of
`torso_height_px`, so their individual scores are not independent.

## Computational Cost

**Time:** ~2-5 minutes on validation set (depends on val set size and n_repeats)

**Memory:** Same as training (all features loaded into memory once)

**Parallelization:** Permutation is done sequentially per feature (no GPU speedup), but batch inference uses GPU if available.

## Configuration Options

### Change number of repeats

Edit `src/train.py`:

```python
importances, baseline_f1 = compute_permutation_importance(
 model=model,
 val_loader=dl_va,
 threshold=global_best_thresh,
 device=device,
 n_repeats=10 # Increase for more stable estimates (slower)
)
```

**Trade-off:**
- `n_repeats=5`: Fast, reasonable stability (default)
- `n_repeats=10`: Slower, more stable estimates
- `n_repeats=1`: Fastest, less reliable (not recommended)

### Disable feature importance

Comment out the feature importance section in `src/train.py` (lines 327-348).

## Retraining Notes

**Important:** After changing `window_len` or the feature set:

1. **Retrain the model** (required - model input shape changes with feature count or window length)
2. **Feature importance will be recomputed automatically**
3. **Compare results** with the previous run to track ranking changes

Expected changes when window length increases:
- Temporal features (derivatives, trends) may become more important with longer context
- Relative ranking of top features should remain broadly stable
- Absolute importance values may shift

## Advanced Usage

### Standalone Feature Importance Script

You can also compute feature importance separately without retraining:

```python
from src.feature_importance import compute_permutation_importance, print_importance_ranking
from src.model import HazardGRU
from src.config import Config
import torch

# Load trained model
cfg = Config()
model = HazardGRU(input_dim=118, hidden=cfg.gru_hidden, dropout=cfg.dropout)
model.load_state_dict(torch.load("outputs/checkpoints/hazard_gru.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load validation data (build_dataset_per_video, then create val_loader)
# ...

# Compute importance
importances, baseline_f1 = compute_permutation_importance(
 model=model,
 val_loader=val_loader,
 threshold=0.35, # Use your best threshold
 device=device,
 n_repeats=5
)

print_importance_ranking(importances, baseline_f1)
```

## Troubleshooting

### "CUDA out of memory"

Solution: Reduce batch size or use CPU

```python
# In src/feature_importance.py, line ~90
batch_size = 32 # Reduce from val_loader.batch_size
```

### Features have negative importance

This is normal! Negative importance means:
- Feature is redundant (covered by other features)
- Feature adds noise
- Feature is not learned by the model

Example: If `edge_crop_left` has -0.005 importance, removing it slightly improves F1 (model handles missing data better without this hint).

### Importance percentages don't sum to 100%

This is expected because:
- Only **positive** drops are counted
- Negative drops are excluded from percentage calculation
- Total represents useful features only

## References

- **Permutation Importance**: Breiman, L. (2001). Random Forests. Machine Learning.
- **Feature Engineering**: Our custom radial/tangential decomposition
- **Implementation**: Based on sklearn's permutation_importance but adapted for temporal windows

---

Feature importance is computed on the validation set using the best-F1
checkpoint, so it reflects behaviour on held-out windows rather than on the
training data.
