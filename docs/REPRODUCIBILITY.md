# Reproducibility Guide

This document explains how to ensure deterministic, reproducible results across multiple training and evaluation runs.

## Problem Statement

You may observe different `max_hazard` values for the same video across different training runs, even when using the same model architecture and training data. This non-deterministic behavior makes it difficult to:

- Compare model improvements
- Debug issues
- Reproduce published results
- Ensure consistent deployment behavior

## Root Causes

### 1. **Optical Flow Random Sampling** (PRIMARY ISSUE)

**Location**: `src/flow.py`

The optical flow computation samples random points within and outside the person's bounding box:

```python
# Sample 200 random points inside bbox
xs = np.random.randint(x1, x2, size=(200,))
ys = np.random.randint(y1, y2, size=(200,))

# Sample 200 random points in background
idx = np.random.choice(len(xs), size=(200,), replace=False)
```

**Impact**:
- Different random points → different optical flow vectors → different flow features → different predictions
- This happens during **feature extraction**, not just training
- Even with the same trained model, evaluation gives different results

**Why it matters**:
- Optical flow and interaction features dominate importance — `expansion_proximity` (11.1%), `translation_lower` (9.2%), `translation_torso` (8.3%)
- Small variations in flow sampling can significantly affect predictions

### 2. **PyTorch Non-Determinism**

**Sources**:
- CUDA operations (cuDNN backend uses non-deterministic algorithms by default)
- DataLoader shuffling (different order each epoch)
- Dropout (random during training, disabled during eval)

**Impact**:
- Different model weights after training
- Different validation metrics
- Different optimal threshold selection

### 3. **Python Built-in Random**

Some operations may use Python's built-in `random` module instead of NumPy or PyTorch.

## Solution

### Comprehensive Seed Setting

We created `src/utils.py` with a centralized seeding function:

```python
def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility across all libraries."""
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
```

### Critical Timing

**IMPORTANT**: Seeds must be set **BEFORE** any feature extraction or model operations:

```python
# ✅ CORRECT: Set seed FIRST
set_seed(42, deterministic=True)
cfg = Config()
detector = PoseDetector()
video_data = build_dataset_per_video(detector, cfg)  # Uses random sampling

# ❌ WRONG: Set seed AFTER imports/operations
cfg = Config()
detector = PoseDetector()
video_data = build_dataset_per_video(detector, cfg)  # Random sampling NOT seeded
set_seed(42, deterministic=True)  # Too late!
```

### Updated Scripts

All three main scripts now use proper seeding:

1. **src/train.py**:
   ```python
   from .utils import set_seed, print_seed_info

   # Set seed BEFORE building dataset
   set_seed(seed=42, deterministic=True)
   print_seed_info(seed=42, deterministic=True)

   # Now build dataset (uses random flow sampling)
   video_data = build_dataset_per_video(detector, cfg)
   ```

2. **src/evaluate.py**:
   ```python
   from .utils import set_seed, print_seed_info

   # Set seed BEFORE processing videos
   set_seed(seed=42, deterministic=True)
   print_seed_info(seed=42, deterministic=True)

   # Now evaluate videos (uses random flow sampling)
   result = evaluate_video(video_path, ...)
   ```

3. **src/infer.py**:
   ```python
   from .utils import set_seed

   # Set seed BEFORE inference
   set_seed(seed=42, deterministic=True)

   # Now process video (uses random flow sampling)
   hazard = model(input)
   ```

## Verification

### Test Reproducibility

Run evaluation twice on the same video and verify identical results:

```bash
# Run 1
python -m src.evaluate holdout/ > eval_run1.log

# Run 2
python -m src.evaluate holdout/ > eval_run2.log

# Compare
diff eval_run1.log eval_run2.log
# Should show NO differences (except timestamps)
```

### Check Specific Video

```python
# Extract max_hazard for specific video from logs
grep "h15.mp4" eval_run1.log
grep "h15.mp4" eval_run2.log

# Should show IDENTICAL max_hazard values
# Example:
# h15.mp4 | max_hazard=0.856  (both runs)
```

### Training Reproducibility

Train model twice and verify:

```bash
# Run 1
python -m src.train
mv outputs/checkpoints/hazard_gru.pt outputs/checkpoints/model_run1.pt

# Run 2
python -m src.train
mv outputs/checkpoints/hazard_gru.pt outputs/checkpoints/model_run2.pt

# Compare model weights (should be identical)
python -c "
import torch
w1 = torch.load('outputs/checkpoints/model_run1.pt')
w2 = torch.load('outputs/checkpoints/model_run2.pt')
for key in w1.keys():
    assert torch.allclose(w1[key], w2[key], atol=1e-6), f'{key} differs'
print('✓ Models are identical')
"
```

## Performance Impact

### Deterministic Mode Trade-offs

**Advantages**:
- ✅ Fully reproducible results
- ✅ Easier debugging
- ✅ Consistent evaluation metrics
- ✅ Comparable experiments

**Disadvantages**:
- ❌ ~5-10% slower training (CUDA determinism)
- ❌ ~2-5% slower inference (CUDA determinism)

### Disabling Deterministic Mode (Not Recommended)

If you need maximum speed and don't care about reproducibility:

```python
# In your script (NOT recommended for reproducibility)
set_seed(seed=42, deterministic=False)

# This will:
# - Still seed NumPy (optical flow will be deterministic)
# - NOT enforce CUDA determinism (training will be faster but non-reproducible)
```

## Common Issues

### Issue: Still Getting Different Results

**Causes**:
1. Seed not set before feature extraction
2. Different PyTorch/CUDA versions
3. Different hardware (CPU vs GPU)
4. External non-determinism (e.g., timestamp-based operations)

**Solutions**:
```python
# 1. Check seed timing
# Make sure set_seed() is called FIRST thing in main()

# 2. Check versions
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"

# 3. Force CPU for exact reproducibility
device = torch.device("cpu")  # Instead of cuda

# 4. Check for external randomness
# Search for any time-based operations or external random sources
```

### Issue: Training Still Non-Deterministic on GPU

**Cause**: Some CUDA operations have no deterministic implementation

**Solution**:
```python
# Option 1: Force CPU training (slower but deterministic)
device = torch.device("cpu")

# Option 2: Accept slight variation (<0.1% in metrics)
# Document the variation in your results

# Option 3: Use a specific PyTorch version known to be deterministic
pip install torch==2.0.1  # Example
```

### Issue: Different Results Between CPU and GPU

This is **expected**. CPU and GPU use different numerical precision and algorithms.

**Solution**:
```python
# Pick one device and stick with it for all experiments
device = torch.device("cpu")  # OR torch.device("cuda")

# Document which device was used in your results
```

## Best Practices

### 1. Always Set Seed First

```python
def main():
    # FIRST THING: Set seed
    set_seed(42, deterministic=True)

    # THEN: Do everything else
    cfg = Config()
    detector = PoseDetector()
    model = load_model()
    ...
```

### 2. Document Your Environment

Save environment information:

```bash
# Save versions
pip freeze > requirements_exact.txt

# Save hardware info
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
" > environment.txt
```

### 3. Use Version Control

```bash
# Tag your exact code version
git tag -a v1.0 -m "Reproducible training run"
git push origin v1.0
```

### 4. Log Random State

```python
# At start of training/evaluation
print(f"NumPy random state: {np.random.get_state()[1][0]}")
print(f"PyTorch random state: {torch.initial_seed()}")
```

## Summary

### What Changed

1. ✅ Created `src/utils.py` with centralized seeding
2. ✅ Updated `src/train.py` to set seed before dataset building
3. ✅ Updated `src/evaluate.py` to set seed before video processing
4. ✅ Updated `src/infer.py` to set seed before inference
5. ✅ Enabled CUDA deterministic mode by default

### Expected Behavior

**Before fix**:
```
Run 1: h15.mp4 max_hazard=0.856
Run 2: h15.mp4 max_hazard=0.849
Run 3: h15.mp4 max_hazard=0.862
```

**After fix**:
```
Run 1: h15.mp4 max_hazard=0.856
Run 2: h15.mp4 max_hazard=0.856
Run 3: h15.mp4 max_hazard=0.856
```

### Performance Cost

- Training: ~5-10% slower (due to CUDA determinism)
- Evaluation: ~2-5% slower (due to CUDA determinism)
- **Worth it** for reproducible research and debugging

---

For questions about reproducibility, open an issue on GitHub with the `reproducibility` label.
