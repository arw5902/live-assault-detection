# Quick Start Guide

Get up and running with the Pre-contact Detection System in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- Webcam or video files for testing

## Installation (3 steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download YOLOv8-Pose Model

```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt -O models/yolov8m-pose.pt
```

Or download manually from: https://github.com/ultralytics/assets/releases

### 3. Verify Installation

```bash
python -c "from src.model import HazardGRU; print('✓ Installation successful!')"
```

## Usage Examples

### Inference on Video

```bash
python -m src.infer path/to/video.mp4
```

### Training (if you have data)

```bash
# 1. Prepare data in this structure:
# data/
#   safe/
#     s1.mp4
#     s2.mp4
#   attack/
#     a1.mp4
#     a2.mp4

# 2. Run training
python -m src.train

# Output: outputs/checkpoints/hazard_gru_YYYYMMDD_HHMMSS.pt
#         outputs/checkpoints/meta.json  (stores model filename + best threshold)
```

### Evaluation (if you have holdout data)

```bash
# 1. Prepare holdout data:
# holdout/
#   labels.json
#   safe/
#     s1.mp4
#   attack/
#     h1.mp4

# 2. Run evaluation
python -m src.evaluate holdout/

# Output: Detailed performance metrics
```

## Data Format

### Training Data

**Attack videos MUST be trimmed:**
- Start frame: Onset (wind-up begins)
- End frame: End of violence
- Every frame contains attack behavior

**Safe videos:**
- Normal behavior only
- No attack or aggressive movement

### Holdout Data

**labels.json format:**
```json
{
  "h1.mp4": {
    "category": "attack",
    "onset_frame": 126,
    "attack_frame": 150
  },
  "s1.mp4": {
    "category": "safe"
  }
}
```

## Configuration

Edit `src/config.py` to customize:

```python
# Quick tweaks
window_len = 5              # Temporal window (frames)
safe_window_stride = 2      # Stride for safe videos
focal_gamma = 2.0           # Focus on hard examples
focal_alpha = 0.75          # Attack class weight
early_thresh = 0.50         # THREAT threshold (tuned during training)
```

## Common Issues

### "Cannot open video"
- Check codec: Use H.264/MP4
- Verify path is correct
- Ensure OpenCV installed with video support

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "CUDA not available" (optional)
```bash
# For GPU support, install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Low detection rate
- Check video quality (need clear person)
- Verify lighting (avoid darkness)
- Ensure person visible and reasonably sized

## Performance Tips

### Faster Inference
```python
# In config.py or runtime:
# Use smaller model (if available)
# Reduce processing resolution
# Increase frame_stride (process every 4th frame instead of 3rd)
```

### Better Accuracy
```python
# Collect more training data
# Increase window_len to 8-10 frames
# Lower threshold for more sensitivity
# Add more diverse attack scenarios
```

## Next Steps

1. **Read the full [README.md](README.md)** for detailed usage
2. **Check [DESIGN.md](docs/DESIGN.md)** for architecture details
3. **See [CONTRIBUTING.md](CONTRIBUTING.md)** to contribute
4. **Experiment** with your own data!

## Quick Commands Reference

| Task | Command |
|------|---------|
| Training | `python -m src.train` |
| Evaluation | `python -m src.evaluate holdout/` |
| Inference (video) | `python -m src.infer video.mp4` |
| Inference (simulate live) | `python -m src.infer video.mp4 --simulate-live` |
| Evaluate dataset | `python -m src.infer --eval-dir holdout/` |
| Live detection (Pi 5) | `python -m src.infer 0 --picamera2 --pi` |
| Check GPU | `python -c "import torch; print(torch.cuda.is_available())"` |
| Test imports | `python -c "from src.model import HazardGRU"` |

## Support

- **Questions**: Open an issue with label `question`
- **Bugs**: Open an issue with label `bug`
- **Features**: Open an issue with label `enhancement`

---

**Time to first inference: < 5 minutes** ⚡

Happy detecting! 🎯
