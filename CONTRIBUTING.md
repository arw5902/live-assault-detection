# Contributing to Real-Time Physical Threat Detection System

Thank you for your interest in contributing to the Real-Time Physical Threat Detection System! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Feature Requests](#feature-requests)
- [Bug Reports](#bug-reports)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility for mistakes and learn from them

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/arw5902/live-assault-detection.git
   cd live-assault-detection
   ```

3. **Set up remote upstream**:
   ```bash
   git remote add upstream https://github.com/originalrepo/live-assault-detection.git
   ```

4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Environment Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download models:
   ```bash
   mkdir -p models
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt -O models/yolov8m-pose.pt
   ```

### Verify Installation

Run a quick test to ensure everything is set up:

```bash
# Test inference on sample video (if you have one)
python -m src.infer data/safe/sample.mp4

# Or test model loading
python -c "from src.model import HazardGRU; print('Model imports successfully')"
```

## How to Contribute

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Performance**: Optimize existing code
5. **Tests**: Add or improve test coverage
6. **Examples**: Add usage examples or tutorials

### Contribution Workflow

1. **Check existing issues** to see if someone is already working on it
2. **Create an issue** if one doesn't exist (for features/bugs)
3. **Discuss your approach** in the issue before starting work
4. **Implement your changes** following coding standards
5. **Test thoroughly** to ensure nothing breaks
6. **Submit a pull request** with clear description

## Coding Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Write **docstrings** for all public functions/classes
- Keep functions **focused and small** (< 50 lines preferred)
- Use **meaningful variable names**

Example:
```python
def extract_features(
    frame: np.ndarray,
    detection: Dict[str, Any],
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from frame and pose detection.

    Args:
        frame: Input frame (BGR format)
        detection: Pose detection result
        config: Configuration object

    Returns:
        features: Feature vector (59-dim)
        mask: Validity mask (59-dim)
    """
    # Implementation
    ...
```

### Code Organization

- **One class per file** (unless closely related)
- **Group related functions** in modules
- **Import order**: stdlib → third-party → local
- **Avoid circular imports**

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain WHY, not WHAT
- **README updates**: Update if adding features
- **DESIGN.md updates**: Update if changing architecture

Example docstring:
```python
def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """
    Compute Focal Loss for binary classification.

    Focal Loss down-weights easy examples and focuses on hard examples,
    making it effective for class-imbalanced datasets.

    Args:
        pred: Predicted probabilities [batch_size] in range [0, 1]
        target: Ground truth binary labels [batch_size] in {0, 1}
        gamma: Focusing parameter (default: 2.0). Higher values increase
            focus on hard examples.
        alpha: Weighting factor for positive class (default: 0.75)

    Returns:
        Scalar loss value

    References:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_features.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test edge cases and error conditions

Example test:
```python
def test_windowize_short_sequence():
    """Test windowize with sequence shorter than window length."""
    X = np.random.randn(3, 59)  # Only 3 frames
    M = np.ones_like(X)
    y = np.array([0, 0, 1])

    Xw, Mw, yw = windowize(X, M, y, window_len=5, stride=1)

    # Should return empty arrays
    assert Xw.shape[0] == 0
    assert Mw.shape[0] == 0
    assert yw.shape[0] == 0
```

## Submitting Changes

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG** (if exists)
5. **Create pull request** with description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Other (describe)

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No breaking changes (or documented)
```

### Commit Messages

Use clear, descriptive commit messages:

```
✅ Good:
- "Add focal loss implementation for class imbalance"
- "Fix carry-forward imputation for missing keypoints"
- "Improve video-level train/val split algorithm"

❌ Bad:
- "Update code"
- "Fix bug"
- "Changes"
```

Format: `<type>: <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

## Feature Requests

To request a new feature:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with label `enhancement`
3. **Describe the feature** clearly:
   - What problem does it solve?
   - What is the expected behavior?
   - Are there any alternatives?
4. **Discuss the approach** before implementing

## Bug Reports

To report a bug:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with label `bug`
3. **Provide details**:
   - Description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Error messages and stack traces
4. **Minimal reproducible example** if possible

### Bug Report Template

```markdown
## Description
Clear description of the bug

## To Reproduce
Steps to reproduce:
1. ...
2. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8 or CPU-only]

## Error Message
```
Paste error message/stack trace here
```

## Additional Context
Any other relevant information
```

## Areas for Contribution

### High Priority

- [ ] Multi-person tracking and detection
- [ ] Model quantization for edge deployment
- [ ] Real-time visualization GUI
- [ ] REST API for deployment
- [ ] More comprehensive test suite

### Medium Priority

- [ ] Attention mechanism for temporal modeling
- [ ] Audio features integration
- [ ] Transfer learning experiments
- [ ] Mobile/web deployment examples

### Documentation

- [ ] Video tutorials
- [ ] More usage examples
- [ ] Performance benchmarks
- [ ] Architecture diagrams
- [ ] API reference

## Questions?

If you have questions:

1. Check the [README](README.md) and [DESIGN.md](docs/DESIGN.md)
2. Search existing issues
3. Create a new issue with label `question`
4. Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing! 🎉
