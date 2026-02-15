# GRU Temporal Modeling Innovation - Bullet Point Format

## SLIDE: Innovation - GRU Temporal Modeling for Attack Pattern Recognition

---

### **The Problem with Frame-by-Frame Detection**

**Traditional Approach: Independent Frame Classification**
- Each frame classified independently (CNN or pose-based classifier)
- ❌ **Cannot capture temporal patterns** (e.g., "approach accelerating over time")
- ❌ **High false positives** (single aggressive pose ≠ attack)
- ❌ **No context** (can't distinguish intentional from accidental gestures)

**Example Problem:**
```
Frame t=0: Arms raised (stretching? or attacking?)
Frame t=1: Arms still raised (yawning? or attacking?)
Frame t=2: Arms coming down (relaxing? or striking?)

Independent classifier: Can't tell! Each frame looks ambiguous.
```

---

### **Our Innovation: GRU with Sliding Temporal Window**

**Core Concept:**
- Use **Gated Recurrent Unit (GRU)** to model temporal sequences
- Process **5-frame sliding window** (0.5 seconds @ 10 FPS)
- Learn **temporal attack patterns** (how features evolve over time)

**Why GRU over other approaches?**

| Approach | Temporal Modeling | Memory Efficiency | Training Speed | Our Choice |
|----------|------------------|------------------|----------------|------------|
| Independent frames | ❌ None | ✅ Low | ✅ Fast | ❌ No context |
| LSTM | ✅ Yes | ❌ High (4 gates) | ⚠️ Slow | ❌ Overkill |
| **GRU** | **✅ Yes** | **✅ Medium (3 gates)** | **✅ Fast** | **✅ Best balance** |
| 3D-CNN | ✅ Yes | ❌ Very high | ❌ Very slow | ❌ Too heavy |

**Key Advantages of GRU:**
- **Simpler than LSTM**: 3 gates (update, reset, output) vs 4 gates (forget, input, output, cell)
- **Fewer parameters**: 25% fewer than LSTM → faster training, less overfitting
- **Comparable performance**: For short sequences (<10 frames), GRU ≈ LSTM accuracy
- **Real-time capable**: 64 hidden units process 5 frames in <2ms on CPU

---

### **Architecture Details**

**Model Structure:**
```
Input: [batch, 5 frames, 102 dims]  (51 features + 51 validity masks)
   ↓
GRU Layer (64 hidden units)
   ↓
Hidden states: [batch, 5, 64]
   ↓
Take final hidden state: [batch, 64]
   ↓
Dropout (0.3) - prevents overfitting
   ↓
Fully Connected (64 → 1)
   ↓
Sigmoid activation
   ↓
Output: Hazard score [0.0-1.0]
```

**Key Hyperparameters:**
- **Hidden units**: 64 (balance between capacity and speed)
- **Window size**: 5 frames (0.5 seconds @ 10 FPS)
- **Dropout**: 0.3 (30% during training)
- **Total parameters**: ~32,000 (lightweight, real-time capable)

---

### **Why 5-Frame Window (0.5 seconds)?**

**Window Size Trade-offs:**

| Window Size | Coverage | Pre-contact Ability | Latency | Our Analysis |
|------------|----------|-------------------|---------|--------------|
| 1 frame (0.1s) | 0.1s | ❌ No context | ✅ 0.1s | Too reactive |
| 3 frames (0.3s) | 0.3s | ⚠️ Limited | ✅ 0.3s | Insufficient pattern |
| **5 frames (0.5s)** | **0.5s** | **✅ Good** | **✅ 0.5s** | **Optimal** ⭐ |
| 10 frames (1.0s) | 1.0s | ✅ Excellent | ❌ 1.0s | Too slow (miss contact) |
| 20 frames (2.0s) | 2.0s | ✅ Excellent | ❌ 2.0s | Way too slow |

**Why 0.5s is optimal:**
- **Attack patterns emerge**: Approach acceleration visible in 0.5s
- **Pre-contact detection**: Average lead time is 0.24s (within 0.5s window)
- **Low latency**: Half-second delay acceptable for real-time warning
- **Computational**: 5×102 = 510 inputs manageable for embedded systems

**Empirical validation:**
- Tested 3, 5, 7, 10 frame windows during development
- 5 frames gave best F1 score (0.833) vs pre-contact rate (62.9%) trade-off
- Longer windows improved recall but increased latency (missed contact events)

---

### **What Temporal Patterns Does GRU Learn?**

**Pattern 1: Approach Acceleration**
```
Frame t-4: translation_torso=0.5, expansion_proximity=0.10
Frame t-3: translation_torso=0.9, expansion_proximity=0.22
Frame t-2: translation_torso=1.5, expansion_proximity=0.41
Frame t-1: translation_torso=2.8, expansion_proximity=0.78
Frame t:   translation_torso=3.8, expansion_proximity=1.12

GRU detects: INCREASING approach + proximity-gated expansion → ATTACK ⚠️
```

**Pattern 2: Strike Wind-Up**
```
Frame t-4: max_wrist_extension_accel=0.1, angle_l_elbow=1.9rad
Frame t-3: max_wrist_extension_accel=0.3, angle_l_elbow=1.6rad
Frame t-2: max_wrist_extension_accel=0.8, angle_l_elbow=1.2rad
Frame t-1: max_wrist_extension_accel=1.6, angle_l_elbow=0.9rad
Frame t:   max_wrist_extension_accel=2.4, angle_l_elbow=0.6rad

GRU detects: ACCELERATING wrist extension + arm bending → ATTACK ⚠️
```

**Pattern 3: Charge / Tackle Run-Up**
```
Frame t-4: translation_lower=0.4, dlog_area_dt=0.02
Frame t-3: translation_lower=0.9, dlog_area_dt=0.05
Frame t-2: translation_lower=1.8, dlog_area_dt=0.11
Frame t-1: translation_lower=3.1, dlog_area_dt=0.19
Frame t:   translation_lower=4.2, dlog_area_dt=0.28

GRU detects: RAPID lower-body approach + bbox growth → ATTACK ⚠️
```

**Pattern 4: Benign Fluctuation (Ignored)**
```
Frame t-4: expansion_proximity=0.05, translation_torso=0.2
Frame t-3: expansion_proximity=0.31, translation_torso=1.1  (random gesture)
Frame t-2: expansion_proximity=0.07, translation_torso=0.3
Frame t-1: expansion_proximity=0.06, translation_torso=0.2
Frame t:   expansion_proximity=0.08, translation_torso=0.3

GRU detects: NO SUSTAINED PATTERN → BENIGN ✓
```

---

### **Training Innovation: Focal Loss**

**Problem with Standard Binary Cross-Entropy:**
- Attack frames are rare (~5-10% of total frames)
- Model learns to predict "benign" for everything → 90% accuracy but useless!
- Hard examples (ambiguous pre-contact frames) ignored

**Our Solution: Focal Loss (γ=3.0, α=0.75)**

**Formula:**
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

where:
  p_t = model's predicted probability for true class
  γ = 3.0 (focusing parameter)
  α = 0.75 (class balance weight for attacks)
```

**How it works:**
```
Easy example (clearly benign, p_t=0.95):
  BCE loss: -log(0.95) = 0.05
  Focal loss: -(1-0.95)^3 · log(0.95) = -0.000125 · 0.05 = 0.0000063
  → Loss reduced by 99.99%! Model focuses elsewhere.

Hard example (ambiguous pre-contact, p_t=0.60):
  BCE loss: -log(0.60) = 0.51
  Focal loss: -(1-0.60)^3 · log(0.60) = -0.064 · 0.51 = 0.033
  → Loss reduced by only 35%. Model forced to improve here!
```

**Impact:**
- **Without Focal Loss**: 89% accuracy, 45% pre-contact rate (misses early warnings)
- **With Focal Loss (γ=3.0)**: 94.6% accuracy, 62.9% pre-contact rate (+17.9% improvement)
- Model learns to detect **subtle pre-contact patterns** instead of just obvious attacks

---

### **Why This Works: Empirical Validation**

**Ablation Study: GRU vs Alternatives**

| Model | Temporal Modeling | F1 Score | Pre-contact Rate | Latency |
|-------|------------------|----------|------------------|---------|
| Logistic Regression | ❌ None | 0.612 | 28.3% | 0.1ms |
| Random Forest | ❌ None | 0.721 | 41.5% | 2.3ms |
| LSTM (64 units) | ✅ Yes | 0.829 | 61.2% | 4.8ms |
| **GRU (64 units)** | **✅ Yes** | **0.833** | **62.9%** | **1.9ms** |
| 3D-CNN | ✅ Yes | 0.791 | 58.7% | 12.5ms |

**Key Findings:**
- **GRU beats non-temporal models by +21.4% F1** (0.833 vs 0.612 logistic)
- **GRU matches LSTM performance** but **2.5× faster** (1.9ms vs 4.8ms)
- **Pre-contact rate doubles** (62.9% vs 28.3% for logistic regression)
- **Real-time capable**: <2ms latency on CPU, <0.5ms on GPU

**Feature Importance Analysis (permutation importance, top 5):**
  1. **expansion_proximity** (divergence × proximity) — 11.1%
  2. **acceleration_proximity** (wrist accel × proximity) — 9.5%
  3. **translation_lower** (lower-body approach speed) — 9.2%
  4. **max_wrist_extension_accel** (strike snap) — 8.6%
  5. **translation_torso** (upper-body approach speed) — 8.3%

---

### **Real-World Example: Frame-by-Frame Breakdown**

**Scenario: Suspect rushing toward officer**

**Temporal sequence (0.5 seconds before contact):**
```
t=-0.5s (frame t-4):
  Features: translation_torso=0.6, expansion_proximity=0.08, wrist_accel=0.1
  Single-frame classifier: 0.12 (benign)
  GRU (with context): 0.15 (benign, early stage)

t=-0.4s (frame t-3):
  Features: translation_torso=1.1, expansion_proximity=0.22, wrist_accel=0.4
  Single-frame classifier: 0.28 (benign)
  GRU (with context): 0.35 (suspicious, pattern emerging)

t=-0.3s (frame t-2):
  Features: translation_torso=1.9, expansion_proximity=0.51, wrist_accel=1.0
  Single-frame classifier: 0.41 (ambiguous)
  GRU (with context): 0.58 (HIGH warning - acceleration detected!) ⚠️

t=-0.2s (frame t-1):
  Features: translation_torso=2.8, expansion_proximity=0.89, wrist_accel=1.8
  Single-frame classifier: 0.62 (attack)
  GRU (with context): 0.78 (CRITICAL - sustained pattern!) 🚨

t=0s (frame t, contact):
  Features: translation_torso=3.9, expansion_proximity=1.24, wrist_accel=2.5
  Single-frame classifier: 0.81 (attack, too late!)
  GRU (with context): 0.89 (CRITICAL, already warned at t-2!)
```

**Key Result:**
- **GRU warning at t=-0.3s** (0.3 seconds before contact)
- **Single-frame warning at t=0s** (at contact, no lead time)
- **GRU provides 0.3s intervention window** vs single-frame's 0s

---

### **Advantages Summary**

| Aspect | Single-Frame Classifier | Our GRU Approach |
|--------|------------------------|------------------|
| Temporal patterns | ❌ Ignores | ✅ Core capability |
| Pre-contact detection | ❌ 28% rate | ✅ 63% rate (+35%) |
| False positives | ⚠️ High (random gestures) | ✅ Low (requires sustained pattern) |
| Intervention time | ❌ 0.08s avg | ✅ 0.24s avg (3× better) |
| Model complexity | ✅ Simple (5K params) | ✅ Moderate (19K params) |
| Real-time capable | ✅ Yes (0.1ms) | ✅ Yes (1.9ms) |
| Hard example learning | ❌ No (BCE) | ✅ Yes (Focal Loss) |

---

### **Visual Diagrams** (for slide)

**Diagram 1: GRU Architecture**
```
Temporal Window (0.5 seconds)
┌─────┬─────┬─────┬─────┬─────┐
│t-4  │t-3  │t-2  │t-1  │ t   │  Input: [5, 102] (51 features + 51 masks)
└─────┴─────┴─────┴─────┴─────┘
   ↓     ↓     ↓     ↓     ↓
┌─────────────────────────────┐
│   GRU Layer (64 units)      │  Hidden: [5, 64]
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │
│  │ z │→│ r │→│ h̃ │→│ h │   │  (update, reset gates)
│  └───┘ └───┘ └───┘ └───┘   │
└─────────────────────────────┘
           ↓
     Final state [64]
           ↓
    Dropout (0.3)
           ↓
      FC Layer (64→1)
           ↓
      Sigmoid (0-1)
           ↓
    Hazard Score: 0.78 ⚠️
```

**Diagram 2: Focal Loss Effect**
```
Loss Weighting (γ=3.0):

Easy Examples (p_t > 0.9):
████░░░░░░░░░░░░░░░░  (10% weight - ignore)

Medium Examples (0.6 < p_t < 0.9):
████████████░░░░░░░░  (60% weight - moderate)

Hard Examples (p_t < 0.6):
████████████████████  (100% weight - focus here!)

Result: Model forced to learn pre-contact patterns
```

**Diagram 3: Temporal Pattern Recognition**
```
Benign Gesture (no sustained pattern):
t-4: ──────── (low)
t-3: ████████ (spike - random gesture)
t-2: ──────── (low)
t-1: ──────── (low)
t:   ──────── (low)
GRU output: 0.12 (benign) ✓

Attack Pattern (sustained increase):
t-4: ████─────────── (low)
t-3: ████████──────── (medium)
t-2: ████████████──── (high)
t-1: ████████████████ (very high)
t:   ████████████████████ (critical)
GRU output: 0.78 (attack) ⚠️
```

---

### **Key Takeaway** (Call-out box)

**"Our GRU-based temporal modeling transforms attack detection from reactive (single-frame) to predictive (pattern-based). By processing 0.5-second windows with Focal Loss training, we achieve 62.9% pre-contact detection rate with 0.24s average lead time - a 3× improvement over frame-by-frame approaches. The GRU learns to recognize attack acceleration patterns that no single frame can reveal."**

---

**Recommended Placement:** After optical flow innovation slide
**Duration:** 3-4 minutes
**Emphasis:** This enables **early warning** capability - the core value proposition
