# System Pipeline Diagram - PowerPoint Guide

This guide shows you how to create a professional pipeline diagram in PowerPoint that fits on one slide.

---

## Layout Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PRE-CONTACT DETECTION SYSTEM PIPELINE                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  [Video] → [Preprocessing] → [Pose Detection] → [Features] → [GRU] → [Alert] │
│   30 FPS      ↓ 10 FPS         17 Keypoints      51-dim     0-1      3 Levels│
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## PowerPoint Instructions

### Slide Dimensions
- **Layout**: Blank slide
- **Orientation**: Landscape
- **Margins**: 0.5 inch all sides

### Color Scheme
- **Primary Blue**: RGB(0, 120, 212) - for main boxes
- **Accent Orange**: RGB(255, 140, 0) - for highlights
- **Gray**: RGB(128, 128, 128) - for labels
- **Green**: RGB(16, 124, 16) - for output/success
- **White**: RGB(255, 255, 255) - for text

---

## Step-by-Step Building Instructions

### 1. Title (Top)
```
Position: Top center
Font: Arial Bold, 24pt
Text: "Pre-contact Detection System Pipeline"
Color: Dark Blue RGB(0, 51, 102)
```

### 2. Main Pipeline Flow (6 boxes, left to right)

Create 6 rounded rectangles in a horizontal line:

#### Box 1: VIDEO INPUT
```
Position: Left side
Size: 1.5" wide × 1.2" tall
Fill: Light Blue RGB(173, 216, 230)
Border: 2pt, Dark Blue
Text (3 lines):
  "VIDEO"
  "INPUT"
  "30 FPS"
Font: Arial Bold, 14pt
Icon: 🎥 (or video camera icon)
```

#### Box 2: PREPROCESSING
```
Position: After Box 1
Size: 1.5" wide × 1.2" tall
Fill: Light Gray RGB(220, 220, 220)
Border: 2pt, Gray
Text (3 lines):
  "PRE-"
  "PROCESSING"
  "↓ 10 FPS"
Font: Arial Bold, 12pt
Icon: ⚙️ (or gear icon)
```

#### Box 3: POSE DETECTION
```
Position: After Box 2
Size: 1.5" wide × 1.2" tall
Fill: Light Orange RGB(255, 228, 196)
Border: 2pt, Orange
Text (4 lines):
  "POSE"
  "DETECTION"
  "YOLOv8m"
  "Hailo NPU"
Font: Arial Bold, 12pt
Icon: 🤸 (or person icon)
Sub-label below: "17 keypoints"
```

#### Box 4: FEATURE EXTRACTION
```
Position: After Box 3
Size: 1.5" wide × 1.2" tall
Fill: Light Purple RGB(221, 160, 221)
Border: 2pt, Purple
Text (4 lines):
  "FEATURE"
  "EXTRACTION"
  "51-dim"
  "Pose+Flow+Interaction"
Font: Arial Bold, 12pt
Icon: 📊 (or chart icon)
```

#### Box 5: GRU MODEL
```
Position: After Box 4
Size: 1.5" wide × 1.2" tall
Fill: Light Blue RGB(173, 216, 230)
Border: 2pt, Dark Blue
Text (4 lines):
  "GRU"
  "MODEL"
  "5-frame"
  "window"
Font: Arial Bold, 12pt
Icon: 🧠 (or brain icon)
Sub-label below: "64 hidden units"
```

#### Box 6: ALERT OUTPUT
```
Position: After Box 5 (right side)
Size: 1.5" wide × 1.2" tall
Fill: Light Green RGB(144, 238, 144)
Border: 2pt, Dark Green
Text (3 lines):
  "ALERT"
  "LEVEL"
  "0.0-1.0"
Font: Arial Bold, 14pt
Icon: ⚠️ (or alert icon)
```

### 3. Arrows Between Boxes

Draw thick arrows (3pt) connecting each box:
```
Arrow style: Block arrows or thick lines with arrowheads
Color: Dark Gray RGB(64, 64, 64)
Width: 3pt
```

### 4. Detail Boxes (Below Main Pipeline)

Create 3 detailed explanation boxes below the main pipeline:

#### Detail Box A: FEATURES (Below Box 4)
```
Position: Below Feature Extraction box
Size: 3" wide × 1.5" tall
Fill: Very Light Purple RGB(240, 230, 250)
Border: 1pt, Purple (dashed)
Text (bullet points, 10pt Arial):
  "• Reliability/Bbox: 17 features
  • Upper/Lower Pose: 15 features
  • Flow (torso+lower+bg): 10 features
  • Posture + Dynamics: 6 features
  • Interaction: 3 features
  = 51 features + 51 masks → 102-dim"
```

#### Detail Box B: TEMPORAL WINDOW (Below Box 5)
```
Position: Below GRU box
Size: 2" wide × 1.5" tall
Fill: Very Light Blue RGB(230, 240, 255)
Border: 1pt, Blue (dashed)
Text (centered, 10pt Arial):
  "Sliding Window

  [t-4][t-3][t-2][t-1][t]

  0.5 seconds
  @ 10 FPS"
```

#### Detail Box C: WARNING LEVELS (Below Box 6)
```
Position: Below Alert box
Size: 2" wide × 1.5" tall
Fill: Very Light Green RGB(240, 255, 240)
Border: 1pt, Green (dashed)
Text (3 lines, 10pt Arial):
  "PRE-CONTACT: 0.50
   HIGH: 0.60
   CRITICAL: 0.80"
Color code each line:
  - PRE: Yellow RGB(255, 255, 0)
  - HIGH: Orange RGB(255, 140, 0)
  - CRITICAL: Red RGB(255, 0, 0)
```

### 5. Performance Metrics (Top Right Corner)

Create a small info box:
```
Position: Top right corner
Size: 2" wide × 1" tall
Fill: Very Light Gray RGB(245, 245, 245)
Border: 1pt, Gray
Text (small, 9pt Arial):
  "Performance:
   • Detection: 94.6%
   • Lead Time: 0.24s
   • FP Rate: 0%"
```

### 6. Hardware Labels (Bottom)

Add small labels showing where processing happens:

```
Below Boxes 1-2: "CPU"
Below Box 3: "Hailo NPU" (in orange box)
Below Boxes 4-6: "CPU/GPU"

Style: Small rounded rectangles, 8pt font
Colors: Light yellow for CPU, Light orange for NPU
```

---

## Alternative: Simplified Version (Even More Compact)

If the above is too detailed, use this simplified version:

### Simplified Layout

```
┌─────────────────────────────────────────────────────────────────┐
│              PRE-CONTACT DETECTION PIPELINE                      │
│                                                                   │
│  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐           │
│  │Video│ →  │Pose │ →  │Feat │ →  │ GRU │ →  │Alert│           │
│  │30fps│    │YOLOv8│   │51-d │    │5×102│    │0-1  │           │
│  └─────┘    └─────┘    └─────┘    └─────┘    └─────┘           │
│                                                                   │
│  Details:                                                         │
│  • 10 FPS processing (downsample from 30)                        │
│  • 17 keypoints + optical flow → 51 features + 51 masks = 102-d │
│  • GRU: 5-frame window (0.5s), 64 hidden units                  │
│  • Output: PRE (0.50) | HIGH (0.60) | CRITICAL (0.80)           │
│                                                                   │
│  Performance: 94.6% detection | 0.24s lead | 0% false positives │
└─────────────────────────────────────────────────────────────────┘
```

**Create this in PowerPoint:**
1. 5 boxes across the middle
2. Arrows between them
3. Text details below
4. Performance metrics at bottom

---

## Visual Enhancement Tips

### Icons to Use (Insert → Icons in PowerPoint)
- Video camera for Video Input
- Gear/settings for Preprocessing
- Person/skeleton for Pose Detection
- Graph/chart for Features
- Brain/circuit for GRU
- Warning triangle for Alert

### Color Gradient (Optional)
Apply a subtle gradient from left (cool blue) to right (warm orange/green) to show progression from input to output.

### Animation (Optional)
If presenting:
1. Fade in each box sequentially (left to right)
2. Arrows appear after boxes
3. Detail boxes fade in last
4. Use "Appear" animation, 0.5s delay between elements

---

## Exact Dimensions for Perfect Fit

For a standard 16:9 slide (10" × 7.5"):

```
Main Pipeline Boxes:
- X positions: 0.75", 2.5", 4.25", 6.0", 7.75", 9.5"
- Y position: 1.5" (all boxes aligned)
- Width: 1.5" each
- Height: 1.2" each
- Spacing: 0.25" between boxes

Arrows:
- Start: Right edge of box
- End: Left edge of next box
- Y position: 2.1" (center of boxes)

Detail Boxes:
- Y position: 4.0"
- Heights: 1.5" each

Performance Box:
- Position: (8.5", 0.5")
- Size: 2" × 1"
```

---

## Color Palette Reference

Copy these RGB values into PowerPoint's custom colors:

```
Main Colors:
- Video Input:     RGB(173, 216, 230) - Light Blue
- Preprocessing:   RGB(220, 220, 220) - Light Gray
- Pose Detection:  RGB(255, 228, 196) - Peach
- Features:        RGB(221, 160, 221) - Plum
- GRU:             RGB(173, 216, 230) - Light Blue
- Alert:           RGB(144, 238, 144) - Light Green

Borders:
- Standard:        RGB(64, 64, 64) - Dark Gray
- Highlight:       RGB(255, 140, 0) - Orange

Text:
- Headers:         RGB(0, 51, 102) - Navy
- Body:           RGB(0, 0, 0) - Black
- Labels:         RGB(96, 96, 96) - Gray

Warning Levels:
- PRE-CONTACT:    RGB(255, 255, 0) - Yellow
- HIGH:           RGB(255, 140, 0) - Orange
- CRITICAL:       RGB(255, 0, 0) - Red
```

---

## Final Checklist

Before finalizing your slide:

- [ ] All text is readable from 10 feet away (minimum 12pt font)
- [ ] Color contrast is sufficient (dark text on light backgrounds)
- [ ] Arrows clearly show flow direction
- [ ] Icons are consistent in style
- [ ] No overcrowding (white space is good)
- [ ] Technical terms are accurate (GRU not LSTM, 10 FPS not 30)
- [ ] Numbers are up-to-date (94.6%, 0.24s, 0%)

---

## Quick Start: Pre-made Template

If you want to save time, here's a text-based template you can copy into PowerPoint's "Designer" feature:

```
Slide Title: Pre-contact Detection System Pipeline

Main content:
Video (30fps) → Preprocessing (10fps) → YOLOv8 Pose (Hailo NPU) → Features (51-dim + 51 masks) → GRU (5-frame window) → Alert (0-1 score)

Key Details:
• Processing: 10 FPS (downsampled from 30 FPS input)
• Pose: 17 keypoints from YOLOv8m on Hailo-8L NPU
• Features: 51-dim (pose + flow + posture + dynamics + interaction) + 51 validity masks = 102-dim model input
• Model: GRU with 64 hidden units, 5-frame sliding window (0.5s)
• Output: Multi-level warnings (PRE: 0.50, HIGH: 0.60, CRITICAL: 0.80)

Performance: 94.6% detection rate | 0.24s lead time | 0% false positives
```

PowerPoint Designer will auto-generate layouts - choose the horizontal flow diagram option.

---

For questions or if you need the diagram in a different format (SVG, PNG, etc.), let me know!
