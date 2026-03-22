# Video Tampering Detector

A forensic video analysis tool that detects tampering, splicing, and frame manipulation using **Temporal Differencing**, **Frame Drop Detection**, and **Static Noise Analysis** — with a composite tampering score and visual heatmap output.

---

##  Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tampering Score](#-tampering-score)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [CLI Arguments](#-cli-arguments)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## Features

-  **Frame Drop Detection** — Catches missing/cut frames via timestamp gap analysis
-  **Static Noise Analysis** — Detects frozen/pasted-over regions using pixel std deviation
-  **Temporal Differencing** — Flags sudden pixel-level anomalies (3σ spike detection)
-  **Composite Tampering Score** — Weighted 0–100% score with risk tier classification
-  **Heatmap Generation** — JET colormap overlay showing *where* tampering occurred spatially
- **Rich Terminal Output** — Color-coded report with visual progress bars and score meter
- **Optional Diff Video** — Save frame-by-frame difference as `output_diff.mp4`

---

##  How It Works

The detector runs three independent analysis signals on every frame:

### Signal A — Frame Drop Detection (Weight: 35%)
Compares the timestamp gap between consecutive frames against the expected frame duration (`1000ms / FPS`). If a gap exceeds **1.9×** the expected duration, frames are flagged as missing — a strong indicator of a cut or splice.

### Signal B — Static Noise Analysis (Weight: 30%)
Measures the standard deviation of pixel intensity inside a Region of Interest (ROI) per frame. A `std_dev < 1.0` means near-zero noise — indicating a **static image was pasted** over the video. High noise variance across frames also flags possible re-encoding or sensor switching.

### Signal C — Temporal Diff Spikes (Weight: 35%)
Computes `absdiff` between every consecutive frame pair. Frames more than **3 standard deviations above the mean** (3σ) are flagged as anomalous spikes — these mark the exact moments an edit, insert, or splice occurred.

---

##  Tampering Score

The three signals are combined into a single weighted composite score:

```
Final Score = (Score_A × 0.35) + (Score_B × 0.30) + (Score_C × 0.35) × 100
```

| Score Range | Risk Level | Meaning |
|:-----------:|:----------:|---------|
| 0% – 9%     | ✅ CLEAN   | No significant tampering indicators |
| 10% – 34%   | 🟡 LOW     | Minor anomalies, possibly compression artifacts |
| 35% – 69%   | 🟠 MEDIUM  | Moderate signals — manual review recommended |
| 70% – 100%  | 🔴 HIGH    | Strong tampering indicators — video likely manipulated |

---

##  Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python  | 3.8+    | Runtime |
| opencv-python | ≥ 4.5 | Video I/O, frame processing, heatmap generation |
| numpy   | ≥ 1.21  | Array math, statistical analysis |

---

##  Installation

### Option 1 — Virtual Environment (Recommended)

```bash
# 1. Clone or download the project
git clone https://github.com/your-username/video-tampering-detector.git
cd video-tampering-detector

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install opencv-python numpy
```

### Option 2 — Conda Environment

```bash
# 1. Create and activate a conda environment
conda create -n tampering-detector python=3.10
conda activate tampering-detector

# 2. Install dependencies
pip install opencv-python numpy
```

### Option 3 — System-wide (Quick)

```bash
pip install opencv-python numpy
```

> ⚠️ **Note for Windows users:** If you see `ModuleNotFoundError: No module named 'cv2'`, make sure you have activated your virtual environment **before** running the script. Packages installed outside the venv are not accessible inside it.

### Verify Installation

```bash
python -c "import cv2; import numpy; print('cv2:', cv2.__version__, '| numpy:', numpy.__version__)"
```

Expected output:
```
cv2: 4.x.x | numpy: 1.x.x
```

---

##  Usage

### Basic Usage

```bash
python tampering_detector.py --video your_video.mp4
```

### With All Options

```bash
python tampering_detector.py --video your_video.mp4 --save-diff --roi 300 350 400 450 --diff-threshold 25 --jump-threshold 1.9
```

### Examples

```bash
# Analyze a CCTV clip
python tampering_detector.py --video cctv_footage.mp4

# Analyze with a custom noise ROI (focus on road area)
python tampering_detector.py --video road_cam.mp4 --roi 200 300 100 300

# Save the temporal difference video too
python tampering_detector.py --video clip.mp4 --save-diff

# Lower the diff threshold for more sensitive detection
python tampering_detector.py --video clip.mp4 --diff-threshold 15
```

---

##  Output Files

| File | Description |
|------|-------------|
| `output_heatmap.jpg` | JET colormap heatmap overlay — **Blue** = low change, **Red** = high change. Always generated. |
| `output_diff.mp4` | Frame-by-frame temporal difference video. Generated only with `--save-diff`. |

### Reading the Heatmap

- **Blue regions** → low inter-frame change → normal/authentic
- **Green/Yellow regions** → moderate change → could be motion or mild edit
- **Red regions** → high accumulated change → likely location of tampering/splicing

---

##  CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | `str` | *(required)* | Path to the input video file |
| `--save-diff` | flag | `False` | Save temporal diff video as `output_diff.mp4` |
| `--roi` | `int int int int` | Center 100×100 | Noise analysis region: `Y1 Y2 X1 X2` |
| `--diff-threshold` | `float` | `30` | Avg pixel diff value to flag a frame as high-change |
| `--jump-threshold` | `float` | `1.9` | Timestamp gap multiplier to detect frame drops |

### ROI (Region of Interest) Guide

The `--roi` argument lets you focus the noise analysis on a specific part of the frame:

```
--roi Y1 Y2 X1 X2
```

Example — focus on top-left 200×200 area:
```bash
--roi 0 200 0 200
```

Example — focus on center of a 1080p video:
```bash
--roi 440 640 760 1160
```

---

##  Project Structure

```
video-tampering-detector/
│
├── tampering_detector.py   # Main detection engine
├── README.md               # This file
│
├── output_heatmap.jpg      # Generated after each run
└── output_diff.mp4         # Generated with --save-diff
```

---

## 🗺️ Roadmap

- [x] Terminal-based detection engine
- [x] Composite tampering score (0–100%)
- [x] Heatmap generation
- [x] Frame drop detection
- [x] Static noise analysis
- [x] 3σ temporal diff spike detection
- [x] Tkinter GUI interface
- [x] Per-second timeline chart of tampering intensity
- [ ] PDF/JSON report export
- [x] Batch processing multiple videos
- [ ] Audio track discontinuity detection

---

##  License

MIT License — free to use, modify, and distribute.

---

> Built with Python + OpenCV · Forensic Video Analysis Tool
