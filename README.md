# PrameyaEngine

PrameyaEngine is a Python-based forensic framework for **image and video authentication** built around the **Six Pillars of Detection**. It produces a single, normalized output per analyzed frame (or per aggregated video run):

- **Probability of Tampering**: a float in **[0, 1]** computed as a weighted average of the six pillar scores.

**Images** are analyzed directly via `PrameyaEngine(image_path)`. **Videos** are supported by **sampling frames** (for example with OpenCV `VideoCapture`), writing each frame to a temporary image file or path the engine can read, and then **aggregating** frame-level scores (mean, maximum, or fraction above a threshold) for a clip-level verdict.

This repository contains a working implementation of all six pillars on **raster frames** (the same logic applies whether the source is a still photo or a decoded video frame):

- **ELA**: Error Level Analysis (Pillow-based recompression + difference variance)
- **DQ**: Double Quantization (8×8 DCT with `scipy.fftpack` and AC histogram “holes”)
- **PRNU**: Sensor fingerprinting heuristic (noise residual via OpenCV NLM denoise + abnormality scoring)
- **CFA**: Bayer-grid artifact heuristic (Laplacian local variance + cross-channel disagreement)
- **SVD**: Copy-move detection (overlapping blocks + SVD singular vectors + lexicographic sorting)
- **Metadata & Hex Audit**: EXIF + raw bytes scan for editor footprints and missing camera tags (most meaningful on **exported frame files** such as JPEG/PNG; container-level metadata is not read by the engine)

> Note: This is a **forensic heuristic framework**. Scores depend on image type, compression, and content; calibrate against your dataset. For video, temporal consistency (frame-to-frame jumps) is not a separate pillar—combine frame scores with your own temporal rules if needed.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start (images)

```python
from prameya_engine import PrameyaEngine

engine = PrameyaEngine("path/to/image.jpg")

prob_tampering = engine.generate_truth_score()
print("Probability of Tampering:", prob_tampering)

# You can also call pillars directly:
print("ELA:", engine.detect_ela())
print("DQ :", engine.detect_double_quantization())
print("PRNU:", engine.detect_prnu())
print("CFA :", engine.detect_cfa())
print("SVD :", engine.detect_svd())
print("META:", engine.audit_metadata())
```

## Video tampering (frame-based analysis)

The engine operates on **single images**. For video, decode frames with OpenCV, save selected frames to disk (JPEG/PNG), run `PrameyaEngine` on each path, then aggregate.

Example: sample every *n*-th frame and use the **mean** tampering probability across sampled frames (adjust stride and aggregation to match your latency and accuracy needs):

```python
import os
import tempfile

import cv2

from prameya_engine import PrameyaEngine


def analyze_video(video_path: str, frame_stride: int = 30) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    scores = []
    frame_index = 0

    with tempfile.TemporaryDirectory() as tmp:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                frame_path = os.path.join(tmp, f"f_{frame_index:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                engine = PrameyaEngine(frame_path)
                scores.append(engine.generate_truth_score())
            frame_index += 1

    cap.release()

    if not scores:
        return {
            "frames_decoded": frame_index,
            "samples": 0,
            "mean_probability": None,
            "max_probability": None,
        }

    return {
        "frames_decoded": frame_index,
        "samples": len(scores),
        "mean_probability": sum(scores) / len(scores),
        "max_probability": max(scores),
        "per_frame_scores": scores,
    }


# result = analyze_video("path/to/video.mp4", frame_stride=30)
# print(result["mean_probability"], result["max_probability"])
```

**Practical notes for video:**

- **Stride**: smaller stride = more frames = slower but smoother coverage; use keyframe-only steps for speed if edits are localized in time.
- **Aggregation**: `max` is sensitive to a single bad frame; **mean** or **95th percentile** often behave better for long clips.
- **ELA / DQ / metadata**: re-saving frames as JPEG in the snippet above introduces compression; for fair ELA/DQ on raw decode, consider lossless PNG intermediates or extend the engine to accept in-memory arrays (not in this repo today).
- **Copy-move / splicing**: `detect_svd()` and CFA-style cues can flag duplicated or inconsistent regions **within a frame**; cross-frame deepfake detection would require additional temporal models beyond these pillars.

### Visual tamper maps for video

Call `generate_visual_report()` on **individual frames** of interest (e.g., frames where `generate_truth_score()` exceeds a threshold) to produce ELA heatmaps and SVD maps under `results/`, same as for still images.

## Visual tamper maps for dashboards (images / per frame)

`PrameyaEngine` can generate **localized tamper maps** suitable for UI overlays and save them as PNG files in a `results` folder:

```python
from prameya_engine import PrameyaEngine

engine = PrameyaEngine("path/to/image.jpg")

report = engine.generate_visual_report(results_dir="results")

print("ELA heatmap PNG:", report["ela_heatmap_path"])
print("SVD tamper map PNG:", report["svd_tamper_map_path"])
print("SVD matching blocks:", report["svd_matches"])  # List[(y1, x1, y2, x2)]
```

- **ELA pillar**: saves a **grayscale heatmap** PNG (`*_ELA_tamper_map.png`) where **brighter pixels indicate higher probability of tampering**.
- **SVD (Copy-Move) pillar**:
  - returns a list of matching block coordinate pairs: `(y1, x1, y2, x2)` in image pixel space;
  - saves a **binary tamper map** PNG (`*_SVD_tamper_map.png`) where bright regions correspond to detected copy-move blocks.

These PNGs and coordinates can be consumed directly by your dashboard to draw overlays or highlight suspicious regions.

## File overview

- `prameya_engine.py`: The full implementation (`PrameyaEngine`).
- `test_bench.py`: CLI to run all six pillars on three image paths (`--original`, `--edited`, `--screenshot`) and print a short report.
- `requirements.txt`: Runtime dependencies.

## Public API (what to call)

### Engine

- `PrameyaEngine(image_path: str)`: loads the image via OpenCV (`cv2.imread`).

### Pillar methods (each returns a score in **[0, 1]**)

- `detect_ela()` → calls `run_ela()`
- `detect_double_quantization()` → calls `run_double_quantization()`
- `detect_prnu()` → calls `run_prnu()`
- `detect_cfa()` → calls `run_cfa()`
- `detect_svd()` → calls `run_svd_copy_move()`
- `audit_metadata()` → calls `run_metadata_hex_audit()`

### Aggregation (final score)

#### `generate_truth_score(weights: dict[str, float] | None = None) -> float`

Computes a **weighted average** of the six pillar scores.

- **Expected weight keys**:
  - `ela`
  - `dq`
  - `prnu`
  - `cfa`
  - `svd`
  - `metadata`

Example with custom weights:

```python
weights = {"ela": 1.0, "dq": 1.2, "prnu": 1.0, "cfa": 0.9, "svd": 1.4, "metadata": 0.8}
print(engine.generate_truth_score(weights))
```

## Pillar details (what each one does)

### 1) ELA (Error Level Analysis)

- **Implementation**: `run_ela(quality=90, scale=15.0)`
- **How it works**:
  - Loads the image with Pillow, resaves it to JPEG at `quality=90` into memory
  - Computes pixel-wise absolute difference between original and resaved
  - Scales the difference image (brightness enhancement) so compression artifacts become visible
  - Computes a **normalized tamper score** from the **variance** of the enhanced difference map

### 2) DQ (Double Quantization)

- **Implementation**: `run_double_quantization(block_size=8, bins=50)`
- **How it works**:
  - Converts to grayscale
  - Splits into non-overlapping **8×8** blocks
  - Computes **2D DCT** per block using `scipy.fftpack.dct` (row DCT then column DCT)
  - Collects **AC coefficients** (all DCT terms except DC)
  - Builds a histogram of AC magnitudes and estimates a score from periodic “holes” (zero-count bins)

### 3) PRNU (Sensor fingerprinting)

- **Implementation**: `run_prnu(h=10.0, template_window_size=7, search_window_size=21)`
- **How it works** (no-reference heuristic):
  - Uses `cv2.fastNlMeansDenoising` to estimate a “clean” version of the image
  - Computes residual: `residual = original - denoised`
  - Scores “abnormality” using:
    - residual energy ratio (too low or too high can be suspicious)
    - neighbor correlation in residual (structured residuals can be suspicious)

### 4) CFA (Bayer grid artifacts)

- **Implementation**: `run_cfa(ksize=3, blur_ksize=7)`
- **How it works**:
  - Computes Laplacian high-frequency response per color channel
  - Measures local variance of Laplacian magnitude
  - Computes a cross-channel disagreement map; higher disagreement increases the score

### 5) SVD (Copy-move detection)

- **Implementation**: `run_svd_copy_move(block_size=16, step=8, k=8, feature_decimals=3, min_shift=12)`
- **How it works**:
  - Extracts **overlapping blocks** (sliding window)
  - Computes **SVD** per block; keeps the top `k` singular values
  - Normalizes + rounds (quantizes) the singular vectors to stabilize matching
  - Uses **lexicographical sorting** and compares neighbors in sorted order
  - Matches with small spatial shift are ignored; repeated distant matches raise the score

### 6) Metadata & Hex audit

- **Implementation**: `run_metadata_hex_audit()`
- **How it works**:
  - Reads EXIF tags via `PIL.ExifTags` and searches values for editor footprints such as:
    - Photoshop, GIMP, Canva (plus a few common editors)
  - Scans raw file bytes (ASCII-ish + basic UTF-16LE decode) for the same footprints using regex
  - Flags missing camera-ish tags that are often present in real camera photos:
    - Make, Model, DateTimeOriginal, ExifVersion, ExposureTime, ISO, FNumber, FocalLength, etc.

## Test bench (images)

Compare three images from the command line:

```bash
python test_bench.py --original path/to/original.jpg --edited path/to/edited.jpg --screenshot path/to/screenshot.jpg
```

## Repro / sanity check script

Create `check_engine.py` next to `prameya_engine.py`:

```python
from prameya_engine import PrameyaEngine

img = "path/to/image.jpg"
engine = PrameyaEngine(img)

print("ELA     :", engine.detect_ela())
print("DQ      :", engine.detect_double_quantization())
print("PRNU    :", engine.detect_prnu())
print("CFA     :", engine.detect_cfa())
print("SVD     :", engine.detect_svd())
print("Metadata:", engine.audit_metadata())
print("TOTAL   :", engine.generate_truth_score())
```
