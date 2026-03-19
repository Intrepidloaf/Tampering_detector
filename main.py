"""
Video Tampering Detector
========================
Detects video tampering using:
  1. Temporal Differencing         - Frame-to-frame pixel change analysis
  2. Frame Drop Detection          - Timestamp gap analysis
  3. Static Noise Analysis         - Zero-noise region detection (paste-over edits)
  4. Heatmap Generation            - Visual overlay of tampered regions saved as image

Output:
  - Terminal report
  - output_heatmap.jpg  → Cumulative heatmap of suspicious regions
  - output_diff.mp4     → Temporal difference video (optional)

Usage:
  python tampering_detector.py --video <path_to_video> [--save-diff] [--roi y1 y2 x1 x2]
"""

import cv2
import numpy as np
import argparse
import sys
import os
from collections import defaultdict


# ─────────────────────────────────────────────
# ANSI Colors for terminal output
# ─────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def print_header():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════╗
║       VIDEO TAMPERING DETECTOR v1.1          ║
║  Temporal Differencing + Tampering Score     ║
╚══════════════════════════════════════════════╝{RESET}
""")


def print_section(title):
    print(f"\n{BOLD}{CYAN}── {title} {'─' * (42 - len(title))}{RESET}")


# ─────────────────────────────────────────────
# CORE DETECTION ENGINE
# ─────────────────────────────────────────────

def analyze_video(video_path, roi=None, save_diff=False, diff_threshold=30, jump_threshold=1.9):
    """
    Main analysis function.

    Args:
        video_path      : Path to the input video
        roi             : [y1, y2, x1, x2] region of interest for noise check.
                          If None, uses center of frame.
        save_diff       : Whether to save the temporal difference video
        diff_threshold  : Pixel diff value above which a pixel is "changed" (0-255)
        jump_threshold  : Frame gap multiplier to flag as a drop (default 1.9x)

    Returns:
        dict: Summary of findings
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{RED}[ERROR] Cannot open video: {video_path}{RESET}")
        sys.exit(1)

    # ── Video Properties ──────────────────────
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print_section("VIDEO METADATA")
    print(f"  File          : {os.path.basename(video_path)}")
    print(f"  Resolution    : {width}x{height}")
    print(f"  FPS           : {fps:.2f}")
    print(f"  Total Frames  : {total_frames}")
    print(f"  Duration      : {duration_sec:.2f}s")

    # ── ROI Setup ─────────────────────────────
    # Default: center 100x100 region for noise analysis
    if roi is None:
        cy, cx = height // 2, width // 2
        roi = [cy - 50, cy + 50, cx - 50, cx + 50]
    y1, y2, x1, x2 = roi
    print(f"  Noise ROI     : y[{y1}:{y2}], x[{x1}:{x2}]")

    # ── VideoWriter (optional diff video) ─────
    writer = None
    if save_diff:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter("output_diff.mp4", fourcc, fps, (width, height))

    # ── Heatmap accumulator ───────────────────
    # Accumulates absolute difference across all frames
    heatmap_acc = np.zeros((height, width), dtype=np.float64)

    # ── State Variables ───────────────────────
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print(f"{RED}[ERROR] Video is empty or unreadable.{RESET}")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    expected_ms  = 1000.0 / fps  # Expected time between frames in ms
    frame_count  = 1

    # Results
    drops           = []       # List of (frame, prev_ms, curr_ms, lost_ms)
    noise_samples   = []       # Std dev of ROI per frame
    high_diff_frames = []      # Frames with high average pixel difference
    frame_diffs     = []       # Average diff per frame (for statistics)

    print_section("ANALYZING FRAMES")
    print(f"  {DIM}Processing {total_frames} frames...{RESET}\n")

    # ── Main Loop ─────────────────────────────
    while True:
        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            break

        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Frame Drop Detection
        delta = curr_time - prev_time
        if frame_count > 1 and delta > (expected_ms * jump_threshold):
            lost = delta - expected_ms
            drops.append((frame_count, prev_time, curr_time, delta))
            print(f"  {RED}[DROP]{RESET} Frame {frame_count:5d} | "
                  f"{prev_time/1000:.3f}s → {curr_time/1000:.3f}s | "
                  f"Gap: {delta:.1f}ms (~{int(delta/expected_ms)} frames lost)")

        # 2. Temporal Differencing
        diff = cv2.absdiff(curr_gray, prev_gray)
        heatmap_acc += diff.astype(np.float64)

        avg_diff = np.mean(diff)
        frame_diffs.append(avg_diff)

        if avg_diff > diff_threshold:
            high_diff_frames.append((frame_count, avg_diff, curr_time))

        # 3. Noise / Static Region Check (ROI)
        roi_patch = curr_gray[y1:y2, x1:x2]
        if roi_patch.size > 0:
            std_dev = float(np.std(roi_patch))
            noise_samples.append(std_dev)

        # Write diff frame if requested
        if writer is not None:
            diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            writer.write(diff_color)

        # Progress indicator every 100 frames
        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:.0f}%  Frame {frame_count}/{total_frames}", end="\r")

        prev_gray = curr_gray
        prev_time = curr_time
        frame_count += 1

    print(f"\n  {GREEN}✓ Scan complete. {frame_count} frames processed.{RESET}")

    cap.release()
    if writer:
        writer.release()
        print(f"  {DIM}Diff video saved → output_diff.mp4{RESET}")

    # ─────────────────────────────────────────
    # GENERATE HEATMAP
    # ─────────────────────────────────────────
    heatmap_path = generate_heatmap(heatmap_acc, prev_frame, high_diff_frames, width, height)

    # ─────────────────────────────────────────
    # BUILD SUMMARY REPORT
    # ─────────────────────────────────────────
    summary = build_report(
        video_path, fps, total_frames, duration_sec,
        drops, noise_samples, frame_diffs, high_diff_frames,
        diff_threshold, heatmap_path
    )

    return summary


# ─────────────────────────────────────────────
# HEATMAP GENERATION
# ─────────────────────────────────────────────

def generate_heatmap(heatmap_acc, last_frame, high_diff_frames, width, height):
    """
    Normalize the accumulated difference map and apply a color heatmap.
    Overlays it semi-transparently on the last video frame.
    """

    # Normalize to 0-255
    norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
    norm_uint8 = norm.astype(np.uint8)

    # Apply JET colormap (blue=low change, red=high change)
    heatmap_color = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

    # Blend with last frame for context
    if last_frame is not None and last_frame.shape[:2] == (height, width):
        overlay = cv2.addWeighted(last_frame, 0.4, heatmap_color, 0.6, 0)
    else:
        overlay = heatmap_color

    # Draw markers on high-diff frame positions (sample up to 20)
    # These are the frames most likely to contain tampering
    # We mark the top 10% of frames by diff value
    if high_diff_frames:
        top_frames = sorted(high_diff_frames, key=lambda x: x[1], reverse=True)[:20]
        for (fnum, avg_diff, ts) in top_frames:
            label = f"F{fnum} ({avg_diff:.0f})"
            # Place label at top of frame, staggered
            y_pos = 30 + (fnum % 8) * 22
            cv2.putText(overlay, label, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Add legend bar
    legend_h = 30
    legend = np.zeros((legend_h, width, 3), dtype=np.uint8)
    for i in range(width):
        val = int((i / width) * 255)
        color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        legend[:, i] = color
    cv2.putText(legend, "LOW CHANGE", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(legend, "HIGH CHANGE", (width - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    final = np.vstack([overlay, legend])

    # Add title bar
    title_bar = np.zeros((40, width, 3), dtype=np.uint8)
    cv2.putText(title_bar, "TAMPERING HEATMAP - Temporal Difference Accumulation",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
    final = np.vstack([title_bar, final])

    out_path = "output_heatmap.jpg"
    cv2.imwrite(out_path, final)
    return out_path


# ─────────────────────────────────────────────
# TAMPERING SCORE ENGINE
# ─────────────────────────────────────────────

def compute_tampering_score(total_frames, drops, noise_samples, frame_diffs, high_diff_frames):
    """
    Computes a composite tampering percentage (0–100%) from 3 independent signals.
    Each signal contributes a weighted sub-score:

      Signal A — Frame Drops          (weight: 35%)
        Score = min(1.0, drop_count / max_tolerable_drops)
        Rationale: Even 1–2 hard drops in a short video is highly suspicious.
        Max tolerable: ceil(total_frames / 300) drops (1 per ~10s at 30fps)

      Signal B — Static Noise         (weight: 30%)
        Score = fraction of frames with ROI std_dev < 1.0  (near-zero = pasted image)
        Also penalizes inconsistent noise (high std variance → possible re-encoding)

      Signal C — Temporal Diff Spikes (weight: 35%)
        Score = fraction of frames that are 3σ above the mean diff
        3σ events should be extremely rare in unedited footage; clusters = edits

    Final score is clamped to [0, 100] and classified into risk tiers.

    Returns:
        dict with per-signal scores, final_score, and risk_level
    """

    score_a = score_b = score_c = 0.0

    # ── Signal A: Frame Drops ──────────────────
    if total_frames > 0 and drops:
        max_tolerable = max(1, total_frames // 300)
        score_a = min(1.0, len(drops) / max_tolerable)

    # ── Signal B: Static Noise ─────────────────
    if noise_samples:
        n = len(noise_samples)
        frozen_frames = sum(1 for s in noise_samples if s < 1.0)
        frozen_ratio = frozen_frames / n

        # Also check for high variance (inconsistent noise = possible splicing)
        noise_std = float(np.std(noise_samples))
        noise_variance_penalty = min(0.4, noise_std / 100.0)  # caps at 0.4

        score_b = min(1.0, frozen_ratio + noise_variance_penalty)

    # ── Signal C: Temporal Diff Spikes ─────────
    if frame_diffs and len(frame_diffs) > 2:
        mean_d = np.mean(frame_diffs)
        std_d  = np.std(frame_diffs)
        sigma3 = mean_d + 3 * std_d

        spike_frames = sum(1 for d in frame_diffs if d > sigma3)
        spike_ratio  = spike_frames / len(frame_diffs)

        # Scale: >5% spike frames is very suspicious → map to 1.0
        score_c = min(1.0, spike_ratio / 0.05)

    # ── Weighted Composite ─────────────────────
    W_A, W_B, W_C = 0.35, 0.30, 0.35
    raw = (score_a * W_A + score_b * W_B + score_c * W_C)
    final_score = round(min(100.0, raw * 100), 2)

    # ── Risk Tier ──────────────────────────────
    if final_score >= 70:
        risk_level = "HIGH"
        risk_color = RED
    elif final_score >= 35:
        risk_level = "MEDIUM"
        risk_color = YELLOW
    elif final_score >= 10:
        risk_level = "LOW"
        risk_color = YELLOW
    else:
        risk_level = "CLEAN"
        risk_color = GREEN

    return {
        "score_a_drops":  round(score_a * 100, 1),
        "score_b_noise":  round(score_b * 100, 1),
        "score_c_spikes": round(score_c * 100, 1),
        "final_score":    final_score,
        "risk_level":     risk_level,
        "risk_color":     risk_color,
    }


def print_tampering_score(score_data, total_frames, drops, noise_samples, frame_diffs):
    """Renders the tampering score as a visual terminal meter."""

    print_section("TAMPERING SCORE BREAKDOWN")

    sa = score_data["score_a_drops"]
    sb = score_data["score_b_noise"]
    sc = score_data["score_c_spikes"]
    final = score_data["final_score"]
    risk  = score_data["risk_level"]
    color = score_data["risk_color"]

    def mini_bar(pct, width=20):
        filled = int((pct / 100) * width)
        bar_color = RED if pct >= 70 else YELLOW if pct >= 35 else GREEN
        return f"{bar_color}{'█' * filled}{'░' * (width - filled)}{RESET}"

    print(f"\n  {'Signal':<30} {'Score':>6}   Bar")
    print(f"  {'─'*30} {'─'*6}   {'─'*20}")
    print(f"  {'[A] Frame Drop Score (35%)':<30} {sa:>5.1f}%   {mini_bar(sa)}")
    print(f"  {'[B] Static Noise Score (30%)':<30} {sb:>5.1f}%   {mini_bar(sb)}")
    print(f"  {'[C] Diff Spike Score (35%)':<30} {sc:>5.1f}%   {mini_bar(sc)}")
    print(f"\n  {'─'*55}")

    # Big composite score display
    bar_width = 40
    filled = int((final / 100) * bar_width)
    big_bar = f"{color}{'█' * filled}{'░' * (bar_width - filled)}{RESET}"

    print(f"\n  {BOLD}COMPOSITE TAMPERING SCORE{RESET}")
    print(f"\n  [{big_bar}]  {color}{BOLD}{final:.1f}%{RESET}")
    print()

    # Risk label with box
    tier_msgs = {
        "CLEAN":  "No significant tampering indicators found.",
        "LOW":    "Minor anomalies present. Could be compression artifacts.",
        "MEDIUM": "Moderate tampering signals. Manual review recommended.",
        "HIGH":   "Strong tampering indicators. Video likely manipulated.",
    }
    msg = tier_msgs[risk]

    box_w = 52
    print(f"  {color}┌{'─' * box_w}┐{RESET}")
    print(f"  {color}│  RISK LEVEL: {BOLD}{risk:<8}{RESET}{color}  {msg:<{box_w - 26}}│{RESET}")
    print(f"  {color}└{'─' * box_w}┘{RESET}")


# ─────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────

def build_report(video_path, fps, total_frames, duration_sec,
                 drops, noise_samples, frame_diffs, high_diff_frames,
                 diff_threshold, heatmap_path):

    print_section("FRAME DROP ANALYSIS")
    if drops:
        print(f"  {RED}⚠  {len(drops)} frame drop(s) detected{RESET}")
        for d in drops:
            fnum, t0, t1, delta = d
            print(f"     Frame {fnum:5d}: {t0/1000:.3f}s → {t1/1000:.3f}s  |  Gap = {delta:.1f}ms")
    else:
        print(f"  {GREEN}✓  No frame drops detected{RESET}")

    print_section("NOISE / STATIC ANALYSIS (ROI)")
    verdict_noise = "NORMAL"
    if noise_samples:
        avg_noise = np.mean(noise_samples)
        min_noise = np.min(noise_samples)
        std_noise = np.std(noise_samples)

        print(f"  Average Std Dev  : {avg_noise:.2f}")
        print(f"  Minimum Std Dev  : {min_noise:.2f}")
        print(f"  Noise Variance   : {std_noise:.2f}")

        if min_noise < 1.0:
            verdict_noise = "SUSPICIOUS"
            print(f"  {RED}⚠  Near-zero noise frames found.")
            print(f"     Possible static image pasted over video (edit/splice).{RESET}")
        elif std_noise > 15:
            verdict_noise = "INCONSISTENT"
            print(f"  {YELLOW}⚠  High noise variance detected.")
            print(f"     Possibly different camera sensors or re-encoded segments.{RESET}")
        else:
            print(f"  {GREEN}✓  Sensor noise consistent throughout. Likely authentic.{RESET}")

    print_section("TEMPORAL DIFFERENCE ANALYSIS")
    verdict_diff = "NORMAL"
    if frame_diffs:
        global_avg = np.mean(frame_diffs)
        global_std = np.std(frame_diffs)
        spike_threshold = global_avg + (3 * global_std)  # 3-sigma spikes

        spikes = [(fnum, diff, ts) for fnum, diff, ts in high_diff_frames if diff > spike_threshold]

        print(f"  Global Avg Diff  : {global_avg:.2f}")
        print(f"  Global Std Dev   : {global_std:.2f}")
        print(f"  Spike Threshold  : {spike_threshold:.2f}  (3σ above mean)")
        print(f"  Frames > {diff_threshold} diff  : {len(high_diff_frames)}")
        print(f"  3σ Spike Frames  : {len(spikes)}")

        if spikes:
            verdict_diff = "SUSPICIOUS"
            print(f"\n  {RED}⚠  Anomalous spike frames (possible cut/splice/insert):{RESET}")
            for fnum, diff, ts in spikes[:15]:
                print(f"     Frame {fnum:5d}  |  Diff={diff:.1f}  |  @{ts/1000:.3f}s")
            if len(spikes) > 15:
                print(f"     ... and {len(spikes)-15} more.")
        else:
            print(f"  {GREEN}✓  No anomalous spikes detected in temporal differences.{RESET}")

    # ── Tampering Score ───────────────────────
    score_data = compute_tampering_score(
        total_frames, drops, noise_samples, frame_diffs, high_diff_frames
    )
    print_tampering_score(score_data, total_frames, drops, noise_samples, frame_diffs)

    # ── Final Verdict ─────────────────────────
    print_section("FINAL VERDICT")

    flags = []
    if drops:
        flags.append(f"Frame drops ({len(drops)})")
    if verdict_noise == "SUSPICIOUS":
        flags.append("Static-image overlay detected")
    if verdict_noise == "INCONSISTENT":
        flags.append("Noise inconsistency (possible re-encoding)")
    if verdict_diff == "SUSPICIOUS":
        flags.append(f"Temporal diff spikes")

    if flags:
        print(f"  {RED}{BOLD}  ██ TAMPERED / SUSPICIOUS ██{RESET}")
        for f in flags:
            print(f"  {RED}   → {f}{RESET}")
    else:
        print(f"  {GREEN}{BOLD}  ✓✓ VIDEO APPEARS AUTHENTIC{RESET}")
        print(f"  {GREEN}   No significant anomalies detected.{RESET}")

    print(f"\n  {CYAN}Heatmap saved → {heatmap_path}{RESET}")
    print(f"  {DIM}Open the heatmap image to see spatial distribution of changes.{RESET}")
    print()

    return {
        "drops": drops,
        "verdict_noise": verdict_noise,
        "verdict_diff": verdict_diff,
        "flags": flags,
        "heatmap_path": heatmap_path,
        "tampering_score": score_data["final_score"],
        "risk_level": score_data["risk_level"],
        "score_breakdown": score_data,
    }


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Tampering Detector — Temporal Differencing + Noise Analysis"
    )
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--save-diff", action="store_true",
                        help="Save temporal difference video as output_diff.mp4")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("Y1", "Y2", "X1", "X2"),
                        help="Region of interest for noise analysis [y1 y2 x1 x2]")
    parser.add_argument("--diff-threshold", type=float, default=30,
                        help="Avg pixel diff to flag a frame as high-change (default: 30)")
    parser.add_argument("--jump-threshold", type=float, default=1.9,
                        help="Timestamp gap multiplier to detect frame drops (default: 1.9)")
    return parser.parse_args()


if __name__ == "__main__":
    print_header()
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"{RED}[ERROR] File not found: {args.video}{RESET}")
        sys.exit(1)

    result = analyze_video(
        video_path     = args.video,
        roi            = args.roi,
        save_diff      = args.save_diff,
        diff_threshold = args.diff_threshold,
        jump_threshold = args.jump_threshold,
    )