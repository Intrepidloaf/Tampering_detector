from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from prameya_engine import PrameyaEngine


@dataclass(frozen=True)
class Case:
    name: str
    image_path: str
    expected: str


def _safe_call(fn: Callable[[], float]) -> Tuple[float | None, str | None]:
    try:
        return float(fn()), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def analyze_image(case: Case) -> Dict[str, object]:
    engine = PrameyaEngine(case.image_path)

    pillar_fns: Dict[str, Callable[[], float]] = {
        "ELA": engine.detect_ela,
        "DQ": engine.detect_double_quantization,
        "PRNU": engine.detect_prnu,
        "CFA": engine.detect_cfa,
        "SVD": engine.detect_svd,
        "Metadata": engine.audit_metadata,
    }

    pillar_scores: Dict[str, float] = {}
    pillar_errors: Dict[str, str] = {}

    for pillar, fn in pillar_fns.items():
        score, err = _safe_call(fn)
        if err is not None or score is None:
            pillar_errors[pillar] = err or "Unknown error"
        else:
            pillar_scores[pillar] = max(0.0, min(1.0, float(score)))

    total_score, total_err = _safe_call(engine.generate_truth_score)

    top_pillar = None
    if pillar_scores:
        top_pillar = max(pillar_scores.items(), key=lambda kv: kv[1])[0]

    return {
        "case": case,
        "total_score": total_score,
        "total_error": total_err,
        "pillar_scores": pillar_scores,
        "pillar_errors": pillar_errors,
        "top_pillar": top_pillar,
    }


def print_report(result: Dict[str, object]) -> None:
    case: Case = result["case"]  # type: ignore[assignment]
    total_score = result["total_score"]
    total_error = result["total_error"]
    pillar_scores: Dict[str, float] = result["pillar_scores"]  # type: ignore[assignment]
    pillar_errors: Dict[str, str] = result["pillar_errors"]  # type: ignore[assignment]
    top_pillar = result["top_pillar"]

    print("=" * 72)
    print(f"CASE: {case.name}")
    print(f"Path: {case.image_path}")
    print(f"Expected: {case.expected}")
    print("-" * 72)

    if total_error is not None or total_score is None:
        print(f"TOTAL Probability of Tampering: ERROR ({total_error})")
    else:
        print(f"TOTAL Probability of Tampering: {total_score:.4f}")

    print("\nPillar scores (higher = more suspicious):")
    if not pillar_scores and pillar_errors:
        print("  (no pillar scores available)")
    else:
        for k in ["ELA", "DQ", "PRNU", "CFA", "SVD", "Metadata"]:
            if k in pillar_scores:
                print(f"  - {k:<8} {pillar_scores[k]:.4f}")
            elif k in pillar_errors:
                print(f"  - {k:<8} ERROR ({pillar_errors[k]})")
            else:
                print(f"  - {k:<8} (missing)")

    if top_pillar is not None:
        print(f"\nMost artifacts detected by: {top_pillar}")
    else:
        print("\nMost artifacts detected by: (unavailable)")

    if pillar_errors:
        print("\nNotes:")
        print("  - Some pillars errored. This can happen for unsupported formats,")
        print("    missing metadata, or images too small for certain block-based checks.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PrameyaEngine test bench: run the 6 pillars on three image types and "
            "report which pillar detected the most artifacts per case."
        )
    )
    parser.add_argument("--original", required=True, help="Path to an original camera photo (expected low/safe).")
    parser.add_argument("--edited", required=True, help="Path to a Photoshop-edited JPEG (expected high/tampered).")
    parser.add_argument("--screenshot", required=True, help="Path to a screenshot/social-media capture (expected elevated).")
    args = parser.parse_args()

    cases = [
        Case("Original camera photo", args.original, "Low / Safe"),
        Case("Photoshop-edited JPEG", args.edited, "High / Tampered"),
        Case("Screenshot / Social media post", args.screenshot, "Medium–High (often metadata missing)"),
    ]

    for case in cases:
        result = analyze_image(case)
        print_report(result)

    print("=" * 72)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

