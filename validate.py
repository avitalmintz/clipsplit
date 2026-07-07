"""Validation harness. Renders a synthetic screen recording with KNOWN scene
cuts, runs ClipSplit's detection, and reports REAL boundary precision / recall
and timing error, plus how well the auto-names recovered the on-screen headings.

Detection has no ground truth on a real recording, so we plant our own: each
scene switches at a known timestamp with a known heading. A detected boundary
counts as a true positive if it lands within TOLERANCE seconds of a planted cut.

Run:  ./venv/bin/python validate.py
"""

from __future__ import annotations

import os
import sys

from clipsplit import synthgen
from clipsplit.detect import DetectParams, find_boundaries, score_video
from clipsplit.pipeline import analyze


SYNTH_DIR = os.path.join(os.path.dirname(__file__), "synthetic_out")
# 4fps sampling reports a boundary up to ~0.25s late; allow a little encoder slack.
TOLERANCE = 0.4

# A richer scene list than the module default, including a couple of harder
# cases: two visually similar light scenes back to back, and a very short scene.
VALIDATION_SCENES = [
    ((245, 245, 245), "Project Overview", 3.0),
    ((40, 30, 30), "Terminal Session", 2.5),
    ((250, 240, 220), "Settings Panel", 2.0),
    ((30, 40, 60), "Database Schema", 2.5),
    ((248, 248, 240), "Release Notes", 2.0),        # light -> light (hard)
    ((235, 235, 250), "Final Results", 3.0),
]


def _match(found, planted, tol):
    """Greedy nearest-match. Returns (tp, matched_errors, unmatched_found)."""
    remaining = list(found)
    tp = 0
    errors = []
    for p in planted:
        best = None
        for f in remaining:
            if abs(f - p) <= tol and (best is None or abs(f - p) < abs(best - p)):
                best = f
        if best is not None:
            tp += 1
            errors.append(abs(best - p))
            remaining.remove(best)
    return tp, errors, remaining


def main():
    os.makedirs(SYNTH_DIR, exist_ok=True)
    video_path = os.path.join(SYNTH_DIR, "validation.mp4")
    truth = synthgen.generate(video_path, scenes=VALIDATION_SCENES)

    params = DetectParams()
    score = score_video(video_path, params)
    found = find_boundaries(score, params)

    planted = truth.boundaries
    tp, errors, extra = _match(found, planted, TOLERANCE)
    fn = len(planted) - tp
    fp = len(extra)

    recall = tp / len(planted) if planted else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    mean_err = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0.0

    print("\n" + "=" * 78)
    print("ClipSplit boundary detection on synthetic recording")
    print("=" * 78)
    print(f"video: {os.path.basename(video_path)}  "
          f"({truth.duration:.1f}s, {len(truth.scenes)} scenes, fps={truth.fps:.0f})")
    print(f"planted boundaries: {[round(b, 2) for b in planted]}")
    print(f"detected boundaries:{[round(b, 2) for b in found]}")
    print("-" * 78)
    print(f"{'metric':<22}{'value':>12}")
    print("-" * 78)
    print(f"{'true positives':<22}{tp:>12}")
    print(f"{'false negatives':<22}{fn:>12}")
    print(f"{'false positives':<22}{fp:>12}")
    print(f"{'recall':<22}{recall:>12.2f}")
    print(f"{'precision':<22}{precision:>12.2f}")
    print(f"{'mean timing error (s)':<22}{mean_err:>12.3f}")
    print(f"{'max timing error (s)':<22}{max_err:>12.3f}  (tolerance {TOLERANCE}s)")
    print("-" * 78)

    # naming quality: how many scene headings did OCR recover into clip names?
    result = analyze(video_path, params)
    headings = [s["heading"] for s in truth.scenes]
    joined = " ".join(c.name.lower().replace("_", " ") for c in result.clips)
    recovered = [h for h in headings if h.lower() in joined]
    name_rate = len(recovered) / len(headings) if headings else float("nan")
    print(f"clips produced: {len(result.clips)} (expected {len(truth.scenes)})")
    print(f"auto-names: {[c.name for c in result.clips]}")
    print(f"headings recovered by OCR into names: {len(recovered)}/{len(headings)} "
          f"({name_rate:.0%})")
    print("-" * 78)

    # Pass bar: catch every planted cut (recall == 1.0), no spurious cuts
    # (precision == 1.0), timing within tolerance, correct clip count, and the
    # majority of headings recovered into names.
    passed = (
        recall >= 1.0 and
        precision >= 1.0 and
        max_err <= TOLERANCE and
        len(result.clips) == len(truth.scenes) and
        name_rate >= 0.6
    )
    print("Pass bar: recall=1.0, precision=1.0, timing<=tolerance, "
          "clip count exact, name recovery>=60%")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 78 + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
