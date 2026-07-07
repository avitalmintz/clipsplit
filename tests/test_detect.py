"""Boundary detection on a synthetic recording with KNOWN cuts.

We render a video whose scenes switch at known timestamps, run detection, and
check that every planted boundary is found within a small tolerance and that no
spurious extra boundaries appear.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from clipsplit import synthgen  # noqa: E402
from clipsplit.detect import detect_clips, find_boundaries, score_video  # noqa: E402


# 4fps sampling means a boundary can be reported up to ~0.25s late; allow a bit
# more slack for encoder timing.
TOLERANCE = 0.4


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    out = str(d / "demo.mp4")
    truth = synthgen.generate(out)
    return out, truth


def _match(found, planted, tol):
    """Greedy match: for each planted boundary, is there a found one within tol?"""
    remaining = list(found)
    hits = 0
    for p in planted:
        best = None
        for f in remaining:
            if abs(f - p) <= tol:
                if best is None or abs(f - p) < abs(best - p):
                    best = f
        if best is not None:
            hits += 1
            remaining.remove(best)
    return hits, remaining  # hits, unmatched found (potential false positives)


def test_all_planted_boundaries_found(synth):
    path, truth = synth
    score = score_video(path)
    found = find_boundaries(score)
    hits, extra = _match(found, truth.boundaries, TOLERANCE)
    assert hits == len(truth.boundaries), (
        f"missed boundaries: found {found}, planted {truth.boundaries}")


def test_no_spurious_boundaries(synth):
    path, truth = synth
    score = score_video(path)
    found = find_boundaries(score)
    _, extra = _match(found, truth.boundaries, TOLERANCE)
    assert not extra, f"spurious boundaries detected: {extra}"


def test_clip_count_matches_scenes(synth):
    path, truth = synth
    clips, _ = detect_clips(path)
    assert len(clips) == len(truth.scenes), (
        f"expected {len(truth.scenes)} clips, got {len(clips)}")


def test_clips_tile_the_timeline(synth):
    """Clips must be contiguous and cover [0, duration] with no gaps/overlaps."""
    path, truth = synth
    clips, _ = detect_clips(path)
    assert clips[0].start == 0.0
    assert abs(clips[-1].end - truth.duration) < 0.1
    for a, b in zip(clips, clips[1:]):
        assert abs(a.end - b.start) < 1e-6, "clips are not contiguous"


def test_static_video_makes_one_clip(tmp_path):
    """A recording that never changes should yield a single clip, not noise."""
    out = str(tmp_path / "static.mp4")
    # one scene only
    synthgen.generate(out, scenes=[((200, 200, 200), "Just One Screen", 5.0)])
    clips, _ = detect_clips(out)
    assert len(clips) == 1
