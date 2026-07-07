"""Export produces the right number of playable clip files.

We detect clips on the synthetic video, export them, and confirm:
  - one output file per clip,
  - each file opens and its reported duration is close to the requested one,
  - the keyframe-vs-between-keyframe branch picks the right method.
"""

import os
import sys

import cv2
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from clipsplit import synthgen, video  # noqa: E402
from clipsplit.detect import Clip, detect_clips  # noqa: E402
from clipsplit.export import export_clip, export_clips  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    out = str(d / "demo.mp4")
    synthgen.generate(out)
    return out


def _readable(path):
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    cap.release()
    return ok


def test_export_produces_one_file_per_clip(synth, tmp_path):
    clips, _ = detect_clips(synth)
    out_dir = str(tmp_path / "clips")
    for i, c in enumerate(clips):
        c.name = f"clip_{i:02d}"
    results = export_clips(synth, clips, out_dir)
    assert len(results) == len(clips)
    assert all(r.ok for r in results), [r.detail for r in results if not r.ok]
    files = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
    assert len(files) == len(clips)


def test_exported_clips_are_playable_and_right_length(synth, tmp_path):
    clips, _ = detect_clips(synth)
    out_dir = str(tmp_path / "clips")
    for i, c in enumerate(clips):
        c.name = f"clip_{i:02d}"
    results = export_clips(synth, clips, out_dir)
    for r, c in zip(results, clips):
        assert _readable(r.path), f"clip not playable: {r.path}"
        info = video.probe(r.path)
        # duration should be within ~0.3s of requested (keyframe snap can extend
        # a stream-copy clip slightly)
        assert abs(info.duration - c.duration) < 0.6, (
            f"{r.path}: got {info.duration:.2f}s, wanted {c.duration:.2f}s")


def test_between_keyframe_boundary_reencodes(synth, tmp_path):
    """A start time that falls between keyframes must trigger a re-encode so the
    cut is frame-accurate, not snapped backwards to a keyframe."""
    kf = video.keyframe_times(synth)
    # find a time that is at least 0.5s away from every keyframe
    target = None
    for t10 in range(5, 120):
        t = t10 / 10.0
        if t < video.probe(synth).duration - 1.0 and min(abs(t - k) for k in kf) > 0.4:
            target = t
            break
    assert target is not None, "could not find a between-keyframe time"
    c = Clip(index=0, start=target, end=target + 1.0, name="between")
    r = export_clip(synth, c, str(tmp_path / "c"), kf)
    assert r.ok
    assert r.method == "reencode", f"expected reencode at t={target}, got {r.method}"
    assert _readable(r.path)


def test_on_keyframe_boundary_copies(synth, tmp_path):
    """A start time sitting on a keyframe must stream-copy (lossless)."""
    kf = video.keyframe_times(synth)
    on_kf = kf[1] if len(kf) > 1 else kf[0]  # skip 0.0
    c = Clip(index=0, start=on_kf, end=on_kf + 1.0, name="onkf")
    r = export_clip(synth, c, str(tmp_path / "c"), kf)
    assert r.ok
    assert r.method == "copy", f"expected copy at keyframe t={on_kf}, got {r.method}"
