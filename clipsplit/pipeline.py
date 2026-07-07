"""End-to-end pipeline: video path -> named clips (not yet exported).

This is the orchestration the UI and validate.py both call. It runs detection,
grabs a representative frame for each clip, OCRs it, and assigns a name.

All on-device: OpenCV for frames, Apple Vision for OCR. No network.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from . import naming, video
from .detect import Clip, DetectParams, SceneScore, detect_clips


@dataclass
class ClipResult:
    clips: list[Clip]
    score: SceneScore
    info: video.VideoInfo


def _representative_frame(cap: cv2.VideoCapture, fps: float, clip: Clip):
    """Grab a frame a little way INTO the clip.

    Not the very first frame: a scene transition (a fade, a window sliding in)
    can leave the first frame half-rendered. A short offset lands on the
    settled content whose title we want to read.
    """
    offset = min(0.6, max(0.1, clip.duration * 0.25))
    t = clip.start + offset
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    if not ok:
        # fall back to the start
        cap.set(cv2.CAP_PROP_POS_MSEC, clip.start * 1000.0)
        ok, frame = cap.read()
    return frame if ok else None


def analyze(path: str, params: DetectParams | None = None,
            do_naming: bool = True) -> ClipResult:
    """Detect clips and (optionally) name each from its most prominent text."""
    params = params or DetectParams()
    info = video.probe(path)
    clips, score = detect_clips(path, params)

    if do_naming:
        cap = cv2.VideoCapture(path)
        for clip in clips:
            frame = _representative_frame(cap, info.fps, clip)
            clip.name = naming.name_for_clip(frame, clip.index, clip.start)
        cap.release()
    else:
        for clip in clips:
            clip.name = naming.timecode_name(clip.index, clip.start)

    return ClipResult(clips=clips, score=score, info=info)
