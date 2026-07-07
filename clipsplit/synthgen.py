"""Generate a synthetic screen recording with KNOWN scene boundaries.

Detection has no ground truth on a real recording, so for testing and
validation we render our own. Each "scene" is a full-screen card with a distinct
background color and a big heading, plus a subtle animated element (a moving
cursor block and a running clock) so the video is not perfectly static within a
scene and the detector has to distinguish real turnovers from small motion.

The generator returns the exact boundary timestamps and the heading text of
each scene, so validate.py can measure boundary precision/recall and the tests
can check clip count and names.

This module is test-only. The detector never imports it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import video


# Each scene: a background color (BGR), a heading, and how long it lasts.
DEFAULT_SCENES = [
    ((245, 245, 245), "Project Overview", 3.0),        # light "slide"
    ((40, 30, 30), "Terminal Session", 2.5),            # dark "terminal"
    ((250, 240, 220), "Settings Panel", 3.0),           # cream "app"
    ((30, 40, 60), "Database Schema", 2.5),             # navy "editor"
    ((235, 235, 250), "Final Results", 3.0),            # lavender "report"
]


@dataclass
class SynthScene:
    start: float
    end: float
    heading: str
    bg: tuple


@dataclass
class SynthTruth:
    video_path: str
    fps: float
    width: int
    height: int
    duration: float
    boundaries: list          # interior boundary times (start of each scene>0)
    scenes: list              # list of dicts (start, end, heading, bg)


def _draw_scene(w: int, h: int, bg, heading: str, frame_idx: int, fps: float) -> np.ndarray:
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    # pick a heading color that contrasts with the background
    lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    ink = (30, 30, 30) if lum > 128 else (235, 235, 235)

    # big heading near the top (title-like position)
    cv2.putText(img, heading, (int(w * 0.06), int(h * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX, w / 640.0 * 1.4, ink,
                max(2, int(w / 640.0 * 3)), cv2.LINE_AA)

    # a few smaller body lines so a scene has realistic secondary text
    body = ["line one of content", "another row here", "status: running"]
    for i, line in enumerate(body):
        cv2.putText(img, line, (int(w * 0.06), int(h * 0.4 + i * h * 0.09)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 640.0 * 0.7, ink,
                    max(1, int(w / 640.0 * 1.5)), cv2.LINE_AA)

    # animated element 1: a blinking cursor block (small, must NOT trigger a cut)
    if (frame_idx // int(max(1, fps / 2))) % 2 == 0:
        cx = int(w * 0.06)
        cy = int(h * 0.4 + len(body) * h * 0.09)
        cv2.rectangle(img, (cx, cy - int(h * 0.03)), (cx + int(w * 0.02), cy),
                      ink, -1)

    # animated element 2: a running clock (tiny text change per frame)
    t = frame_idx / fps
    cv2.putText(img, f"{t:6.2f}s", (int(w * 0.78), int(h * 0.95)),
                cv2.FONT_HERSHEY_SIMPLEX, w / 640.0 * 0.6, ink,
                max(1, int(w / 640.0 * 1.2)), cv2.LINE_AA)
    return img


def generate(out_path: str, scenes=None, fps: float = 30.0,
             width: int = 640, height: int = 360) -> SynthTruth:
    """Render the synthetic video to `out_path` (mp4) and return ground truth.

    We render raw frames to an intermediate lossless-ish AVI with OpenCV, then
    re-encode to H.264 mp4 with the bundled ffmpeg so the file has real
    keyframes and looks like a QuickTime export. (OpenCV's mp4 writer support is
    unreliable across builds; piping through ffmpeg is deterministic.)
    """
    scenes = scenes or DEFAULT_SCENES
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # feed raw BGR frames to ffmpeg over stdin -> H.264 mp4
    ff = video.ffmpeg_exe()
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        # force a keyframe interval so keyframe-snapping logic has something to
        # work with, similar to a real screen recording
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-g", str(int(fps * 2)),  # keyframe every ~2s
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    truth_scenes: list[SynthScene] = []
    boundaries: list[float] = []
    t_cursor = 0.0
    global_idx = 0
    for si, (bg, heading, dur) in enumerate(scenes):
        start = t_cursor
        end = t_cursor + dur
        truth_scenes.append(SynthScene(start=round(start, 3), end=round(end, 3),
                                       heading=heading, bg=tuple(bg)))
        if si > 0:
            boundaries.append(round(start, 3))
        n_frames = int(round(dur * fps))
        for f in range(n_frames):
            img = _draw_scene(width, height, bg, heading, global_idx, fps)
            proc.stdin.write(img.tobytes())
            global_idx += 1
        t_cursor = end

    proc.stdin.close()
    proc.wait()

    duration = round(t_cursor, 3)
    truth = SynthTruth(
        video_path=out_path, fps=fps, width=width, height=height,
        duration=duration, boundaries=boundaries,
        scenes=[asdict(s) for s in truth_scenes],
    )
    # write a sidecar ground-truth json next to the video
    gt_path = os.path.splitext(out_path)[0] + "_truth.json"
    with open(gt_path, "w") as fh:
        json.dump(asdict(truth), fh, indent=2)
    return truth


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_out/demo.mp4"
    t = generate(out)
    print(f"Wrote {out}  ({t.duration:.1f}s, {len(t.scenes)} scenes)")
    print("Boundaries:", t.boundaries)
