"""Video I/O helpers built on OpenCV + the bundled static ffmpeg.

There is no system ffmpeg/ffprobe on this machine, so:
  - frame reading and basic metadata (fps, dimensions, frame count) come from
    OpenCV.
  - keyframe (I-frame) timestamps and the actual clip cutting come from the
    static ffmpeg binary shipped inside the `imageio-ffmpeg` wheel.

Nothing here touches the network.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

import cv2
import imageio_ffmpeg


def ffmpeg_exe() -> str:
    """Absolute path to the bundled static ffmpeg binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def probe(path: str) -> VideoInfo:
    """Read basic metadata with OpenCV. Falls back to a sane fps if the
    container does not report one."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0  # QuickTime screen recordings are almost always 30 or 60
    return VideoInfo(path=path, fps=fps, frame_count=frame_count,
                     width=width, height=height)


def keyframe_times(path: str) -> list[float]:
    """Return the presentation timestamps (seconds) of every keyframe (I-frame).

    Lossless `-c copy` cutting can only START a clip on a keyframe, so knowing
    where the keyframes are lets us decide, per boundary, whether a stream-copy
    cut lands close enough or whether that boundary needs a re-encode. We ask
    ffmpeg to dump per-packet flags as JSON and keep the packets whose flags
    contain 'K' (keyframe).
    """
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-c", "copy", "-f", "null", "-",
    ]
    # ffmpeg cannot itself emit keyframe JSON (that is ffprobe's job, which we
    # do not have), so we detect keyframes by decoding just the video packets
    # and reading their pict_type via the debug output. Simpler and robust:
    # use the -skip_frame nokey trick with showinfo, which prints a line per
    # decoded keyframe including its pts_time.
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "info",
        "-skip_frame", "nokey",
        "-i", path,
        "-vf", "showinfo",
        "-an", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    times: list[float] = []
    # showinfo prints lines like: [Parsed_showinfo_0 @ ...] n:0 pts:0 pts_time:0 ...
    for m in re.finditer(r"pts_time:([0-9]+\.?[0-9]*)", proc.stderr):
        times.append(float(m.group(1)))
    times = sorted(set(times))
    if not times:
        times = [0.0]
    return times


def read_frame_at(path: str, t: float):
    """Read a single BGR frame near time t (seconds). Returns None on failure."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
