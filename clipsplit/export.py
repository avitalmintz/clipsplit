"""Cut clips out of the source video with the bundled static ffmpeg.

THE HARD PART
-------------
Cutting a video losslessly means `-c copy`: copy the already-encoded packets
straight through, no re-encode, so it is fast and pixel-for-pixel identical.
But H.264/HEVC packets depend on earlier frames, so a decoder can only START
playback from a keyframe (I-frame). ffmpeg with `-c copy` therefore snaps a cut
to the nearest keyframe at or before the requested start time. Keyframes in a
screen recording are often 2-10 seconds apart, so a stream-copy cut can drift
by seconds from the boundary we detected. For an auto-splitter that is wrong:
the clip would begin in the middle of the previous scene.

How we handle it, per clip:
  1. Ask ffmpeg where the keyframes are (see video.keyframe_times).
  2. If a keyframe sits within `snap_tolerance` seconds of the clip start, the
     boundary IS effectively on a keyframe: stream-copy it. Lossless and fast.
  3. Otherwise the boundary falls between keyframes, so a stream copy would
     drift. We re-encode THAT clip only, which lets ffmpeg place a fresh
     keyframe exactly at the cut for a frame-accurate start. Re-encoding is
     slower and not bit-identical, but it is the correct trade for accuracy.

Either way audio is carried through (`-c:a copy` on stream-copy clips, AAC on
re-encoded clips), and each exported clip is an independent, playable file.

Nothing here touches the network.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from . import video
from .detect import Clip


@dataclass
class ExportResult:
    clip: Clip
    path: str
    method: str      # "copy" (lossless) or "reencode" (frame-accurate)
    ok: bool
    detail: str = ""


def _nearest_keyframe_delta(start: float, keyframes: list[float]) -> float:
    """Smallest absolute distance from `start` to any keyframe."""
    if not keyframes:
        return float("inf")
    return min(abs(start - k) for k in keyframes)


def _run(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # keep the last chunk of stderr for a useful error message
        tail = "\n".join(proc.stderr.strip().splitlines()[-4:])
        return False, tail
    return True, ""


def export_clip(src: str, clip: Clip, out_dir: str, keyframes: list[float],
                snap_tolerance: float = 0.25, container: str = "mp4") -> ExportResult:
    """Export a single clip, choosing lossless copy vs frame-accurate re-encode."""
    os.makedirs(out_dir, exist_ok=True)
    stem = clip.name or f"clip_{clip.index:02d}"
    out_path = os.path.join(out_dir, f"{stem}.{container}")
    # avoid clobbering if two clips sanitize to the same name
    if os.path.exists(out_path):
        out_path = os.path.join(out_dir, f"{stem}_{clip.index:02d}.{container}")

    ff = video.ffmpeg_exe()
    duration = clip.duration
    delta = _nearest_keyframe_delta(clip.start, keyframes)

    if delta <= snap_tolerance:
        # Boundary is on (or within tolerance of) a keyframe: lossless copy.
        # -ss before -i seeks by keyframe, which is exactly what we want here.
        cmd = [
            ff, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{clip.start:.3f}",
            "-i", src,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            out_path,
        ]
        ok, err = _run(cmd)
        if ok:
            return ExportResult(clip, out_path, "copy", True,
                                f"lossless stream copy (keyframe within {snap_tolerance:.2f}s)")
        # if copy fails (rare container edge cases), fall through to re-encode
    # Boundary falls between keyframes: re-encode this clip so the cut is
    # frame-accurate. -ss AFTER -i decodes up to the exact time; a fresh
    # keyframe is written at the start.
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-i", src,
        "-ss", f"{clip.start:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]
    ok, err = _run(cmd)
    method = "reencode"
    detail = "frame-accurate re-encode (boundary between keyframes)"
    if not ok:
        return ExportResult(clip, out_path, method, False, err)
    return ExportResult(clip, out_path, method, True, detail)


def export_clips(src: str, clips: list[Clip], out_dir: str,
                 snap_tolerance: float = 0.25, container: str = "mp4",
                 progress=None) -> list[ExportResult]:
    """Export a list of clips. `progress` is an optional callback(i, n, clip)."""
    keyframes = video.keyframe_times(src)
    results: list[ExportResult] = []
    for i, clip in enumerate(clips):
        if progress:
            progress(i, len(clips), clip)
        results.append(export_clip(src, clip, out_dir, keyframes,
                                    snap_tolerance=snap_tolerance, container=container))
    return results
