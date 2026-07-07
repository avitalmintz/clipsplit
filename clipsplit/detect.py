"""Scene-change detection for screen recordings.

A screen recording is not a movie. Most of the time the frame barely changes
(a cursor moves, a caret blinks), and then occasionally the WHOLE screen turns
over: an app switch, a new slide, a big scroll or navigation jump. We want the
big turnovers and nothing else.

The signal we use is a combination of two per-frame-pair differences, both
computed on a small grayscale copy so the whole thing runs fast:

  1. Structural difference: mean absolute pixel difference between consecutive
     sampled frames. This spikes when a large fraction of the screen changes
     (a window covering the screen, a slide flip).

  2. Histogram distance: correlation distance between the two frames' grayscale
     histograms. This catches changes in overall tone/layout (light editor ->
     dark terminal) even when the pixel-diff is confused by scrolling text.

We combine them, run the score over time, and mark a boundary where the score
crosses an adaptive threshold AND is a local peak, then enforce a minimum clip
length so a single busy transition does not shatter into ten micro-clips.

Everything is on-device; there is no network anywhere in this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import video


# Small analysis resolution. Screen recordings are large (often 2880x1800 on a
# Retina Mac); we only need the gross structure of the change, so downscale
# hard. Aspect ratio is not preserved on purpose: a fixed grid makes the
# per-cell diffs comparable frame to frame.
_ANALYSIS_W = 160
_ANALYSIS_H = 90


@dataclass
class DetectParams:
    sample_fps: float = 4.0        # how many frames per second to analyze
    min_clip_seconds: float = 1.5  # boundaries closer than this are merged
    # A boundary is declared where the combined score exceeds a ROBUST
    # threshold: median + z * MAD of the score series, AND clears an absolute
    # floor. We use median/MAD, not mean/std, on purpose: the boundary spikes
    # are exactly the outliers we are hunting, and they wildly inflate the std,
    # which would push a mean+std threshold ABOVE the spikes and miss them. The
    # median and MAD describe the quiet background between cuts and are immune
    # to a handful of large spikes.
    z_threshold: float = 6.0
    abs_floor: float = 0.18        # ignore tiny scores even if locally large


@dataclass
class SceneScore:
    times: np.ndarray   # timestamp (s) of each analyzed frame
    scores: np.ndarray  # combined change score aligned to `times` (scores[0]=0)


def _prep(frame) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_ANALYSIS_W, _ANALYSIS_H), interpolation=cv2.INTER_AREA)


def _pair_score(a: np.ndarray, b: np.ndarray) -> float:
    """Combined change score for a consecutive pair of prepped frames.

    Structural term: fraction of the frame whose brightness moved by more than
    a noticeable amount. Using a per-pixel threshold before averaging makes the
    term robust to codec noise and a moving cursor (a few pixels) while staying
    sensitive to a big region turning over.
    """
    diff = cv2.absdiff(a, b)
    changed = (diff > 25).mean()  # fraction of pixels that meaningfully changed

    # Histogram term: 1 - correlation of the two grayscale histograms.
    ha = cv2.calcHist([a], [0], None, [64], [0, 256])
    hb = cv2.calcHist([b], [0], None, [64], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    hist_term = max(0.0, 1.0 - corr)

    # Weight structure a bit more: a full window swap moves most pixels, which
    # is the signal we most want. Histogram catches tone flips scrolling misses.
    return float(0.7 * changed + 0.3 * hist_term)


def score_video(path: str, params: DetectParams | None = None) -> SceneScore:
    """Walk the video at `sample_fps` and compute the change score over time."""
    params = params or DetectParams()
    info = video.probe(path)
    step = max(1, int(round(info.fps / params.sample_fps)))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    times: list[float] = []
    scores: list[float] = []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / info.fps
            cur = _prep(frame)
            if prev is None:
                scores.append(0.0)
            else:
                scores.append(_pair_score(prev, cur))
            times.append(t)
            prev = cur
        idx += 1
    cap.release()

    return SceneScore(times=np.asarray(times), scores=np.asarray(scores))


def find_boundaries(score: SceneScore, params: DetectParams | None = None) -> list[float]:
    """Turn the score series into a sorted list of boundary timestamps (s).

    A boundary is a local peak in the score that clears both an adaptive
    threshold (mean + z*std over the series) and an absolute floor, with a
    minimum-clip-length constraint so close peaks collapse to one cut.
    """
    params = params or DetectParams()
    s = score.scores
    t = score.times
    if len(s) < 3:
        return []

    # Robust background statistics (median + scaled MAD), immune to the spikes
    # we are detecting. 1.4826 makes MAD a consistent estimator of std for
    # normal noise, so z_threshold reads in familiar "sigma" units.
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    if mad <= 1e-6:
        mad = float(s.std()) or 1e-6  # degenerate: perfectly flat background
    thresh = max(params.abs_floor, med + params.z_threshold * mad)

    # candidate peaks: score above threshold and >= its immediate neighbors
    candidates: list[tuple[float, float]] = []  # (time, score)
    for i in range(1, len(s) - 1):
        if s[i] >= thresh and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            candidates.append((float(t[i]), float(s[i])))
    # also allow the very last frame to be a peak (end-of-recording change)
    if len(s) >= 2 and s[-1] >= thresh and s[-1] >= s[-2]:
        candidates.append((float(t[-1]), float(s[-1])))

    # enforce minimum spacing, keeping the strongest peak in each cluster
    candidates.sort(key=lambda c: c[0])
    boundaries: list[float] = []
    last_kept_time = -1e9
    last_kept_score = -1.0
    for bt, bs in candidates:
        if bt - last_kept_time >= params.min_clip_seconds:
            boundaries.append(bt)
            last_kept_time = bt
            last_kept_score = bs
        else:
            # too close to the previous kept boundary: keep the stronger one
            if bs > last_kept_score:
                boundaries[-1] = bt
                last_kept_time = bt
                last_kept_score = bs
    return boundaries


@dataclass
class Clip:
    index: int
    start: float
    end: float
    name: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def boundaries_to_clips(boundaries: list[float], total_duration: float,
                        params: DetectParams | None = None) -> list[Clip]:
    """Convert interior boundary timestamps into a full [0, duration] segmentation.

    Each boundary starts a new clip. A trailing tiny segment (shorter than the
    minimum clip length) is merged back into the previous clip so we never emit
    a fraction-of-a-second sliver at the end.
    """
    params = params or DetectParams()
    cuts = [0.0] + [b for b in boundaries if 0.0 < b < total_duration] + [total_duration]
    cuts = sorted(set(round(c, 3) for c in cuts))

    clips: list[Clip] = []
    for i in range(len(cuts) - 1):
        clips.append(Clip(index=i, start=cuts[i], end=cuts[i + 1]))

    # merge a too-short trailing clip into its predecessor
    if len(clips) >= 2 and clips[-1].duration < params.min_clip_seconds:
        clips[-2].end = clips[-1].end
        clips.pop()

    # renumber after any merge
    for i, c in enumerate(clips):
        c.index = i
    return clips


def detect_clips(path: str, params: DetectParams | None = None) -> tuple[list[Clip], SceneScore]:
    """Full detection: score the video, find boundaries, build the segmentation."""
    params = params or DetectParams()
    info = video.probe(path)
    score = score_video(path, params)
    boundaries = find_boundaries(score, params)
    clips = boundaries_to_clips(boundaries, info.duration, params)
    return clips, score
