"""Name a clip from the most prominent on-screen text in it.

We OCR a representative frame of the clip with Apple Vision (on-device, via
ocrmac), then pick the single most "title-like" text line and sanitize it into
a filename. When OCR yields nothing usable we fall back to a timecode name.

"Prominent" is not just "biggest box". A giant faint watermark should lose to a
crisp window title. We score each recognized line by a blend of:
  - text HEIGHT (bigger glyphs read as more important) — the dominant term,
  - OCR confidence,
  - a small bonus for lines near the top of the frame (titles/headings live
    there),
  - a small penalty for very long lines (a paragraph is not a title).

Everything here runs on-device; there is no network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import ocrmac.ocrmac as _ocr


@dataclass
class TextLine:
    text: str
    confidence: float
    box: tuple[int, int, int, int]  # top-left pixel box (x1, y1, x2, y2)


def ocr_frame(frame_bgr: np.ndarray) -> list[TextLine]:
    """Run Apple Vision OCR on a single BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    w, h = pil.size
    engine = _ocr.OCR(
        pil,
        recognition_level="accurate",
        language_preference=["en-US"],
        detail=True,
    )
    out: list[TextLine] = []
    for text, conf, nbox in engine.recognize():
        x1, y1, x2, y2 = _ocr.convert_coordinates_pil(nbox, w, h)
        out.append(TextLine(
            text=text.strip(),
            confidence=float(conf),
            box=(int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))),
        ))
    return out


# Lines that are pure chrome / not useful as a name.
_JUNK_RE = re.compile(r"^[\W_]*$")


def _line_score(line: TextLine, frame_h: int) -> float:
    x1, y1, x2, y2 = line.box
    height = max(1, y2 - y1)
    # normalize glyph height as a fraction of frame height so it is resolution
    # independent
    h_frac = height / max(1, frame_h)
    # top bonus: 1.0 at the very top, decaying to ~0 at the bottom third
    top_pos = y1 / max(1, frame_h)
    top_bonus = max(0.0, 1.0 - top_pos * 1.5)
    # length penalty: titles are short; a long paragraph line is unlikely a name
    n = len(line.text)
    length_pen = 0.0 if n <= 40 else min(0.5, (n - 40) / 120.0)

    return (h_frac * 6.0) + (line.confidence * 0.8) + (top_bonus * 0.5) - length_pen


def prominent_text(lines: list[TextLine], frame_h: int) -> str | None:
    """Return the raw text of the most title-like line, or None."""
    usable = [l for l in lines if l.text and not _JUNK_RE.match(l.text) and len(l.text) >= 2]
    if not usable:
        return None
    best = max(usable, key=lambda l: _line_score(l, frame_h))
    return best.text


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_US = re.compile(r"_+")


def sanitize_name(text: str, max_len: int = 48) -> str:
    """Turn arbitrary on-screen text into a safe, readable filename stem."""
    # strip trailing app-name tails common in window titles ("Doc - App")
    text = re.sub(r"\s+[\-–—|]\s+.*$", "", text) if " - " in text or " | " in text else text
    text = _ILLEGAL.sub(" ", text)
    text = text.strip()
    # collapse whitespace to single underscores
    text = re.sub(r"\s+", "_", text)
    text = _MULTI_US.sub("_", text).strip("_.")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_.")
    return text


def timecode_name(index: int, start: float) -> str:
    """Fallback name: clip_{index}_{mm-ss}."""
    mm = int(start) // 60
    ss = int(start) % 60
    return f"clip_{index:02d}_{mm:02d}-{ss:02d}"


def name_for_clip(frame_bgr: np.ndarray | None, index: int, start: float) -> str:
    """Derive a clip name: prominent on-screen text if usable, else timecode."""
    fallback = timecode_name(index, start)
    if frame_bgr is None:
        return fallback
    frame_h = frame_bgr.shape[0]
    lines = ocr_frame(frame_bgr)
    raw = prominent_text(lines, frame_h)
    if not raw:
        return fallback
    stem = sanitize_name(raw)
    if not stem or len(stem) < 2:
        return fallback
    # prefix with the index so clips stay ordered on disk and names stay unique
    return f"{index:02d}_{stem}"
