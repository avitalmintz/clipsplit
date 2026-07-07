"""Naming: filename sanitization, prominence scoring, and end-to-end names.

The sanitizer and prominence picker are pure functions we test directly. The
end-to-end name (which runs Apple Vision OCR on the synthetic frames) is checked
against the known scene headings.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from clipsplit import naming, synthgen  # noqa: E402
from clipsplit.naming import TextLine, prominent_text, sanitize_name, timecode_name  # noqa: E402
from clipsplit.pipeline import analyze  # noqa: E402


def test_sanitize_basic():
    assert sanitize_name("Project Overview") == "Project_Overview"


def test_sanitize_strips_illegal_chars():
    out = sanitize_name('report: q3/2025 "final"')
    assert "/" not in out and ":" not in out and '"' not in out


def test_sanitize_strips_app_name_tail():
    # window titles often look like "Document - AppName"
    assert sanitize_name("Budget 2025 - Numbers") == "Budget_2025"
    assert sanitize_name("index.py | VS Code") == "index.py"


def test_sanitize_length_cap():
    long = "x" * 200
    assert len(sanitize_name(long)) <= 48


def test_timecode_fallback_format():
    assert timecode_name(3, 75.0) == "clip_03_01-15"


def test_prominence_prefers_bigger_top_text():
    lines = [
        TextLine(text="tiny footer note", confidence=0.9, box=(10, 380, 300, 392)),
        TextLine(text="Big Heading", confidence=0.9, box=(10, 20, 300, 70)),
    ]
    assert prominent_text(lines, frame_h=400) == "Big Heading"


def test_prominence_ignores_junk_lines():
    lines = [
        TextLine(text="-----", confidence=0.9, box=(10, 20, 300, 70)),
        TextLine(text="Real Title", confidence=0.8, box=(10, 90, 300, 130)),
    ]
    assert prominent_text(lines, frame_h=400) == "Real Title"


def test_prominence_none_when_empty():
    assert prominent_text([], frame_h=400) is None


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    out = str(d / "demo.mp4")
    synthgen.generate(out)
    return out


def test_names_come_from_on_screen_headings(synth):
    """The auto-name of each clip should contain that scene's heading (OCR is
    imperfect, so we match loosely: the heading words appear in the name)."""
    result = analyze(synth)
    headings = ["Project Overview", "Terminal Session", "Settings Panel",
                "Database Schema", "Final Results"]
    # every clip got a non-empty name
    assert all(c.name for c in result.clips)
    joined = " ".join(c.name.lower().replace("_", " ") for c in result.clips)
    # at least 4 of the 5 headings should be recovered by OCR (allow one miss)
    recovered = sum(1 for h in headings if h.lower() in joined)
    assert recovered >= 4, f"only recovered {recovered} headings from names: " \
                           f"{[c.name for c in result.clips]}"
