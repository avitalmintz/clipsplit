"""ClipSplit review UI. 100% local. Binds to 127.0.0.1, share=False, no browser.

Flow:
  1. Pick a screen-recording video (.mov / .mp4).
  2. "Detect clips": ClipSplit finds where the on-screen content meaningfully
     changes, splits the timeline into segments, and auto-names each from its
     most prominent on-screen text (Apple Vision OCR, on-device).
  3. Review each clip: thumbnail, start/end timecodes, duration, and an
     editable name. Toggle which clips to keep.
  4. "Export selected" / "Export all": cut the clips to an output folder
     (lossless stream copy where the boundary lands on a keyframe, a
     frame-accurate re-encode where it does not) and report where they went.

Nothing you process leaves your machine. OCR is Apple Vision (on-device); all
detection and cutting run in-process with a bundled static ffmpeg. No network.

Run:  ./venv/bin/python app.py  [optional video path to prefill]
"""

from __future__ import annotations

import base64
import html
import io
import os
import sys

import cv2
import gradio as gr
from PIL import Image

from clipsplit import export as export_mod
from clipsplit import pipeline as pipeline_mod
from clipsplit import video as video_mod
from clipsplit.detect import DetectParams


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.video_path: str = ""
        self.clips = []                      # list[Clip]
        self.info = None                     # VideoInfo
        self.thumbs: dict[int, str] = {}     # clip index -> data-uri thumbnail


STATE = State()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tc(seconds: float) -> str:
    """Format seconds as m:ss.d timecode."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


def _thumb_uri(path: str, t: float, size: int = 240) -> str:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return ""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------

def do_detect(video_path: str, progress=gr.Progress()):
    path = os.path.expanduser((video_path or "").strip())
    if not path or not os.path.isfile(path):
        return (_warn("Pick a screen-recording video (.mov or .mp4) first."),
                gr.update(visible=False), *[gr.update()] * (MAX_CLIPS * _WIDGETS_PER_CLIP))

    STATE.video_path = path
    progress(0.1, desc="Analyzing scene changes")
    result = pipeline_mod.analyze(path, DetectParams())
    STATE.clips = result.clips
    STATE.info = result.info

    progress(0.8, desc="Building thumbnails")
    STATE.thumbs = {}
    for c in STATE.clips:
        t = c.start + min(0.6, max(0.1, c.duration * 0.25))
        STATE.thumbs[c.index] = _thumb_uri(path, t)

    n_copy = "lossless where the cut lands on a keyframe, frame-accurate re-encode otherwise"
    summary = _summary_html(n_copy)
    updates = _render_clip_updates()
    return summary, gr.update(visible=True), *updates


def _summary_html(export_note: str) -> str:
    info = STATE.info
    n = len(STATE.clips)
    dur = info.duration if info else 0.0
    return (
        f'<div class="cs-summary">'
        f'<div class="cs-bignum">{n} clip{"s" if n != 1 else ""} detected</div>'
        f'<div class="cs-subline">from a {_tc(dur)} recording &middot; '
        f'{info.width}&times;{info.height} @ {info.fps:.0f}fps</div>'
        f'<div class="cs-tinyline">Names are the most prominent on-screen text at each '
        f'cut (Apple Vision OCR, on-device). Edit any name below, untick clips you do '
        f'not want, then export. Cutting is {export_note}.</div>'
        f'</div>'
    )


def _warn(msg: str) -> str:
    return f'<div class="cs-summary cs-warn"><div class="cs-subline">{html.escape(msg)}</div></div>'


# ---------------------------------------------------------------------------
# Dynamic clip rows. Gradio needs a fixed number of components, so we build a
# pool of MAX_CLIPS rows and show/hide + fill them per detection.
# ---------------------------------------------------------------------------

MAX_CLIPS = 40
# per clip: [row(Group visibility), thumb HTML, name Textbox, keep Checkbox]
_WIDGETS_PER_CLIP = 4


def _clip_meta_html(c) -> str:
    return (
        f'<div class="cs-meta">'
        f'<span class="cs-idx">#{c.index:02d}</span>'
        f'<span class="cs-time">{_tc(c.start)} &rarr; {_tc(c.end)}</span>'
        f'<span class="cs-dur">{c.duration:.1f}s</span>'
        f'</div>'
    )


def _thumb_html(idx: int) -> str:
    uri = STATE.thumbs.get(idx, "")
    if not uri:
        return '<div class="cs-thumb cs-thumb-empty">no frame</div>'
    return f'<img class="cs-thumb" src="{uri}" alt="clip thumbnail"/>'


def _render_clip_updates():
    """Return the flat list of gr.update()s for all pooled clip widgets."""
    updates = []
    for i in range(MAX_CLIPS):
        if i < len(STATE.clips):
            c = STATE.clips[i]
            combined = _thumb_html(c.index) + _clip_meta_html(c)
            updates.append(gr.update(visible=True))                 # row
            updates.append(gr.update(value=combined))               # thumb+meta html
            updates.append(gr.update(value=c.name, visible=True))   # name textbox
            updates.append(gr.update(value=True, visible=True))     # keep checkbox
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value="", visible=False))
            updates.append(gr.update(value=False, visible=False))
    return updates


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _default_out_dir() -> str:
    if STATE.video_path:
        base = os.path.splitext(os.path.basename(STATE.video_path))[0]
        return os.path.join(os.path.dirname(STATE.video_path), f"{base}_clips")
    return os.path.join(os.getcwd(), "clipsplit_out")


def _do_export(out_dir: str, names: list, keeps: list, only_selected: bool,
               progress=gr.Progress()):
    if not STATE.clips:
        return _warn("Detect clips first.")
    out_dir = os.path.expanduser((out_dir or "").strip()) or _default_out_dir()

    # apply edited names back onto the clips
    chosen = []
    for i, c in enumerate(STATE.clips):
        if i < len(names) and names[i] and names[i].strip():
            c.name = names[i].strip()
        keep = (keeps[i] if i < len(keeps) else True)
        if (not only_selected) or keep:
            chosen.append(c)

    if not chosen:
        return _warn("No clips selected to export. Tick at least one.")

    def _prog(i, n, clip):
        progress((i, n), desc=f"Exporting {clip.name}")

    results = export_mod.export_clips(STATE.video_path, chosen, out_dir, progress=_prog)

    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    n_copy = sum(1 for r in ok if r.method == "copy")
    n_reenc = sum(1 for r in ok if r.method == "reencode")

    rows = []
    for r in results:
        badge = ('<span class="cs-badge cs-copy">lossless</span>' if r.method == "copy"
                 else '<span class="cs-badge cs-reenc">re-encode</span>')
        status = "&#10003;" if r.ok else "&#10007;"
        rows.append(
            f'<div class="cs-exrow">{status} {badge}'
            f'<code>{html.escape(os.path.basename(r.path))}</code>'
            f'<span class="cs-exnote">{html.escape(r.detail)}</span></div>'
        )

    head = (f'<div class="cs-exhead">Exported {len(ok)} clip{"s" if len(ok)!=1 else ""} to '
            f'<code>{html.escape(out_dir)}</code>'
            f'<div class="cs-tinyline">{n_copy} lossless stream-copy, {n_reenc} '
            f'frame-accurate re-encode.'
            + (f' {len(fail)} failed.' if fail else '')
            + '</div></div>')
    return f'<div class="cs-export cs-ok">{head}{"".join(rows)}</div>'


def export_selected(out_dir, *args):
    names = list(args[:MAX_CLIPS])
    keeps = list(args[MAX_CLIPS:MAX_CLIPS * 2])
    return _do_export(out_dir, names, keeps, only_selected=True)


def export_all(out_dir, *args):
    names = list(args[:MAX_CLIPS])
    keeps = list(args[MAX_CLIPS:MAX_CLIPS * 2])
    return _do_export(out_dir, names, keeps, only_selected=False)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
:root {
  --cs-bg: #f7f7f5;
  --cs-card: #ffffff;
  --cs-border: #e3e3de;
  --cs-ink: #1d1d1b;
  --cs-muted: #6b6b66;
  --cs-accent: #4f46e5;
  --cs-ok: #16a34a;
  --cs-warn: #d97706;
}
.gradio-container { background: var(--cs-bg) !important; max-width: 1080px !important; }
footer { display: none !important; }

.cs-hero { padding: 4px 2px 0; }
.cs-hero h1 { font-size: 30px; margin: 0 0 4px; letter-spacing: -0.5px; }
.cs-hero .cs-tag { color: var(--cs-muted); font-size: 15px; margin: 0; }
.cs-local { display: inline-block; margin-top: 8px; padding: 3px 10px; border-radius: 999px;
  background: #eef7f0; color: var(--cs-ok); font-size: 12.5px; font-weight: 600;
  border: 1px solid #cdebd6; }

.cs-section-label { font-weight: 700; font-size: 13px; color: var(--cs-muted);
  text-transform: uppercase; letter-spacing: 0.6px; margin: 18px 0 6px; }

.cs-summary { background: var(--cs-card); border: 1px solid var(--cs-border);
  border-left: 5px solid var(--cs-accent); border-radius: 12px; padding: 16px 18px; margin: 8px 0; }
.cs-summary.cs-warn { border-left-color: var(--cs-warn); }
.cs-bignum { font-size: 21px; font-weight: 800; letter-spacing: -0.3px; }
.cs-subline { margin-top: 5px; font-size: 14.5px; color: var(--cs-ink); }
.cs-tinyline { margin-top: 8px; font-size: 12.5px; color: var(--cs-muted); line-height: 1.5; }

/* clip rows */
.cs-cliprow { background: var(--cs-card); border: 1px solid var(--cs-border);
  border-radius: 12px; padding: 10px 12px !important; margin-bottom: 8px; }
.cs-thumb { width: 100%; border-radius: 8px; border: 1px solid var(--cs-border);
  display: block; background: #eee; }
.cs-thumb-empty { height: 90px; display: flex; align-items: center; justify-content: center;
  color: var(--cs-muted); font-size: 12px; }
.cs-meta { display: flex; align-items: center; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
.cs-idx { font-weight: 800; color: var(--cs-accent); font-size: 13px; }
.cs-time { color: var(--cs-ink); font-size: 12.5px; font-family: ui-monospace, monospace; }
.cs-dur { color: var(--cs-muted); font-size: 12px; }

/* export report */
.cs-export { background: var(--cs-card); border: 1px solid var(--cs-border);
  border-left: 5px solid var(--cs-ok); border-radius: 12px; padding: 14px 16px; margin-top: 10px; }
.cs-exhead { font-size: 15px; font-weight: 700; margin-bottom: 8px; }
.cs-exhead code { background: #f2f2ee; padding: 1px 6px; border-radius: 5px; font-size: 12.5px;
  font-weight: 500; }
.cs-exrow { display: flex; align-items: center; gap: 8px; padding: 5px 0;
  border-top: 1px solid var(--cs-border); font-size: 12.5px; flex-wrap: wrap; }
.cs-exrow code { background: #f2f2ee; padding: 1px 6px; border-radius: 5px; }
.cs-exnote { color: var(--cs-muted); font-size: 11.5px; }
.cs-badge { display: inline-block; padding: 1px 7px; border-radius: 6px; font-size: 10.5px;
  font-weight: 800; letter-spacing: 0.3px; }
.cs-copy { background: #eef7f0; color: var(--cs-ok); border: 1px solid #cdebd6; }
.cs-reenc { background: #eff5ff; color: #2563eb; border: 1px solid #cadcf8; }
"""


def build_ui(prefill: str = ""):
    with gr.Blocks(title="ClipSplit") as demo:
        gr.HTML(
            '<div class="cs-hero">'
            '<h1>&#9986; ClipSplit</h1>'
            '<p class="cs-tag">Auto-split a screen recording into named clips. '
            'Detects where the screen changes, cuts there, and names each clip from '
            'its on-screen text.</p>'
            '<span class="cs-local">On-device Apple Vision OCR &middot; bundled ffmpeg '
            '&middot; no network calls</span>'
            '</div>'
        )

        gr.HTML('<div class="cs-section-label">1 &middot; Choose a recording</div>')
        with gr.Row():
            video_in = gr.Textbox(
                label=None, value=prefill, container=False, scale=4,
                placeholder="paste a path to a .mov or .mp4 screen recording")
            detect_btn = gr.Button("Detect clips", variant="primary", scale=1)

        summary = gr.HTML()

        # clip rows section (hidden until detection)
        with gr.Column(visible=False) as clips_section:
            gr.HTML('<div class="cs-section-label">2 &middot; Review clips</div>')
            row_groups = []
            thumb_htmls = []
            name_boxes = []
            keep_boxes = []
            for i in range(MAX_CLIPS):
                with gr.Row(visible=False, elem_classes=["cs-cliprow"]) as rg:
                    with gr.Column(scale=2, min_width=180):
                        th = gr.HTML()
                    with gr.Column(scale=4):
                        nm = gr.Textbox(label="Clip name", visible=False,
                                        container=True, interactive=True)
                    with gr.Column(scale=1, min_width=90):
                        kp = gr.Checkbox(label="Export", value=True, visible=False)
                row_groups.append(rg)
                thumb_htmls.append(th)
                name_boxes.append(nm)
                keep_boxes.append(kp)

            gr.HTML('<div class="cs-section-label">3 &middot; Export</div>')
            with gr.Row():
                out_dir_in = gr.Textbox(
                    label="Output folder (blank = <video>_clips next to the source)",
                    value="", container=True, scale=3)
                export_sel_btn = gr.Button("Export selected", variant="primary", scale=1)
                export_all_btn = gr.Button("Export all", scale=1)
            export_report = gr.HTML()

        # flatten pooled widgets in the order _render_clip_updates produces
        clip_outputs = []
        for i in range(MAX_CLIPS):
            clip_outputs += [row_groups[i], thumb_htmls[i], name_boxes[i], keep_boxes[i]]

        detect_btn.click(
            do_detect, [video_in],
            [summary, clips_section, *clip_outputs],
        )
        export_sel_btn.click(
            export_selected, [out_dir_in, *name_boxes, *keep_boxes], [export_report])
        export_all_btn.click(
            export_all, [out_dir_in, *name_boxes, *keep_boxes], [export_report])

    return demo


if __name__ == "__main__":
    prefill = sys.argv[1] if len(sys.argv) > 1 else ""
    demo = build_ui(prefill)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                inbrowser=False, css=CSS,
                theme=gr.themes.Soft(font=["system-ui", "sans-serif"]))
