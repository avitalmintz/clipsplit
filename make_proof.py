"""Render the README proof image from a REAL detection run.

Runs ClipSplit on the synthetic demo, then composites the actual clip thumbnails
and their auto-derived names / timecodes into a single review-panel image that
mirrors the app UI. Everything shown (thumbnails, names, timecodes, methods) is
produced by the real pipeline, not mocked.

Run:  ./venv/bin/python make_proof.py
"""

from __future__ import annotations

import os

import cv2
from PIL import Image, ImageDraw, ImageFont

from clipsplit import export as export_mod
from clipsplit import pipeline as pipeline_mod

HERE = os.path.dirname(__file__)
VIDEO = os.path.join(HERE, "synthetic_out", "demo.mp4")
OUT = os.path.join(HERE, "assets", "proof_detected.png")

BG = (247, 247, 245)
CARD = (255, 255, 255)
BORDER = (227, 227, 222)
INK = (29, 29, 27)
MUTED = (107, 107, 102)
ACCENT = (79, 70, 229)
OK = (22, 163, 74)


def _font(size, bold=False):
    paths = ([
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ])
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _tc(sec):
    m = int(sec) // 60
    return f"{m}:{sec - m*60:05.2f}"


def main():
    if not os.path.exists(VIDEO):
        from clipsplit import synthgen
        synthgen.generate(VIDEO)

    result = pipeline_mod.analyze(VIDEO)
    clips = result.clips
    kf = export_mod.video.keyframe_times(VIDEO)

    # grab a real thumbnail for each clip
    thumbs = []
    cap = cv2.VideoCapture(VIDEO)
    for c in clips:
        t = c.start + min(0.6, max(0.1, c.duration * 0.25))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.thumbnail((200, 130))
            thumbs.append(im)
        else:
            thumbs.append(Image.new("RGB", (200, 113), (230, 230, 230)))
    cap.release()

    # decide export method per clip (for the badge)
    methods = []
    for c in clips:
        delta = min(abs(c.start - k) for k in kf) if kf else 1e9
        methods.append("lossless" if delta <= 0.25 else "re-encode")

    # layout
    W = 900
    pad = 28
    header_h = 150
    row_h = 150
    H = header_h + len(clips) * row_h + pad
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title = _font(30, bold=True)
    f_sub = _font(15)
    f_label = _font(12, bold=True)
    f_name = _font(17, bold=True)
    f_meta = _font(13)
    f_badge = _font(11, bold=True)

    # header
    d.text((pad, 24), "ClipSplit", font=f_title, fill=INK)
    d.text((pad, 66), f"{len(clips)} clips detected from a {_tc(result.info.duration)} "
                      f"recording · named from on-screen text (Apple Vision OCR)",
           font=f_sub, fill=MUTED)
    n_loss = methods.count("lossless")
    d.text((pad, 92), f"{n_loss} of {len(clips)} cut losslessly (boundary on a keyframe); "
                      f"the rest frame-accurate re-encode.",
           font=f_sub, fill=MUTED)
    d.line([(pad, 130), (W - pad, 130)], fill=BORDER, width=1)

    y = header_h
    for i, (c, th, method) in enumerate(zip(clips, thumbs, methods)):
        rx, ry, rw = pad, y, W - 2 * pad
        rh = row_h - 14
        d.rounded_rectangle([rx, ry, rx + rw, ry + rh], radius=12,
                            fill=CARD, outline=BORDER, width=1)
        # thumbnail
        tx, ty = rx + 14, ry + (rh - th.height) // 2
        img.paste(th, (tx, ty))
        d.rectangle([tx, ty, tx + th.width, ty + th.height], outline=BORDER, width=1)

        # text block
        bx = tx + th.width + 24
        d.text((bx, ry + 20), f"#{c.index:02d}", font=f_label, fill=ACCENT)
        d.text((bx + 46, ry + 18), c.name, font=f_name, fill=INK)
        d.text((bx, ry + 52), f"{_tc(c.start)}  to  {_tc(c.end)}",
               font=f_meta, fill=INK)
        d.text((bx, ry + 74), f"{c.duration:.1f}s", font=f_meta, fill=MUTED)

        # method badge (right side)
        col = OK if method == "lossless" else (37, 99, 235)
        bg = (238, 247, 240) if method == "lossless" else (239, 245, 255)
        bw = 92
        bxr = rx + rw - bw - 16
        d.rounded_rectangle([bxr, ry + 20, bxr + bw, ry + 42], radius=6,
                            fill=bg, outline=col, width=1)
        tw = d.textlength(method, font=f_badge)
        d.text((bxr + (bw - tw) / 2, ry + 25), method, font=f_badge, fill=col)

        y += row_h

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print(f"Wrote {OUT}  ({len(clips)} clips)")
    for c, m in zip(clips, methods):
        print(f"  #{c.index:02d} {c.name:26} {_tc(c.start)}-{_tc(c.end)}  {m}")


if __name__ == "__main__":
    main()
