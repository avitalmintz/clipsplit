"""Generate the synthetic demo recording used by the README / a first run.

Writes synthetic_out/demo.mp4 and its ground-truth sidecar. This is the same
generator the tests and validate.py use; run it once so you have a video to
point the UI at.

Run:  ./venv/bin/python make_demo.py
"""

import os

from clipsplit import synthgen

OUT = os.path.join(os.path.dirname(__file__), "synthetic_out", "demo.mp4")

if __name__ == "__main__":
    truth = synthgen.generate(OUT)
    print(f"Wrote {OUT}")
    print(f"  {truth.duration:.1f}s, {len(truth.scenes)} scenes, fps={truth.fps:.0f}")
    print(f"  planted boundaries: {truth.boundaries}")
    print(f"  headings: {[s['heading'] for s in truth.scenes]}")
    print("\nNow launch the UI and point it at this file:")
    print(f"  ./venv/bin/python app.py {OUT}")
