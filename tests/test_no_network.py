"""The local premise: ClipSplit's detect + split + name path opens no sockets.

We install a guard that raises if any Python socket is created, then run the
full analysis (detection + OCR naming) over the synthetic recording. If any
dependency in that path tried to phone home, this test would fail.

Caveat (same as LeakShot): Apple Vision OCR runs in a separate on-device system
process, so its work happens outside this Python process and the socket guard
cannot observe it. Apple documents Vision as fully on-device; this test covers
everything on the Python side (OpenCV, numpy, our code, the ffmpeg subprocess
for keyframe probing).
"""

import os
import socket
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from clipsplit import synthgen  # noqa: E402
from clipsplit.pipeline import analyze  # noqa: E402


class NetworkAccessError(AssertionError):
    pass


class _BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise NetworkAccessError(
            "A network socket was created during analysis. ClipSplit must be 100% local.")


def _block_connect(*args, **kwargs):
    raise NetworkAccessError("An outbound connection was attempted during analysis.")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    out = str(d / "demo.mp4")
    synthgen.generate(out)
    return out


def test_no_socket_opened_during_analysis(synth, monkeypatch):
    monkeypatch.setattr(socket, "socket", _BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", _block_connect)
    monkeypatch.setattr(socket.socket, "connect", _block_connect, raising=False)

    result = analyze(synth)

    # sanity: analysis actually did work, so the guard was exercised on the real
    # path, not a no-op.
    assert len(result.clips) >= 2, "analysis produced too few clips; guard not exercised"
    assert all(c.name for c in result.clips), "clips were not named; naming path skipped"
