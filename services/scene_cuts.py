"""Detect hard cuts and camera repositioning, so they cannot be "flaws".

THE PROBLEM THIS FIXES
    Latent kinetic error measures how far the movement departed from its own
    recent trend. A hard cut departs from it completely -- the entire frame
    changes at once -- so a scene change produces the largest spike in the
    clip, every time, by a wide margin.

    Left alone, that spike wins worst-moment selection. SAM then segments the
    title card, and Gemini is asked to coach a graphic. Observed exactly that
    on stitched footage: two of three chapters reported their proof moment as
    a "title graphic". Gemini was honest about it, which is the only reason it
    did not become a fabricated fault.

WHY A COLOUR HISTOGRAM AND NOT THE EMBEDDINGS
    The V-JEPA embedding covers a 0.75 s window of 16 frames, so a cut inside
    a window is smeared across it and the embedding cannot say where -- or
    whether -- it happened. A cut is a FRAME-level event and needs a
    frame-level signal.

    An HSV histogram is the right one: it ignores motion (a fast rep barely
    moves the colour distribution) and reacts violently to the whole frame
    being replaced. It is also nearly free, since it runs on frames the
    sampler has already decoded.

MEASURED SEPARATION (this project's clips, cosine distance on consecutive
frames)
    continuous footage : median 0.0002, p99 0.062, max 0.309
    genuine hard cuts  : 0.990, 0.981, 1.000

    Two orders of magnitude between the two populations, which is why a fixed
    threshold works here at all. 0.65 sits in the empty middle: it caught
    every real cut and produced zero false positives on continuous footage.
"""

import cv2
import numpy as np

# Cosine distance between consecutive frame histograms above which the frame
# is a discontinuity. See the measured separation above before changing it.
CUT_THRESHOLD = 0.65

# Histogram resolution per HSV channel. 8 is plenty: this has to distinguish
# "different scene" from "same scene", not identify what is in the frame.
BINS = 8

# Frames are shrunk before histogramming. The colour distribution of a scene
# survives it and the cost drops by two orders of magnitude.
THUMB = (160, 90)


def frame_signature(frame_bgr):
    """BGR frame -> L2-normalised HSV histogram."""
    hsv = cv2.cvtColor(cv2.resize(frame_bgr, THUMB), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [BINS] * 3,
                     [0, 180, 0, 256, 0, 256]).ravel()
    return h / (np.linalg.norm(h) + 1e-8)


def distance(a, b):
    """Cosine distance between two signatures. 0 = identical scene."""
    return float(1.0 - np.dot(a, b))


def scan_window(frames_rgb):
    """The 16 frames of one window -> (first_sig, last_sig, max_internal_cut).

    Frames arrive RGB from the sampler (it converts on decode), so they are
    converted back for cv2's HSV call. Cheaper than making the sampler hand
    out both.
    """
    if frames_rgb is None or len(frames_rgb) == 0:
        return None, None, 0.0
    sigs = [frame_signature(f[:, :, ::-1].copy()) for f in frames_rgb]
    worst = 0.0
    for a, b in zip(sigs[:-1], sigs[1:]):
        worst = max(worst, distance(a, b))
    return sigs[0], sigs[-1], worst


class CutTracker:
    """Accumulates window scans and reports which windows are discontinuous.

    Used from Stage A, where every window is decoded anyway, so detection adds
    a histogram per frame and nothing else.
    """

    def __init__(self, threshold=CUT_THRESHOLD):
        self.threshold = threshold
        self._prev_last = None
        self.records = []          # per window: {internal, boundary, cut}

    def add(self, frames_rgb):
        """Scan one window. Returns True if it contains a discontinuity."""
        first, last, internal = scan_window(frames_rgb)
        boundary = 0.0
        if first is not None and self._prev_last is not None:
            boundary = distance(self._prev_last, first)
        if last is not None:
            self._prev_last = last
        cut = max(internal, boundary) > self.threshold
        self.records.append({
            "internal": round(internal, 4),
            "boundary": round(boundary, 4),
            "cut": bool(cut),
        })
        return cut

    def cut_windows(self):
        """Indices of windows containing a cut."""
        return {i for i, r in enumerate(self.records) if r["cut"]}

    def excluded_windows(self, pad=1):
        """Windows unusable as a worst-moment candidate.

        Cuts are padded by one either side. A cut at window i corrupts the
        embedding delta arriving at i AND the one leaving it, so the two
        neighbours are just as untrustworthy as the cut itself -- and the
        spike in kinetic error typically lands on a neighbour rather than on i.

        """
        out = set()
        n = len(self.records)
        for i in self.cut_windows():
            for j in range(max(0, i - pad), min(n, i + pad + 1)):
                out.add(j)
        return out

    def summary(self):
        cuts = sorted(self.cut_windows())
        return {
            "threshold": self.threshold,
            "windows_scanned": len(self.records),
            "cuts_detected": len(cuts),
            "cut_windows": cuts,
            "excluded_windows": sorted(self.excluded_windows()),
            "max_boundary_distance": (
                round(max((r["boundary"] for r in self.records), default=0.0), 4)),
            "max_internal_distance": (
                round(max((r["internal"] for r in self.records), default=0.0), 4)),
        }
