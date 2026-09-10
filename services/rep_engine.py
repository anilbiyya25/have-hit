"""STAGE A (second half) -- repetition structure from the embedding sequence.

Reps are periodic: a squat's embedding leaves a stance, travels to the bottom
and comes back, and the next rep retraces it. That shows up as an oscillation in
the V-JEPA embedding trajectory, and it does so without anything here knowing
what a squat is -- the same machinery counts tennis serves.

METHOD
  1. Project the (n, 768) trajectory onto its first principal component. Reps
     are the dominant variance in a set of reps, so PC1 is the axis the movement
     actually oscillates along. This is a big reduction, and it is the right
     one: counting on all 768 dimensions at once means counting noise too.
  2. Autocorrelate that 1-D signal to find the dominant period.
  3. Take extrema spaced at roughly half that period as rep turning points.

A NOTE ON RESOLUTION, WHICH IS THE REAL LIMIT
  A rep lasts 2-4 s. Sampling in 2 s windows gives about one window per rep,
  which cannot resolve a rep any more than one sample per cycle can resolve a
  sine wave -- this is plain Nyquist. So the pipeline samples finer for rep work
  (see REP_WINDOW_SECONDS) and, when the footage still does not support it, this
  module says so in `confidence` rather than returning a rep count that looks
  authoritative and is not.
"""

import numpy as np

# Target window length for rep work. Short enough to put 3-5 windows inside a
# typical 2-4 s rep, which is the minimum for locating a turning point rather
# than merely detecting that one happened.
REP_WINDOW_SECONDS = 0.75

# Below this many windows per period, the sampling cannot honestly resolve
# individual reps and the count is reported as low confidence.
MIN_WINDOWS_PER_PERIOD = 3.0


def _pc1(Z):
    """Project an (n, D) trajectory onto its first principal component.

    Uses the covariance eigenvector rather than a full SVD: D is 768 but n is
    typically under 150, so the (n, n) Gram route is far cheaper. Sign is fixed
    so the signal starts below its mean, making "descend then return" the
    canonical rep shape and keeping peak/trough labels stable across clips.
    """
    X = Z - Z.mean(axis=0, keepdims=True)
    # Gram matrix trick: eigenvectors of X X^T map to those of X^T X.
    G = X @ X.T
    vals, vecs = np.linalg.eigh(G)
    s = vecs[:, -1] * np.sqrt(max(float(vals[-1]), 0.0))
    if len(s) and s[0] > s.mean():
        s = -s
    return s.astype(np.float32), float(vals[-1] / (vals.sum() + 1e-8))


def _dominant_period(s):
    """Dominant period of a 1-D signal, in samples, via autocorrelation."""
    n = len(s)
    if n < 4:
        return None, 0.0
    x = s - s.mean()
    denom = float((x * x).sum()) + 1e-8
    ac = np.correlate(x, x, mode="full")[n - 1:] / denom

    # Skip lag 0 and walk past the first zero crossing, so the self-similarity
    # at tiny lags does not masquerade as a period.
    start = 1
    while start < len(ac) and ac[start] > 0:
        start += 1
    if start >= len(ac) - 1:
        return None, 0.0

    # Only consider periods that could fit at least twice in the clip; a "period"
    # longer than half the footage has not been observed to repeat even once.
    hi = max(start + 1, n // 2)
    seg = ac[start:hi]
    if not len(seg):
        return None, 0.0
    lag = int(np.argmax(seg)) + start
    return lag, float(seg.max())


def _extrema(s, min_gap, minima=True):
    """Indices of local minima (or maxima) of `s`, at least `min_gap` apart."""
    sign = 1.0 if minima else -1.0
    x = s * sign
    idx = []
    for i in range(1, len(x) - 1):
        if x[i] <= x[i - 1] and x[i] <= x[i + 1]:
            if not idx or i - idx[-1] >= min_gap:
                idx.append(i)
            elif x[i] < x[idx[-1]]:
                idx[-1] = i          # keep the more extreme of two close ones
    return idx


def detect_reps(embeddings, times, kinetic_error=None):
    """Embedding sequence -> rep boundaries, count, cadence, best/worst.

    `kinetic_error` (Stage B) is optional. When present it scores each rep, so
    best_rep/worst_rep reflect measured trajectory consistency rather than an
    arbitrary pick.
    """
    Z = np.stack(embeddings).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    n = len(Z)

    empty = {
        "total_reps": 0, "reps": [], "cadence_per_min": 0.0,
        "confidence": "none", "period_windows": None,
        "pc1_variance_ratio": 0.0, "best_rep": None, "worst_rep": None,
        "note": "too few windows to detect repetition",
    }
    if n < 6:
        return empty

    s, var_ratio = _pc1(Z)
    period, strength = _dominant_period(s)
    if period is None or period < 2:
        return {**empty, "note": "no periodic structure found; likely a single "
                                 "continuous movement rather than a rep set"}

    gap = max(2, int(period * 0.6))
    # Troughs are the bottom of each rep -- one per repetition performed. Peaks
    # are the standing/reset positions between them, so they delimit reps.
    #
    # Counting trough-to-trough SPANS instead would report N-1 for N reps: six
    # squats reach the bottom six times but leave only five gaps between those
    # bottoms. The count a user checks against is bottoms.
    troughs = _extrema(s, gap, minima=True)
    peaks = _extrema(s, gap, minima=False)
    if not troughs:
        return {**empty, "period_windows": period, "pc1_variance_ratio":
                round(var_ratio, 4),
                "note": "periodicity detected but no turning points found"}

    win_s = float(times[0][1] - times[0][0]) or 1.0
    windows_per_period = float(period)

    # Honest confidence. Resolution first: below Nyquist-ish sampling the count
    # is a guess no matter how clean the autocorrelation looked.
    if windows_per_period < MIN_WINDOWS_PER_PERIOD:
        confidence = "low"
    elif strength >= 0.45 and var_ratio >= 0.30:
        confidence = "high"
    elif strength >= 0.25:
        confidence = "medium"
    else:
        confidence = "low"

    reps = []
    for r, bottom in enumerate(troughs, start=1):
        # Bound each rep by the reset positions either side of its bottom.
        # Where the clip starts or ends mid-rep there is no enclosing peak, so
        # the clip edge stands in rather than dropping an observed repetition.
        before = [p for p in peaks if p < bottom]
        after = [p for p in peaks if p > bottom]
        a = before[-1] if before else 0
        b = after[0] if after else n - 1

        t_start = float(times[a][0])
        t_end = float(times[b][1])
        rep = {
            "rep_number": r,
            "t_start": round(t_start, 2),
            # The bottom of the movement: deepest point of a squat, the catch
            # of a swim stroke, the trough of the dominant oscillation.
            "t_inflection": round(float(times[bottom][0]), 2),
            "t_end": round(t_end, 2),
            "duration_s": round(t_end - t_start, 2),
            "timestamp": _mmss(t_start),
        }
        if kinetic_error is not None and len(kinetic_error):
            lo, hi = min(a, len(kinetic_error) - 1), min(b, len(kinetic_error))
            seg = np.asarray(kinetic_error[lo:max(lo + 1, hi)])
            rep["kinetic_error"] = round(float(seg.mean()), 4) if seg.size else 0.0
        reps.append(rep)

    span = reps[-1]["t_end"] - reps[0]["t_start"]
    cadence = round(len(reps) / span * 60.0, 1) if span > 0 else 0.0

    best = worst = None
    if reps and "kinetic_error" in reps[0]:
        ranked = sorted(reps, key=lambda r: r["kinetic_error"])
        # timestamp stays the MM:SS.s string the response schema specifies.
        # timestamp_s is added alongside it because callers that need to SEEK
        # to this rep -- the keyframe grab, the player -- were re-parsing the
        # display string to get a number back, and one of them fed the string
        # straight into a numeric comparison.
        #
        # It is the INFLECTION, not t_start: the bottom of a squat is the
        # frame that shows the rep, while its start is a lifter standing
        # still, which is the same picture for every rep in the set.
        # Both fields name the SAME instant. They did not at first: timestamp
        # was copied from the rep record, where it is the rep's start, while
        # timestamp_s was the inflection -- so a report showed "00:10.6" and
        # seeked to 11.41, and two fields with the same name meant two
        # different moments. Formatted from one number here so they cannot
        # drift again.
        best = {"rep_number": ranked[0]["rep_number"],
                "timestamp": _mmss(ranked[0]["t_inflection"]),
                "timestamp_s": ranked[0]["t_inflection"]}
        worst = {"rep_number": ranked[-1]["rep_number"],
                 "timestamp": _mmss(ranked[-1]["t_inflection"]),
                 "timestamp_s": ranked[-1]["t_inflection"]}

    return {
        "total_reps": len(reps),
        "reps": reps,
        "cadence_per_min": cadence,
        "confidence": confidence,
        "period_windows": period,
        "period_seconds": round(period * win_s, 2),
        "periodicity_strength": round(strength, 4),
        "pc1_variance_ratio": round(var_ratio, 4),
        "windows_per_rep": round(windows_per_period, 2),
        "best_rep": best,
        "worst_rep": worst,
        "note": ("sampling resolves fewer than "
                 f"{MIN_WINDOWS_PER_PERIOD} windows per rep; count is "
                 "approximate" if confidence == "low" else ""),
    }


def _mmss(seconds):
    """Seconds -> MM:SS.s, the timestamp format the response schema asks for."""
    m, s = divmod(max(float(seconds), 0.0), 60.0)
    return f"{int(m):02d}:{s:04.1f}"
