"""Slice a long session into exercise chapters and rest periods.

A ten-minute upload is not one movement. It is a few sets of one lift, a rest,
a few sets of another, and some walking around in between. Analysing it as a
single block produces one averaged score describing nothing that happened.

METHOD: SEGMENT ON SMOOTHED EMBEDDINGS, NAME WITH LABELS
    The tempting approach is to segment on the classifier's label sequence,
    since the label ought to be stable across a set. Measured, it is not: on a
    three-exercise clip the dominant label held only 26-42% of the windows in
    its own block, and run-length encoding that found two exercises where
    there were three. A 44%-accurate head is simply too noisy to cut on, and
    no amount of median filtering rescues a signal that is wrong most of the
    time.

    The embeddings those labels are derived from are much better behaved. The
    obstacle to using them directly is that the biggest embedding changes
    inside a set are the REPS, so raw change-point detection chops a set into
    reps rather than finding where squats became bench press.

    Smoothing removes exactly that. A moving average over a window longer than
    one rep cancels the rep oscillation and leaves the slower drift, and what
    remains when the rep cycle is averaged out is the change of exercise.
    Boundaries come from peaks in the smoothed trajectory; the noisy labels
    are then used only to NAME each finished block, where a plurality vote
    over dozens of windows is reliable even when any single vote is not.

REST
    Rest is not an exercise the classifier knows, so it cannot be labelled. It
    is detected geometrically instead: consecutive windows whose embeddings
    barely move are someone standing still, whatever label the head guessed.
"""

import numpy as np

# Windows in the median filter. At 0.75 s per window, 7 spans ~5 s -- longer
# than any single rep, shorter than any real set, which is the gap this has to
# thread.
LABEL_SMOOTH = 7

# A block shorter than this is not a set, it is misclassification. Merged into
# whichever neighbour it resembles rather than reported as an exercise.
MIN_BLOCK_S = 10.0

# Rest detection. A window whose latent speed sits below this fraction of the
# clip's median counts as "not moving much".
REST_SPEED_RATIO = 0.55
MIN_REST_S = 6.0


# Moving-average span for the embedding trajectory, in windows. At 0.75 s per
# window, 9 spans ~6.75 s. Retained for reference and for the rest detector's
# neighbourhood; boundary detection no longer uses it -- see below.
EMB_SMOOTH = 9

# A boundary must beat this many robust deviations above the median change to
# count. Relative, never an absolute cosine value: consecutive V-JEPA windows
# of ANY human movement sit between 0.980 and 0.997 cosine similarity, so a
# fixed cut such as "< 0.65" fires exactly zero times on real footage. Measured
# across the five reference clips below.
BOUNDARY_Z = 4.5

# Smoothing the trajectory before differencing it was the original method, on
# the reasoning that a moving average cancels the rep oscillation and leaves
# the exercise change. It does cancel the reps. It also smears the boundary
# across the smoothing window and buries weaker ones entirely.
#
# MEASURED, five labelled 30 s clips (three with a real cut at 15.0 s, two
# single-exercise where any cut is a false positive):
#
#     clip                truth   SMOOTHED           RAW z-scored
#     two_squat_pullup    15.0    17.30  err 2.30    15.04  err 0.04  z 11.76
#     two_bench_squat     15.0    MISSED, one block  15.04  err 0.04  z  5.70
#     two_pullup_bench    15.0    18.02  err 3.02    15.02  err 0.02  z 10.03
#     one_pullup          none    FALSE SPLIT 13.52  rejected         z  3.52
#     one_bench           none    correct            rejected         z  3.65
#                                 -------            -------
#                                 0 of 5             5 of 5
#
# The rep-chopping the smoothing existed to prevent is handled structurally
# instead, by MIN_BLOCK_S: a rep-scale peak cannot become a boundary because
# it would leave a block shorter than a set. That constraint was already here
# and was doing the work the smoothing was credited with.
#
# The gate at 4.5 sits between the weakest real boundary (5.70) and the
# strongest false one (3.65). That is a 1.56x separation on FIVE clips, which
# is a real gap but a narrow one -- an order of magnitude less headroom than
# the scene-cut threshold enjoys. Widen the reference set before trusting it
# on footage unlike these.
USE_SMOOTHED_BOUNDARIES = False


def _smooth_embeddings(Z, k=EMB_SMOOTH):
    """Centred moving average over the trajectory, edges clamped."""
    n = len(Z)
    if n <= 2:
        return Z
    k = max(3, min(k, n))
    half = k // 2
    out = np.empty_like(Z)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = Z[lo:hi].mean(axis=0)
    return out


def boundary_scores(Z):
    """Per-transition robust z-score of consecutive-window dissimilarity.

    `1 - cos(z_t, z_{t+1})`, then scored against the clip's OWN median and MAD
    rather than an absolute number. Returned separately from the picking so it
    can be inspected and reported -- the response carries it, which is the
    only way to tell a confident split from a marginal one after the fact.
    """
    n = len(Z)
    if n < 2:
        return np.zeros(0), np.zeros(0)
    A = np.asarray(Z, dtype=np.float64)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    change = 1.0 - np.sum(A[:-1] * A[1:], axis=1)
    med = float(np.median(change))
    mad = float(np.median(np.abs(change - med)))
    if mad < 1e-12:
        return change, np.zeros_like(change)
    return change, (change - med) / (1.4826 * mad)


def _embedding_boundaries(Z, times, min_gap_windows):
    """Indices where consecutive windows stop resembling each other."""
    n = len(Z)
    if n < max(4, min_gap_windows * 2):
        return []
    if USE_SMOOTHED_BOUNDARIES:
        S = _smooth_embeddings(Z)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        change = 1.0 - np.sum(S[:-1] * S[1:], axis=1)
    else:
        change, _ = boundary_scores(Z)
    if not len(change):
        return []

    med = float(np.median(change))
    mad = float(np.median(np.abs(change - med)))
    if mad < 1e-9:
        return []
    z = (change - med) / (1.4826 * mad)

    # Walk peaks strongest-first, keeping a minimum block length between them
    # so a single broad transition cannot yield three boundaries in a row.
    picked = []
    for idx in np.argsort(z)[::-1]:
        i = int(idx)
        if z[i] < BOUNDARY_Z:
            break
        b = i + 1
        if b < min_gap_windows or b > n - min_gap_windows:
            continue
        if all(abs(b - p) >= min_gap_windows for p in picked):
            picked.append(b)
    return sorted(picked)


def _median_filter(labels, k=LABEL_SMOOTH):
    """Mode filter over a sliding window. Kills isolated misclassifications.

    The mode, not the median: labels are categorical, and the median of
    ["squat", "bench", "row"] by string order is meaningless.
    """
    n = len(labels)
    if n < 3:
        return list(labels)
    half = max(1, k // 2)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = labels[lo:hi]
        out.append(max(set(window), key=window.count))
    return out


def _runs(labels):
    """[(label, start_idx, end_idx_exclusive)] for consecutive equal labels."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((labels[start], start, i))
            start = i
    return runs


def _rest_mask(speed, times):
    """Boolean per window: True where the athlete was essentially still.

    Uses the median rather than the mean as the reference: a set of explosive
    reps drags the mean up enough that genuine rest stops looking slow by
    comparison, while the median is unmoved by it.
    """
    n = len(times)
    mask = [False] * n
    if len(speed) < 3:
        return mask
    ref = float(np.median(speed))
    if ref <= 1e-8:
        return mask
    slow = [bool(s < ref * REST_SPEED_RATIO) for s in speed]
    # speed[i] describes i -> i+1, so it is one shorter than the window list.
    slow.append(slow[-1] if slow else False)

    win_s = float(times[0][1] - times[0][0]) or 0.75
    need = max(2, int(round(MIN_REST_S / win_s)))
    i = 0
    while i < len(slow):
        if not slow[i]:
            i += 1
            continue
        j = i
        while j < len(slow) and slow[j]:
            j += 1
        # Only sustained stillness is rest. A single slow window is the top of
        # a rep, where the bar changes direction and the body genuinely pauses.
        if j - i >= need:
            for k in range(i, min(j, n)):
                mask[k] = True
        i = j
    return mask


# Candidate classes handed to Stage D per chapter. Three because the head's
# top-1 is right about 44% of the time and its top-3 substantially more often,
# so a shortlist constrains the naming without pinning it to a coin flip.
TOPK = 3


def chapter_candidates(gym_timeline, s, e, k=TOPK):
    """Top-k classes over one chapter's windows. -> [{exercise, confidence}].

    Ranked by MEAN PROBABILITY MASS across every window in the chapter, not by
    how often a class won. Those differ, and the difference matters: a class
    that is second by a hair in every window is a far better candidate than one
    that wins three windows out of forty and is near-zero in the rest, and a
    vote count cannot see that.

    The figure is a lower bound on the true mean, because only each window's
    top-3 survived from inference. That is deliberate -- keeping all 27 per
    window to make the tail exact would multiply the stored timeline by nine to
    refine a number nothing reads below third place.
    """
    total = {}
    n = max(e - s, 1)
    for w in gym_timeline[s:e]:
        pairs = w.get("top_k") or [[w.get("exercise"), w.get("confidence", 0.0)]]
        for label, prob in pairs:
            if label:
                total[label] = total.get(label, 0.0) + float(prob or 0.0)
    ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"exercise": label, "confidence": round(mass / n, 4)}
            for label, mass in ranked]


def _merge_short(blocks, times):
    """Fold sub-MIN_BLOCK_S EXERCISE blocks into a neighbour.

    Rest blocks are exempt. They already cleared MIN_REST_S, which is
    deliberately lower -- a 9 s gap between sets is a real rest, and applying
    the exercise minimum to it deletes every rest period in a normal session.
    That is precisely what happened before this exemption existed: three sets
    separated by two rests came back as three uninterrupted exercise blocks
    with the rests absorbed into them.
    """
    if len(blocks) <= 1:
        return blocks
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i, b in enumerate(blocks):
            dur = times[b["end"] - 1][1] - times[b["start"]][0]
            if dur >= MIN_BLOCK_S or b["label"] == "__rest__":
                continue
            # Merge into the LONGER neighbour: absorbing a stray block into an
            # established set is safer than letting it split a real one.
            prev_len = (times[blocks[i - 1]["end"] - 1][1]
                        - times[blocks[i - 1]["start"]][0]) if i > 0 else -1
            next_len = (times[blocks[i + 1]["end"] - 1][1]
                        - times[blocks[i + 1]["start"]][0]) if i < len(blocks) - 1 else -1
            target = i - 1 if prev_len >= next_len else i + 1
            if target < 0 or target >= len(blocks):
                target = i + 1 if i == 0 else i - 1
            lo = min(blocks[target]["start"], b["start"])
            hi = max(blocks[target]["end"], b["end"])
            blocks[target] = {**blocks[target], "start": lo, "end": hi}
            blocks.pop(i)
            changed = True
            break
    return blocks


def detect_chapters(embeddings, times, gym_timeline, latent_speed):
    """-> [{kind, label, start, end, t_start, t_end, confidence}] in time order.

    `kind` is "exercise" or "rest". Indices are into the window arrays, so a
    caller can slice embeddings/times per chapter and run the normal per-block
    analysis on each.
    """
    n = len(times)
    if n == 0:
        return []

    labels = [w["exercise"] for w in gym_timeline]
    confs = [w["confidence"] for w in gym_timeline]
    rest = _rest_mask(latent_speed, times)

    win_s = float(times[0][1] - times[0][0]) or 0.75
    min_gap = max(2, int(round(MIN_BLOCK_S / win_s)))

    Z = np.stack(embeddings).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8

    # Primary cuts: where consecutive windows stop resembling each other.
    # Secondary cuts: the edges of sustained stillness, which the trajectory
    # alone would not mark because standing still is a smooth state, not a
    # change.
    cuts = set(_embedding_boundaries(Z, times, min_gap))
    _, zscores = boundary_scores(Z)
    for i in range(1, n):
        if rest[i] != rest[i - 1]:
            cuts.add(i)

    edges = [0] + sorted(cuts) + [n]
    blocks = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        # Rest for the block if most of its windows were still. A couple of
        # slow windows inside a working set must not turn the set into a rest.
        rest_frac = sum(rest[a:b]) / float(b - a)
        seg = labels[a:b]
        blocks.append({
            "label": "__rest__" if rest_frac > 0.5
                     else max(set(seg), key=seg.count),
            "start": a, "end": b,
        })

    # Adjacent blocks that ended up with the same name are one block: a
    # trajectory wobble mid-set should not split a set in two.
    merged = []
    for b in blocks:
        if merged and merged[-1]["label"] == b["label"]:
            merged[-1]["end"] = b["end"]
        else:
            merged.append(b)
    blocks = _merge_short(merged, times)

    out = []
    for b in blocks:
        s, e = b["start"], b["end"]
        seg_labels = [labels[i] for i in range(s, e)]
        seg_conf = [confs[i] for i in range(s, e)]
        is_rest = b["label"] == "__rest__"
        # Recompute the majority from the RAW labels inside the final span:
        # merging may have changed which label actually dominates, and the
        # smoothed value is no longer authoritative once spans move.
        dominant = (max(set(seg_labels), key=seg_labels.count)
                    if seg_labels else None)
        agree = ([l for l in seg_labels if l == dominant])
        out.append({
            "kind": "rest" if is_rest else "exercise",
            "label": None if is_rest else dominant,
            "start": s,
            "end": e,
            "windows": e - s,
            "t_start": round(float(times[s][0]), 2),
            "t_end": round(float(times[e - 1][1]), 2),
            "duration_s": round(float(times[e - 1][1] - times[s][0]), 2),
            # Two different things, both worth having: how often the head
            # agreed with itself, and how sure it was when it did.
            "label_agreement": round(len(agree) / max(len(seg_labels), 1), 3),
            "mean_confidence": (round(float(np.mean([c for c, l in
                                                     zip(seg_conf, seg_labels)
                                                     if l == dominant])), 4)
                                if agree else 0.0),
            # The shortlist Stage D is constrained to. Empty for rest blocks,
            # which are not an exercise and must not be given a name to pick.
            "vjepa_candidates": ([] if is_rest
                                 else chapter_candidates(gym_timeline, s, e)),
            # Read from the FINAL span index rather than carried on the block:
            # merging moves starts, and a confidence copied before the merge
            # would describe a boundary that no longer exists there.
            "boundary_z": (round(float(zscores[s - 1]), 2)
                           if s and s - 1 < len(zscores) else None),
        })
    return out


def summarize(chapters):
    """Counts for the response header."""
    ex = [c for c in chapters if c["kind"] == "exercise"]
    return {
        "total_exercises_detected": len(ex),
        "rest_periods": len(chapters) - len(ex),
        "distinct_exercises": len({c["label"] for c in ex if c["label"]}),
    }
