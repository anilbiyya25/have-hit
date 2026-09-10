"""STAGE B -- latent trajectory physics over the V-JEPA embedding sequence.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The brief for this stage named LeWorldModel (LeWM, Maes/Le Lidec/Scieur/LeCun/
Balestriero 2026) as the dynamics evaluator. LeWM is real, and the *idea* this
stage borrows from it is real: a JEPA world model's next-embedding prediction
residual -- its "surprise" -- reliably marks physically implausible events.

LeWM itself cannot do this job, for three independent reasons:

  1. Its released checkpoints are trained on control environments -- Two-Room,
     Reacher, Push-T, OGBench-Cube. None of them contain a human, let alone a
     barbell or a tennis racket.
  2. Its predictor is ACTION-CONDITIONED: p(z_next | z, action). A video of a
     person has no action vector to condition on. There is nothing to pass.
  3. It trains its own encoder end-to-end from pixels and is explicitly not
     built to ingest foreign embeddings. V-JEPA's 768-d vectors live in a
     different latent space, so feeding them to LeWM's predictor would not
     raise -- it would quietly emit numbers that mean nothing.

That last one is the dangerous failure: a "physics score" that is actually
noise is worse than no score, because nothing downstream can tell.

So this module implements the CONCEPT in the latent space we actually have.
V-JEPA 2.1 is itself a joint-embedding predictive architecture, and its
embedding moves when the body's motion pattern moves, so a trajectory model
fitted over its embedding sequence measures real changes in the movement.

It is named for what it does. If a LeWM checkpoint trained on human sport video
ever exists, `LatentDynamicsEvaluator.evaluate()` is the seam to swap: same
inputs, same output contract.

WHAT IT MEASURES
----------------
Given z_0..z_n (one L2-normalised embedding per time window):

  speed[i]     = ||z[i+1] - z[i]||            how fast the motion pattern moves
  accel[i]     = ||z[i+2] - 2z[i+1] + z[i]||  momentum deviation (2nd difference)
  surprise[i]  = ||z[i+1] - predict(z[i-k..i])||

`predict` is a ridge-regularised local linear extrapolation over the previous
`k` windows. The choice of k matters: at k=2 a linear fit reduces exactly to
the second difference, so surprise would be a rename of accel rather than an
independent signal. k >= 3 fits a genuine local trend, so the residual measures
departure from the movement's own recent behaviour -- which is the quantity
LeWM's surprise captures.

All of it is unitless and relative to this clip. It is a latent-space geometry,
not metres per second, and nothing here should ever be reported as physical
velocity.
"""

import numpy as np

# Windows of history the local linear model fits before extrapolating. 4 is a
# compromise: shorter collapses toward the second difference (see above), longer
# smooths over the short transitions that are exactly what we want to catch.
HISTORY = 4

# Ridge term. The design matrix is tiny (HISTORY x 2) and can be near-singular
# on held positions where every row is nearly identical, which would send the
# extrapolation to infinity and manufacture a "flaw" out of stillness.
RIDGE = 1e-3

# How many frames Stage C is allowed to segment. The brief says 1-3; more than
# that stops being a highlight reel and starts being a slideshow.
MAX_FLAGS = 3


def _robust_z(x):
    """Median/MAD z-score. Returns zeros for a flat or empty series.

    Mean/stddev would be self-defeating here: one genuine spike inflates the
    stddev enough to push its own z-score back toward the middle, so the very
    events this stage exists to find would rank themselves down. The median and
    MAD are unmoved by a handful of outliers, which is the whole point.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad < 1e-8:
        return np.zeros_like(x)
    # 1.4826 makes MAD a consistent estimator of sigma for Gaussian data.
    return (x - med) / (1.4826 * mad)


def _linear_extrapolate(hist):
    """Least-squares linear fit over `hist` rows -> predicted next row.

    Fits each of the 768 dimensions against time independently, which is a
    first-order model of "the movement continues doing what it was doing".
    Solved in closed form rather than with lstsq: the design matrix is shared
    across all dimensions, so one 2x2 inverse handles the whole embedding.
    """
    k = len(hist)
    t = np.arange(k, dtype=np.float32)
    # Design matrix [t, 1]; normal equations with a ridge term on the diagonal.
    A = np.stack([t, np.ones(k, dtype=np.float32)], axis=1)     # (k, 2)
    G = A.T @ A + RIDGE * np.eye(2, dtype=np.float32)           # (2, 2)
    coef = np.linalg.solve(G, A.T @ hist)                       # (2, D)
    return np.array([k, 1.0], dtype=np.float32) @ coef          # (D,)


class LatentDynamicsEvaluator:
    """Trajectory consistency over a V-JEPA embedding sequence.

    Stateless and CPU-only -- it runs on embeddings that already exist, so it
    costs no VRAM and cannot interfere with the sequential GPU handoff the rest
    of the pipeline depends on.
    """

    def __init__(self, history=HISTORY, max_flags=MAX_FLAGS):
        self.history = history
        self.max_flags = max_flags

    def evaluate(self, embeddings, times, excluded_windows=None):
        """embeddings (list of (768,) arrays) + times -> dynamics report.

        Returns per-window kinetic error plus the 1-3 windows with the highest
        error, which are what Stage C segments and Stage D explains.

        `excluded_windows` are indices that must never be SELECTED as a worst
        moment -- scene cuts and their neighbours, from services.scene_cuts.
        Their kinetic error is still computed and still reported, because it
        is real: the trajectory genuinely did jump. It is simply not evidence
        about the athlete, and a hard cut outscores every human movement in
        the clip, so without this the flaw is always the edit.
        """
        excluded = set(excluded_windows or ())
        Z = np.stack(embeddings).astype(np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        n = len(Z)

        speed = (np.linalg.norm(np.diff(Z, axis=0), axis=1)
                 if n > 1 else np.zeros(0, dtype=np.float32))
        accel = (np.linalg.norm(np.diff(Z, n=2, axis=0), axis=1)
                 if n > 2 else np.zeros(0, dtype=np.float32))

        # Surprise is defined only where a full history exists. Windows before
        # that get 0.0 rather than a guess -- padding them with a fabricated
        # value would let the opening of every clip look like a flaw.
        surprise = np.zeros(max(n - 1, 0), dtype=np.float32)
        for i in range(self.history, n):
            pred = _linear_extrapolate(Z[i - self.history:i])
            pred /= np.linalg.norm(pred) + 1e-8
            surprise[i - 1] = float(np.linalg.norm(Z[i] - pred))

        # Kinetic error combines "the movement broke from its own trend"
        # (surprise) with "momentum changed abruptly" (accel). Surprise leads
        # because accel alone fires on every normal direction change at the top
        # and bottom of a rep, which is correct movement, not a fault.
        sz = _robust_z(surprise)
        az = _robust_z(accel)
        m = min(len(sz), len(az)) if len(az) else len(sz)
        kinetic = (0.7 * sz[:m] + 0.3 * az[:m]) if m else np.zeros(0, np.float32)

        flags = self._flag(kinetic, times, excluded)
        # The single worst moment of kinetic breakdown -- what Stage C
        # segments and what visual_proof reports. None on clean footage, which
        # is a real answer: not every clip contains a fault.
        worst = next((f for f in flags if f["rank"] == 1), None)

        return {
            # Named so no caller mistakes this for LeWM's own output.
            "evaluator": "vjepa_latent_dynamics_v1",
            "windows": n,
            "history_windows": self.history,
            "units": "unitless latent-space geometry; NOT m/s or degrees",
            "latent_speed": [round(float(v), 4) for v in speed],
            "momentum_deviation": [round(float(v), 4) for v in accel],
            "surprise": [round(float(v), 4) for v in surprise],
            "kinetic_error_score": [round(float(v), 4) for v in kinetic],
            "mean_kinetic_error": (round(float(kinetic.mean()), 4)
                                   if len(kinetic) else 0.0),
            "peak_kinetic_error": (round(float(kinetic.max()), 4)
                                   if len(kinetic) else 0.0),
            "flagged_frames": flags,
            "worst_moment": worst,
        }

    def _flag(self, kinetic, times, excluded=()):
        """Top-scoring windows, spaced apart, as timestamped flags."""
        if not len(kinetic):
            return []
        # Spacing stops all three flags landing on one transition and its two
        # neighbours, which would segment the same instant three times.
        min_gap = max(1, len(kinetic) // (self.max_flags * 2))
        picked = []
        for idx in np.argsort(kinetic)[::-1]:
            i = int(idx)
            # A flat clip has nothing to flag. Reporting the argmax of noise
            # would invent a fault in clean footage.
            if kinetic[i] <= 0:
                break
            # kinetic[i] is the transition from window i into window i+1, so a
            # cut in EITHER makes this score an artefact of the edit.
            if i in excluded or (i + 1) in excluded:
                continue
            if all(abs(i - p) >= min_gap for p in picked):
                picked.append(i)
            if len(picked) == self.max_flags:
                break

        # `picked` is in descending severity, so rank comes from THAT order.
        # The list is then emitted in time order for display. Deriving rank
        # from the time-sorted position instead -- which this did until the
        # pipeline started selecting "the single worst" by rank -- makes rank 1
        # the earliest flag rather than the worst one.
        severity_rank = {idx: r for r, idx in enumerate(picked, start=1)}

        out = []
        for i in sorted(picked):
            # kinetic[i] describes the transition INTO window i+1, so the
            # moment of interest is that window's start, not window i's.
            w = min(i + 1, len(times) - 1)
            t0, t1 = times[w]
            out.append({
                "rank": severity_rank[i],
                "window": w,
                "timestamp": round(float(t0), 2),
                "t_start": round(float(t0), 2),
                "t_end": round(float(t1), 2),
                "kinetic_error": round(float(kinetic[i]), 4),
                # Ranked severity, not a physical claim. Stage D is told to
                # treat it as a pointer to look, never as a diagnosis.
                "severity": ("critical" if kinetic[i] >= 3.0
                             else "warning" if kinetic[i] >= 1.5
                             else "minor"),
            })
        return out
