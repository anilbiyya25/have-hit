"""STAGE C -- SAM 2.1 keyframe segmentation on the frames Stage B flagged.

WHY THE PROMPTS ARE A GRID AND NOT "the knee"
---------------------------------------------
SAM segments what it is pointed at. It has no vocabulary, so it cannot be asked
for "the knee" or "the barbell" -- a text prompt is not something this model
takes. Nothing upstream of here knows where the knee is either: Stage A gives a
whole-clip exercise label and Stage B gives a timestamp, neither of which is a
location in the frame.

So this stage does the part it can do honestly: prompt a point grid over the
region the athlete occupies, collect the distinct objects SAM finds there, and
hand them on as CANDIDATE regions with their boxes. Stage D -- which does have
the vocabulary, because it can see -- picks the candidate that matches the fault
it observes and names it.

That ordering keeps each model inside its competence. The alternative, guessing
a point and asserting the resulting mask is "the knee", would produce a
confident box around whatever happened to be under the guess.

VRAM
----
The card is a 4 GB RTX 3050. This model is loaded on demand and released the
moment the stage is done, so it is never resident at the same time as V-JEPA.
`load()`/`unload()` are explicit rather than automatic so the orchestrator keeps
control of the ordering.
"""

import cv2
import numpy as np

from services.keyframes import mask_to_box, normalize_box

# Tiny is the deliberate pick: 39M params, ~289 MB VRAM measured on this card.
# The small/base variants are better at fine boundaries and do not fit next to
# everything else this pipeline needs.
MODEL_ID = "facebook/sam2.1-hiera-tiny"

# Point grid over the middle of the frame. Athletes are near-centred in phone
# footage and the edges are floor, ceiling and wall, so sampling them wastes
# prompts on masks that are discarded a few lines later.
#
# 4x4 rather than 3x3: the pipeline segments ONE keyframe now, not three, and
# the cost here is dominated by set_image (the image encoder, run once per
# frame) rather than by the prompts. So a denser grid finds smaller parts --
# a knee, a wrist -- for roughly the same time a coarse grid took on 3 frames.
GRID = 4
GRID_INSET = 0.15

# A mask covering almost the whole frame is the background; a speck is noise.
MIN_AREA_FRAC = 0.004
MAX_AREA_FRAC = 0.60

# Two masks overlapping this much are the same object found from two points.
DEDUPE_IOU = 0.75

# Candidates kept per keyframe. Enough to contain the athlete, a limb and the
# equipment; few enough that Stage D's prompt stays readable. Raised alongside
# the denser grid so the extra small-part masks are not immediately discarded.
MAX_CANDIDATES = 7


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if not inter:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


class Segmenter:
    """Lazy SAM 2.1 wrapper. Import, weights and VRAM all deferred to load()."""

    def __init__(self, model_id=MODEL_ID, device="cuda"):
        self.model_id = model_id
        self.device = device
        self.predictor = None
        self.error = None

    def load(self):
        """Bring the model into VRAM. Returns True on success.

        Never raises. Segmentation is an enhancement -- if the weights are
        missing or the card is full, the report should still be produced with
        Stage D's own boxes rather than the whole request failing.
        """
        if self.predictor is not None:
            return True
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor.from_pretrained(
                self.model_id, device=self.device)
            self.error = None
            return True
        except Exception as exc:                              # noqa: BLE001
            self.error = f"{exc.__class__.__name__}: {exc}"
            self.predictor = None
            return False

    def unload(self):
        """Release the model and hand the VRAM back."""
        if self.predictor is None:
            return
        self.predictor = None
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:                                     # noqa: BLE001
            pass

    def _grid_points(self, h, w):
        xs = np.linspace(w * GRID_INSET, w * (1 - GRID_INSET), GRID)
        ys = np.linspace(h * GRID_INSET, h * (1 - GRID_INSET), GRID)
        return [(float(x), float(y)) for y in ys for x in xs]

    def segment_frame(self, frame_bgr):
        """One BGR frame -> candidate regions, best first.

        Each entry carries a normalised box in Gemini's [ymin,xmin,ymax,xmax]
        order plus a coarse polygon, so the frontend can draw either.
        """
        if self.predictor is None:
            return []
        import torch

        h, w = frame_bgr.shape[:2]
        # cvtColor, not frame[:, :, ::-1]. The slice is a view with a NEGATIVE
        # stride, and torch.from_numpy rejects those outright -- which showed
        # up here as every keyframe silently returning zero candidates.
        # cvtColor returns a fresh contiguous buffer.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        found = []
        try:
            # no_grad, NOT inference_mode. set_image caches the image embedding
            # on the predictor and predict() reuses it across calls; tensors
            # created under inference_mode carry a flag that makes that reuse
            # illegal, which surfaces as "CUDA error: invalid resource handle"
            # rather than anything naming the real cause. Measured on this box:
            # inference_mode and autocast(bf16/fp16) both fail, no_grad works.
            with torch.no_grad():
                self.predictor.set_image(rgb)    # runs the image encoder once
                for (px, py) in self._grid_points(h, w):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([[px, py]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=True,
                    )
                    for m, sc in zip(masks, scores):
                        m = m.astype(bool)
                        frac = float(m.sum()) / float(h * w)
                        if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
                            continue
                        found.append((float(sc), frac, m))
        except Exception as exc:                              # noqa: BLE001
            self.error = f"{exc.__class__.__name__}: {exc}"
            return []

        # Best-scoring first, then drop anything that duplicates a mask already
        # kept. Greedy is fine at this scale and keeps the highest-confidence
        # version of each object.
        found.sort(key=lambda t: t[0], reverse=True)
        kept = []
        for sc, frac, m in found:
            if any(_iou(m, k[2]) > DEDUPE_IOU for k in kept):
                continue
            kept.append((sc, frac, m))
            if len(kept) == MAX_CANDIDATES:
                break

        out = []
        for i, (sc, frac, m) in enumerate(kept):
            box = mask_to_box(m)
            if box is None:
                continue
            ax_deg, ax_elong = _axis(m)
            out.append({
                "candidate_id": i,
                "sam_score": round(sc, 4),
                "area_fraction": round(frac, 4),
                "normalized_bbox": normalize_box(box, (h, w)),
                "polygon": _polygon(m, h, w),
                "centroid": _centroid(m, h, w),
                "axis_deg": ax_deg,
                "elongation": ax_elong,
            })
        return out

    def segment_keyframes(self, frames):
        """[(t, frame)] -> [{timestamp, candidates}], loading once for all."""
        if not self.load():
            return [], self.error
        try:
            return [{"timestamp": round(t, 2),
                     "frame_shape": [int(f.shape[0]), int(f.shape[1])],
                     "candidates": self.segment_frame(f)}
                    for t, f in frames], self.error
        finally:
            self.unload()


# approxPolyDP tolerance, as a fraction of the contour perimeter, and the
# vertex ceiling.
#
# 0.01 was too aggressive once the polygon started being DRAWN rather than
# merely carried: it reduced a whole limb to four points, so the "contour" was
# a quadrilateral -- no better than the box it was supposed to replace, and
# measured at 4 points on a real keyframe. 0.004 keeps the shape of a thigh or
# a barbell while still discarding the pixel-level jitter of a mask edge.
#
# The cap is what actually bounds render cost, and it is generous because the
# canvas redraws this every frame: 40 lineTo calls is nothing, 500 is not.
POLY_EPSILON = 0.004
POLY_MAX_POINTS = 40


def _polygon(mask, h, w, max_points=POLY_MAX_POINTS):
    """Normalised [x, y] outline of a mask, for canvas rendering.

    NOTE THE ORDER. These are [x, y] pairs, while normalized_bbox next to them
    is [ymin, xmin, ymax, xmax] -- Gemini's convention, which the box has to
    keep because Gemini both reads and writes it. The polygon is ours alone and
    uses the order a canvas draws in. Anything consuming both has to convert.
    """
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, POLY_EPSILON * peri, True).reshape(-1, 2)
    if len(approx) > max_points:
        step = len(approx) / float(max_points)
        approx = approx[[int(i * step) for i in range(max_points)]]
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in approx]


def _axis(mask):
    """Orientation of the mask's long axis, in degrees from TRUE VERTICAL.

    Second-order central image moments, which is the closed-form answer for
    the axis of an area:  theta = 0.5 * atan2(2*mu11, mu20 - mu02).

    WHAT THIS IS AND IS NOT
        It is the tilt of the segmented REGION -- exact, measured, reproducible.
        It is NOT a spine, a femur or a bar path, and nothing in this system
        knows where those are. There is no pose model here: SAM produced this
        mask from a bare point grid and has no idea what it enclosed.

        The overlay that draws this says "region axis", never "your spine",
        for exactly that reason. A cyan skeleton labelled with joint names,
        drawn over a body nothing located, would look far more authoritative
        than it could possibly be.

    Sign convention: positive is clockwise from vertical as the viewer sees
    it, so it matches the direction the athlete appears to lean.
    """
    mo = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if not mo.get("m00"):
        return None, None
    mu20, mu02, mu11 = mo["mu20"], mo["mu02"], mo["mu11"]
    denom = mu20 - mu02
    if abs(mu11) < 1e-9 and abs(denom) < 1e-9:
        return None, None                     # circular: no meaningful axis
    theta = 0.5 * np.arctan2(2.0 * mu11, denom)      # from the +x axis
    # Rotate into "degrees from vertical" and fold onto (-90, 90]: an axis is
    # a line, not an arrow, so 170 deg and -10 deg are the same tilt.
    deg = float(np.degrees(theta)) + 90.0
    while deg > 90.0:
        deg -= 180.0
    while deg <= -90.0:
        deg += 180.0

    # Elongation. A near-round mask has an axis in the arithmetic but not in
    # any useful sense, and drawing a confident tilt line through a blob is
    # the same category of fabrication this function's docstring warns about.
    common = np.sqrt(max((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2, 0.0))
    lam1 = (mu20 + mu02 + common) / 2.0
    lam2 = (mu20 + mu02 - common) / 2.0
    elong = float(np.sqrt(lam1 / lam2)) if lam2 > 1e-9 else None
    return round(deg, 2), (round(elong, 3) if elong else None)


def _centroid(mask, h, w):
    """Normalised [cx, cy] centre of MASS of a mask, for pinning a label.

    Image moments, not the centre of the bounding box. For anything bent --
    an arm at the elbow, a leg mid-squat, the classic L shape -- the box centre
    sits in the empty corner OUTSIDE the mask, which would pin the anatomical
    label to a patch of gym floor next to the joint it names.
    """
    m = mask.astype(np.uint8)
    mo = cv2.moments(m, binaryImage=True)
    if not mo.get("m00"):
        return None
    return [round(float(mo["m10"] / mo["m00"]) / w, 4),
            round(float(mo["m01"] / mo["m00"]) / h, 4)]
