"""Pull single frames out of a video at given timestamps.

Stage C segments these frames and Stage D looks at them, so both need the same
decode. Doing it once here keeps the two stages from disagreeing about which
pixels a flagged moment refers to.
"""

import cv2
import numpy as np

# Longest side of the JPEG sent to Gemini. Big enough to judge a joint angle,
# small enough that three keyframes plus a video do not blow the request out.
JPEG_MAX_SIDE = 768
JPEG_QUALITY = 82


def grab(path, timestamps):
    """[(t, BGR frame)] for each timestamp that could be decoded.

    Seeks per timestamp. That is the opposite of what sample_timeline does, and
    deliberately: there are at most three of these, so 3 seeks beat re-decoding
    the whole file, whereas 960 seeks would not.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out = []
    try:
        for t in timestamps:
            idx = int(round(float(t) * fps))
            if total:
                # Clamp inside the file. A flag on the final window can land one
                # frame past the end, where read() fails and the keyframe would
                # silently go missing.
                idx = max(0, min(idx, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Seek failed (variable-GOP files do this). Fall back to the
                # first decodable frame from the start rather than dropping it.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
            out.append((float(t), frame))
    finally:
        cap.release()
    return out


def to_jpeg(frame, max_side=JPEG_MAX_SIDE, quality=JPEG_QUALITY):
    """BGR frame -> JPEG bytes, downscaled so the long side fits `max_side`."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def normalize_box(box, shape):
    """Pixel [x0,y0,x1,y1] -> normalised [ymin,xmin,ymax,xmax] in 0..1.

    The odd axis order is not a mistake: it is the convention Gemini uses for
    spatial output, and the response schema follows it so the frontend has one
    box format to render rather than two.
    """
    h, w = shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    return [round(max(0.0, min(1.0, y0 / h)), 4),
            round(max(0.0, min(1.0, x0 / w)), 4),
            round(max(0.0, min(1.0, y1 / h)), 4),
            round(max(0.0, min(1.0, x1 / w)), 4)]


def mask_to_box(mask):
    """Binary mask -> tight pixel bounding box, or None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
