"""Turn a video file or a live frame buffer into a fixed-length clip.

V-JEPA 2 wants a clip of NUM_FRAMES frames, not a single image. Everything
here exists to produce a uint8 array of shape (NUM_FRAMES, H, W, 3) that the
video processor can consume.
"""

import cv2
import numpy as np

import config

# Below this length, decoding straight through is cheaper than seeking to each
# wanted frame; above it, the discarded frames outweigh the seek cost.
# ~40 s at 30 fps, which covers anything a mobile client posts.
SEQUENTIAL_MAX_FRAMES = 1200


def sample_clip_from_video(path, num_frames=config.NUM_FRAMES):
    """Evenly sample num_frames across the whole video.

    Returns (num_frames, H, W, 3) uint8 RGB, or None if unreadable.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Some containers report no frame count; fall back to reading all.
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        return np.stack([frames[i] for i in idx])

    idx = np.linspace(0, total - 1, num_frames).round().astype(int)

    # Short clip: read straight through and keep the frames we want. Each
    # cap.set() seek forces the decoder back to a keyframe and re-decodes
    # forward, so 16 seeks through a few hundred frames costs several times
    # what one linear pass does. This is the path the API takes -- phones send
    # seconds, not whole tutorials.
    if total <= SEQUENTIAL_MAX_FRAMES:
        wanted = set(int(i) for i in idx)
        grabbed, pos = {}, 0
        while pos <= int(idx[-1]):
            ok, f = cap.read()
            if not ok:
                break
            if pos in wanted:
                grabbed[pos] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pos += 1
        cap.release()
        if not grabbed:
            return None
        # idx can repeat an index when the clip is shorter than num_frames.
        last = None
        frames = []
        for i in idx:
            last = grabbed.get(int(i), last)
            if last is None:
                last = next(iter(grabbed.values()))
            frames.append(last)
        return np.stack(frames)

    # Long video: seeking beats decoding thousands of frames we discard.
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            # Reuse the previous frame rather than dropping the clip.
            if not frames:
                cap.release()
                return None
            frames.append(frames[-1])
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.stack(frames)


def sample_windows_from_video(path, num_windows=3, num_frames=config.NUM_FRAMES):
    """Sample several temporal windows from one video.

    With only ~16 clips per exercise, one embedding per video is not enough to
    train on. Different windows of the same video show different points in the
    movement, so this multiplies usable samples without new footage.

    Returns a list of (num_frames, H, W, 3) uint8 arrays, possibly shorter than
    num_windows if the video is too short or unreadable.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        cap.release()
        clip = sample_clip_from_video(path, num_frames)
        return [clip] if clip is not None else []

    # Split the video into num_windows contiguous spans and sample each one.
    windows = []
    span = total / num_windows
    for w in range(num_windows):
        start, end = int(w * span), int((w + 1) * span) - 1
        if end <= start:
            continue
        idx = np.linspace(start, end, num_frames).round().astype(int)

        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                if not frames:
                    break
                frames.append(frames[-1])
                continue
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

        if len(frames) == num_frames:
            windows.append(np.stack(frames))

    cap.release()
    return windows


def sample_timeline(path, num_frames=config.NUM_FRAMES, window_seconds=2.0,
                    max_windows=60):
    """Walk a full-length video, yielding (t_start, t_end, frames) per window.

    A GENERATOR, deliberately. Returning a list would hold every window in
    memory at once: 60 windows x 16 frames of 720p is ~2.6 GB, which this
    machine does not have. Yielding one at a time keeps the peak at a single
    window while the caller embeds it and lets it go.

    Windows are contiguous and non-overlapping, so together they cover the
    whole file with no gaps -- that is what makes the result a timeline rather
    than a sample. `max_windows` bounds GPU time on long footage by widening
    each window instead of adding more of them.

    Decoding is one sequential pass. Seeking to 16 frames x 60 windows would
    be ~960 seeks, each forcing the decoder back to a keyframe, and seek
    accuracy is unreliable on variable-GOP files -- exactly the containers a
    phone produces.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0                      # some containers do not report it
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        # No frame count. Fall back to a single whole-file clip rather than
        # guessing a duration we cannot know.
        cap.release()
        clip = sample_clip_from_video(path, num_frames)
        if clip is not None:
            yield 0.0, 0.0, clip
        return

    duration = total / fps
    n = int(duration // window_seconds) or 1
    n = min(n, max_windows)
    span = total / n

    # Frame index -> which windows want it. A frame can serve only one window
    # here (spans do not overlap), but the map keeps the scan a single pass.
    wanted = {}
    for w in range(n):
        start, end = int(w * span), max(int(w * span), int((w + 1) * span) - 1)
        for i in np.linspace(start, end, num_frames).round().astype(int):
            wanted.setdefault(int(i), []).append(w)

    buckets = {w: [] for w in range(n)}
    last_frame = None
    pos = 0
    max_wanted = max(wanted)

    while pos <= max_wanted:
        ok, f = cap.read()
        if not ok:
            break
        if pos in wanted:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            last_frame = rgb
            for w in wanted[pos]:
                buckets[w].append(rgb)
                # Emit as soon as a window is complete, then drop it, so only
                # one finished window is ever held.
                if len(buckets[w]) == num_frames:
                    frames = np.stack(buckets[w])
                    buckets[w] = []
                    yield (w * span) / fps, ((w + 1) * span) / fps, frames
        pos += 1

    cap.release()

    # A truncated or short file can leave a window part-filled. Pad it with its
    # own last frame rather than dropping the tail of the workout.
    for w in range(n):
        held = buckets.get(w)
        if held and last_frame is not None:
            held += [held[-1]] * (num_frames - len(held))
            yield (w * span) / fps, ((w + 1) * span) / fps, np.stack(held)


class FrameBuffer:
    """Rolling buffer for live camera input.

    Push frames as they arrive; once it holds num_frames, `clip()` returns a
    model-ready window. Used by the realtime backend.
    """

    def __init__(self, num_frames=config.NUM_FRAMES):
        self.num_frames = num_frames
        self.frames = []

    def push(self, frame_rgb):
        self.frames.append(frame_rgb)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    @property
    def ready(self):
        return len(self.frames) == self.num_frames

    def clip(self):
        return np.stack(self.frames) if self.ready else None
