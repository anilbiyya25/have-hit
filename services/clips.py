"""Cut the isolated flaw micro-clip out of a source recording.

WHY A REAL FILE AND NOT A TIME RANGE ON THE FULL VIDEO
    The player could seek the full video and loop a range -- that is what the
    Focus Loop button did, and it worked. It stops working the moment the
    source is gone: an upload is deleted after analysis and recording chunks
    are deleted when the session closes, so the history drawer had a flaw
    timestamp and nothing to play it from. It is also the wrong thing to send
    to a phone on gym wi-fi: looping three seconds out of a 90 MB file means
    downloading enough of that file to seek into it.

    A 3-second clip is tens of kilobytes, loads instantly, loops natively and
    survives the source being deleted.

WHY RE-ENCODE INSTEAD OF STREAM-COPY
    A stream copy cannot start anywhere except a keyframe. Phone video
    routinely runs 2-10 second GOPs, so "copy from t-1.5s" really means "copy
    from the keyframe before that", which can be seconds early -- and the clip
    exists precisely to be centred on one moment. Cutting mid-GOP instead
    gives a clip whose first frames reference data that was not copied, which
    decodes to smears. Re-encoding three seconds is cheap and exact.

WHY PyAV AND NOT cv2.VideoWriter
    VideoWriter's H.264 support depends on an opencv_videoio_ffmpeg DLL being
    present and picking a codec by FourCC guess; when it fails it does so
    silently, returning a writer that accepts every frame and produces an
    unplayable file. PyAV links its own libav and raises. It also lets frames
    from two different source files be fed to one encoder, which is what a
    chunked recording needs when the flaw lands near a chunk boundary.
"""

import time
from fractions import Fraction
from pathlib import Path

import av

# Half-width of the window, so the clip is 2 x this. 1.5 s either side is
# enough to see the approach, the fault and the recovery -- the fault itself
# is a moment, and a moment with no run-up is unreadable.
PAD_S = 1.5

# Encoded height cap. The hero viewport is a phone-width panel, not a cinema,
# and the encode is on the critical path of a report the athlete is waiting
# on. 720 keeps a knee joint legible while keeping a 1080p source from costing
# four times the encode for detail the viewport cannot show.
MAX_HEIGHT = 720

# CRF 26 / veryfast, not the usual 23 / medium. This clip is watched once, in
# a loop, at 0.75x, with a contour drawn over it -- fidelity beyond "the joint
# is clearly visible" buys nothing and costs latency.
CRF = "26"
PRESET = "veryfast"

# Frame rate ceiling. A 120 fps slow-motion phone clip re-encoded at source
# rate is 360 frames for three seconds of footage, all of it played back at
# 0.75x anyway.
MAX_FPS = 30

CLIP_DIR = Path(__file__).resolve().parent.parent / "static" / "clips"

# Clips are derived data and the only thing stopping them accumulating without
# limit. Keeping the newest N is enough: a clip is regenerated from the source
# on demand for as long as the source exists, and after that the session is
# history, where the stored keyframes carry the visual proof.
KEEP_CLIPS = 80


def clip_path(session_id, chapter_index):
    """Where one chapter's flaw clip lives. Deterministic, so a re-analysis of
    the same session overwrites rather than accumulating."""
    safe = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in "-_")
    return CLIP_DIR / f"{safe}_flaw_{int(chapter_index)}.mp4"


def window_for(proof_ts, duration):
    """The [start, end] to cut, clamped into the file.

    Returns (start, end, offset) where offset is where the flaw sits INSIDE
    the clip. That is normally PAD_S, but a fault in the first or last second
    and a half of a recording cannot be centred, and the overlay has to know
    where the moment actually landed -- assuming 1.5 s there would draw the
    contour over the wrong frame.
    """
    duration = float(duration or 0.0)
    span = 2 * PAD_S
    # Clamp the moment into the file FIRST. A proof timestamp past the end --
    # which a flag on the final analysis window can produce -- otherwise yields
    # an offset pointing outside the clip it is an offset into.
    t = max(0.0, float(proof_ts))
    if duration > 0:
        t = min(t, duration)

    start = t - PAD_S
    end = t + PAD_S
    # Slide, do not truncate. A fault in the first second and a half cannot be
    # centred, but it can still get its full three seconds by taking them all
    # from after the moment; the same in reverse at the end of the file. The
    # earlier version only handled the end, so a fault at 0.4 s produced a
    # 1.9 s clip -- short, and silently so.
    if start < 0:
        start, end = 0.0, span
    if duration > 0 and end > duration:
        end = duration
        start = max(0.0, end - span)
    if duration > 0:
        end = min(end, duration)
    return round(start, 3), round(end, 3), round(min(max(t - start, 0.0),
                                                     end - start), 3)


def _even(n):
    """H.264 4:2:0 needs even dimensions; an odd one is an encoder error."""
    return max(2, int(n) // 2 * 2)


def source_fps(path, default=30.0):
    """The source's frame rate, capped at MAX_FPS.

    Declaring a fixed output rate was a real bug and an invisible one: the
    reference clip is 24 fps, so its 72 decoded frames were muxed as 30 fps
    and came out a 2.4 s clip playing 25% fast. The clip is meant to show
    mechanics at 0.75x, and it was silently showing them at 0.94x.
    """
    try:
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            rate = st.average_rate or st.guessed_rate
            if rate:
                return min(float(rate), float(MAX_FPS))
    except Exception:                                         # noqa: BLE001
        pass
    return min(default, float(MAX_FPS))


def _decode_span(path, start, end):
    """Frames from one file between two timestamps, as av.VideoFrames.

    Seeking lands on the keyframe at or before `start`, so the decoded frames
    before it are dropped here rather than encoded -- that is the difference
    between an exact cut and a stream copy's approximate one.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base or Fraction(1, 1000)
        if start > 0:
            try:
                container.seek(int(start / tb), stream=stream)
            except av.AVError:
                pass                      # unseekable: decode from the top
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            if t < start:
                continue
            if t > end:
                break
            yield t, frame


def _prune(keep=KEEP_CLIPS):
    """Drop the oldest clips beyond `keep`. Best-effort and never fatal: a
    clip that cannot be deleted is wasted disk, not a failed analysis."""
    try:
        files = sorted(CLIP_DIR.glob("*_flaw_*.mp4"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def extract(spans, dest, max_height=MAX_HEIGHT):
    """Write one clip from one or more source spans.

    `spans` is [(path, start, end)]. More than one is the chunked-recording
    case: chunks are 10 s and a flaw 1 s into one of them needs the tail of
    its predecessor, so frames from two files are fed to a single encoder and
    come out as one continuous clip.

    Returns {"path", "frames", "duration_s", "encode_ms"} or None if nothing
    decoded -- an empty clip is worse than no clip, because the player would
    show a black box where the proof is supposed to be.
    """
    t0 = time.perf_counter()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    out = ostream = None
    written = 0
    try:
        for path, start, end in spans:
            if not path or not Path(path).exists() or end <= start:
                continue
            for src_t, frame in _decode_span(path, start, end):
                if ostream is None:
                    fps = source_fps(path)
                    tb = Fraction(1, 1000)      # ms: exact enough for any rate
                    scale = min(1.0, max_height / float(frame.height or 1))
                    w = _even(round((frame.width or 2) * scale))
                    h = _even(round((frame.height or 2) * scale))
                    # +faststart puts the moov atom first so the browser can
                    # start playing on the first bytes instead of waiting for
                    # the whole file -- the clip is meant to be up the instant
                    # the report is.
                    out = av.open(str(dest), "w",
                                  options={"movflags": "+faststart"})
                    ostream = out.add_stream("libx264",
                                             rate=Fraction(fps).limit_denominator(1001))
                    ostream.width, ostream.height = w, h
                    ostream.pix_fmt = "yuv420p"
                    ostream.time_base = tb
                    ostream.options = {"crf": CRF, "preset": PRESET}
                    step = 1.0 / fps

                # Decimate a high-rate source down to the declared rate rather
                # than encoding every frame of a 120 fps slow-motion capture.
                want = written * step
                if (src_t - spans[0][1]) + 1e-6 < want:
                    continue

                if (frame.width, frame.height) != (ostream.width, ostream.height):
                    frame = frame.reformat(width=ostream.width,
                                           height=ostream.height)
                # PTS is reassigned from a running frame count in the output's
                # own time base. The source stamps are absolute positions in
                # the original file; muxing those into a clip that starts at
                # zero gave a container reporting a 0.03 s duration and a
                # frame rate of 1/0, which the browser cannot loop.
                frame.pts = int(round(written * step / tb))
                frame.time_base = tb
                for packet in ostream.encode(frame):
                    out.mux(packet)
                written += 1

        if ostream is None:
            return None
        for packet in ostream.encode():       # flush the encoder's queue
            out.mux(packet)
    except Exception:
        # A half-written MP4 is worse than none: it has a valid header and
        # plays as a black frame, which reads as "the analysis found nothing".
        try:
            if out is not None:
                out.close()
                out = None
        finally:
            dest.unlink(missing_ok=True)
        raise
    finally:
        if out is not None:
            out.close()

    _prune()
    return {
        "path": str(dest),
        "frames": written,
        "fps": round(1.0 / step, 3),
        "duration_s": round(written * step, 2),
        "encode_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def extract_flaw(session_id, chapter_index, proof_ts, duration, source,
                 locate=None):
    """The whole job for one chapter: work out the window, cut it, describe it.

    `source` is either a path to a single file (the upload path) or the chunk
    index (the recording path), in which case `locate` maps an absolute
    session time to (chunk, local_time) -- chunk_pipeline owns that mapping
    and this module should not learn it twice.

    Returns the dict that goes into visual_proof, or None. Never raises: the
    report is worth delivering without its clip, and the player falls back to
    the full video when flaw_clip_url is absent.
    """
    start, end, offset = window_for(proof_ts, duration)
    if end <= start:
        return None

    if locate is None:
        spans = [(source, start, end)]
    else:
        # Split the window at chunk boundaries. Walked in small steps rather
        # than solved analytically because chunk durations are measured, not
        # nominal -- a dropped frame makes the last chunk short.
        spans, cursor = [], start
        while cursor < end - 1e-3:
            chunk, local = locate(source, cursor)
            if not chunk or not chunk.get("path"):
                break
            span_end = min(end, chunk["t_end"])
            spans.append((chunk["path"], local, local + (span_end - cursor)))
            cursor = span_end
        if not spans:
            return None

    try:
        info = extract(spans, clip_path(session_id, chapter_index))
    except Exception as exc:                                  # noqa: BLE001
        print(f"  flaw clip {session_id}#{chapter_index} failed: {exc}",
              flush=True)
        return None
    if not info:
        return None

    return {
        "flaw_clip_url": (f"/api/v1/session/{session_id}"
                          f"/flaw-clip/{int(chapter_index)}"),
        "flaw_clip_start_s": start,
        "flaw_clip_end_s": end,
        # Where the flagged moment sits INSIDE the clip. The overlay is drawn
        # against this, not against the session timeline.
        "flaw_relative_ts": offset,
        "flaw_clip_duration_s": info["duration_s"],
        "flaw_clip_ms": info["encode_ms"],
    }


def purge_session(session_id):
    """Delete a session's clips. Called when its history row is deleted."""
    safe = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in "-_")
    n = 0
    try:
        for p in CLIP_DIR.glob(f"{safe}_flaw_*.mp4"):
            p.unlink(missing_ok=True)
            n += 1
    except OSError:
        pass
    return n


def stats():
    """Count and total size on disk, for the startup diagnostic."""
    try:
        files = list(CLIP_DIR.glob("*_flaw_*.mp4"))
        return {"clips": len(files),
                "bytes": sum(f.stat().st_size for f in files)}
    except OSError:
        return {"clips": 0, "bytes": 0}
