"""Rolling chunk ingestion: do Stage A while the user is still recording.

THE LATENCY PROBLEM THIS SOLVES, AND THE ONE IT DOES NOT
    A finished 60 s set costs roughly:

        Stage A  V-JEPA over every window     ~6 s
        Stage B  latent dynamics              ~0.1 s
        Stage C  SAM on the worst keyframes   ~3.6 s
        Stage D  Gemini                       ~49-74 s

    Stage A is the only part that can be moved earlier, because it is the only
    part that depends solely on footage already captured. Feeding it 10 s
    chunks as they are recorded means that by the time the user presses stop,
    every embedding already exists and Stage A costs ~0 s.

    That is worth doing and it is not enough on its own. Gemini is 80-95% of
    the wall clock and cannot start before the last chunk exists. So finalize
    returns the LOCAL report immediately -- chapters, reps, worst moments, SAM
    proof, deterministic cues, all of it real -- and the Gemini narration
    arrives afterwards via `GET /api/v1/session/{id}`. The user sees a full
    report in about a second; the prose upgrades in place when it lands.

WHY CHUNKS ARE SEPARATE FILES, NOT A SPLIT STREAM
    MediaRecorder.start(timeslice) emits fragments where only the FIRST holds
    the container header, so an individual fragment is not decodable and the
    server cannot process one as it arrives. The client therefore runs a fresh
    start/stop per chunk, making each an independent, complete video.

    That also removes any need to concatenate. Stage C pulls its keyframe from
    whichever chunk contains the moment, and Stage D uploads only those
    chunks -- less to send than the whole session, and no ffmpeg dependency.
"""

import threading
import time
import uuid
from pathlib import Path

import numpy as np

from services import (analysis_pipeline, chapters, clips, keyframes as kf,
                      scene_cuts)
from services.segmentation import Segmenter
from services.latent_dynamics import LatentDynamicsEvaluator
from services.rep_engine import detect_reps, _mmss

# Sessions older than this with no activity are swept. Long enough that a
# genuine pause between sets never loses a session, short enough that abandoned
# recordings do not accumulate video on disk.
SESSION_TTL_S = 3600

# Hard cap on retained chunks (~17 min at 10 s each). Beyond this the earliest
# chunk files are dropped: their embeddings are already extracted and kept, so
# only the ability to pull a keyframe from that far back is lost.
MAX_CHUNKS = 100


class Session:
    """One recording in progress. Embeddings accumulate; video stays on disk."""

    def __init__(self, session_id, tmpdir):
        self.id = session_id
        self.dir = Path(tmpdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.created = time.time()
        self.touched = time.time()

        self.chunks = []          # [{index, path, t_start, t_end, duration}]
        self.embeddings = []
        self.times = []           # global (t_start, t_end) per window
        self.gym_timeline = []
        self.next_offset = 0.0    # where the next chunk begins, in session time
        self.errors = []
        # Session-global window indices that contain a scene cut, so they can
        # never be chosen as a worst moment.
        self.excluded = set()

        # Filled by finalize(); read by the status endpoint.
        self.result = None
        self.gemini_state = "idle"   # idle | running | done | failed

    def touch(self):
        self.touched = time.time()

    @property
    def duration(self):
        return self.next_offset

    def chunk_for(self, t):
        """Which chunk contains session-time `t`, and the local offset in it."""
        for c in self.chunks:
            if c["t_start"] <= t < c["t_end"]:
                return c, t - c["t_start"]
        if self.chunks and t >= self.chunks[-1]["t_end"]:
            c = self.chunks[-1]
            return c, max(0.0, t - c["t_start"])
        return (self.chunks[0], t) if self.chunks else (None, t)

    def cleanup(self):
        for c in self.chunks:
            # `path` is None for chunks already released -- by the MAX_CHUNKS
            # eviction, or by release_video() once Stage D is done. Path(None)
            # raises TypeError, which this except would NOT have caught, so a
            # long session's cleanup used to die before reaching its own rmdir.
            if not c.get("path"):
                continue
            try:
                Path(c["path"]).unlink()
            except OSError:
                pass
        try:
            self.dir.rmdir()
        except OSError:
            pass

    def release_video(self):
        """Delete the footage but keep the session pollable. -> bytes freed.

        Called once Stage D has finished with it. Nothing downstream needs the
        video again: the proof keyframes are already JPEGs in workouts.db and
        `result` holds the whole report. A vault upload can be 256 MB, and
        waiting out the hour-long TTL to reclaim that is a lot of disk to hold
        for nothing.
        """
        freed = 0
        for c in self.chunks:
            p = c.get("path")
            if not p:
                continue
            try:
                f = Path(p)
                freed += f.stat().st_size
                f.unlink()
            except OSError:
                pass
            c["path"] = None
        return freed


class SessionStore:
    """Thread-safe registry of in-flight sessions."""

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._sessions = {}

    def create(self, prefix=""):
        """A new session. `prefix` tags where it came from.

        Vault uploads use "upload_" so a stored row is still identifiable as an
        upload after the session is gone -- db.clear_test_data keys its
        exclusion on exactly that prefix.
        """
        sid = f"{prefix}{uuid.uuid4().hex[:16]}"
        with self._lock:
            self._sweep_locked()
            s = Session(sid, self.root / sid)
            self._sessions[sid] = s
        return s

    def get(self, sid):
        with self._lock:
            s = self._sessions.get(sid)
        if s:
            s.touch()
        return s

    def drop(self, sid):
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s:
            s.cleanup()

    def _sweep_locked(self):
        dead = [k for k, v in self._sessions.items()
                if time.time() - v.touched > SESSION_TTL_S]
        for k in dead:
            self._sessions.pop(k).cleanup()

    def stats(self):
        with self._lock:
            return {"active_sessions": len(self._sessions),
                    "ids": list(self._sessions)}


def ingest_chunk(session, path, encoder, head, window_seconds,
                 max_windows=32):
    """Stage A over ONE chunk. Blocking; the caller runs it off the event loop.

    Window times are rebased onto session time as they are produced, so the
    accumulated arrays look exactly like a single-file Stage A run and the rest
    of the pipeline needs no special case for chunked input.

    `infer_window` and `sample_timeline` used to be passed in. They are gone:
    the batched embed helper does both jobs, this module already imports
    analysis_pipeline directly, so the injection was ceremony around an import
    that was never circular.
    """
    # A tracker per chunk, not per session: chunk boundaries are recording
    # artefacts, not scene changes, so carrying the previous chunk's last
    # frame across the join would report a cut at every 10 s mark.
    tracker = scene_cuts.CutTracker()

    # Batched, via the same helper the whole-file path uses, so the recorder
    # and the vault upload cannot drift on how a window becomes an embedding.
    embeddings, times, timeline = analysis_pipeline.embed_timeline(
        str(path), encoder, head, window_seconds, max_windows, cuts=tracker)
    local_end = max((t1 for _, t1 in times), default=0.0)

    if not embeddings:
        raise ValueError("chunk could not be decoded as video")

    with session.lock:
        base = session.next_offset
        idx = len(session.chunks)
        offset = len(session.embeddings)
        for local_i in tracker.excluded_windows():
            session.excluded.add(offset + local_i)
        for (t0, t1), w in zip(times, timeline):
            g0, g1 = base + t0, base + t1
            session.times.append((g0, g1))
            session.gym_timeline.append({"t_start": round(g0, 2),
                                         "t_end": round(g1, 2), **w})
        session.embeddings.extend(embeddings)
        session.chunks.append({
            "index": idx, "path": str(path),
            "t_start": round(base, 2), "t_end": round(base + local_end, 2),
            "duration": round(local_end, 2), "windows": len(embeddings),
        })
        session.next_offset = base + local_end

        # Drop the oldest chunk FILES past the cap. Embeddings survive, so the
        # analysis is unaffected -- only a keyframe from that far back becomes
        # unavailable, and the report has already moved past it.
        if len(session.chunks) > MAX_CHUNKS:
            for old in session.chunks[:-MAX_CHUNKS]:
                if old.get("path"):
                    try:
                        Path(old["path"]).unlink()
                    except OSError:
                        pass
                    old["path"] = None
        session.touch()
    return len(embeddings)


def finalize_local(session, segmenter=None, enable_sam=True, max_proof=4):
    """Everything except Gemini: chapters, reps, worst moments, SAM, cues.

    This is what the caller returns immediately. It contains every measured
    quantity in the full report -- the only thing missing is the prose.
    """
    with session.lock:
        embeddings = list(session.embeddings)
        times = list(session.times)
        gym_timeline = list(session.gym_timeline)
        chunks = list(session.chunks)
        excluded = set(session.excluded)

    if not embeddings:
        raise ValueError("no chunks were ingested for this session")

    duration = round(float(times[-1][1]), 2)
    dynamics = LatentDynamicsEvaluator().evaluate(embeddings, times, excluded)
    reps = detect_reps(embeddings, times, dynamics["kinetic_error_score"])
    spans = chapters.detect_chapters(embeddings, times, gym_timeline,
                                     dynamics["latent_speed"])

    ev = LatentDynamicsEvaluator()
    analysed = []
    for i, ch in enumerate(spans):
        if ch["kind"] != "exercise" or ch["end"] - ch["start"] < 2:
            analysed.append({**ch, "chapter_index": i, "dynamics": None,
                             "reps": None})
            continue
        s, e = ch["start"], ch["end"]
        # Chapter-local indices; global ones would mask the wrong windows.
        local_excl = {g - s for g in excluded if s <= g < e}
        d = ev.evaluate(embeddings[s:e], times[s:e], local_excl)
        analysed.append({**ch, "chapter_index": i, "dynamics": d,
                         "reps": detect_reps(embeddings[s:e], times[s:e],
                                             d["kinetic_error_score"])})

    # --- Stage C, on the chunk that actually holds each moment -------------
    wanted = [c for c in analysed
              if c.get("dynamics") and c["dynamics"].get("worst_moment")]
    wanted.sort(key=lambda c: c["dynamics"]["peak_kinetic_error"], reverse=True)
    wanted = wanted[:max_proof]

    segments, proof_frames, sam_error = [], [], None
    # Best-rep frames are kept in their OWN list, not merged into
    # proof_frames. proof_frames is the Gemini payload: every entry is a
    # flagged moment carrying SAM candidates, and slipping a clean frame with
    # no candidates into it would put "here is a fault, boxes below" in front
    # of the model for a rep that has none.
    best_frames = []
    if wanted:
        # A shared, already-loaded segmenter is the fast path and the normal
        # one. Constructing our own is the fallback for when startup could not
        # load it, and only then do we own it enough to unload it.
        owned = segmenter is None and enable_sam
        seg = (Segmenter() if owned else segmenter) if enable_sam else None
        loaded = bool(seg and (seg.predictor is not None or seg.load()))
        try:
            # Group the grabs by chunk so each file is opened once. Opening a
            # VideoCapture per chapter put three opens on the critical path,
            # and this sits inside the latency the athlete is waiting on.
            by_chunk = {}
            for c in wanted:
                t = c["dynamics"]["worst_moment"]["timestamp"]
                chunk, local_t = _locate(chunks, t)
                if chunk and chunk.get("path"):
                    by_chunk.setdefault(chunk["path"], []).append(
                        (c["chapter_index"], round(t, 2), chunk["index"], local_t))

            for path, items in by_chunk.items():
                got = kf.grab(path, [lt for _, _, _, lt in items])
                for (ch_idx, t, chunk_idx, _), (_, frame) in zip(items, got):
                    cands = seg.segment_frame(frame) if loaded else []
                    segments.append({
                        "chapter_index": ch_idx,
                        "timestamp": t,
                        "chunk_index": chunk_idx,
                        "frame_shape": [int(frame.shape[0]),
                                        int(frame.shape[1])],
                        "candidates": cands,
                    })
                    jpg = kf.to_jpeg(frame)
                    if jpg:
                        proof_frames.append((ch_idx, t, jpg, cands))
            # Chapter order, not chunk order: the timeline and the prompt both
            # read these positionally.
            segments.sort(key=lambda s: s["chapter_index"])
            proof_frames.sort(key=lambda p: p[0])

            # The good rep, from the same chunked source. SAM is deliberately
            # NOT run on these: there is nothing to segment on a rep that went
            # well, and the second image encoder pass would cost ~2.5 s to
            # outline a limb nobody is being asked to look at.
            best_by_chunk = {}
            for c in wanted:
                bt = ((c.get("reps") or {}).get("best_rep") or {}).get("timestamp_s")
                if bt is None:
                    continue
                chunk, local_t = _locate(chunks, bt)
                if chunk and chunk.get("path"):
                    best_by_chunk.setdefault(chunk["path"], []).append(
                        (c["chapter_index"], round(bt, 2), local_t))
            for path, items in best_by_chunk.items():
                for (ch_idx, t, _), (_, frame) in zip(
                        items, kf.grab(path, [lt for _, _, lt in items])):
                    jpg = kf.to_jpeg(frame)
                    if jpg:
                        best_frames.append((ch_idx, t, jpg))
            best_frames.sort(key=lambda p: p[0])

            # The isolated 3 s flaw clip per chapter. Cut from the CHUNKS,
            # which is why clips.extract_flaw is handed _locate: a 10 s chunk
            # and a +/-1.5 s window mean roughly a third of flaws sit close
            # enough to a boundary that the clip spans two files, and the
            # extractor feeds both into one encoder.
            #
            # Chunks are deleted when the session closes, so this is the last
            # moment the footage exists.
            by_ch = {sg["chapter_index"]: sg for sg in segments}
            for c in wanted:
                ch_idx = c["chapter_index"]
                meta = clips.extract_flaw(
                    session.id, ch_idx,
                    c["dynamics"]["worst_moment"]["timestamp"],
                    duration, chunks, locate=_locate)
                if meta and ch_idx in by_ch:
                    by_ch[ch_idx]["flaw_clip"] = meta
        finally:
            if seg:
                sam_error = seg.error
                # Only unload what we created. Unloading the shared instance
                # would hand back the 5 s load cost on the very next finalize,
                # which is the whole thing this avoids.
                if owned:
                    seg.unload()

    # Dominant classifier reading over the whole session. Cheap here and the
    # measurements panel shows it for both paths; computing it in the endpoint
    # instead would mean two copies of the same tally.
    counts = {}
    for w in gym_timeline:
        counts[w["exercise"]] = counts.get(w["exercise"], 0) + 1
    dominant = max(counts, key=counts.get) if counts else None
    conf = [w["confidence"] for w in gym_timeline if w["exercise"] == dominant]

    return {
        "session_id": session.id,
        "session_duration_s": duration,
        **chapters.summarize(spans),
        "chunks": len(chunks),
        "windows": len(embeddings),
        "embedding_dim": int(np.asarray(embeddings[0]).shape[0]),
        "gym_dominant": dominant,
        "gym_confidence": round(float(np.mean(conf)), 4) if conf else 0.0,
        "analysed": analysed,
        "segments": segments,
        "proof_frames": proof_frames,
        "best_frames": best_frames,
        "chunk_index": chunks,
        "excluded_windows": sorted(excluded),
        "dynamics": dynamics,
        "reps": reps,
        "spans": spans,
        "gym_timeline": gym_timeline,
    }


def extract_context_frames(analysed, chunks):
    """Per-chapter keyframes for the Gemini upgrade. -> [(chapter, t, jpg)].

    Runs in the BACKGROUND task, never inside finalize. Doing it inline cost
    2.4 s -- nine VideoCapture opens for a three-chapter session -- and pushed
    a 1.97 s finalize to 4.36 s. Nothing on screen at that moment needs these
    frames; only the narration does, and that is already asynchronous.

    Grabs are grouped by chunk so each file is opened once rather than once
    per timestamp, which is most of what made the inline version slow.
    """
    wanted = {}
    for c in analysed:
        if c["kind"] != "exercise":
            continue
        for t in analysis_pipeline.chapter_sample_times(c):
            chunk, local_t = _locate(chunks, t)
            if chunk and chunk.get("path"):
                wanted.setdefault(chunk["path"], []).append(
                    (c["chapter_index"], round(t, 2), local_t))

    out = []
    for path, items in wanted.items():
        got = kf.grab(path, [lt for _, _, lt in items])
        for (ch_idx, t, _), (_, frame) in zip(items, got):
            jpg = kf.to_jpeg(frame)
            if jpg:
                out.append((ch_idx, t, jpg))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _locate(chunks, t):
    for c in chunks:
        if c["t_start"] <= t < c["t_end"]:
            return c, t - c["t_start"]
    if chunks and t >= chunks[-1]["t_end"]:
        return chunks[-1], max(0.0, t - chunks[-1]["t_start"])
    return (chunks[0], t) if chunks else (None, t)


def proof_chunk_paths(segments, chunks, limit=2):
    """Chunk files worth uploading to Gemini: the ones holding the proofs.

    Uploading the whole session would send minutes of video to narrate a few
    seconds of it. The chunks containing the flagged moments carry the
    evidence, and they are a fraction of the bytes.
    """
    idxs, seen = [], set()
    for s in segments:
        ci = s.get("chunk_index")
        if ci is not None and ci not in seen:
            seen.add(ci)
            idxs.append(ci)
    by_index = {c["index"]: c for c in chunks}
    return [by_index[i]["path"] for i in idxs[:limit]
            if by_index.get(i) and by_index[i].get("path")]
