"""Have Hit inference API.

    .\\run.ps1 -m uvicorn app:app --host 0.0.0.0 --port 8000
    .\\run.ps1 app.py                     # same thing, with reload off

ONE asynchronous analysis architecture, serving two user-facing modes:

  RECORD  the phone is propped up, a set is recorded, Stop uploads the whole
          clip. Highlight reel, rep timestamps, form breakdown come back.
  VAULT   any pre-recorded clip -- gym lift, golf swing, tennis serve -- gets
          the same treatment.

Both are the same request. The client is the only thing that differs, so there
is one endpoint and one code path rather than two that drift apart:

    POST /api/v1/analyze-workout   video -> the full report (see services/)
    POST /coach/speak              a coaching line -> audio/wav
    GET  /health                   liveness, warm state, Gemini availability
    GET  /classes                  label list and their ids
    GET  /analytics                logged sets, oldest first
    POST /analytics                append one set
    GET  /                         the console (dashboard.html)
    GET  /full                     instrumented console (dashboard_full.html)
    GET  /console                  the older test console (index.html)

The four analysis stages live in services/analysis_pipeline.py. app.py owns
HTTP, the model lifecycle and concurrency; it does not own analysis logic.

WHAT WAS REMOVED, AND WHY IT MATTERS
    The rolling 1500 ms /predict stream is gone. It was the only fully local,
    zero-cost, offline path in the product -- every request now leaves the
    machine for Gemini and costs money. That is a deliberate trade for depth
    over immediacy, but it is a real loss and worth remembering before anyone
    asks why the app needs internet to count a squat.

CONCURRENCY AND VRAM (4 GB RTX 3050)
    The whole pipeline is serialised behind one semaphore. Analysis loads
    V-JEPA and then SAM, and two concurrent requests on this card is an OOM,
    not throughput. Requests queue; they do not race.

    The encoder loads ONCE at startup and is warmed with a synthetic clip --
    the first CUDA forward costs ~1.8 s while kernels compile, and making a
    real user pay that makes every latency number a lie.

Blocking work runs in a worker thread so the event loop keeps serving while
the GPU is busy: "async" here buys concurrency of waiting, not a faster pass.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
import config
import voice
from models.classifier import load_classifier
from services import chunk_pipeline, clips, db, health, local_coach
from services.analysis_pipeline import (STAGE_A_BATCH, chat_reply,
                                        WINDOW_SECONDS as ANALYZE_WINDOW_SECONDS,
                                        _build_timeline as build_timeline,
                                        run_analysis, stage_d_timeline)
from services.segmentation import Segmenter

HERE = Path(__file__).parent
ANALYTICS_PATH = HERE / "user_analytics.json"

# Full-length uploads. Bounded because the file is written to disk before cv2
# can open it, and this machine's commit limit is not generous.
MAX_ANALYZE_BYTES = 256 * 1024 * 1024

# One rolling chunk is ~10 s of phone video. Generous, but far below the
# whole-file cap: a chunk this large means the client is not chunking.
MAX_CHUNK_BYTES = 48 * 1024 * 1024

# Window ceiling for a manual upload, and the single biggest lever on latency:
# Stage A is ~139 ms per window here and 85-90% of the local response, so this
# number very nearly IS the response time.
#
# Lowering it is not a free win, because sample_timeline holds the window COUNT
# and WIDENS each window to cover the file -- and every analysis constant
# downstream is written in windows, not seconds. chapters.EMB_SMOOTH is 9
# windows: 6.75 s at the normal 0.75 s width, but 14.9 s at 1.65 s, which is
# wider than a whole set and smooths away the transitions it exists to find.
#
# MEASURED on the three-exercise reference clip (66 s; true chapters are
# 0-28 s, 28-43 s, 43-66 s):
#
#   cap  wall    chapters found   boundaries          SAM boxes
#    96  16.2s   3  correct       0-28 28-43 43-66    3/3
#    64  11.4s   4  WRONG         splits a set in two 4/4
#    48   8.6s   3  wrong         0-38 38-51 51-66    3/3
#    40   7.6s   3  wrong         0-16 16-38 38-66    3/3
#    32   5.7s   2  LOST ONE      0-33 33-66          2/2
#    24   6.2s   3  wrong         0-33 33-44 44-66    3/3, two with 0 reps
#    16   3.2s   1  COLLAPSED     0-66                0/1  -- no keyframe
#     8   1.8s   1  COLLAPSED     0-66                0/1  -- no keyframe
#
# 96 is the only setting that gets the timeline right, and every setting fast
# enough to approach 2.5 s returns neither chapters nor a bounding box. A wrong
# timeline delivered in six seconds is worse than a right one in fourteen,
# because the athlete acts on it. Per-request override: ?max_windows=N.
UPLOAD_MAX_WINDOWS = 96

# In-flight recording sessions. Video lives under the project cache rather
# than the system temp dir, because C: is where the pagefile lives and is the
# drive this project must stay off.
SESSIONS = chunk_pipeline.SessionStore(
    Path(os.environ.get("TMP") or (HERE / ".cache" / "tmp")) / "hh_sessions")

# The text-model fallback chain moved to services/analysis_pipeline.py, which
# is the only thing that calls a text model now. It exists because
# gemini-2.5-flash was hardcoded, then went 404 "no longer available to new
# users" while still being listed by models.list() -- visible is not callable,
# and only a real request settles it.
#
# The speech model, voice and PCM format live in voice.py, shared with
# live_coach.py. A text model cannot emit audio; the TTS variant can.
GEMINI_TIMEOUT_S = 45

# The analysis pipeline talks to a multimodal model and can take a while: the
# upload, server-side transcoding and a long-context video read all sit inside
# it. Everything about how that request is built lives in services/.
ANALYZE_TIMEOUT_S = 360

# NOTE ON PRIVACY: /api/v1/analyze-workout sends the uploaded clip, and three
# keyframes from it, to Google. That is now the ONLY analysis path, so unlike
# the previous architecture there is no local-only mode left -- every workout
# a user records leaves this machine. /coach/speak still sends a sentence and
# nothing more. Everything the prompt is built from lives in services/.

# The history panel is a rolling view, not an archive. Keeping the newest N
# bounds both the file and the payload the console parses on every append.
MAX_SETS = 200

# Serialises read-modify-write on the analytics file. Appends are rare (one per
# completed set) but two tabs streaming at once would otherwise interleave and
# lose an entry. Module level, not in STATE, because STATE is cleared on
# shutdown and this must outlive that.
ANALYTICS_LOCK = asyncio.Lock()

STATE = {"encoder": None, "head": None, "analysis": None, "warm": False,
         "startup_health": None, "segmenter": None}


def _load():
    from models.vjepa_wrapper import VJEPAEncoder

    enc = VJEPAEncoder()
    for p in enc.model.parameters():
        p.requires_grad = False
    head = load_classifier()
    return enc, head


def _warmup(enc, head):
    """Pay the CUDA kernel-compile cost now instead of on a user's request.

    BOTH shapes get warmed. cuDNN picks and caches an algorithm per input
    shape, so a B=1 warmup leaves the batched path cold and the first upload
    pays the compile -- measured at roughly a second, landing squarely inside
    the latency this sprint exists to reduce.
    """
    dummy = np.zeros((config.NUM_FRAMES, 240, 320, 3), dtype=np.uint8)
    for _ in range(2):
        _infer(enc, head, dummy)
    # Batched shape, as Stage A will actually call it.
    for _ in range(2):
        enc.embed_many([dummy] * STAGE_A_BATCH)


def _infer(enc, head, frames):
    """Blocking. frames -> (label, confidence, probabilities)."""
    emb = enc.embed(frames)
    with torch.no_grad():
        logits = head(torch.tensor(emb, device=config.DEVICE).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(probs.argmax())
    return config.ID_TO_LABEL[idx], float(probs[idx]), probs


def _gemini_key():
    # One definition of "which env var holds the key", in voice.py.
    return voice.api_key()


def _gemini_speech(text):
    """Blocking. Coaching line -> WAV bytes, or None if TTS is unavailable.

    The call itself lives in voice.py so this and live_coach.py cannot drift
    apart on model id, voice, or PCM format.
    """
    if not voice.available():
        return None
    return voice.synthesize(text)


def _read_analytics():
    """Current log, or an empty one if the file is missing or corrupt.

    A malformed file must not take the console down with it: the history panel
    is a nicety, while /predict is the product.
    """
    if not ANALYTICS_PATH.exists():
        return {"version": 1, "sets": []}
    try:
        data = json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "sets": []}
    if not isinstance(data, dict) or not isinstance(data.get("sets"), list):
        return {"version": 1, "sets": []}
    return {"version": data.get("version", 1), "sets": data["sets"]}


def _write_analytics(data):
    """Write via a temp file and os.replace so a crash cannot truncate the log.

    Writing in place would leave a half-written file on a kill, and the next
    read would silently discard every set logged so far.
    """
    tmp = ANALYTICS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, ANALYTICS_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before anything is loaded, so the numbers describe the machine this is
    # about to run on rather than the machine after two backbones landed on
    # it. If the load below dies of commit pressure, this block is already in
    # the log and names the reason -- which is the entire point of running it
    # first. Cached so /health can serve it without re-measuring disks.
    #
    # run_server.py prints the same block before the app is even imported, and
    # sets HH_PREFLIGHT_DONE when it has. The check still RUNS in that case --
    # /health serves this snapshot -- it just does not print a second identical
    # copy thirty lines below the first.
    quiet = bool(os.environ.get("HH_PREFLIGHT_DONE"))
    STATE["startup_health"] = health.log_startup_diagnostics(
        printer=(lambda *a, **k: None) if quiet else print)

    try:
        info = await asyncio.to_thread(db.init)
        print(f"  workouts.db ready: {info['sessions']} stored session(s)")
    except Exception as exc:                                  # noqa: BLE001
        # History is a convenience. Losing it must not stop the analysis
        # engine from serving, which is the actual product.
        print(f"  workouts.db unavailable ({exc}); history disabled")

    print("loading V-JEPA 2.1 encoder + head...")
    t0 = time.time()
    enc, head = await asyncio.to_thread(_load)
    STATE["encoder"], STATE["head"] = enc, head
    # One analysis at a time. A run loads V-JEPA and then SAM, so two in
    # parallel would exceed 4 GB; queueing is the difference between a slow
    # second request and two failed ones.
    STATE["analysis"] = asyncio.Semaphore(1)
    print(f"  loaded in {time.time() - t0:.1f}s, warming up...")

    t1 = time.time()
    await asyncio.to_thread(_warmup, enc, head)
    STATE["warm"] = True
    print(f"  warm in {time.time() - t1:.1f}s -- ready on {config.DEVICE}")

    # SAM stays resident. Loading it costs ~5 s, and doing that per request
    # put the entire Stage C bill inside the latency the user feels: finalize
    # measured 3.2 s of which 5.0 s-worth was load/unload churn, against 1.5 s
    # of actual segmentation. Paid once here, at boot, where nobody is waiting.
    #
    # This is a considered exception to "never hold two backbones at once".
    # That rule exists to avoid OOM on a 4 GB card, and the numbers say this
    # is not the case it was written for: sam2.1-hiera-tiny is 39M params and
    # 289 MB, against V-JEPA's ~1.2 GB, leaving well over a gigabyte spare.
    # The VRAM figures print below on every boot so a regression is visible.
    t2 = time.time()
    seg = Segmenter()
    STATE["segmenter"] = seg if seg.load() else None
    if STATE["segmenter"] is None:
        print(f"  SAM unavailable ({seg.error}); Stage C will load per request")
    else:
        print(f"  SAM resident in {time.time() - t2:.1f}s")

    post = health.memory_status()
    gpu = health.gpu_status()
    print(f"  commit after load: {post['commit_used_gb']}/"
          f"{post['commit_limit_gb']} GB ({post['commit_percent']}%)")
    print(f"  VRAM after load  : {gpu['vram_used_gb']}/{gpu['vram_total_gb']} GB "
          f"({gpu['vram_free_gb']} GB free)")
    yield
    seg = STATE.get("segmenter")
    if seg:
        seg.unload()
    STATE.clear()


app = FastAPI(title="Have Hit", version="1.0", lifespan=lifespan)

# The mobile client is served from a different origin, so the browser/webview
# needs these. Credentials stay off: the endpoint takes no cookies or auth, so
# allowing them alongside "*" would be both invalid and pointless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# dashboard.html loads its JS from here. Served by the API rather than inlined
# so the browser can cache it between loads, and so the markup stays readable.
# check_dir=False: a missing static/ must not stop the API from booting, since
# every endpoint except the console works without it.
app.mount("/static", StaticFiles(directory=str(HERE / "static"), check_dir=False),
          name="static")


# Cache-busting token stamped into __ASSET_V__ in the served markup.
#
# WHY THIS EXISTS
#   /static was served with no Cache-Control at all, which does NOT mean "do
#   not cache" -- it means the browser is free to invent a freshness lifetime
#   from Last-Modified (RFC 9111 heuristic caching, commonly 10% of the file's
#   age). So an edited app.js kept being served from disk cache for hours, and
#   the service worker made it worse rather than better: its network-first
#   fetch() goes through that same HTTP cache, so "fresh from the network" was
#   returning the stale copy too. The symptom is the whole point of this
#   change: new UI code shipped and the browser kept running the old one.
#
# WHY MTIME AND NOT A BOOT TIMESTAMP
#   A per-boot or per-request timestamp busts the cache every single load,
#   which is not cache-busting, it is cache-disabling -- every reload re-pulls
#   72 KB that has not changed. mtime changes exactly when the file changes,
#   which is exactly when the cache should miss.
def _asset_version():
    stamp = 0.0
    for rel in ("static/js/app.js",):
        f = HERE / rel
        if f.exists():
            stamp = max(stamp, f.stat().st_mtime)
    return str(int(stamp))


# no-cache is NOT no-store: the browser may keep the copy, it just has to
# revalidate before using it, so an unchanged file still costs a 304 and no
# body. Applied to the page (it carries the version token, so a stale page
# pins a stale script) and to /static (the token only works if the request
# actually reaches the server).
_NO_CACHE = "no-cache, must-revalidate"


@app.middleware("http")
async def _revalidate_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = _NO_CACHE
    return resp


def _page(name):
    """Serve a static page from the API itself.

    Opening these as a file:// URL makes the browser send Origin: null and treat
    every request as cross-origin. Serving them here makes the page and the
    endpoint same-origin, so the console works even if CORS is later tightened.

    __ASSET_V__ is substituted here rather than by a template engine because
    that is the only dynamic thing in these pages, and the file stays openable
    on its own for editing.
    """
    page = HERE / name
    if not page.exists():
        raise HTTPException(404, f"{name} not found")
    html = page.read_text(encoding="utf-8").replace("__ASSET_V__", _asset_version())
    return HTMLResponse(html, headers={"Cache-Control": _NO_CACHE})


@app.get("/")
async def dashboard():
    """The coaching console."""
    return _page("dashboard.html")


@app.get("/manifest.json")
async def manifest():
    """The PWA manifest, with the media type the spec requires.

    Served by hand rather than from the /static mount because StaticFiles
    guesses the type from the extension and returns application/json. Chrome
    tolerates that; the spec says application/manifest+json, and some
    installability checks and manifest linters do not.
    """
    p = HERE / "static" / "manifest.json"
    if not p.exists():
        raise HTTPException(404, "manifest not found")
    return Response(content=p.read_bytes(),
                    media_type="application/manifest+json",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/sw.js")
async def service_worker():
    """The service worker, which MUST be served from the root.

    A worker's scope cannot rise above its own path: served from
    /static/sw.js it could only ever control /static/*, so it would never see
    a request for "/" and the app would not be installable. Same file, root
    URL.

    no-store on the worker itself is deliberate. It is the one file that
    decides how every other file is cached, so a stale copy of it is a bug
    that cannot fix itself.
    """
    p = HERE / "static" / "sw.js"
    if not p.exists():
        raise HTTPException(404, "service worker not found")
    return Response(content=p.read_bytes(), media_type="text/javascript",
                    headers={"Cache-Control": "no-store",
                             "Service-Worker-Allowed": "/"})


@app.get("/console")
async def legacy_console():
    """Retired streaming console. index.html is now a redirect to /.

    The route stays so old bookmarks land on the working page rather than a
    404. dashboard_full.html is not served at all any more: it was built
    entirely around the 1500 ms /predict stream and has no endpoint left to
    call. The file is kept on disk as reference, not wired up.
    """
    return _page("index.html")


@app.get("/health")
async def health_check(verbose: bool = False):
    """Liveness, plus live memory and VRAM when asked.

    The short form stays cheap because the console polls it while the model
    loads. `?verbose=1` adds the live commit/VRAM figures and the warnings
    recorded at startup -- enough to answer "why did that run die" without
    going to the server log.
    """
    body = {
        "status": "ok" if STATE.get("warm") else "loading",
        "device": config.DEVICE,
        "warm": bool(STATE.get("warm")),
        "classes": config.NUM_CLASSES,
        # So the console can show voice as off rather than letting the user
        # press a button that will only ever 503.
        "gemini": bool(_gemini_key()),
    }
    if verbose:
        started = STATE.get("startup_health") or {}
        body["memory"] = health.memory_status()
        body["gpu"] = health.gpu_status()
        body["startup_warnings"] = started.get("warnings", [])
    return body


@app.get("/classes")
async def classes():
    return {"classes": config.EXERCISES,
            "ids": {str(i): n for i, n in config.ID_TO_LABEL.items()}}


@app.post("/api/v1/analyze-workout")
async def analyze_workout(request: Request, sync: bool = False,
                          max_windows: int = None,
                          file: UploadFile = File(None)):
    """Analyse an uploaded file. Returns the LOCAL report, fast.

    A manual upload is handled as a session with exactly ONE chunk, which makes
    it the same code path the recorder already uses: Stage A, then chapters,
    Stage B and SAM, then the local coach -- and Gemini afterwards, in the
    background, exactly as /session/finalize does it.

    That unification is the point. Before this the endpoint blocked on Gemini
    for 49-74 s, so a vault upload took over a minute to show anything while a
    recorded set showed everything in about one second, from the same stages in
    the same order. Two paths that should behave identically and did not.

    Accepts multipart 'file' or a raw body, because mobile webviews send both
    shapes in practice.

    `?sync=1` keeps the old behaviour: one blocking call that waits for Gemini
    and returns the narrated report. Kept for scripted use, where a caller with
    no way to poll would rather wait than get a report it must then chase.

    `?max_windows=N` trades analysis resolution for speed on this one request.
    Stage A is ~139 ms per window on this card, so N is very nearly the latency
    in units of 139 ms -- and it is the ONLY large lever, since Stage A is 88%
    of the local response. Fewer windows means each one covers more time, and
    the downstream constants are written in windows: see UPLOAD_MAX_WINDOWS for
    what that costs.

    Serialised behind one semaphore. On a 4 GB card two concurrent runs are an
    OOM, so the second caller waits rather than crashing both.
    """
    if STATE.get("encoder") is None:
        raise HTTPException(503, "model still loading")

    raw = await file.read() if file is not None else await request.body()
    if not raw:
        raise HTTPException(400, "empty body: send a video file")
    if len(raw) > MAX_ANALYZE_BYTES:
        raise HTTPException(
            413, f"file larger than {MAX_ANALYZE_BYTES // 1024 // 1024} MB")

    mime = (getattr(file, "content_type", None)
            or request.headers.get("content-type") or "video/mp4")
    if not mime.startswith("video/"):
        # Raw-body posts arrive as octet-stream, and MediaRecorder tags its
        # output with a codec suffix the Files API will not accept.
        mime = "video/webm" if "webm" in mime else "video/mp4"
    mime = mime.split(";")[0].strip()

    suffix = {"video/quicktime": ".mov", "video/webm": ".webm",
              "video/x-matroska": ".mkv"}.get(mime, ".mp4")

    if sync:
        return await _analyze_blocking(raw, mime, suffix)

    t0 = time.perf_counter()
    # The upload lives in the session directory, not the system temp dir,
    # because the background Gemini task needs it AFTER this response has been
    # sent -- and because Session.cleanup() then deletes it for us when the
    # session is dropped or expires. A tempfile deleted in a finally: block
    # here would be gone before Stage D could read it.
    session = SESSIONS.create(prefix="upload_")
    path = session.dir / f"chunk_0000{suffix}"
    await asyncio.to_thread(path.write_bytes, raw)

    t_write = time.perf_counter()
    try:
        async with STATE["analysis"]:
            t_queue = time.perf_counter()
            try:
                # Clamped, not trusted. Below 8 windows there is no trajectory
                # left to fit a trend to, and above 96 Stage A runs for over
                # thirteen seconds while holding the GPU semaphore.
                windows_cap = (UPLOAD_MAX_WINDOWS if max_windows is None
                               else max(8, min(96, int(max_windows))))
                await asyncio.to_thread(
                    chunk_pipeline.ingest_chunk, session, path,
                    STATE["encoder"], STATE["head"], ANALYZE_WINDOW_SECONDS,
                    windows_cap)
                t_a = time.perf_counter()
                local = await asyncio.to_thread(
                    chunk_pipeline.finalize_local, session,
                    STATE.get("segmenter"))
                t_bc = time.perf_counter()
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except HTTPException:
        # Nothing will ever poll a session whose first request failed, and it
        # holds the whole uploaded file on disk until the TTL sweep.
        SESSIONS.drop(session.id)
        raise

    timeline, _ = build_timeline(local["analysed"], local["segments"], None)
    payload = _session_payload(session, local, timeline, "local-fallback",
                               "local (gemini pending)", t0)
    # Per phase, because "it took N seconds" is not actionable: the brief for
    # this endpoint was written against a guess about which phase was the bill,
    # and the guess was wrong. stage_a is the encoder; stage_bc is dynamics,
    # chaptering and SAM together.
    payload["timing_ms"].update({
        "upload_to_disk": round((t_write - t0) * 1000, 1),
        "gpu_queue_wait": round((t_queue - t_write) * 1000, 1),
        "stage_a_vjepa": round((t_a - t_queue) * 1000, 1),
        "stage_bc_sam": round((t_bc - t_a) * 1000, 1),
    })
    session.result = payload

    # Persisted WITH the proof keyframes, before returning. The UI draws its
    # boxes from these, and the history drawer has nothing to draw on later if
    # they are not stored now -- the upload is deleted when the session closes.
    try:
        await asyncio.to_thread(
            db.save_session, payload, _frame_rows(local))
    except Exception as exc:                                  # noqa: BLE001
        print(f"  history write failed: {exc}", flush=True)

    if voice.available():
        session.gemini_state = "running"
        asyncio.create_task(_upgrade_with_gemini(session, local, t0,
                                                 mime_type=mime))
    else:
        session.gemini_state = "skipped: GEMINI_API_KEY is not set"

    return JSONResponse({**payload, "gemini_state": session.gemini_state})


def _frame_rows(local):
    """Both keyframe kinds as db.save_session rows.

    proof_frames carry SAM candidates as a fourth element and best_frames do
    not, so they cannot simply be concatenated -- this is where the two shapes
    become one, and the only place that knows both.
    """
    rows = [(ci, "worst", ts, jpg) for ci, ts, jpg, _ in (local.get("proof_frames") or [])]
    rows += [(ci, "best", ts, jpg) for ci, ts, jpg in (local.get("best_frames") or [])]
    return rows


async def _analyze_blocking(raw, mime, suffix):
    """The old synchronous path: wait for Gemini, return the narrated report.

    Reached only via ?sync=1. Uses run_analysis, which does its own whole-clip
    Stage B and reports per-stage timings the chunked path has no equivalent
    for, so this is also the honest way to measure the pipeline end to end.
    """
    # Minted BEFORE the analysis, not after: run_analysis names the flaw clips
    # it writes after this id, and a clip cannot be filed under an id that does
    # not exist yet.
    sid = f"upload_{uuid.uuid4().hex[:12]}"
    fd, path = tempfile.mkstemp(suffix=suffix,
                                dir=os.environ.get("TMP") or None)
    os.close(fd)
    try:
        await asyncio.to_thread(Path(path).write_bytes, raw)
        async with STATE["analysis"]:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_analysis, path, STATE["encoder"], STATE["head"],
                        mime_type=mime, size_bytes=len(raw), session_id=sid,
                    ),
                    timeout=ANALYZE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    504, f"analysis timed out after {ANALYZE_TIMEOUT_S}s")
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    result.setdefault("session_id", sid)
    try:
        await asyncio.to_thread(db.save_session, result)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  history write failed: {exc}", flush=True)
    return JSONResponse(result)


@app.post("/api/v1/session/chunk")
async def session_chunk(request: Request, file: UploadFile = File(None),
                        session_id: str = None):
    """Ingest one ~10 s segment while recording is still going.

    Stage A runs on it immediately, so by the time the user presses stop the
    embeddings for the whole set already exist and finalize has nothing left
    to extract. Send `session_id` from the second chunk onwards; omit it on
    the first and one is allocated and returned.

    Each chunk must be a COMPLETE video file, not a MediaRecorder fragment --
    see the note in services/chunk_pipeline.py about container headers.
    """
    if STATE.get("encoder") is None:
        raise HTTPException(503, "model still loading")

    raw = await file.read() if file is not None else await request.body()
    if not raw:
        raise HTTPException(400, "empty body: send a video chunk")
    if len(raw) > MAX_CHUNK_BYTES:
        raise HTTPException(413, f"chunk larger than "
                                 f"{MAX_CHUNK_BYTES // 1024 // 1024} MB")

    sid = session_id or request.headers.get("x-session-id")
    session = SESSIONS.get(sid) if sid else None
    if sid and session is None:
        raise HTTPException(404, f"unknown or expired session: {sid}")
    if session is None:
        session = SESSIONS.create()

    mime = (getattr(file, "content_type", None) or "video/webm").split(";")[0]
    suffix = ".mp4" if "mp4" in mime else ".webm"
    path = session.dir / f"chunk_{len(session.chunks):04d}{suffix}"
    await asyncio.to_thread(path.write_bytes, raw)

    # Serialised with full analyses: this is a GPU forward like any other, and
    # two on a 4 GB card is an OOM.
    try:
        async with STATE["analysis"]:
            windows = await asyncio.to_thread(
                chunk_pipeline.ingest_chunk, session, path,
                STATE["encoder"], STATE["head"], ANALYZE_WINDOW_SECONDS)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"session_id": session.id, "chunk_index": len(session.chunks) - 1,
            "windows_added": windows, "total_windows": len(session.embeddings),
            "session_duration_s": round(session.duration, 2)}


@app.post("/api/v1/session/finalize")
async def session_finalize(request: Request, session_id: str = None):
    """Close a session and return the report.

    Returns as soon as the LOCAL analysis is done -- chapters, reps, worst
    moments, SAM proof and deterministic cues, every one of them measured.
    Gemini then runs in the background because it costs 49-74 s and nothing
    would be gained by making the user watch it; poll
    GET /api/v1/session/{id} for the narrated version.
    """
    body = {}
    try:
        body = await request.json()
    except Exception:                                         # noqa: BLE001
        pass
    sid = (session_id or (body or {}).get("session_id")
           or request.headers.get("x-session-id"))
    if not sid:
        raise HTTPException(400, "session_id is required")
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(404, f"unknown or expired session: {sid}")

    t0 = time.perf_counter()
    try:
        async with STATE["analysis"]:
            local = await asyncio.to_thread(
                chunk_pipeline.finalize_local, session, STATE.get("segmenter"))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    timeline, _ = build_timeline(local["analysed"], local["segments"], None)
    payload = _session_payload(session, local, timeline, "local-fallback",
                               "local (gemini pending)", t0)
    session.result = payload

    # Persist immediately, with the proof keyframes: the recording chunks are
    # deleted when the session is dropped, and these frames are the only thing
    # a stored bounding box can ever be drawn on again.
    try:
        await asyncio.to_thread(
            db.save_session, payload, _frame_rows(local))
    except Exception as exc:                                  # noqa: BLE001
        print(f"  history write failed: {exc}")

    if voice.available():
        session.gemini_state = "running"
        asyncio.create_task(_upgrade_with_gemini(session, local, t0))

    return JSONResponse(payload)


@app.get("/api/v1/history")
async def history_list(limit: int = 50, offset: int = 0):
    """Past sessions, newest first. Summaries only -- no timelines."""
    return await asyncio.to_thread(db.list_sessions, min(max(limit, 1), 200),
                                   max(offset, 0))


@app.get("/api/v1/analytics/form-health")
async def analytics_form_health(window_days: int = 7):
    """Rolling form-health index and fatigue picture over recent sessions.

    Everything here is computed from stored scores at request time rather than
    maintained as a running column. A stored aggregate has to be recomputed
    whenever a session is deleted or a Gemini upgrade rewrites a score, and
    getting that wrong leaves a number that is confidently stale. Fifty-eight
    sessions cost a few milliseconds to sum.
    """
    return await asyncio.to_thread(db.form_health,
                                   window_days=max(1, min(window_days, 90)))


@app.get("/api/v1/history/export")
async def history_export():
    """Every stored session as one downloadable JSON document.

    MUST stay declared above /api/v1/history/{sid}: FastAPI matches routes in
    declaration order, so the other way round "export" is captured as a session
    id and this endpoint is unreachable.
    """
    data = await asyncio.to_thread(db.export_all)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return JSONResponse(
        data,
        headers={"Content-Disposition":
                 f'attachment; filename="have-hit-workouts-{stamp}.json"'})


@app.post("/api/v1/history/clear-test-data")
async def history_clear_test_data(confirm: bool = False):
    """Purge stored sessions that never produced a usable proof keyframe.

    Defaults to a DRY RUN. Without `?confirm=1` it reports exactly which
    sessions match and deletes nothing, so the count in the confirmation
    dialog is measured rather than guessed and a mistake costs one wasted
    request instead of a user's history.
    """
    report = await asyncio.to_thread(db.clear_test_data, not confirm)
    if report["deleted"]:
        # Same reasoning as the single delete: the clips are footage and go
        # with the row. Only on a real run -- a dry run must delete nothing,
        # including files it does not mention.
        removed = 0
        for sid in (c["session_id"] for c in report.get("sessions") or ()):
            removed += await asyncio.to_thread(clips.purge_session, sid)
        report["clips_removed"] = removed
        print(f"  history: purged {report['deleted']} frameless session(s), "
              f"{removed} clip file(s)", flush=True)
    return report


@app.get("/api/v1/history/{sid}")
async def history_detail(sid: str):
    """One stored session: full timeline, bboxes and proof frame URLs."""
    row = await asyncio.to_thread(db.get_session, sid)
    if row is None:
        raise HTTPException(404, f"no stored session: {sid}")
    return row


class ChatTurn(BaseModel):
    role: str = "user"
    text: str = ""


class ChatRequest(BaseModel):
    chapter_idx: int = 0
    message: str = ""
    # Not in the brief's request shape, and additive rather than a change to
    # it: without prior turns "and how do I fix THAT" has no antecedent, so
    # every follow-up would be answered as if it were the first question.
    # Sent by the client, not held server-side -- a chat that survives a
    # server restart is not worth a session table.
    history: list[ChatTurn] = []


@app.post("/api/v1/session/{sid}/chat")
async def session_chat(sid: str, body: ChatRequest):
    """Ask the coach about one analysed set.

    Grounded on the STORED report, not on the video: by the time anyone asks a
    question the footage has been deleted, and the numbers in the database are
    the same ones on the user's screen. Nothing here touches the GPU, so it
    does not queue behind an analysis.

    This is the third path that sends user data off this machine, after
    /api/v1/analyze-workout and /coach/speak, and the first that sends
    something the user typed or said out loud. Worth knowing before pointing
    it at anything private.
    """
    text = (body.message or "").strip()
    if not text:
        raise HTTPException(422, "message is empty")

    session = await asyncio.to_thread(db.get_session, sid)
    if not session:
        raise HTTPException(404, f"no stored session: {sid}")

    result, source = await asyncio.to_thread(
        chat_reply, session, body.chapter_idx, text,
        [t.model_dump() for t in body.history])
    if result is None:
        # 503, not 500: the request was fine and the upstream was not, and the
        # client should offer a retry rather than an apology.
        raise HTTPException(503, source)
    return {**result, "source": source}


@app.get("/api/v1/session/{sid}/flaw-clip/{chapter_index}")
async def flaw_clip(sid: str, chapter_index: int):
    """The isolated 3-second clip of one chapter's breakdown moment.

    FileResponse rather than reading the bytes: it sets Content-Length and
    handles Range requests, and a <video> element issues a range request for
    every seek and every loop restart. Serving the whole file for each of
    those would re-send the clip on every lap of an autoplaying loop.

    Cached hard. The clip is derived from footage that no longer exists by the
    time anyone can ask for it, so it cannot change under this URL.
    """
    p = clips.clip_path(sid, chapter_index)
    if not p.exists():
        raise HTTPException(404, "no clip for that chapter")
    return FileResponse(
        p, media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/v1/session/{sid}/frame/{kind}/{chapter_index}")
async def session_frame(sid: str, kind: str, chapter_index: int):
    """One stored keyframe: the flawed moment, or the rep that went well.

    These two are what the side-by-side comparison card is made of, and
    serving them from the database rather than re-seeking the video is what
    makes that card work in the history drawer at all -- by then the recording
    has been deleted and the browser has no video to grab a frame from.

    Cached hard: a stored keyframe is immutable. It is keyed by session,
    chapter and kind, and nothing ever rewrites one under the same key.
    """
    if kind not in ("best", "worst"):
        raise HTTPException(404, "kind must be 'best' or 'worst'")
    jpg = await asyncio.to_thread(db.get_frame, sid, chapter_index, kind)
    if not jpg:
        raise HTTPException(404, "no stored frame")
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/v1/history/{sid}/frame/{chapter_index}")
async def history_frame(sid: str, chapter_index: int):
    """The stored proof keyframe, so past visual proof can be re-drawn.

    The recording itself is long gone -- chunks are deleted when a session
    closes -- so this frame is the only thing a historical bounding box can be
    drawn on.
    """
    jpg = await asyncio.to_thread(db.get_frame, sid, chapter_index)
    if jpg is None:
        raise HTTPException(404, "no stored frame")
    # Immutable once written, so let the browser keep it.
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=31536000"})


@app.delete("/api/v1/history/{sid}")
async def history_delete(sid: str):
    if not await asyncio.to_thread(db.delete_session, sid):
        raise HTTPException(404, f"no stored session: {sid}")
    # Take the flaw clips with it. They are footage of the user, kept outside
    # the database as files, and a delete that leaves them on disk is not a
    # delete -- the eviction cap would eventually reach them, which is not the
    # same as honouring the request.
    gone = await asyncio.to_thread(clips.purge_session, sid)
    return {"deleted": sid, "clips_removed": gone}


@app.get("/api/v1/session/{sid}")
async def session_status(sid: str):
    """The latest report for a session, narrated if Gemini has finished."""
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(404, f"unknown or expired session: {sid}")
    if session.result is None:
        return {"session_id": sid, "state": "recording",
                "chunks": len(session.chunks),
                "windows": len(session.embeddings),
                "session_duration_s": round(session.duration, 2)}
    return JSONResponse({**session.result, "gemini_state": session.gemini_state})


@app.delete("/api/v1/session/{sid}")
async def session_delete(sid: str):
    """Drop a session and its video files."""
    if SESSIONS.get(sid) is None:
        raise HTTPException(404, f"unknown or expired session: {sid}")
    SESSIONS.drop(sid)
    return {"deleted": sid}


async def _upgrade_with_gemini(session, local, t0, mime_type="video/webm"):
    """Background: narrate an already-returned report, then store it.

    Failure here is not an error state. The local report is already in the
    user's hands and remains valid; this only ever adds prose.
    """
    try:
        paths = chunk_pipeline.proof_chunk_paths(local["segments"],
                                                 session.chunks)
        if not paths:
            session.gemini_state = "skipped: no proof chunk available"
            return
        # No whole-session video exists -- each chunk is its own recording --
        # so one proof chunk goes up for motion context and the per-chapter
        # keyframes carry the rest. Sending every chunk would upload the
        # entire session to narrate a few seconds of it.
        # Extracted here, not in finalize: nine VideoCapture opens cost 2.4 s
        # and the user is already looking at a finished local report.
        context = await asyncio.to_thread(
            chunk_pipeline.extract_context_frames,
            local["analysed"], local["chunk_index"])
        report, source = await asyncio.to_thread(
            stage_d_timeline, paths[0], mime_type,
            Path(paths[0]).stat().st_size, local["analysed"],
            local["proof_frames"], local["session_duration_s"], context)
        if report is None:
            session.gemini_state = f"failed: {source}"
            return
        timeline, mode = build_timeline(local["analysed"], local["segments"],
                                        report)
        prior = ((session.result or {}).get("timing_ms") or {}).get("finalize")
        session.result = _session_payload(session, local, timeline, mode,
                                          source, t0, finalize_ms=prior)
        session.gemini_state = "done"
        # Stage D was the last thing that needed the footage. The session stays
        # alive and pollable; only the video goes.
        freed = session.release_video()
        if freed > 1024 * 1024:
            print(f"  released {freed / 1024 / 1024:.0f} MB of footage "
                  f"for session {session.id}", flush=True)
        # Second write, same row. Upserts rather than replaces, so the frames
        # stored above survive (see the note in services/db.py).
        try:
            await asyncio.to_thread(db.save_session, session.result)
        except Exception as exc:                              # noqa: BLE001
            print(f"  history upgrade write failed: {exc}")
    except Exception as exc:                                  # noqa: BLE001
        session.gemini_state = f"failed: {exc.__class__.__name__}: {exc}"


def _session_payload(session, local, timeline, mode, source, t0,
                     finalize_ms=None):
    primary = max(timeline, key=lambda b: b["duration_s"], default=None) or {}
    return {
        "session_id": session.id,
        "session_duration_s": local["session_duration_s"],
        "total_exercises_detected": local["total_exercises_detected"],
        "rest_periods": local["rest_periods"],
        "distinct_exercises": local["distinct_exercises"],
        "timeline": timeline,
        "analysis_source": source,
        "analysis_mode": mode,
        "activity_detected": primary.get("exercise_name"),
        "form_score": primary.get("form_score"),
        "total_reps": sum(b.get("total_reps") or 0 for b in timeline),
        "best_rep": primary.get("best_rep"),
        "worst_rep": primary.get("worst_rep"),
        "visual_proof": primary.get("visual_proof"),
        "actionable_cue": primary.get("actionable_cue"),
        "synthesis_note": (local_coach.OFFLINE_NOTE
                           if mode == "local-fallback" else None),
        "local_measurements": {
            "duration_s": local["session_duration_s"],
            "windows": local["windows"],
            "window_seconds": ANALYZE_WINDOW_SECONDS,
            "embedding_dim": local.get("embedding_dim"),
            "chunks": local["chunks"],
            # Both paths report this now. It was missing here, so the console's
            # measurements panel showed an em dash for every recorded set while
            # showing a real reading for uploads -- same pipeline, same number,
            # visible in one place only.
            "gym_classifier": {
                "note": "27-class gym model, ~44% val accuracy. Meaningless "
                        "off gym footage.",
                "dominant": local.get("gym_dominant"),
                "mean_confidence": local.get("gym_confidence"),
                "timeline": local.get("gym_timeline") or [],
            },
            "latent_dynamics": local["dynamics"],
            "repetitions": local["reps"],
            "chapters": [{k: v for k, v in c.items()
                          if k not in ("dynamics", "reps")}
                         for c in local["analysed"]],
            "segmentation": {"model": "sam2.1-hiera-tiny",
                             "keyframes": local["segments"]},
        },
        # The LOCAL finalize, always -- not the elapsed time at the moment this
        # payload happened to be built. The Gemini upgrade rebuilds the payload
        # with the same t0, so measuring afresh there reported "finalize:
        # 95984 ms" for a finalize that took 2.1 s, in the panel whose whole
        # job is to say how fast the local stages were.
        "timing_ms": {"finalize": (finalize_ms if finalize_ms is not None
                                   else round((time.perf_counter() - t0) * 1000, 1))},
    }


@app.post("/coach/speak")
async def coach_speak(request: Request):
    """Pipeline 1's voice: a recognised exercise -> spoken coaching, as WAV.

    Kept server-side because the API key belongs on the server, not in a page
    any viewer can read. Only the coaching sentence goes to Google -- never
    webcam frames, which never leave this machine.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "body must be JSON")

    exercise = (body or {}).get("exercise")
    if exercise not in config.LABEL_TO_ID:
        raise HTTPException(422, f"unknown exercise: {exercise!r}")

    try:
        conf = float((body or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    if not _gemini_key():
        raise HTTPException(503, "voice unavailable: GEMINI_API_KEY is not set")

    line = f"You are doing {exercise.replace('_', ' ')}. Confidence {int(conf * 100)} percent."
    try:
        wav = await asyncio.wait_for(
            asyncio.to_thread(_gemini_speech, line), timeout=GEMINI_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "gemini tts timed out")
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(502, f"gemini tts failed: {exc.__class__.__name__}: {exc}")

    if wav is None:
        raise HTTPException(503, "voice unavailable: google-genai not installed")

    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Coach-Line": line})


@app.get("/analytics")
async def analytics():
    """Logged sets, oldest first."""
    return _read_analytics()


@app.post("/analytics")
async def log_set(request: Request):
    """Append one completed set and return the trimmed log.

    The console posts a set once the same exercise leads for several
    consecutive windows. Fields are validated rather than trusted: this file is
    read back and rendered, and an unknown label here would show up as an
    exercise the model cannot actually predict.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")

    exercise = body.get("exercise")
    if exercise not in config.LABEL_TO_ID:
        raise HTTPException(422, f"unknown exercise: {exercise!r}")

    def num(key, lo, hi, cast=float):
        try:
            return max(lo, min(hi, cast(body.get(key, 0) or 0)))
        except (TypeError, ValueError):
            return lo

    entry = {
        # The server stamps the time. A client clock can be anything at all,
        # and the panel sorts and renders "x minutes ago" off this.
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exercise": exercise,
        "confidence": round(num("confidence", 0.0, 1.0), 4),
        "reps": num("reps", 0, 9999, int),
        "windows": num("windows", 0, 9999, int),
        "duration_s": round(num("duration_s", 0.0, 86400.0), 1),
    }

    async with ANALYTICS_LOCK:
        data = _read_analytics()
        data["sets"] = (data["sets"] + [entry])[-MAX_SETS:]
        await asyncio.to_thread(_write_analytics, data)

    return data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
