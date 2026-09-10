"use strict";
/* Have Hit console.
 *
 * Two capture modes feeding one endpoint, plus the result player. Extracted
 * from dashboard.html so the markup stays readable and this file can be
 * cached by the browser between page loads.
 */

const API = "";                       // same-origin: the page is served by the API
const $ = (id) => document.getElementById(id);

/* Everything model-generated is escaped before it reaches innerHTML. Neither
   Gemini output nor a filename is trusted markup -- an activity name
   containing a tag would otherwise execute here. */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));

/* "01:23.4" or a bare number of seconds -> seconds. The schema asks for MM:SS.s
   strings but the local measurements carry plain floats, and both end up as
   seek targets. */
/* Prefer the numeric field the rep engine now emits; fall back to parsing the
   MM:SS.s display string for payloads written before it existed (the history
   drawer replays stored JSON, so those are still in circulation). */
function repSeconds(rep, fallbackString) {
  const n = rep && rep.timestamp_s;
  return Number.isFinite(n) ? n : toSeconds(fallbackString || (rep || {}).timestamp);
}

function toSeconds(ts) {
  if (ts == null) return null;
  if (typeof ts === "number") return ts;
  const s = String(ts).trim();
  const m = s.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (m) return parseInt(m[1], 10) * 60 + parseFloat(m[2]);
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : null;
}
function fmtTime(sec) {
  if (sec == null || !Number.isFinite(sec)) return "--:--";
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

/* ─────────────────────────── engine status ─────────────────────────── */
async function pingHealth() {
  try {
    const r = await fetch(API + "/health");
    const h = await r.json();
    const ok = h.status === "ok";
    $("engineDot").className = `w-1.5 h-1.5 rounded-full ${ok ? "bg-volt" : "bg-warn"}`;
    $("engineTxt").className = `text-[10px] font-bold tracking-[0.14em] ${ok ? "text-volt/80" : "text-warn/80"}`;
    $("engineTxt").textContent = ok
      ? `READY · ${h.classes} CLASSES${h.gemini ? "" : " · OFFLINE MODE"}`
      : "MODEL LOADING…";
    if (!ok) setTimeout(pingHealth, 2500);
  } catch {
    $("engineDot").className = "w-1.5 h-1.5 rounded-full bg-crit";
    $("engineTxt").textContent = "API OFFLINE";
    setTimeout(pingHealth, 3000);
  }
}

/* ─────────────────────────────── tabs ─────────────────────────────── */
function wireTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach(b => {
        const on = b === btn;
        b.className = `tab flex-1 sm:flex-none px-4 sm:px-5 py-2.5 rounded-xl text-[12px] font-bold tracking-tight transition-all ${
          on ? "bg-volt text-black" : "text-white/50 hover:text-white/80"}`;
      });
      const tab = btn.dataset.tab;
      $("paneRecord").classList.toggle("hidden", tab !== "record");
      $("paneVault").classList.toggle("hidden", tab !== "vault");
    };
  });
}

/* ───────────────────────── record mode ───────────────────────── */
let stream = null, recorder = null, chunks = [], recStart = 0, recTimer = null;

/* Default to the REAR camera on touch devices. The phone gets propped against
   a wall or a bench facing the lifter, who cannot see the screen from there
   anyway -- and the rear sensor is the better one on essentially every phone.
   Desktops keep the front camera, which is the only one they have. */
const isTouch = matchMedia("(pointer: coarse)").matches;
let facing = isTouch ? "environment" : "user";

/* Capture resolution. Phones default to 1080p or 4K, and every pixel above
   this is thrown away twice over: the encoder pays for it, the network pays
   for it, and then V-JEPA downsamples to 256x256 anyway. 640x480 is already
   more than the model consumes, so asking for more buys nothing but latency. */
const CAP_W = 640, CAP_H = 480;

/* ~800 kbps at 640x480 is visually fine for movement analysis and keeps a
   10 s chunk near 1 MB, which uploads over gym wifi inside the recording
   window rather than queueing up behind it. */
const CHUNK_BITRATE = 800000;

async function openCamera() {
  stopStream();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        // Ideal, not exact: `exact` throws OverconstrainedError on any device
        // without that camera, which is most laptops.
        facingMode: { ideal: facing },
        width: { ideal: CAP_W }, height: { ideal: CAP_H },
        frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });
    $("cam").srcObject = stream;
    $("camIdle").classList.add("hidden");
    $("recBtn").disabled = false;
    $("camFlip").classList.toggle("hidden", !hasMultipleCameras);
    // Report what was actually granted, not what was requested: a device may
    // ignore the constraint, and claiming 640x480 while streaming 1080p would
    // hide the very problem these constraints exist to prevent.
    const s = stream.getVideoTracks()[0]?.getSettings?.() || {};
    $("camLabel").textContent =
      `${facing === "environment" ? "Rear" : "Front"} · ${s.width || "?"}×${s.height || "?"}`;
    $("camFlipLabel").textContent = facing === "environment" ? "Front" : "Back";
  } catch (e) {
    showError(`Camera unavailable: ${e.message}. Chrome needs https:// or localhost for camera access.`);
  }
}

function stopStream() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}

let hasMultipleCameras = false;
async function detectCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    hasMultipleCameras = devices.filter(d => d.kind === "videoinput").length > 1;
  } catch { hasMultipleCameras = false; }
}

/* Container preference for uploaded chunks.
     VP8 WebM first  — cheapest to encode on a phone, and verified decodable
                       server-side by this project's OpenCV build.
     VP9             — better compression, materially slower to encode on
                       mid-range hardware, so it sits behind VP8.
     MP4/H.264       — Safari and iOS, which do not offer WebM at all.
   Bitrate, not codec, is what actually bounds the upload; see CHUNK_BITRATE. */
function pickMime() {
  const want = ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm",
                "video/mp4;codecs=h264", "video/mp4"];
  for (const m of want) if (MediaRecorder.isTypeSupported(m)) return m;
  return "";
}

function recorderOptions() {
  const mimeType = pickMime();
  const o = { videoBitsPerSecond: CHUNK_BITRATE };
  if (mimeType) o.mimeType = mimeType;
  return o;
}

/* Rolling chunk length. Each chunk is a COMPLETE recording (its own
   start/stop cycle) because a MediaRecorder timeslice fragment carries no
   container header after the first and the server cannot decode one. */
const CHUNK_MS = 10000;

let sessionId = null, recording = false, fullRecorder = null, fullChunks = [];

function startRec() {
  recording = true;
  sessionId = null;
  fullChunks = [];
  /* A SECOND recorder runs across the whole take, purely so the results page
     has something to play. The uploaded chunks cannot serve that purpose:
     they are separate files, and concatenating separate WebM/MP4 containers
     does not produce a playable one. This costs a second encoder pass, which
     is the cheapest of the available options. */
  fullRecorder = new MediaRecorder(stream, recorderOptions());
  fullRecorder.ondataavailable = e => { if (e.data && e.data.size) fullChunks.push(e.data); };
  fullRecorder.start();

  recStart = Date.now();
  $("recBadge").classList.remove("hidden");
  $("recBtn").textContent = "Stop & Analyse";
  $("recBtn").className = recBtnClass("bg-rec text-white");
  recTimer = setInterval(() => {
    const t = (Date.now() - recStart) / 1000;
    $("recTime").textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  }, 200);

  cycleChunk();
}

/* One 10 s chunk, uploaded as soon as it closes, then the next. Stage A runs
   server-side while the user keeps lifting, so by Stop there is nothing left
   to extract. */
function cycleChunk() {
  if (!recording || !stream) return;
  const rec = new MediaRecorder(stream, recorderOptions());
  const parts = [];
  rec.ondataavailable = e => { if (e.data && e.data.size) parts.push(e.data); };
  rec.onstop = () => {
    const blob = new Blob(parts, { type: rec.mimeType || "video/webm" });
    if (blob.size > 1024) uploadChunk(blob);
    if (recording) cycleChunk();
  };
  rec.start();
  recorder = rec;
  setTimeout(() => { if (rec.state === "recording") rec.stop(); }, CHUNK_MS);
}

async function uploadChunk(blob) {
  const fd = new FormData();
  fd.append("file", blob, "chunk.webm");
  try {
    const headers = sessionId ? { "X-Session-Id": sessionId } : {};
    const r = await fetch(API + "/api/v1/session/chunk",
                          { method: "POST", body: fd, headers });
    if (!r.ok) return;                 // a dropped chunk costs coverage, not the set
    const j = await r.json();
    sessionId = j.session_id;
    $("camLabel").textContent =
      `${j.total_windows} windows analysed · ${j.session_duration_s.toFixed(0)}s`;
  } catch { /* offline mid-set: keep recording, finalize on what arrived */ }
}

function stopRec() {
  recording = false;
  clearInterval(recTimer);
  $("recBadge").classList.add("hidden");
  $("recBtn").textContent = "Start Recording";
  $("recBtn").className = recBtnClass("bg-volt text-black");

  // Close the in-flight chunk so its footage is not lost.
  if (recorder && recorder.state === "recording") recorder.stop();

  if (fullRecorder && fullRecorder.state === "recording") {
    fullRecorder.onstop = () => {
      const blob = new Blob(fullChunks, { type: fullRecorder.mimeType || "video/webm" });
      // Give the last chunk a moment to reach the server before finalizing;
      // finalizing first would analyse a session missing its final seconds.
      setTimeout(() => finalizeSession(blob), 900);
    };
    fullRecorder.stop();
  }
}

/* Stop -> report. The server returns as soon as the LOCAL analysis is done,
   so this resolves in about two seconds; the narration arrives later. */
async function finalizeSession(playbackBlob) {
  if (!sessionId) {
    // Nothing reached the server. Fall back to the whole-file endpoint so the
    // set is still analysed, just without the head start.
    if (playbackBlob && playbackBlob.size > 1024) analyze(playbackBlob);
    else showError("No footage was captured.");
    return;
  }
  $("errBox").classList.add("hidden");
  $("report").classList.add("hidden");
  $("progress").classList.remove("hidden");
  $("progLabel").textContent = "Finalising…";

  const t0 = performance.now();
  try {
    const r = await fetch(API + "/api/v1/session/finalize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch {}
      showError(detail);
      return;
    }
    const data = await r.json();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = playbackBlob ? URL.createObjectURL(playbackBlob) : null;
    $("progress").classList.add("hidden");
    $("progTime").textContent = ((performance.now() - t0) / 1000).toFixed(2) + "s";
    renderReport(data, objectUrl);
    if (data.analysis_mode === "local-fallback") pollUpgrade(sessionId);
  } catch (e) {
    showError(e.message);
  }
}

/* Poll for the narrated version. Only ever ADDS to what is already on screen,
   so a failure here is a badge change and nothing more. */
async function pollUpgrade(sid, attempt = 0) {
  if (attempt > 45) { setBadge("stalled"); return; }
  setBadge("refining");
  try {
    const r = await fetch(`${API}/api/v1/session/${sid}`);
    if (!r.ok) { setBadge("stalled"); return; }
    const d = await r.json();
    const st = d.gemini_state || "";
    if (st === "done" && d.analysis_mode === "gemini") { applyUpgrade(d); return; }
    if (st.startsWith("failed") || st.startsWith("skipped")) {
      setBadge("failed", st);
      return;
    }
  } catch { /* transient: keep polling */ }
  setTimeout(() => pollUpgrade(sid, attempt + 1), 3000);
}

const recBtnClass = (colour) =>
  `w-full px-5 py-4 rounded-2xl ${colour} text-[14px] font-black tracking-tight ` +
  `disabled:opacity-30 disabled:cursor-not-allowed transition hover:brightness-110`;

/* ───────────────────────── vault mode ───────────────────────── */
function wireVault() {
  $("fileInput").onchange = e => {
    if (!e.target.files[0]) return;
    primeSpeech();               // picking a file is a gesture; use it
    analyze(e.target.files[0]);
  };
  ["dragenter", "dragover"].forEach(ev => $("drop").addEventListener(ev, e => {
    e.preventDefault(); $("drop").classList.add("border-volt/60", "bg-volt/[0.03]");
  }));
  ["dragleave", "drop"].forEach(ev => $("drop").addEventListener(ev, e => {
    e.preventDefault(); $("drop").classList.remove("border-volt/60", "bg-volt/[0.03]");
  }));
  $("drop").addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) analyze(f);
    else showError("That does not look like a video file.");
  });
}

/* ───────────────────────────── analyse ───────────────────────────── */
let objectUrl = null;

function showError(msg) {
  $("progress").classList.add("hidden");
  $("errBox").classList.remove("hidden");
  $("errMsg").textContent = msg;
}

async function analyze(blob) {
  $("errBox").classList.add("hidden");
  $("report").classList.add("hidden");
  $("progress").classList.remove("hidden");
  $("progStages").innerHTML = ["V-JEPA 2.1", "LATENT DYNAMICS", "SAM 2.1", "COACH"]
    .map(s => `<span class="px-2 py-1 rounded-md bg-white/[0.04] text-white/40">${s}</span>`).join("");
  $("progress").scrollIntoView({ behavior: "smooth", block: "nearest" });

  const t0 = performance.now();
  const tick = setInterval(() => {
    $("progTime").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s";
  }, 100);

  const fd = new FormData();
  fd.append("file", blob, blob.name || "recording.webm");

  try {
    const r = await fetch(API + "/api/v1/analyze-workout", { method: "POST", body: fd });
    clearInterval(tick);
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch {}
      showError(detail);
      return;
    }
    const data = await r.json();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(blob);
    $("progress").classList.add("hidden");
    $("progTime").textContent = ((performance.now() - t0) / 1000).toFixed(2) + "s";
    renderReport(data, objectUrl);
    /* The upload endpoint now returns the LOCAL report and refines in the
       background, exactly like the recorder. Same poller, same badge -- the two
       modes had no business behaving differently, and giving the vault its own
       progress logic is how they would drift again. */
    if (data.session_id && data.analysis_mode === "local-fallback"
        && data.gemini_state === "running") {
      pollUpgrade(data.session_id);
    }
  } catch (e) {
    clearInterval(tick);
    showError(e.message);
  }
}

/* ───────────────────────────── report ───────────────────────────── */
const STATUS = {
  optimal:  { dot: "bg-volt", text: "text-volt", ring: "border-volt/25", label: "OPTIMAL"  },
  warning:  { dot: "bg-warn", text: "text-warn", ring: "border-warn/25", label: "WARNING"  },
  critical: { dot: "bg-crit", text: "text-crit", ring: "border-crit/25", label: "CRITICAL" }
};

/* Report state. TIMELINE is the chapter array; ACTIVE is the chapter whose
   card, chips and overlay box are currently shown. PROOF is a flattened view
   of the active chapter's visual_proof, read by the canvas loop every frame --
   kept separate so the draw loop never walks the whole payload. */
let DATA = null, TIMELINE = [], ACTIVE = 0, PROOF = null;
let SOURCE_URL = null;

/* Milestone ticks on the seek bar: [{t, colour, label}]. Rebuilt per chapter,
   because a mark belongs to the chapter it was measured in -- showing every
   chapter's marks at once turns a 60 s session's rail into a picket fence. */
let MARKS = [];

const MARK_STYLE = {
  best:  { colour: "#C6FF00", label: "Best rep" },
  worst: { colour: "#FFB020", label: "Worst rep" },
  proof: { colour: "#FF4D4D", label: "Proof" },
};

function buildMarks(c) {
  if (!c) return [];
  const out = [];
  const best = repSeconds(c.best_rep, c.best_rep_timestamp);
  const worst = repSeconds(c.worst_rep, c.worst_rep_timestamp);
  if (Number.isFinite(best)) out.push({ t: best, kind: "best" });
  if (Number.isFinite(worst)) out.push({ t: worst, kind: "worst" });
  if (PROOF && Number.isFinite(PROOF.t)) out.push({ t: PROOF.t, kind: "proof" });
  return out;
}

/* Marks are positioned against the VIDEO's duration, not the session's.
   Those differ on the chunked path -- the playback blob is one continuous
   take while the analysed session can be a hair longer or shorter -- and
   using the wrong one puts every tick at a slightly wrong place, which is
   worse than no tick because it looks authoritative. */
function renderSeekMarks() {
  const el = $("seekMarks"), v = $("player");
  if (!el) return;
  const dur = v && Number.isFinite(v.duration) && v.duration > 0
            ? v.duration : (DATA || {}).session_duration_s;
  if (!dur) { el.innerHTML = ""; return; }
  el.innerHTML = MARKS.map(m => {
    const s = MARK_STYLE[m.kind];
    const pct = Math.max(0, Math.min(100, (m.t / dur) * 100));
    return `<span title="${s.label} · ${fmtTime(m.t)}" style="
      position:absolute;left:${pct.toFixed(3)}%;top:50%;transform:translate(-50%,-50%);
      width:3px;height:14px;border-radius:2px;background:${s.colour};
      box-shadow:0 0 6px ${s.colour}99;pointer-events:none;"></span>`;
  }).join("");

  const legend = $("markLegend");
  if (legend) {
    legend.innerHTML = MARKS.map(m => {
      const s = MARK_STYLE[m.kind];
      return `<span class="flex items-center gap-1 whitespace-nowrap">
        <span style="width:3px;height:9px;border-radius:2px;background:${s.colour}"></span>
        ${s.label}</span>`;
    }).join("");
  }
}

function renderReport(d, url) {
  DATA = d;
  TIMELINE = d.timeline || [];
  ACTIVE = 0;
  /* The full recording's URL, kept here rather than read back off the
     <video> element. Reading the sibling's .src worked only because the
     markup happened to be assigned first, and it returns "" for a history
     session where there is no recording at all -- the exact case the hero's
     fallback exists for. */
  SOURCE_URL = url || null;
  heroKey = null;                 // new report: force a rebuild
  CHAT_LOG = {};                  // a new session is a new conversation

  const lm = d.local_measurements || {};
  const dyn = lm.latent_dynamics || {};
  const offline = d.analysis_mode === "local-fallback";
  const multi = TIMELINE.length > 1;

  $("report").innerHTML = `
    <div id="upgradeBadge"></div>

    ${multi ? `
    <div class="rounded-3xl bg-panel border border-white/[0.06] p-4 sm:p-5">
      <div class="flex items-baseline justify-between mb-3 flex-wrap gap-2">
        <span class="text-[10px] font-bold tracking-[0.14em] text-white/35">WORKOUT TIMELINE</span>
        <span class="text-[10px] text-white/30">
          ${TIMELINE.length} exercises${d.rest_periods ? ` &middot; ${d.rest_periods} rest` : ""} &middot; ${fmtTime(d.session_duration_s)}
        </span>
      </div>
      <div id="chapterBar" class="flex gap-2 overflow-x-auto scrollbar-none pb-0.5"></div>
      <div id="chapterTicks" class="flex gap-3 mt-2 text-[9px] text-white/25 overflow-x-auto scrollbar-none"></div>
    </div>` : ""}

    <div id="summaryCard"></div>

    <!-- ══ THE HERO: the flagged three seconds, and nothing else ══
         This is the whole point of the screen. It replaces the full recording
         as the primary viewport: the analysis already found the moment, and
         opening on 90 seconds of uncut footage handed that search back to the
         athlete. Rebuilt per chapter by renderHero(). -->
    <div id="heroCard"></div>

    <!-- Three fields, three questions: what happened, what to think about on
         the next rep, what to train. -->
    <div id="coachCard"></div>

    <!-- Hands-free follow-up about the set on screen. Placed under the
         coaching card because every question it answers is a question the
         coaching card just provoked. -->
    <div id="chatCard"></div>

    <div id="compareCard"></div>

    <!-- ══ THE FULL SET, collapsed ══
         Still here, still complete, still with its scrubber, chapter marks
         and fullscreen -- just no longer the thing you have to get past.
         preload="none" is load-bearing: a <details> that is closed still
         builds its contents, so without it every report would pull the whole
         recording down to show a viewport nobody opened. -->
    <details id="fullVideoDrawer" class="rounded-3xl bg-panel border border-white/[0.06] overflow-hidden group">
      <summary class="px-5 py-4 cursor-pointer flex items-center gap-2 text-[11px] font-bold text-white/45 hover:text-white/75 transition list-none">
        <svg class="transition-transform group-open:rotate-90 shrink-0" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>
        Show full uncut workout recording
        <span class="ml-auto text-[10px] font-normal text-white/25 tabular-nums">${fmtTime(d.session_duration_s)}</span>
      </summary>

      <!-- The canvas is a SIBLING of the video inside this stage, and the
           stage is what goes fullscreen. Fullscreening the <video> itself
           hands the browser its own player and the overlay disappears with
           the rest of the page. -->
      <div id="stage" class="relative bg-black flex items-center justify-center">
        <video id="player" src="${url}" preload="none" playsinline class="w-full max-h-[70vh] object-contain block"></video>
        <!-- Positioned by fit(), not by inset-0: in fullscreen the video is
             letterboxed inside the stage, and a canvas pinned to the stage
             would put every box in the wrong place. -->
        <canvas id="overlay" class="absolute left-0 top-0 pointer-events-none"></canvas>
      </div>

      <div class="flex items-center gap-2 px-3 pt-3">
        <button id="focusBtn" class="flex-1 sm:flex-none px-3.5 py-2.5 rounded-xl text-[11px] font-bold transition active:scale-95 whitespace-nowrap"></button>
        <button id="fullBtn" class="flex-1 sm:flex-none px-3.5 py-2.5 rounded-xl text-[11px] font-bold transition active:scale-95 whitespace-nowrap bg-white/[0.06] text-white/55 hover:text-white">
          &#9654; Full set replay
        </button>
      </div>

      <div id="playerBar" class="px-3 pt-2.5 pb-1">
        <!-- Seek bar. h-6 hit area around a 1.5px rail: the rail is what you
             see, the padding is what a thumb can actually land on. -->
        <div id="seekBar" class="relative h-6 flex items-center cursor-pointer select-none touch-none">
          <div class="absolute inset-x-0 h-1.5 rounded-full bg-white/[0.09]"></div>
          <div id="seekFill" class="absolute left-0 h-1.5 rounded-full bg-volt/80" style="width:0%"></div>
          <div id="seekMarks" class="absolute inset-0"></div>
          <div id="seekHead" class="absolute w-3 h-3 rounded-full bg-white shadow -ml-1.5 pointer-events-none" style="left:0%"></div>
        </div>
        <div class="flex items-center gap-2 pb-1">
          <button id="playBtn" class="w-8 h-8 rounded-lg bg-white/[0.08] hover:bg-white/[0.14] flex items-center justify-center transition active:scale-95"></button>
          <span id="playTime" class="text-[10px] tabular-nums text-white/45">0:00.0 / 0:00.0</span>
          <div id="markLegend" class="ml-auto hidden sm:flex items-center gap-2.5 text-[9px] text-white/30"></div>
          <button id="fsBtn" title="Fullscreen" class="w-8 h-8 rounded-lg bg-white/[0.08] hover:bg-white/[0.14] flex items-center justify-center transition active:scale-95">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"/>
            </svg>
          </button>
        </div>
      </div>

      <div id="chipRow" class="p-3 pt-0 flex gap-2 overflow-x-auto scrollbar-none"></div>
    </details>

    <details class="rounded-3xl bg-panel border border-white/[0.06] overflow-hidden">
      <summary class="px-5 py-4 cursor-pointer text-[10px] font-bold tracking-[0.14em] text-white/35 hover:text-white/60 transition">
        LOCAL MEASUREMENTS · ${lm.windows ?? "?"} WINDOWS · ${esc(d.analysis_source || "")}
      </summary>
      <div class="px-5 pb-5 space-y-4">
        ${sparkline(dyn.kinetic_error_score || [], dyn.flagged_frames || [])}
        <div class="grid sm:grid-cols-3 gap-3 text-[11px]">
          ${miniStat("V-JEPA", `${lm.windows ?? "?"} × ${lm.embedding_dim ?? "?"}d`, `${lm.window_seconds ?? "?"}s windows · ${lm.duration_s ?? "?"}s`)}
          ${miniStat("Gym classifier", esc((lm.gym_classifier || {}).dominant || "—"), `conf ${(lm.gym_classifier || {}).mean_confidence ?? "—"} · weak hint only`)}
          ${miniStat("SAM 2.1", esc((lm.segmentation || {}).model || "—"),
                     (lm.segmentation || {}).error ? esc(lm.segmentation.error).slice(0, 60) : `${((lm.segmentation || {}).keyframes || []).length} keyframe(s)`)}
        </div>
        <div class="text-[10px] text-white/25 leading-relaxed">
          Kinetic error is unitless and relative to this clip. A spike means the motion pattern
          departed from its own recent trend — a change, not necessarily a fault.
          ${Object.entries(d.timing_ms || {}).map(([k, v]) => `${esc(k.replace("stage_", ""))} ${v}ms`).join(" · ")}
        </div>
      </div>
    </details>
  `;

  if (multi) renderChapterBar();
  selectChapter(0, false);
  // Default state for a local-only report. The chunked path overrides this
  // with "refining" the moment it starts polling, so this is what the
  // whole-file path shows when Gemini was unreachable.
  if (offline) setBadge("failed", d.analysis_source);
  // Speak the cue the moment the report lands -- this is the payoff for the
  // fast finalize: the athlete hears it before racking the bar.
  speak((TIMELINE[0] || d).actionable_cue);
  $("report").classList.remove("hidden");
  wirePlayer();
  $("report").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* Chapters as proportional segments of a single bar, so the visual width of a
   block matches how long it actually took. Rest gaps are drawn too -- a
   session with two long rests should look like one. */
/* Chapter pills, replacing a proportional segmented bar.

   The bar showed each chapter sized by duration, which is honest and was
   almost unusable: a 12-second block between two long ones became a 6%-wide
   sliver with its name clipped to "Ba...", and it was also the tap target.
   A session has two to four chapters, not forty, so there is no need to
   compress them -- each gets a full-width-enough pill carrying the name, the
   score and the reps, and the durations move to a line underneath where they
   are read rather than tapped. */
function renderChapterBar() {
  const bar = $("chapterBar");
  if (!bar) return;
  bar.innerHTML = TIMELINE.map((c, i) => {
    const s = c.form_score;
    const tone = s == null ? "text-white/50" : s >= 80 ? "text-volt"
               : s >= 60 ? "text-warn" : "text-crit";
    const dot = s == null ? "bg-white/25" : s >= 80 ? "bg-volt"
              : s >= 60 ? "bg-warn" : "bg-crit";
    return `<button data-chapter="${i}"
      class="chapter shrink-0 flex items-center gap-2 pl-3 pr-3.5 py-2.5 rounded-xl
             border text-left transition active:scale-95 whitespace-nowrap">
      <span class="w-1.5 h-1.5 rounded-full ${dot} shrink-0"></span>
      <span class="text-[10px] font-bold text-white/30 tabular-nums">${i + 1}</span>
      <span class="text-[12px] font-bold capitalize truncate max-w-[190px]">${esc(c.exercise_name)}</span>
      <span class="text-[12px] font-black tabular-nums ${tone}">${s == null ? "&ndash;" : s}</span>
      <span class="text-[10px] text-white/25">&middot; ${c.total_reps || 0} reps</span>
    </button>`;
  }).join("");

  $("chapterTicks").innerHTML = TIMELINE.map((c, i) =>
    `<span class="tabular-nums">${i + 1}. ${fmtTime(c.time_range[0])}&ndash;${fmtTime(c.time_range[1])}</span>`
  ).join("");

  bar.querySelectorAll(".chapter").forEach(b => {
    b.onclick = () => selectChapter(parseInt(b.dataset.chapter, 10), true);
  });
  paintChapterPills();
}

/* Selection state lives here rather than in renderChapterBar so switching
   chapters does not rebuild the pills -- rebuilding them mid-tap loses the
   element under the finger. */
function paintChapterPills() {
  document.querySelectorAll(".chapter").forEach(b => {
    const on = parseInt(b.dataset.chapter, 10) === ACTIVE;
    b.className = b.className
      .replace(/\s*(bg-white\/\[0\.10\]|bg-white\/\[0\.03\]|border-white\/\[0\.18\]|border-white\/\[0\.06\]|opacity-\d+)/g, "")
      + (on ? " bg-white/[0.10] border-white/[0.18]"
            : " bg-white/[0.03] border-white/[0.06] opacity-60");
  });
}

/* Switch the card, chips, proof panel and overlay box to one chapter.
   `seek` is false on first render so loading a report does not autoplay. */
function selectChapter(i, seek) {
  if (!TIMELINE.length) { renderSummary(DATA, null); PROOF = null; return; }
  ACTIVE = Math.max(0, Math.min(i, TIMELINE.length - 1));
  const c = TIMELINE[ACTIVE];

  const vp = c.visual_proof;
  PROOF = vp ? {
    t: Number.isFinite(vp.timestamp_s) ? vp.timestamp_s : toSeconds(vp.timestamp),
    box: Array.isArray(vp.normalized_bbox) && vp.normalized_bbox.length === 4
         ? vp.normalized_bbox.map(clamp01) : null,
    /* [x, y] pairs from SAM's mask, NOT the box's [ymin, xmin, ...] order.
       Three points is a triangle, not an outline, so anything shorter falls
       back to the rectangle rather than drawing a shard. */
    contour: Array.isArray(vp.normalized_contour) && vp.normalized_contour.length >= 3
             ? vp.normalized_contour.map(pt => [clamp01(pt[0]), clamp01(pt[1])]) : null,
    center: Array.isArray(vp.target_joint_center) && vp.target_joint_center.length === 2
            ? vp.target_joint_center.map(clamp01) : null,
    /* The isolated clip and WHERE INSIDE IT the flagged instant sits. rel is
       1.5 only when the moment could be centred -- a fault in the first
       second and a half of a recording cannot be, and assuming 1.5 there
       would draw the contour over the wrong frame of the clip. */
    axisDeg: Number.isFinite(vp.region_axis_deg) ? vp.region_axis_deg : null,
    elongation: Number.isFinite(vp.region_elongation) ? vp.region_elongation : null,
    clip: vp.flaw_clip_url || null,
    rel: Number.isFinite(vp.flaw_relative_ts) ? vp.flaw_relative_ts : null,
    object: vp.target_object, issue: vp.issue
  } : null;
  if (PROOF && PROOF.t == null) PROOF = null;
  MARKS = buildMarks(c);
  renderSeekMarks();

  paintChapterPills();

  renderSummary(DATA, c);
  renderHero(c);
  renderCoach(c);
  renderChat();
  renderChips(c);
  renderComparison(c);

  const v = $("player");
  if (seek && v) {
    v.currentTime = c.time_range[0];
    v.play().catch(() => {});
  }
}

/* ─────────────── the hero card and the coaching card ─────────────── */

/* Rebuilt per chapter rather than updated in place: the hero's <video> has a
   different src for each chapter, and swapping src on a live element leaves
   the old frame on screen until the new one decodes -- which reads as the
   wrong chapter's flaw still being shown. */
let heroKey = null;

function renderHero(c) {
  const host = $("heroCard");
  if (!host) return;
  if (!PROOF) { heroKey = null; host.innerHTML = ""; return; }

  const clip = PROOF.clip;
  // No clip and no full recording is nothing to show -- an empty black stage
  // reads as "the analysis found nothing", which is the opposite of true.
  const src = clip || SOURCE_URL;
  if (!src) { heroKey = null; host.innerHTML = ""; return; }
  /* The Gemini upgrade re-renders every chapter card. Rebuilding the hero
     there would tear down a playing <video> and restart the loop from zero
     for a change that never touches it -- the clip and the contour are
     measured in Stages A-C and the upgrade only rewrites words. Rebuild only
     when something the hero actually shows has changed. */
  const key = `${src}|${PROOF.rel}|${JSON.stringify(PROOF.contour)}|${PROOF.object}`;
  if (key === heroKey) return;
  heroKey = key;

  host.innerHTML = `
    <div class="rounded-3xl bg-panel border border-crit/25 overflow-hidden">
      <div class="flex items-center gap-2 px-4 sm:px-5 pt-4 pb-3">
        <span class="relative flex w-2 h-2 shrink-0">
          <span class="absolute inline-flex w-full h-full rounded-full bg-crit animate-halo"></span>
          <span class="relative inline-flex w-2 h-2 rounded-full bg-crit"></span>
        </span>
        <span class="text-[10px] font-black tracking-[0.14em] text-crit">PEAK BREAKDOWN POINT</span>
        <span class="text-[10px] text-white/30 tabular-nums">at ${fmtTime(PROOF.t)}</span>
        <span id="heroTime" class="ml-auto text-[10px] tabular-nums text-white/30"></span>
      </div>

      <div id="heroStage" class="relative bg-black flex items-center justify-center cursor-pointer">
        <video id="hero" src="${esc(src)}" muted playsinline autoplay
               class="w-full max-h-[62vh] object-contain block"></video>
        <canvas id="heroOverlay" class="absolute left-0 top-0 pointer-events-none"></canvas>
      </div>

      <div class="px-4 sm:px-5 pt-3 flex items-center gap-2 flex-wrap">
        <button id="guideBtn"></button>
        <button id="shareBtn"
          class="px-3 py-2 rounded-xl text-[11px] font-bold transition active:scale-95
                 whitespace-nowrap bg-white/[0.07] text-white/60 hover:text-white">
          &#128247; Share card
        </button>
      </div>

      <div class="px-4 sm:px-5 py-3 flex items-center gap-2 flex-wrap text-[10px] text-white/35">
        ${clip
          ? `<span class="px-2 py-1 rounded-md bg-crit/12 text-crit font-bold">3s isolated &middot; 0.75x &middot; looping</span>`
          : `<span class="px-2 py-1 rounded-md bg-white/[0.07] text-white/50 font-bold">looping the full video &middot; 0.75x</span>
             <span>no isolated clip for this chapter</span>`}
        <span class="ml-auto">tap to pause</span>
      </div>
    </div>`;

  wireHero(clip);
  paintGuideBtn();
  if ($("guideBtn")) $("guideBtn").onclick = toggleGuide;
  if ($("shareBtn")) $("shareBtn").onclick = exportShareCard;
}

/* The coaching card. Three fields because they answer three questions, and a
   single prose blob reliably collapsed into the first of them.

   `corrective_drill` is null whenever Gemini has not answered: nothing local
   looked at the picture, and a drill prescribed from a spike in latent
   kinetic error would be a confident correction for a fault that may not
   exist. The card says it is waiting rather than filling the space. */
function renderCoach(c) {
  const host = $("coachCard");
  if (!host) return;
  const bd = c.mechanical_breakdown, cue = c.actionable_cue, drill = c.corrective_drill;
  if (!bd && !cue && !drill) { host.innerHTML = ""; return; }
  const pending = c.coaching_source !== "gemini";

  const row = (tone, kicker, body, note) => `
    <div class="p-4 sm:p-5 ${tone.wrap}">
      <div class="flex items-center gap-2 mb-1.5">
        <span class="w-1 h-3.5 rounded-full ${tone.bar}"></span>
        <span class="text-[9px] font-black tracking-[0.14em] ${tone.text}">${kicker}</span>
        ${note ? `<span class="ml-auto text-[9px] text-white/25">${esc(note)}</span>` : ""}
      </div>
      <div class="text-[13px] leading-relaxed ${tone.body} pl-3">${esc(body)}</div>
    </div>`;

  host.innerHTML = `
    <div class="rounded-3xl bg-panel border border-white/[0.06] overflow-hidden divide-y divide-white/[0.05]">
      ${bd ? row({ wrap: "", bar: "bg-crit", text: "text-crit",
                   body: "text-white/75" },
                 "MECHANICAL BREAKDOWN", bd,
                 pending ? "measured, not yet reviewed" : "") : ""}
      ${cue ? row({ wrap: "bg-volt/[0.04]", bar: "bg-volt", text: "text-volt",
                    body: "text-volt font-bold text-[14px]" },
                  "CUE FOR YOUR NEXT SET", cue, "") : ""}
      ${drill
        ? row({ wrap: "", bar: "bg-white/40", text: "text-white/50",
                body: "text-white/70" }, "CORRECTIVE DRILL", drill, "")
        : `<div class="p-4 sm:p-5">
             <div class="flex items-center gap-2 mb-1.5">
               <span class="w-1 h-3.5 rounded-full bg-white/15"></span>
               <span class="text-[9px] font-black tracking-[0.14em] text-white/30">CORRECTIVE DRILL</span>
             </div>
             <div class="text-[12px] text-white/30 leading-relaxed pl-3 italic">
               Waiting on the AI coach. Nothing on this machine looked at the
               picture, and a drill prescribed from the movement signal alone
               would be a guess dressed as advice.
             </div>
           </div>`}
    </div>`;
}

function renderSummary(d, c) {
  const score = Number.isFinite((c || d).form_score) ? (c || d).form_score : null;
  const name = (c || d).exercise_name || d.activity_detected || "unidentified";
  const reps = c ? c.total_reps : d.total_reps;
  const conf = c ? c.rep_confidence : (d.local_measurements || {}).repetitions?.confidence;
  const cue = (c || d).actionable_cue;
  const range = c ? `${fmtTime(c.time_range[0])} – ${fmtTime(c.time_range[1])}` : "";

  $("summaryCard").innerHTML = `
    <div class="rounded-3xl bg-panel border border-white/[0.06] p-4 sm:p-5">
      <div class="flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-5">
        <div class="flex items-center gap-4">
          ${scoreRing(score)}
          <div class="min-w-0">
            <!-- The movement name is the card's identity and was a 11px pill,
                 smaller than the rep count beneath it. Promoted to a heading
                 with the pill demoted to a category tag above it. -->
            <span class="inline-block px-2 py-0.5 rounded bg-volt/15 text-volt text-[9px] font-black tracking-[0.12em] uppercase">
              Exercise
            </span>
            <h2 class="text-[17px] sm:text-[19px] font-black capitalize leading-tight mt-1 truncate max-w-[260px]">${esc(name)}</h2>
            <div class="mt-1.5 flex items-baseline gap-1.5">
              <span class="text-[24px] font-black leading-none tabular-nums">${reps ?? "—"}</span>
              <span class="text-[11px] text-white/40 font-bold">reps</span>
            </div>
            <div class="text-[10px] text-white/30 mt-0.5">${esc(conf || "n/a")} confidence${range ? ` · ${range}` : ""}</div>
          </div>
        </div>
        ${cue ? `
        <div class="flex-1 min-w-0 sm:border-l sm:border-white/[0.08] sm:pl-5 pt-3 sm:pt-0 border-t sm:border-t-0 border-white/[0.06]">
          <div class="text-[9px] font-bold tracking-[0.14em] text-volt/70 mb-1.5">ACTIONABLE CUE</div>
          <div class="text-[14px] font-bold text-volt leading-snug">${esc(cue)}</div>
        </div>` : ""}
      </div>
    </div>`;
}

function renderChips(c) {
  const chips = [];
  const best = c.best_rep_timestamp || (c.best_rep || {}).timestamp;
  const worst = c.worst_rep_timestamp || (c.worst_rep || {}).timestamp;
  if (best)  chips.push({ label: "Jump to Best Rep",  t: toSeconds(best),  cls: "bg-volt/15 text-volt border-volt/25" });
  if (worst) chips.push({ label: "Jump to Worst Rep", t: toSeconds(worst), cls: "bg-crit/15 text-crit border-crit/25" });
  if (PROOF) chips.push({ label: "Jump to Worst Moment", t: PROOF.t, cls: "bg-warn/15 text-warn border-warn/25" });

  $("chipRow").innerHTML = chips.filter(x => x.t != null).map(x =>
    `<button data-seek="${x.t}" class="seek shrink-0 px-3.5 py-2 rounded-xl border text-[11px] font-bold whitespace-nowrap transition hover:brightness-125 active:scale-95 ${x.cls}">
       ${x.label} <span class="opacity-60 tabular-nums">${fmtTime(x.t)}</span>
     </button>`).join("");
  bindSeeks();
}

/* Swap in the narrated report WITHOUT touching the <video> element.
   Re-rendering the whole report would recreate the player, resetting playback
   to zero and dropping whatever the user was watching -- the one thing a
   background upgrade must never do. Only the four dynamic panels re-render;
   the player node is left completely alone, so currentTime, play state and
   fullscreen all survive by construction. */
function applyUpgrade(d) {
  DATA = d;
  TIMELINE = d.timeline || [];
  if (TIMELINE.length > 1) renderChapterBar();
  selectChapter(Math.min(ACTIVE, Math.max(TIMELINE.length - 1, 0)), false);
  setBadge("done");
  // Only voiced if the refined cue actually differs from the one already
  // spoken; speak() dedupes, so an unchanged cue stays silent.
  speak((TIMELINE[ACTIVE] || DATA).actionable_cue);
}

const BADGES = {
  refining: { cls: "border-volt/25 bg-volt/[0.07] text-volt",
              text: "⚡ Fast local report ready — AI Coach refining…", spin: true },
  done:     { cls: "border-volt/25 bg-volt/[0.07] text-volt",
              text: "✓ AI Coach analysis applied", spin: false },
  failed:   { cls: "border-warn/25 bg-warn/[0.07] text-warn",
              text: "Local report only — AI Coach unavailable", spin: false },
  stalled:  { cls: "border-white/10 bg-white/[0.03] text-white/45",
              text: "Local report only — AI Coach did not respond", spin: false },
};

function setBadge(state, detail) {
  const el = $("upgradeBadge");
  if (!el) return;
  const b = BADGES[state];
  if (!b) { el.innerHTML = ""; return; }
  el.innerHTML = `
    <div class="flex items-center gap-2.5 rounded-2xl border ${b.cls} px-4 py-2.5">
      ${b.spin ? `<span class="relative flex w-2 h-2 shrink-0">
        <span class="absolute inline-flex w-full h-full rounded-full bg-volt animate-halo"></span>
        <span class="relative inline-flex w-2 h-2 rounded-full bg-volt"></span></span>` : ""}
      <span class="text-[11px] font-bold">${esc(b.text)}</span>
      ${detail ? `<span class="text-[10px] opacity-50 truncate">${esc(String(detail).slice(0, 80))}</span>` : ""}
    </div>`;
}

/* ─────────────────────────── audio coaching ─────────────────────────── */
/* speechSynthesis, not the server TTS. /coach/speak calls Gemini's audio
   model, which costs a round trip measured in seconds and can 503 -- and the
   entire point of the 1.6 s finalize is that the athlete hears the cue while
   still standing at the bar. The browser voice is offline, free and instant.
   The Gemini-refined cue replaces it later, when it exists. */
const AUDIO_KEY = "havehit.muted";
let muted = localStorage.getItem(AUDIO_KEY) === "1";
let lastSpoken = null;

/* ── mobile autoplay unlock ──
   iOS Safari and Chrome on Android refuse synthetic speech that does not
   originate in a user gesture, and the refusal is SILENT: speak() resolves,
   nothing is heard, no error is thrown. Since the cue is spoken from a fetch
   callback about a second after Stop, it is never inside a gesture and would
   be blocked every time on a phone.

   The fix is to run one throwaway utterance from inside a real tap, which
   marks the engine as user-activated for the rest of the page's life.

   A SPACE at volume 0, not an empty string. An empty utterance is dropped
   before it reaches the engine on WebKit, so it never performs the activation
   it was added to perform -- the call succeeds and changes nothing, which is
   the most expensive kind of fix. A single space is real content the engine
   accepts, and volume 0 keeps it inaudible.

   resume() is here because iOS parks the queue when the tab is backgrounded
   (locking the phone mid-set does exactly this) and never restarts it on its
   own; without this the first cue after unlocking the screen is swallowed. */
let audioUnlocked = false;

function primeSpeech() {
  if (!("speechSynthesis" in window)) return;
  try {
    if (!audioUnlocked) {
      const silent = new SpeechSynthesisUtterance(" ");
      silent.volume = 0;
      window.speechSynthesis.speak(silent);
      audioUnlocked = true;
    }
    window.speechSynthesis.resume();
  } catch { /* engine missing or refusing: the pill stays off, nothing breaks */ }
  updateVoicePill();
}

/* The pill claims the voice coach is ACTIVE, so it must only be shown when
   speech would genuinely be heard: engine present, primed by a gesture, not
   muted. Anything looser is an indicator that lies during the one situation
   it exists for. */
function updateVoicePill() {
  const pill = $("voicePill");
  if (!pill) return;
  const live = audioUnlocked && !muted && ("speechSynthesis" in window);
  pill.classList.toggle("hidden", !live);
  pill.classList.toggle("flex", live);
}

/* Explicit "does my earbud hear this" check, before a set rather than after.
   It un-mutes rather than speaking around the mute: pressing a speaker button
   is an unambiguous request for audio, and leaving the toggle off afterwards
   would mean the test passed and the actual cues still arrived in silence. */
function testAudio() {
  primeSpeech();
  if (muted) setMuted(false);
  if (!("speechSynthesis" in window)) {
    showError("This browser has no speech engine, so coaching audio is unavailable.");
    return;
  }
  speak("Voice coach ready. Keep your chest up and drive through the floor.",
        { force: true });
}

function speak(text, { force = false } = {}) {
  if (!text || muted || !("speechSynthesis" in window)) return;
  // The upgrade often returns a cue that means the same thing. Re-speaking an
  // identical string mid-set is noise, so only a genuine change is voiced.
  if (!force && text === lastSpoken) return;
  lastSpoken = text;
  try {
    window.speechSynthesis.cancel();          // drop any queued older cue
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.02; u.pitch = 1.0; u.volume = 1.0;
    u.lang = "en-GB";
    window.speechSynthesis.speak(u);
  } catch { /* unsupported engine: silence is an acceptable degradation */ }
}

function setMuted(v) {
  muted = v;
  localStorage.setItem(AUDIO_KEY, v ? "1" : "0");
  if (v && "speechSynthesis" in window) window.speechSynthesis.cancel();
  updateVoicePill();
  const btn = $("muteBtn");
  if (!btn) return;
  btn.setAttribute("aria-pressed", String(!v));
  btn.title = v ? "Coaching audio off" : "Coaching audio on";
  btn.innerHTML = v
    ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
         <path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="m23 9-6 6M17 9l6 6"/></svg>`
    : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
         <path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a9 9 0 0 1 0 14"/></svg>`;
  btn.className = `w-8 h-8 rounded-lg flex items-center justify-center transition ${
    v ? "bg-white/[0.06] text-white/35" : "bg-volt/15 text-volt"}`;
}

function bindSeeks() {
  document.querySelectorAll(".seek").forEach(b => {
    b.onclick = () => {
      const t = parseFloat(b.dataset.seek);
      const v = $("player");
      if (v && Number.isFinite(t)) { v.currentTime = t; v.play().catch(() => {}); }
    };
  });
}


function scoreRing(score) {
  const s = score == null ? 0 : score;
  const col = s >= 80 ? "#C6FF00" : s >= 60 ? "#FFB020" : "#FF4D4D";
  const circ = 2 * Math.PI * 30;
  return `<svg width="76" height="76" viewBox="0 0 76 76" class="shrink-0 -rotate-90">
      <circle cx="38" cy="38" r="30" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="7"/>
      <circle cx="38" cy="38" r="30" fill="none" stroke="${col}" stroke-width="7" stroke-linecap="round"
        stroke-dasharray="${circ}" stroke-dashoffset="${circ * (1 - s / 100)}"/>
      <text x="38" y="38" transform="rotate(90 38 38)" text-anchor="middle" dominant-baseline="central"
        fill="#fff" font-size="21" font-weight="900" font-family="Geist, system-ui, sans-serif">${score == null ? "—" : score}</text>
    </svg>`;
}

function miniStat(label, value, sub) {
  return `<div class="rounded-xl bg-panel2 border border-white/[0.05] p-3">
    <div class="text-[9px] font-bold tracking-[0.12em] text-white/30 mb-1">${label}</div>
    <div class="text-[13px] font-bold truncate">${value}</div>
    <div class="text-[10px] text-white/30 truncate">${sub}</div>
  </div>`;
}

/* Kinetic error over time, with the flagged windows marked so the flaw
   timestamps line up visually with the curve that produced them. */
function sparkline(series, flags) {
  if (!series.length) return "";
  const W = 700, H = 54;
  const max = Math.max(...series.map(Math.abs), 1);
  const pts = series.map((v, i) =>
    `${(i / Math.max(series.length - 1, 1)) * W},${H - (Math.abs(v) / max) * (H - 8) - 4}`).join(" ");
  const marks = (flags || []).map(f => {
    const x = (f.window / Math.max(series.length, 1)) * W;
    const col = f.severity === "critical" ? "#FF4D4D" : f.severity === "warning" ? "#FFB020" : "#8A8A8A";
    return `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="${col}" stroke-width="1.5" stroke-dasharray="3 3" opacity=".8"/>`;
  }).join("");
  return `<div>
    <div class="text-[9px] font-bold tracking-[0.12em] text-white/30 mb-2">KINETIC ERROR OVER TIME</div>
    <svg viewBox="0 0 ${W} ${H}" class="w-full h-[54px]" preserveAspectRatio="none">
      ${marks}<polyline points="${pts}" fill="none" stroke="#C6FF00" stroke-width="1.5" opacity=".85"/>
    </svg>
  </div>`;
}

/* ─────────── player: transport, scrubber, canvas overlay ─────────── */

/* The proof box is drawn only while the playhead is inside this window of the
   flagged moment, and is HARD ZERO outside it -- the box exists to say "look
   here, now", and one that lingers is just decoration over unrelated footage.
   Within the window it ramps rather than snapping: at 30 fps a hard edge both
   flickers while scrubbing and can fall entirely between two rendered frames. */
const PROOF_WINDOW = 0.40;    // full visibility ends here, alpha reaches 0 here
const PROOF_CORE = 0.25;      // full opacity out to here, then ramps down

function alphaFor(dt) {
  if (dt <= PROOF_CORE) return 1;
  if (dt >= PROOF_WINDOW) return 0;
  return 1 - (dt - PROOF_CORE) / (PROOF_WINDOW - PROOF_CORE);
}

/* Focus Loop: replay a 3 s window around the flagged moment, slowed down.
   1.5 s either side is enough to see the approach, the fault and the recovery
   -- the fault itself is a moment, but a moment with no run-up is unreadable.
   0.75x because at full speed a knee caving in occupies about four frames. */
const FOCUS_PAD = 1.5;
const FOCUS_RATE = 0.75;
let FOCUS = false;

/* The flaw palette, shared by the player overlay and the still panels in the
   comparison card. Defined once at module scope because those two used to
   carry their own copies and drifted apart -- the same joint was outlined in
   two different reds depending on which element you were looking at.

   Red, not the amber this overlay originally used. The seek-bar tick for the
   proof moment was already red while the box over that same moment was amber,
   so the two marks for one thing disagreed. Amber still means "worst REP",
   which is a different claim from "this is the fault". */
const FLAW_FILL = "rgba(244, 63, 94, 0.25)";
const FLAW_LINE = "rgba(244, 63, 94, 0.95)";
const FLAW_GLOW = "#f43f5e";

const ICON_PLAY = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
const ICON_PAUSE = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>`;

/* ────────────── shared flaw renderer: contour, glow, anatomical pin ──────────────

   Extracted out of wirePlayer because there are now TWO surfaces that draw the
   same shape over the same moment -- the 3-second hero clip and the full-set
   player behind the accordion. Two copies of this would drift, and the failure
   mode is quiet: the same joint outlined in two slightly different places
   depending on which one you were looking at. */

function videoBoxIn(rect, vw, vh) {
  /* The video is object-contain, so the picture is letterboxed inside its own
     element. Assuming otherwise offsets every box on portrait footage, which
     is most phone footage. */
  const scale = Math.min(rect.width / (vw || 16), rect.height / (vh || 9));
  const w = (vw || 16) * scale, h = (vh || 9) * scale;
  return { x: (rect.width - w) / 2, y: (rect.height - h) / 2, w, h };
}

function tracePath(ctx, b, proof) {
  /* SAM's contour is [x, y]; the box is [ymin, xmin, ymax, xmax]. Mixing them
     silently draws a shape transposed about the diagonal, which looks
     plausible and is wrong, so the two are converted in separate places and
     never in the same expression. */
  ctx.beginPath();
  if (proof.contour) {
    proof.contour.forEach(([nx, ny], i) => {
      const x = b.x + nx * b.w, y = b.y + ny * b.h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.closePath();
    return true;
  }
  const [ymin, xmin, ymax, xmax] = proof.box;
  ctx.rect(b.x + xmin * b.w, b.y + ymin * b.h,
           (xmax - xmin) * b.w, (ymax - ymin) * b.h);
  return false;
}

function pinPoint(b, proof) {
  if (proof.center) {
    return { x: b.x + proof.center[0] * b.w, y: b.y + proof.center[1] * b.h };
  }
  const [ymin, xmin, ymax, xmax] = proof.box;
  return { x: b.x + ((xmin + xmax) / 2) * b.w,
           y: b.y + ((ymin + ymax) / 2) * b.h };
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* Draw the flaw over one video box. `alpha` is the caller's fade -- the hero
   holds it at 1 near the moment, the full player ramps it across the +/-0.4s
   window. `dim` punches the shape out of a darkened frame; the hero turns it
   off because a 3-second clip IS the flaw and dimming its surroundings just
   makes a small viewport darker. */
function paintFlaw(ctx, rect, vw, vh, proof, alpha, { dim = true, badge = true } = {}) {
  if (!proof || !proof.box || alpha <= 0) return;
  const b = videoBoxIn(rect, vw, vh);
  ctx.save();
  ctx.globalAlpha = alpha;

  if (dim) {
    // Frame rect plus the shape as a second subpath, filled even-odd, so the
    // cut-out is the real mask outline rather than its bounding rectangle.
    ctx.beginPath();
    ctx.rect(b.x, b.y, b.w, b.h);
    if (proof.contour) {
      proof.contour.forEach(([nx, ny], i) => {
        const x = b.x + nx * b.w, y = b.y + ny * b.h;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.closePath();
    } else {
      const [ymin, xmin, ymax, xmax] = proof.box;
      ctx.rect(b.x + xmin * b.w, b.y + ymin * b.h,
               (xmax - xmin) * b.w, (ymax - ymin) * b.h);
    }
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fill("evenodd");
  }

  // Soft wash, then a glowing edge. Both from one path so the fill and the
  // stroke cannot disagree by a pixel.
  tracePath(ctx, b, proof);
  ctx.fillStyle = FLAW_FILL;
  ctx.fill();
  ctx.shadowColor = FLAW_GLOW;
  ctx.shadowBlur = 10;
  ctx.strokeStyle = FLAW_LINE;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.stroke();
  // Second pass with the shadow off keeps the line crisp; a glow drawn in one
  // pass reads as a blurred line rather than a lit one.
  ctx.shadowBlur = 0;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (!badge) { ctx.restore(); return; }

  const pin = pinPoint(b, proof);
  const label = "⚠ " + String(proof.object || "flagged moment")
                             .replace(/_/g, " ").slice(0, 30);
  ctx.font = "700 12px Geist, system-ui, sans-serif";
  const tw = Math.ceil(ctx.measureText(label).width) + 18;
  const th = 24;

  // Offset up-and-right of the joint, then pulled back inside the frame. A
  // badge that runs off the canvas edge simply never appears, and the one time
  // that matters is a fault at the edge of frame.
  let bx = pin.x + 26, by = pin.y - 40;
  if (bx + tw > b.x + b.w) bx = pin.x - 26 - tw;
  if (bx < b.x) bx = b.x + 4;
  if (by < b.y) by = pin.y + 22;

  ctx.beginPath();
  ctx.moveTo(pin.x, pin.y);
  ctx.lineTo(bx < pin.x ? bx + tw : bx, by + th / 2);
  ctx.strokeStyle = FLAW_LINE;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(pin.x, pin.y, 4, 0, Math.PI * 2);
  ctx.fillStyle = FLAW_LINE;
  ctx.fill();

  roundRectPath(ctx, bx, by, tw, th, 6);
  ctx.fillStyle = "rgba(10,10,10,0.92)";
  ctx.fill();
  ctx.strokeStyle = FLAW_LINE;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.fillText(label, bx + 9, by + 16);
  ctx.restore();
}

/* Size a canvas in DEVICE pixels onto a video's CSS box, and return that box.
   Done every paint so a resize, an orientation change and a fullscreen
   transition all land automatically -- the CSS box is the single source of
   truth and nothing has to be told that the layout moved. */
function fitCanvas(ctx, cv, v, host) {
  const r = v.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  cv.style.width = r.width + "px";
  cv.style.height = r.height + "px";
  if (host) {
    const hr = host.getBoundingClientRect();
    cv.style.left = (r.left - hr.left) + "px";
    cv.style.top = (r.top - hr.top) + "px";
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return r;
}

/* ────────────────── alignment guide ("plumb & axis") ──────────────────

   WHAT THIS IS, AND WHAT IT IS NOT

   The brief asked for a "Pro Reference Ghost": a textbook silhouette scaled
   and pinned to the athlete's detected torso and hip bounds, so they can see
   the gap between their joints and ideal ones.

   That cannot be built here, and building something that LOOKS like it would
   be the worst option of the three. There is no pose model anywhere in this
   system. V-JEPA returns a 768-dimensional embedding with no spatial
   decoding; SAM returns a mask from a bare 4x4 point grid and has, in this
   pipeline's own words, no vocabulary -- it does not know a hip from a
   barbell. "Detected torso/hip bounds" do not exist to pin anything to.

   A neon skeleton labelled "ideal knee angle", drawn over a body nothing
   located, would be a confident fabrication -- and a far more damaging one
   than a wrong sentence, because a diagram drawn over the athlete's own
   footage reads as measurement.

   So this draws the two things that ARE measured, and names them for what
   they are:

     PLUMB      true vertical through the region's centre of mass. Not a
                claim about the athlete at all -- it is gravity, and it is
                the reference a coach actually uses by eye.
     AXIS       the long axis of the segmented region, from second-order
                central image moments. Exact to the mask, and honest about
                being the mask's axis rather than a spine.
     GAP        the angle between them, in degrees. This is the "angle gap"
                the brief wanted, and it is a real measurement.

   It declines to draw on a near-round region, where an axis exists in the
   arithmetic and nowhere else. */

const GUIDE_LINE = "rgba(6, 182, 212, 0.95)";
const GUIDE_SOFT = "rgba(6, 182, 212, 0.45)";
const GUIDE_GLOW = "#06b6d4";

// Below this elongation the mask is effectively round and its "long axis" is
// numerical noise. Measured on synthetic shapes: a bar scores ~6-8, a circle
// returns no axis at all, and the ambiguous band sits just above 1.
const GUIDE_MIN_ELONGATION = 1.25;

let GUIDE = false;

function guideAvailable() {
  return !!(PROOF && PROOF.center && Number.isFinite(PROOF.axisDeg)
            && (PROOF.elongation == null
                || PROOF.elongation >= GUIDE_MIN_ELONGATION));
}

function paintGuide(ctx, rect, vw, vh) {
  if (!GUIDE || !guideAvailable()) return;
  const b = videoBoxIn(rect, vw, vh);
  const cx = b.x + PROOF.center[0] * b.w;
  const cy = b.y + PROOF.center[1] * b.h;
  // Half-length of both lines: tied to the region's own size so the guide
  // scales with the athlete in frame rather than being a fixed number of
  // pixels that dwarfs a distant subject and vanishes on a close one.
  const [ymin, xmin, ymax, xmax] = PROOF.box;
  const half = Math.max(24, ((ymax - ymin) * b.h) * 0.62);

  ctx.save();
  ctx.lineCap = "round";

  // Plumb: true vertical. Dashed, because it is a reference and not a
  // measurement of the athlete.
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = GUIDE_SOFT;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy - half);
  ctx.lineTo(cx, cy + half);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axis: the measured one. Solid and lit, because it is real.
  const th = (PROOF.axisDeg * Math.PI) / 180;
  const dx = Math.sin(th) * half, dy = -Math.cos(th) * half;
  ctx.shadowColor = GUIDE_GLOW;
  ctx.shadowBlur = 10;
  ctx.strokeStyle = GUIDE_LINE;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx - dx, cy - dy);
  ctx.lineTo(cx + dx, cy + dy);
  ctx.stroke();
  ctx.shadowBlur = 0;

  // The gap, as a number. The whole point of the overlay is this figure --
  // two lines without it are decoration.
  const gap = Math.abs(PROOF.axisDeg).toFixed(1) + "°";
  const label = "region axis " + gap + " from vertical";
  ctx.font = "700 11px Geist, system-ui, sans-serif";
  const tw = Math.ceil(ctx.measureText(label).width) + 16;
  let lx = cx - tw / 2;
  let ly = cy - half - 26;
  if (ly < b.y) ly = cy + half + 6;
  lx = Math.max(b.x + 2, Math.min(lx, b.x + b.w - tw - 2));
  roundRectPath(ctx, lx, ly, tw, 20, 5);
  ctx.fillStyle = "rgba(8,20,24,0.92)";
  ctx.fill();
  ctx.strokeStyle = GUIDE_SOFT;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = GUIDE_LINE;
  ctx.fillText(label, lx + 8, ly + 14);
  ctx.restore();
}

function paintGuideBtn() {
  const btn = $("guideBtn");
  if (!btn) return;
  const can = guideAvailable();
  btn.disabled = !can;
  btn.style.pointerEvents = can ? "auto" : "none";
  btn.style.opacity = can ? "1" : "0.3";
  btn.title = can ? "Vertical reference and the measured region axis"
                  : "No measurable axis for this region";
  btn.innerHTML = "&#9776; Alignment guide";
  btn.className = "px-3 py-2 rounded-xl text-[11px] font-bold transition "
    + "active:scale-95 whitespace-nowrap "
    + (GUIDE ? "bg-cyan-500 text-black"
             : "bg-cyan-500/15 text-cyan-300 border border-cyan-500/35 hover:bg-cyan-500/25");
}

function toggleGuide() {
  GUIDE = !GUIDE;
  paintGuideBtn();
}

/* ─────────────────── shareable breakdown card ───────────────────

   Composited on an off-screen canvas rather than rasterising the DOM: no
   html2canvas, no CDN, no cross-origin taint, and the layout is fixed at
   1080x1920 instead of inheriting whatever the viewport happened to be.

   The two photographs come from the stored keyframes, which are same-origin
   JPEGs served by this app -- that matters, because a canvas that has drawn a
   cross-origin image cannot be read back and toBlob() throws a SecurityError.
   crossOrigin is set anyway so the rule is explicit rather than incidental. */

const CARD_W = 1080, CARD_H = 1920;

function loadShareImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img.naturalWidth ? img : null);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function drawCover(g, img, x, y, w, h) {
  // object-fit: cover, by hand. Letterboxing a portrait keyframe into a
  // landscape panel leaves two black bars where the athlete should be.
  const s = Math.max(w / img.naturalWidth, h / img.naturalHeight);
  const dw = img.naturalWidth * s, dh = img.naturalHeight * s;
  g.save();
  g.beginPath();
  g.rect(x, y, w, h);
  g.clip();
  g.drawImage(img, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh);
  g.restore();
}

function wrapText(g, text, x, y, maxW, lineH, maxLines) {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  let line = "", n = 0;
  for (let i = 0; i < words.length && n < maxLines; i++) {
    const test = line ? line + " " + words[i] : words[i];
    if (g.measureText(test).width > maxW && line) {
      g.fillText(line, x, y + n * lineH);
      n++;
      line = words[i];
    } else {
      line = test;
    }
  }
  if (line && n < maxLines) { g.fillText(line, x, y + n * lineH); n++; }
  return y + n * lineH;
}

async function exportShareCard() {
  const btn = $("shareBtn");
  const c = TIMELINE[ACTIVE];
  if (!c || !PROOF) return;
  const sid = (DATA || {}).session_id;
  if (btn) { btn.disabled = true; btn.textContent = "Building…"; }

  try {
    const cv = document.createElement("canvas");
    cv.width = CARD_W; cv.height = CARD_H;
    const g = cv.getContext("2d");

    g.fillStyle = "#0A0A0A";
    g.fillRect(0, 0, CARD_W, CARD_H);
    const glow = g.createRadialGradient(CARD_W / 2, 120, 40, CARD_W / 2, 120, 760);
    glow.addColorStop(0, "rgba(198,255,0,0.13)");
    glow.addColorStop(1, "rgba(198,255,0,0)");
    g.fillStyle = glow;
    g.fillRect(0, 0, CARD_W, 900);

    // ---- header ----
    g.fillStyle = "#C6FF00";
    roundRectPath(g, 72, 96, 52, 52, 16);
    g.fill();
    g.strokeStyle = "#0A0A0A"; g.lineWidth = 6; g.lineCap = "round";
    g.beginPath();
    g.moveTo(86, 110); g.lineTo(86, 134); g.moveTo(110, 110); g.lineTo(110, 134);
    g.moveTo(86, 122); g.lineTo(110, 122);
    g.stroke();
    g.fillStyle = "#FFFFFF";
    g.font = "900 34px Geist, system-ui, sans-serif";
    g.fillText("HAVE HIT", 142, 133);
    g.fillStyle = "rgba(255,255,255,0.35)";
    g.font = "700 22px Geist, system-ui, sans-serif";
    g.fillText("AI MOVEMENT COACH", 142, 163);

    // ---- exercise + score ----
    g.fillStyle = "#FFFFFF";
    g.font = "900 74px Geist, system-ui, sans-serif";
    const name = String(c.exercise_name || "Movement");
    g.fillText(name.length > 22 ? name.slice(0, 21) + "…" : name, 72, 300);

    const score = Number.isFinite(c.form_score) ? c.form_score : null;
    const tone = score == null ? "#9CA3AF" : score >= 80 ? "#C6FF00"
               : score >= 60 ? "#FFB020" : "#FF4D4D";
    g.font = "900 130px Geist, system-ui, sans-serif";
    const st = score == null ? "--" : String(score);
    const sw = g.measureText(st).width;
    g.fillStyle = tone;
    g.fillText(st, CARD_W - 72 - sw, 300);
    g.font = "800 22px Geist, system-ui, sans-serif";
    g.fillStyle = "rgba(255,255,255,0.35)";
    const lw = g.measureText("FORM SCORE").width;
    g.fillText("FORM SCORE", CARD_W - 72 - lw, 336);

    // ---- the two panels ----
    const [imgBest, imgWorst] = await Promise.all([
      sid ? loadShareImage(frameUrl(sid, "best", c.chapter_index)) : null,
      sid ? loadShareImage(frameUrl(sid, "worst", c.chapter_index)) : null,
    ]);

    const PX = 72, PW = CARD_W - PX * 2, PH = 470, GAP = 34;
    const panels = [
      { y: 420, img: imgBest, accent: "#C6FF00", kicker: "BEST REP",
        caption: "Optimal form", contour: null },
      { y: 420 + PH + GAP, img: imgWorst, accent: "#f43f5e",
        kicker: "PEAK BREAKDOWN",
        caption: String(PROOF.object || "flagged moment").replace(/_/g, " "),
        contour: PROOF.contour },
    ];

    for (const p of panels) {
      g.fillStyle = "#000000";
      roundRectPath(g, PX, p.y, PW, PH, 28);
      g.fill();
      if (p.img) {
        g.save();
        roundRectPath(g, PX, p.y, PW, PH, 28);
        g.clip();
        drawCover(g, p.img, PX, p.y, PW, PH);
        if (p.contour && p.contour.length >= 3) {
          // The contour is normalised to the FRAME, and the panel is a
          // centre-crop of that frame -- so the same cover transform has to be
          // applied to the polygon, or the outline lands beside the fault.
          const s = Math.max(PW / p.img.naturalWidth, PH / p.img.naturalHeight);
          const dw = p.img.naturalWidth * s, dh = p.img.naturalHeight * s;
          const ox = PX + (PW - dw) / 2, oy = p.y + (PH - dh) / 2;
          g.beginPath();
          p.contour.forEach(([nx, ny], i) => {
            const x = ox + nx * dw, y = oy + ny * dh;
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          });
          g.closePath();
          g.fillStyle = FLAW_FILL; g.fill();
          g.shadowColor = FLAW_GLOW; g.shadowBlur = 26;
          g.strokeStyle = FLAW_LINE; g.lineWidth = 7; g.lineJoin = "round";
          g.stroke();
          g.shadowBlur = 0;
        }
        g.restore();
      } else {
        g.fillStyle = "rgba(255,255,255,0.30)";
        g.font = "700 26px Geist, system-ui, sans-serif";
        g.fillText("keyframe unavailable", PX + 34, p.y + PH / 2);
      }
      g.strokeStyle = p.accent + "66";
      g.lineWidth = 3;
      roundRectPath(g, PX, p.y, PW, PH, 28);
      g.stroke();

      g.fillStyle = "rgba(0,0,0,0.78)";
      const kx = PX + 24, ky = p.y + 24;
      g.font = "900 24px Geist, system-ui, sans-serif";
      const kw = g.measureText(p.kicker).width + 32;
      roundRectPath(g, kx, ky, kw, 46, 12);
      g.fill();
      g.strokeStyle = p.accent + "88"; g.lineWidth = 2;
      roundRectPath(g, kx, ky, kw, 46, 12); g.stroke();
      g.fillStyle = p.accent;
      g.fillText(p.kicker, kx + 16, ky + 32);

      g.fillStyle = "rgba(0,0,0,0.78)";
      g.font = "700 24px Geist, system-ui, sans-serif";
      const cw = g.measureText(p.caption).width + 32;
      const cyy = p.y + PH - 70;
      roundRectPath(g, kx, cyy, Math.min(cw, PW - 48), 46, 12);
      g.fill();
      g.fillStyle = "rgba(255,255,255,0.88)";
      g.fillText(p.caption, kx + 16, cyy + 32);
    }

    // ---- the cue ----
    const cueY = 420 + PH * 2 + GAP + 70;
    g.fillStyle = "rgba(198,255,0,0.08)";
    roundRectPath(g, PX, cueY, PW, 210, 28);
    g.fill();
    g.strokeStyle = "rgba(198,255,0,0.28)"; g.lineWidth = 3;
    roundRectPath(g, PX, cueY, PW, 210, 28); g.stroke();
    g.fillStyle = "rgba(198,255,0,0.65)";
    g.font = "900 22px Geist, system-ui, sans-serif";
    g.fillText("NEXT-SET CUE", PX + 34, cueY + 52);
    g.fillStyle = "#C6FF00";
    g.font = "800 38px Geist, system-ui, sans-serif";
    wrapText(g, c.actionable_cue || "", PX + 34, cueY + 108, PW - 68, 48, 3);

    g.fillStyle = "rgba(255,255,255,0.22)";
    g.font = "600 22px Geist, system-ui, sans-serif";
    g.fillText("V-JEPA 2 · SAM 2.1 · Gemini", PX, CARD_H - 60);

    const blob = await new Promise(r => cv.toBlob(r, "image/png"));
    if (!blob) throw new Error("canvas produced no image");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `have-hit_breakdown_${sid || "session"}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoked on the next tick, not immediately: Safari has not started the
    // download by the time click() returns and a revoked URL cancels it.
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    if (btn) btn.textContent = "✓ Saved";
  } catch (e) {
    if (btn) btn.textContent = "Export failed";
    console.error("share card:", e);
  } finally {
    setTimeout(() => {
      if (btn) { btn.disabled = false; btn.innerHTML = "&#128247; Share card"; }
    }, 2200);
  }
}

/* ─────────────────────── the flaw hero player ───────────────────────

   THE CHANGE THIS IS: the report used to open on the full uncut recording,
   with the flagged moment reachable only by scrubbing to it. That asks the
   athlete to find the thing the analysis already found. The primary viewport
   is now the three seconds that matter, looping, slowed, with the contour on
   it; the full recording is still there, one tap away, for anyone who wants
   the whole set.

   The clip is a real file from the server, so this works in the history
   drawer where the source recording is long deleted. When there is no clip --
   extraction failed, or an old session whose clip has been evicted -- the
   hero falls back to the full video with a clock-enforced loop over the same
   window. Same picture, more bytes; the fallback is real, not a stub. */

const HERO_RATE = 0.75;
let heroRefit = null;

function wireHero(clipUrl) {
  const v = $("hero"), cv = $("heroOverlay"), host = $("heroStage");
  if (!v || !cv) return;
  const ctx = cv.getContext("2d");
  const fallback = !clipUrl;

  v.playbackRate = HERO_RATE;
  // A native `loop` only wraps at the END of the file, which is exactly right
  // for a clip that IS the window and useless for the fallback, where the
  // window is a range inside a longer video.
  v.loop = !fallback;

  function windowFor() {
    if (!PROOF || !Number.isFinite(PROOF.t)) return null;
    const d = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : Infinity;
    return [Math.max(0, PROOF.t - FOCUS_PAD), Math.min(d, PROOF.t + FOCUS_PAD)];
  }

  /* Where the flagged instant sits on THIS element's clock. In the clip that
     is flaw_relative_ts, which is 1.5s only when the moment could be centred
     -- a fault in the first second and a half of a recording cannot be, and
     assuming 1.5 there draws the contour over the wrong frame. */
  function markAt() {
    if (fallback) return PROOF ? PROOF.t : null;
    return PROOF && Number.isFinite(PROOF.rel) ? PROOF.rel : null;
  }

  function hold() {
    if (!fallback || v.seeking) return;
    const w = windowFor();
    if (!w) return;
    if (v.currentTime > w[1] || v.currentTime < w[0] - 0.25) v.currentTime = w[0];
  }

  function paint() {
    hold();
    const r = fitCanvas(ctx, cv, v, host);
    ctx.clearRect(0, 0, r.width, r.height);
    const mark = markAt();
    if (mark == null) return;
    /* The clip is three seconds long and every frame of it is the flagged
       moment's neighbourhood, so the contour holds through a wider band than
       the full player's +/-0.4s -- it would otherwise blink on for a tenth of
       a loop and be missed. It still fades at the edges, because the contour
       was measured on ONE frame and pretending it tracks the whole clip would
       be a claim the pipeline never made. */
    const dt = Math.abs(v.currentTime - mark);
    const a = fallback ? alphaFor(dt)
                       : (dt <= 0.5 ? 1 : dt >= 1.1 ? 0.22 : 1 - (dt - 0.5) / 0.6 * 0.78);
    paintFlaw(ctx, r, v.videoWidth, v.videoHeight, PROOF, a,
              { dim: false });
    // Drawn after, and at full opacity: the guide is a fixed reference and
    // should not fade in and out with the contour it is measured against.
    paintGuide(ctx, r, v.videoWidth, v.videoHeight);
    const badge = $("heroTime");
    if (badge) {
      badge.textContent = fallback
        ? `${fmtTime(v.currentTime)} · full video`
        : `${v.currentTime.toFixed(1)}s / ${(v.duration || 3).toFixed(1)}s · ${HERO_RATE}x`;
    }
  }

  ["timeupdate", "seeked", "seeking", "loadedmetadata", "durationchange",
   "play", "pause", "ratechange"].forEach(ev => v.addEventListener(ev, paint));
  // playbackRate is reset by some browsers when a new source loads, so it is
  // reapplied rather than set once.
  v.addEventListener("loadedmetadata", () => {
    v.playbackRate = HERO_RATE;
    if (fallback) { const w = windowFor(); if (w) v.currentTime = w[0]; }
  });

  let raf = 0;
  (function loop() { paint(); raf = requestAnimationFrame(loop); })();
  const obs = new MutationObserver(() => {
    if (!document.body.contains(cv)) { cancelAnimationFrame(raf); obs.disconnect(); }
  });
  obs.observe($("report"), { childList: true, subtree: true });

  /* One listener pair for the hero, replaced rather than added to. Chapter
     switches rebuild this element, and without the removal every switch left
     another closure painting a canvas that is no longer in the document. */
  if (heroRefit) {
    window.removeEventListener("resize", heroRefit);
    window.removeEventListener("orientationchange", heroRefit);
  }
  heroRefit = () => paint();
  window.addEventListener("resize", heroRefit);
  window.addEventListener("orientationchange", heroRefit);

  const play = () => { v.playbackRate = HERO_RATE; v.play().catch(() => {}); };
  play();
  // Autoplay is refused on some mobile browsers even muted; the first tap
  // anywhere on the hero starts it, and the button says so until it does.
  host && host.addEventListener("click", () => (v.paused ? play() : v.pause()));
  paint();
}

function wirePlayer() {
  const v = $("player"), cv = $("overlay"), stage = $("stage");
  if (!v || !cv) return;
  const ctx = cv.getContext("2d");

  /* ── transport ── */
  const playBtn = $("playBtn"), fsBtn = $("fsBtn");
  const bar = $("seekBar"), fill = $("seekFill"), head = $("seekHead");

  function syncPlayIcon() {
    if (playBtn) playBtn.innerHTML = v.paused ? ICON_PLAY : ICON_PAUSE;
  }
  if (playBtn) playBtn.onclick = () => (v.paused ? v.play().catch(() => {}) : v.pause());
  // Opening the full recording and pressing play means you want to watch the
  // set, not the loop. Both are muted so there is no audio clash, but two
  // videos moving in one viewport is noise.
  v.addEventListener("play", () => { const h = $("hero"); if (h) h.pause(); });
  v.addEventListener("play", syncPlayIcon);
  v.addEventListener("pause", syncPlayIcon);
  syncPlayIcon();

  /* Fullscreen, in the order of what actually exists. The stage is the target
     rather than the video so the overlay canvas comes with it; iOS Safari is
     the exception -- it has no element fullscreen, only the video's own
     native one, and there the overlay CANNOT follow. Better to give iOS its
     native fullscreen than no fullscreen at all, so it is the last resort. */
  if (fsBtn) fsBtn.onclick = () => {
    const doc = document;
    if (doc.fullscreenElement || doc.webkitFullscreenElement) {
      (doc.exitFullscreen || doc.webkitExitFullscreen).call(doc);
      return;
    }
    if (stage && stage.requestFullscreen) stage.requestFullscreen().catch(() => {});
    else if (stage && stage.webkitRequestFullscreen) stage.webkitRequestFullscreen();
    else if (v.webkitEnterFullscreen) v.webkitEnterFullscreen();
  };

  /* ── seek bar ── */
  function duration() {
    return Number.isFinite(v.duration) && v.duration > 0 ? v.duration : 0;
  }
  function seekToClientX(clientX) {
    const r = bar.getBoundingClientRect();
    const d = duration();
    if (!d || !r.width) return;
    v.currentTime = Math.max(0, Math.min(d, ((clientX - r.left) / r.width) * d));
    paint();
  }
  let dragging = false;
  if (bar) {
    /* Pointer events, not mouse+touch pairs: one code path covers finger,
       stylus and mouse, and setPointerCapture keeps the drag alive when the
       finger slides off the 24 px rail, which on a phone it always does. */
    bar.addEventListener("pointerdown", e => {
      dragging = true;
      bar.setPointerCapture(e.pointerId);
      seekToClientX(e.clientX);
    });
    bar.addEventListener("pointermove", e => { if (dragging) seekToClientX(e.clientX); });
    const stop = e => {
      if (!dragging) return;
      dragging = false;
      try { bar.releasePointerCapture(e.pointerId); } catch {}
    };
    bar.addEventListener("pointerup", stop);
    bar.addEventListener("pointercancel", stop);
  }

  function paintScrubber() {
    const d = duration();
    const pct = d ? Math.max(0, Math.min(100, (v.currentTime / d) * 100)) : 0;
    if (fill) fill.style.width = pct + "%";
    if (head) head.style.left = pct + "%";
    const t = $("playTime");
    if (t) t.textContent = `${fmtTime(v.currentTime)} / ${fmtTime(d)}`;
  }

  /* ── canvas ── */

  /* Sized in DEVICE pixels to the element's CSS box every paint, so a resize,
     an orientation change and a fullscreen transition all land automatically
     -- the CSS box is the single source of truth and nothing has to be told
     that the layout moved. The video is object-contain, so the letterboxed
     area is computed rather than assumed; assuming it offsets every box on
     portrait footage, which is most phone footage. */
  // Geometry and drawing come from the shared helpers above -- the hero clip
  // draws the identical shape and two copies would drift.
  const fit = () => fitCanvas(ctx, cv, v, stage || v);

  /* One call into the shared renderer. The 200 lines that used to live here
     -- contour path, glow, badge placement, the even-odd dim -- are now in
     paintFlaw(), shared with the hero clip. This player keeps the dim, since
     it is showing a whole set and the point is to say WHERE to look; the hero
     drops it, because a 3-second clip is already only the flaw. */
  function drawOverlay() {
    const r = fit();
    ctx.clearRect(0, 0, r.width, r.height);
    if (!PROOF || !PROOF.box) return;
    paintFlaw(ctx, r, v.videoWidth, v.videoHeight, PROOF,
              alphaFor(Math.abs(v.currentTime - PROOF.t)), { dim: true });
  }

  /* ── focus loop ── */
  function focusRange() {
    if (!PROOF || !Number.isFinite(PROOF.t)) return null;
    const d = duration() || Infinity;
    return [Math.max(0, PROOF.t - FOCUS_PAD), Math.min(d, PROOF.t + FOCUS_PAD)];
  }

  const FOCUS_ON_CLS  = "bg-crit text-white";
  const FOCUS_OFF_CLS = "bg-crit/15 text-crit border border-crit/35 hover:bg-crit/25";

  function paintFocusBtn() {
    const btn = $("focusBtn"), full = $("fullBtn");
    if (!btn) return;
    const can = !!focusRange();
    btn.disabled = !can;
    btn.style.opacity = can ? "1" : "0.3";
    btn.style.pointerEvents = can ? "auto" : "none";
    // The label never changes with state -- it always names the one thing the
    // button does. What changes is the fill, which shows whether it is on.
    btn.innerHTML = "&#8635; Loop flaw moment " +
      `<span class="opacity-60">3s &middot; ${FOCUS_RATE}x</span>`;
    btn.className = "flex-1 sm:flex-none px-3.5 py-2.5 rounded-xl text-[11px] " +
      "font-bold transition active:scale-95 whitespace-nowrap " +
      (FOCUS ? FOCUS_ON_CLS : FOCUS_OFF_CLS);
    if (full) {
      full.className = "flex-1 sm:flex-none px-3.5 py-2.5 rounded-xl text-[11px] " +
        "font-bold transition active:scale-95 whitespace-nowrap " +
        (FOCUS ? "bg-white/[0.06] text-white/55 hover:text-white"
               : "bg-white/[0.12] text-white");
    }
  }

  function setFocus(on) {
    const range = focusRange();
    if (on && !range) return;
    FOCUS = !!on;
    v.playbackRate = FOCUS ? FOCUS_RATE : 1.0;
    if (FOCUS) {
      v.currentTime = range[0];
      v.play().catch(() => {});
    }
    paintFocusBtn();
  }
  if ($("focusBtn")) $("focusBtn").onclick = () => setFocus(!FOCUS);

  /* Full set replay rewinds to the start of the CHAPTER, not of the file. On
     a three-exercise session, "replay the set" meaning "go back to the squats
     you finished two minutes ago" would be wrong. */
  if ($("fullBtn")) $("fullBtn").onclick = () => {
    setFocus(false);
    const c = TIMELINE[ACTIVE];
    v.currentTime = c && Array.isArray(c.time_range) ? c.time_range[0] : 0;
    v.play().catch(() => {});
  };

  /* Enforced on the clock, not with <video loop>, because loop only wraps at
     the END of the file. Guarded on `seeking` so the wrap-around seek does not
     retrigger itself while the browser is still servicing the previous one. */
  function holdFocus() {
    if (!FOCUS || v.seeking) return;
    const range = focusRange();
    if (!range) { setFocus(false); return; }
    const [a, b2] = range;
    if (v.currentTime > b2 || v.currentTime < a - 0.25) v.currentTime = a;
  }

  function paint() { holdFocus(); drawOverlay(); paintScrubber(); }

  /* Two drivers, because neither is sufficient alone.

     rAF is the SMOOTH one and does the real work: it runs at display rate, so
     the box ramps and the playhead moves without stepping.

     timeupdate/seeked are the CORRECT one: rAF is throttled to a stop in a
     background tab and does not run at all in some browsers while paused, so
     a scrub or a programmatic seek could otherwise leave the canvas showing
     the previous position. timeupdate alone would be no good on its own --
     it fires roughly 4 times a second, and a 0.4 s window sampled at 4 Hz can
     be crossed entirely between two events. */
  ["timeupdate", "seeked", "seeking", "loadedmetadata", "durationchange",
   "play", "pause", "ratechange"].forEach(ev => v.addEventListener(ev, paint));
  // Marks are placed against the video's duration, which is NaN until metadata
  // arrives, so they cannot be laid out at render time.
  v.addEventListener("loadedmetadata", renderSeekMarks);
  v.addEventListener("durationchange", renderSeekMarks);

  let raf = 0;
  function loop() { paint(); raf = requestAnimationFrame(loop); }
  raf = requestAnimationFrame(loop);

  /* Stop the loop when the player goes away, or every re-analysis leaves
     another rAF running against a detached canvas. */
  const obs = new MutationObserver(() => {
    if (!document.body.contains(cv)) { cancelAnimationFrame(raf); obs.disconnect(); }
  });
  obs.observe($("report"), { childList: true, subtree: true });

  const refit = () => { paint(); renderSeekMarks(); };
  window.addEventListener("resize", refit);
  window.addEventListener("orientationchange", refit);
  document.addEventListener("fullscreenchange", refit);
  document.addEventListener("webkitfullscreenchange", refit);

  paintFocusBtn();
  paint();
}

/* ─────────────── best vs worst visual proof comparison ─────────────── */

/* WHERE THE TWO FRAMES COME FROM, AND WHY IT CHANGED
     They used to be grabbed out of the <video> already on the page. That was
     cheap and it worked -- while the video existed. It did not survive into
     the history drawer, where the recording has been deleted, so the one
     place a user goes to review past form showed no comparison at all. And on
     a MediaRecorder WebM, which often carries no seek index, an arbitrary
     seek can hang or land on the wrong frame, so the card was raced against a
     timeout and dropped outright when it lost.

     Both frames are now written to the database during Stage C and served as
     immutable JPEGs. The server already had to decode those exact timestamps
     to run SAM and to build the report, so this costs one extra JPEG encode
     per chapter, not an extra decode pass.

     The in-page video remains the FALLBACK, for any chapter that never got a
     stored frame -- SAM only covers the worst-behaving few. */

const GRAB_TIMEOUT_MS = 4000;

let scratchVideo = null;

function frameUrl(sid, kind, idx) {
  return `/api/v1/session/${encodeURIComponent(sid)}/frame/${kind}/${idx}`;
}

/* Resolves to a drawable <img>, or rejects. A 404 here is an ordinary answer
   -- that chapter has no stored frame of that kind -- not an error worth
   surfacing, so the caller falls through to the video. */
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => (img.naturalWidth ? resolve(img) : reject(new Error("empty")));
    img.onerror = () => reject(new Error("no stored frame"));
    img.src = url;
  });
}

function grabFrameAt(src, t) {
  return new Promise((resolve, reject) => {
    if (!scratchVideo) {
      scratchVideo = document.createElement("video");
      scratchVideo.muted = true;
      scratchVideo.playsInline = true;
      scratchVideo.preload = "auto";
    }
    const v = scratchVideo;
    let timer = null;
    const cleanup = () => {
      clearTimeout(timer);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("error", onError);
    };
    const onSeeked = () => { cleanup(); resolve(v); };
    const onError = () => { cleanup(); reject(new Error("decode failed")); };
    timer = setTimeout(() => { cleanup(); reject(new Error("seek timed out")); },
                       GRAB_TIMEOUT_MS);
    v.addEventListener("seeked", onSeeked, { once: true });
    v.addEventListener("error", onError, { once: true });

    const seek = () => { try { v.currentTime = t; } catch { onError(); } };
    if (v.src === src && v.readyState >= 2) seek();
    else {
      v.src = src;
      v.addEventListener("loadeddata", seek, { once: true });
    }
  });
}

/* Stored JPEG first, live video second. Neither is guaranteed: a chapter that
   SAM skipped has no stored frame, and the history drawer has no video. */
async function panelSource(c, sid, kind, t) {
  if (sid && Number.isFinite(c.chapter_index)) {
    try { return await loadImage(frameUrl(sid, kind, c.chapter_index)); }
    catch { /* fall through to the video */ }
  }
  const player = $("player");
  const src = player && player.src;
  if (src && Number.isFinite(t)) return grabFrameAt(src, t);
  throw new Error("no source");
}

/* `media` is an <img> or a <video>. Both are drawImage-able and both report
   their intrinsic size, so one painter covers the stored and the live case. */
function paintPanel(canvas, media, contour, centre) {
  const vw = media.naturalWidth || media.videoWidth;
  const vh = media.naturalHeight || media.videoHeight;
  if (!canvas || !vw || !vh) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 320;
  const cssH = Math.round(cssW * vh / vw);
  canvas.style.height = cssH + "px";
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.drawImage(media, 0, 0, cssW, cssH);

  if (!contour || contour.length < 3) return;
  g.beginPath();
  contour.forEach(([nx, ny], i) => {
    const x = nx * cssW, y = ny * cssH;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.closePath();
  g.fillStyle = FLAW_FILL;
  g.fill();
  g.shadowColor = FLAW_GLOW;
  g.shadowBlur = 10;
  g.strokeStyle = FLAW_LINE;
  g.lineWidth = 2.5;
  g.lineJoin = "round";
  g.stroke();
  g.shadowBlur = 0;
  if (centre) {
    g.beginPath();
    g.arc(centre[0] * cssW, centre[1] * cssH, 3.5, 0, Math.PI * 2);
    g.fillStyle = FLAW_LINE;
    g.fill();
  }
}

/* Stops a slow frame belonging to a chapter the user has already clicked away
   from repainting the panel they are now looking at. */
let compareToken = 0;

async function renderComparison(c) {
  const host = $("compareCard");
  if (!host) return;
  const mine = ++compareToken;

  const sid = (DATA || {}).session_id;
  const bestT = repSeconds(c.best_rep, c.best_rep_timestamp);
  const worstT = PROOF && Number.isFinite(PROOF.t) ? PROOF.t : null;

  // Both halves or nothing. One panel on its own is not a comparison, it is
  // the keyframe the player is already showing.
  if (worstT == null || !Number.isFinite(bestT)) { host.innerHTML = ""; return; }

  host.innerHTML = `
    <div class="rounded-3xl bg-panel border border-white/[0.06] p-4 sm:p-5">
      <div class="text-[10px] font-bold tracking-[0.14em] text-white/35 mb-3">VISUAL PROOF · BEST vs WORST</div>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div class="rounded-2xl border border-volt/25 bg-black/40 p-3">
          <div class="flex items-center gap-1.5 mb-2">
            <span class="w-1.5 h-1.5 rounded-full bg-volt"></span>
            <span class="text-[10px] font-black tracking-wider text-volt uppercase">Best rep</span>
            <button data-seek="${bestT}" class="seek ml-auto text-[10px] tabular-nums text-white/30 hover:text-white">${fmtTime(bestT)}</button>
          </div>
          <div class="relative rounded-xl overflow-hidden bg-black">
            <canvas id="panelBest" class="w-full block"></canvas>
            <div class="absolute bottom-2 left-2 px-2 py-1 rounded bg-black/75 border border-volt/25 text-[9px] font-bold text-volt">
              &#10003; Cleanest rep of the set
            </div>
          </div>
        </div>
        <div class="rounded-2xl border border-crit/30 bg-black/40 p-3">
          <div class="flex items-center gap-1.5 mb-2">
            <span class="relative flex w-1.5 h-1.5">
              <span class="absolute inline-flex w-full h-full rounded-full bg-crit animate-halo"></span>
              <span class="relative inline-flex w-1.5 h-1.5 rounded-full bg-crit"></span>
            </span>
            <span class="text-[10px] font-black tracking-wider text-crit uppercase">Flaw breakdown</span>
            <button data-seek="${worstT}" class="seek ml-auto text-[10px] tabular-nums text-white/30 hover:text-white">${fmtTime(worstT)}</button>
          </div>
          <div class="relative rounded-xl overflow-hidden bg-black">
            <canvas id="panelWorst" class="w-full block"></canvas>
            <div class="absolute bottom-2 left-2 px-2 py-1 rounded bg-black/75 border border-crit/30 text-[9px] font-bold text-crit truncate max-w-[92%]">
              &#9888; ${esc(String(PROOF.object || "kinetic breakdown").replace(/_/g, " "))}
            </div>
          </div>
        </div>
      </div>
      <div class="text-[11px] text-white/50 mt-3 leading-relaxed">${esc(PROOF.issue || "")}</div>
    </div>`;
  bindSeeks();

  // allSettled, not all: one missing panel must not cost the other.
  const results = await Promise.allSettled([
    panelSource(c, sid, "best", bestT).then(m => [$("panelBest"), m, null, null]),
    panelSource(c, sid, "worst", worstT).then(m => [$("panelWorst"), m, PROOF.contour, PROOF.center]),
  ]);
  if (mine !== compareToken) return;          // chapter changed while loading

  let painted = 0;
  for (const r of results) {
    if (r.status !== "fulfilled") continue;
    paintPanel(...r.value);
    painted++;
  }
  // A card with two black rectangles claims to show proof and shows none.
  if (!painted) host.innerHTML = "";
}

/* ─────────────────────────── history drawer ─────────────────────────── */

/* ─────────────────────── ask the coach ───────────────────────

   A grounded Q&A drawer under the flaw player: tap or hold to speak, the
   question goes to the server with the active chapter, the reply is both
   printed and spoken.

   TWO BROWSER FACTS THAT SHAPE ALL OF THIS

   1. SpeechRecognition needs a SECURE CONTEXT. It works on localhost and over
      https, and it does NOT work over plain http to a LAN address -- which is
      exactly how a phone propped against a bench reaches this server. On that
      setup the microphone is unavailable no matter what the browser supports,
      so the failure is stated in the UI rather than left as a button that does
      nothing. Typing always works.

   2. It is also Chromium/Safari only; Firefox ships no implementation at all.

   The consequence is that voice is an ENHANCEMENT here. The text field is the
   feature, and it is never hidden behind the microphone. */

const CHAT_MAX_TURNS = 40;          // what the log keeps on screen
const CHAT_SEND_TURNS = 6;          // what is replayed to the model

/* Per chapter, so switching sets does not carry the previous set's
   conversation into a question about a different exercise. */
let CHAT_LOG = {};
let CHAT_BUSY = false;
let recog = null;
let recogActive = false;
let recogHeld = false;
let holdTimer = null;

function speechRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function micBlockedReason() {
  if (!speechRecognitionCtor()) return "This browser has no speech recognition";
  // isSecureContext is the actual gate the API checks, so it is the thing to
  // test -- not the protocol, which would wrongly reject localhost.
  if (!window.isSecureContext) return "Microphone needs HTTPS (or localhost)";
  return null;
}

const QUICK_ASKS = [
  "Why did my form break down here?",
  "How do I fix this on my next set?",
  "Am I at risk of injury?",
];

function chatTurns() {
  return (CHAT_LOG[ACTIVE] ||= []);
}

function renderChat() {
  const host = $("chatCard");
  if (!host) return;
  const c = TIMELINE[ACTIVE];
  const sid = (DATA || {}).session_id;
  if (!c || !sid) { host.innerHTML = ""; return; }

  const blocked = micBlockedReason();
  const turns = chatTurns();

  host.innerHTML = `
    <details id="chatDrawer" class="rounded-3xl bg-panel border border-white/[0.06] overflow-hidden" open>
      <summary class="px-5 py-4 cursor-pointer flex items-center gap-2 text-[11px] font-bold text-white/55 hover:text-white transition list-none">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
        </svg>
        Ask the coach
        <span class="text-[10px] font-normal text-white/25 truncate">about ${esc(c.exercise_name || "this set")}</span>
        <span id="chatCount" class="ml-auto text-[10px] text-white/25 tabular-nums">${turns.filter(t => t.role === "user").length || ""}</span>
      </summary>

      <div class="px-4 sm:px-5 pb-4">
        <div id="chatLog" class="space-y-2 max-h-[340px] overflow-y-auto mb-3"></div>

        <div class="flex gap-2 overflow-x-auto scrollbar-none mb-2.5 pb-0.5">
          ${QUICK_ASKS.map((q, i) => `
            <button data-ask="${i}" class="quickask shrink-0 px-3 py-2 rounded-xl bg-white/[0.05]
              hover:bg-white/[0.10] border border-white/[0.07] text-[11px] text-white/60
              hover:text-white transition active:scale-95 whitespace-nowrap">${esc(q)}</button>`).join("")}
        </div>

        <div class="flex items-end gap-2">
          <button id="micBtn" title="${blocked ? esc(blocked) : "Tap to speak, or hold"}"
            class="w-11 h-11 shrink-0 rounded-xl flex items-center justify-center transition active:scale-95"></button>
          <!-- A textarea, not an input: a spoken question transcribes to
               something longer than a search box, and a one-line field hides
               most of it exactly when the user wants to check it before
               sending. -->
          <textarea id="chatInput" rows="1" placeholder="Ask about this set…"
            class="flex-1 min-w-0 resize-none px-3.5 py-3 rounded-xl bg-black/40
                   border border-white/[0.09] text-[13px] text-white placeholder-white/25
                   focus:outline-none focus:border-volt/40 transition"></textarea>
          <button id="chatSend"
            class="w-11 h-11 shrink-0 rounded-xl bg-volt text-black flex items-center justify-center
                   transition active:scale-95 disabled:opacity-30">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 12h14M13 6l6 6-6 6"/>
            </svg>
          </button>
        </div>
        <div id="chatNote" class="text-[9px] text-white/25 mt-2 leading-relaxed"></div>
      </div>
    </details>`;

  renderChatLog();
  paintMicBtn();

  host.querySelectorAll(".quickask").forEach(b => {
    b.onclick = () => askCoach(QUICK_ASKS[parseInt(b.dataset.ask, 10)]);
  });
  const input = $("chatInput");
  if (input) {
    // Enter sends, Shift+Enter breaks the line. On a touch keyboard Enter is
    // a newline, so the send button is the primary control there and this is
    // the desktop shortcut.
    input.onkeydown = (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askCoach(input.value);
      }
    };
    input.oninput = () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 120) + "px";
    };
  }
  if ($("chatSend")) $("chatSend").onclick = () => askCoach(($("chatInput") || {}).value);

  const btn = $("micBtn");
  if (btn && !blocked) {
    // Tap toggles; press-and-hold dictates and sends on release. Both, because
    // a tap is what a thumb does by default and a hold is what a hand does
    // when it is also holding a barbell.
    btn.onclick = () => (recogActive ? stopDictation() : startDictation(false));
    btn.onpointerdown = () => { holdTimer = setTimeout(() => {
      recogHeld = true; startDictation(true);
    }, 260); };
    const release = () => {
      clearTimeout(holdTimer);
      if (recogHeld) { recogHeld = false; stopDictation(); }
    };
    btn.onpointerup = release;
    btn.onpointercancel = release;
    btn.onpointerleave = release;
  }
  setChatNote(blocked
    ? blocked + " — typing works everywhere."
    : "");
}

function paintChatCount() {
  const el = $("chatCount");
  if (!el) return;
  const n = chatTurns().filter(t => t.role === "user").length;
  el.textContent = n ? String(n) : "";
}

function setChatNote(msg, tone) {
  const el = $("chatNote");
  if (!el) return;
  el.textContent = msg || "";
  el.className = "text-[9px] mt-2 leading-relaxed "
    + (tone === "error" ? "text-crit/70" : "text-white/25");
}

function renderChatLog() {
  const log = $("chatLog");
  if (!log) return;
  const turns = chatTurns();
  if (!turns.length) {
    log.innerHTML = `<div class="text-[11px] text-white/25 leading-relaxed py-1">
      Answers come from this set's measurements — the score, the flagged moment
      and the region SAM outlined. Nothing else about you is known here.
    </div>`;
    return;
  }
  log.innerHTML = turns.map(t => t.role === "user"
    ? `<div class="flex justify-end">
         <div class="max-w-[85%] px-3.5 py-2.5 rounded-2xl rounded-br-md bg-white/[0.09] text-[12px] text-white/85 leading-relaxed">${esc(t.text)}</div>
       </div>`
    : t.role === "error"
    ? `<div class="px-3.5 py-2.5 rounded-2xl bg-crit/[0.08] border border-crit/25 text-[11px] text-crit/85 leading-relaxed">${esc(t.text)}</div>`
    : `<div class="flex justify-start">
         <div class="max-w-[88%] px-3.5 py-2.5 rounded-2xl rounded-bl-md bg-volt/[0.07] border border-volt/20 text-[12px] text-white/85 leading-relaxed">
           <div class="text-[9px] font-black tracking-[0.12em] text-volt/60 mb-1">COACH</div>
           ${esc(t.text)}
         </div>
       </div>`).join("")
    + (CHAT_BUSY ? `<div class="flex justify-start">
         <div class="px-3.5 py-2.5 rounded-2xl bg-volt/[0.05] border border-volt/15 flex items-center gap-2">
           <span class="relative flex w-1.5 h-1.5">
             <span class="absolute inline-flex w-full h-full rounded-full bg-volt animate-halo"></span>
             <span class="relative inline-flex w-1.5 h-1.5 rounded-full bg-volt"></span>
           </span>
           <span class="text-[11px] text-white/40">thinking…</span>
         </div>
       </div>` : "");
  log.scrollTop = log.scrollHeight;
}

function paintMicBtn() {
  const btn = $("micBtn");
  if (!btn) return;
  const blocked = micBlockedReason();
  btn.disabled = !!blocked;
  btn.style.opacity = blocked ? "0.3" : "1";
  btn.style.pointerEvents = blocked ? "none" : "auto";
  btn.className = "w-11 h-11 shrink-0 rounded-xl flex items-center justify-center "
    + "transition active:scale-95 "
    + (recogActive ? "bg-crit text-white" : "bg-white/[0.07] text-white/55 hover:text-white");
  btn.innerHTML = recogActive
    ? `<span class="relative flex w-3.5 h-3.5">
         <span class="absolute inline-flex w-full h-full rounded-full bg-white/70 animate-halo"></span>
         <span class="relative inline-flex w-3.5 h-3.5 rounded-full bg-white"></span>
       </span>`
    : `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
         <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
         <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/>
       </svg>`;
}

/* ── dictation ── */

function startDictation(holdMode) {
  const Ctor = speechRecognitionCtor();
  if (!Ctor || recogActive) return;
  // Priming here as well as on the record buttons: this may be the first tap
  // of the session, and the reply is spoken from a fetch callback that is not
  // inside any gesture.
  primeSpeech();

  try {
    recog = new Ctor();
  } catch { setChatNote("Could not start the microphone.", "error"); return; }
  recog.lang = "en-GB";
  recog.continuous = false;
  // Interim results so the field fills as they talk. Without it the box stays
  // empty until the engine finalises, which reads as a dead microphone.
  recog.interimResults = true;
  recog.maxAlternatives = 1;

  let finalText = "";
  recog.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) finalText += r[0].transcript;
      else interim += r[0].transcript;
    }
    const input = $("chatInput");
    if (input) input.value = (finalText + interim).trim();
  };
  recog.onerror = (e) => {
    recogActive = false;
    paintMicBtn();
    const code = (e && e.error) || "unknown";
    setChatNote(
      code === "not-allowed" || code === "service-not-allowed"
        ? "Microphone permission was denied — you can still type."
      : code === "no-speech" ? "Did not catch that. Try again, or type."
      : code === "network" ? "Speech service unreachable — typing works."
      : `Microphone error (${code}) — typing works.`, "error");
  };
  recog.onend = () => {
    recogActive = false;
    paintMicBtn();
    // Auto-send only what the engine actually finalised. Sending an interim
    // transcript means asking a question the user never finished saying.
    const said = finalText.trim();
    if (said) askCoach(said);
    else if (!holdMode) setChatNote("Nothing was transcribed.", "error");
  };

  try {
    recog.start();
    recogActive = true;
    setChatNote("Listening…");
    paintMicBtn();
  } catch {
    recogActive = false;
    setChatNote("Microphone is already in use.", "error");
  }
}

function stopDictation() {
  if (!recog || !recogActive) return;
  // stop(), not abort(): stop delivers the final result and then fires onend,
  // while abort throws the transcript away -- which would silently discard the
  // question the user just asked.
  try { recog.stop(); } catch { /* already stopping */ }
}

/* ── the round trip ── */

async function askCoach(question) {
  const text = String(question || "").trim();
  const sid = (DATA || {}).session_id;
  if (!text || !sid || CHAT_BUSY) return;

  const turns = chatTurns();
  turns.push({ role: "user", text });
  const input = $("chatInput");
  if (input) { input.value = ""; input.style.height = "auto"; }
  CHAT_BUSY = true;
  setChatNote("");
  renderChatLog();

  try {
    const r = await fetch(
      `${API}/api/v1/session/${encodeURIComponent(sid)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_idx: ACTIVE,
          message: text,
          // Everything except the question just pushed on, so the model is not
          // handed the current question twice.
          history: turns.slice(0, -1).slice(-CHAT_SEND_TURNS)
                        .map(t => ({ role: t.role, text: t.text })),
        }),
      });
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch { /* non-JSON */ }
      throw new Error(detail);
    }
    const d = await r.json();
    turns.push({ role: "coach", text: d.reply || "" });
    // force: a coach reply is a direct answer to something just asked, so it
    // is spoken even when it repeats an earlier sentence -- speak()'s dedupe
    // exists for automatic cues, not for answers.
    speak(d.audio_cue || d.reply, { force: true });
  } catch (e) {
    turns.push({ role: "error", text: String(e.message || e) });
  } finally {
    if (turns.length > CHAT_MAX_TURNS) turns.splice(0, turns.length - CHAT_MAX_TURNS);
    CHAT_BUSY = false;
    renderChatLog();
    // Only the count, NOT a rebuild. Re-rendering the drawer would recreate
    // the <details> and the textarea, collapsing an open panel and taking
    // focus off the field mid-conversation.
    paintChatCount();
  }
}

/* ─────────────── form health & fatigue widget ─────────────── */

/* Every number here is computed server-side from stored scores; nothing is
   estimated in the browser. The widget's one real job beyond display is to
   never show a figure the data cannot support -- an index from a single
   session, or a fatigue verdict from a session with one set, is a summary of
   nothing, and the endpoint says so explicitly rather than returning a number
   that looks the same as a real one. */
async function renderHealth() {
  const host = $("healthCard");
  if (!host) return;
  let h;
  try {
    const r = await fetch(`${API}/api/v1/analytics/form-health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    h = await r.json();
  } catch {
    host.innerHTML = "";
    return;
  }

  const idx = h.form_health_index;
  const tone = idx == null ? "text-white/35" : idx >= 80 ? "text-volt"
             : idx >= 60 ? "text-warn" : "text-crit";
  const ring = idx == null ? "#3A3A3A" : idx >= 80 ? "#C6FF00"
             : idx >= 60 ? "#FFB020" : "#FF4D4D";
  const pct = idx == null ? 0 : Math.max(0, Math.min(100, idx));
  // 2*pi*r for r=26
  const circ = 163.4;

  const fatigued = h.fatigue_status === "High Fatigue Degradation";
  const unknown = h.fatigue_status === "Not enough sets to judge";

  const top = (h.recent_exercise_averages || []).slice(0, 3);

  host.innerHTML = `
    <div class="rounded-2xl bg-panel2 border border-white/[0.06] p-4">
      <div class="flex items-center gap-3.5">
        <div class="relative shrink-0" style="width:64px;height:64px">
          <svg width="64" height="64" viewBox="0 0 64 64" class="-rotate-90">
            <circle cx="32" cy="32" r="26" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="6"/>
            <circle cx="32" cy="32" r="26" fill="none" stroke="${ring}" stroke-width="6"
                    stroke-linecap="round" stroke-dasharray="${circ}"
                    stroke-dashoffset="${(circ * (1 - pct / 100)).toFixed(1)}"/>
          </svg>
          <div class="absolute inset-0 flex items-center justify-center">
            <span class="text-[17px] font-black tabular-nums ${tone}">${idx == null ? "—" : Math.round(idx)}</span>
          </div>
        </div>
        <div class="min-w-0 flex-1">
          <div class="text-[11px] font-black tracking-[0.12em] text-white/40">FORM HEALTH</div>
          <div class="text-[10px] text-white/30 mt-0.5 leading-relaxed">
            ${h.note
              ? esc(h.note)
              : `${h.sessions_in_window} session${h.sessions_in_window === 1 ? "" : "s"} ·
                 ${h.window_days}-day window · ${h.halflife_days}-day half-life`}
          </div>
          <div class="mt-2">
            ${unknown
              ? `<span class="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-white/[0.05] text-white/35 text-[10px] font-bold">
                   Fatigue: not enough sets to judge
                 </span>`
              : fatigued
              ? `<span class="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-crit/15 text-crit border border-crit/30 text-[10px] font-bold">
                   &#9888; High fatigue degradation
                   <span class="opacity-70 tabular-nums">−${(h.worst_drop_pct ?? 0).toFixed(0)}% across sets</span>
                 </span>`
              : `<span class="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-volt/12 text-volt border border-volt/25 text-[10px] font-bold">
                   Low fatigue risk
                 </span>`}
          </div>
        </div>
      </div>
      ${top.length ? `
      <div class="mt-3 pt-3 border-t border-white/[0.05] space-y-1.5">
        ${top.map(e => `
          <div class="flex items-center gap-2 text-[10px]">
            <span class="text-white/45 capitalize truncate flex-1">${esc(e.exercise)}</span>
            <span class="text-white/20 tabular-nums">${e.sets} set${e.sets === 1 ? "" : "s"}</span>
            <span class="tabular-nums font-bold ${e.mean_score >= 80 ? "text-volt" : e.mean_score >= 60 ? "text-warn" : "text-crit"}">${e.mean_score.toFixed(0)}</span>
          </div>`).join("")}
      </div>` : ""}
      ${fatigued ? `
      <div class="mt-2.5 text-[9px] text-white/25 leading-relaxed">
        Measured as the drop from the first set's form score to the last, within one
        session. Above ${h.fatigue_threshold_pct}% is called degradation — a threshold, not a diagnosis.
      </div>` : ""}
    </div>`;
}

async function openHistory() {
  const d = $("historyDrawer");
  d.classList.remove("translate-x-full");
  $("historyScrim").classList.remove("hidden");
  $("historyList").innerHTML =
    `<div class="text-[11px] text-white/30 p-4">Loading…</div>`;
  renderHealth();          // independent of the list; neither blocks the other
  try {
    const r = await fetch(API + "/api/v1/history?limit=50");
    const j = await r.json();
    setExportEnabled(j.sessions.length > 0);
    if (!j.sessions.length) {
      $("historyCount").textContent = "none stored";
      $("historyList").innerHTML =
        `<div class="text-[11px] text-white/30 p-4 leading-relaxed">
           No stored workouts yet. Record a set or upload a clip and it will
           appear here.</div>`;
      return;
    }
    $("historyCount").textContent = `${j.total} stored`;
    /* A row is a div wrapping two buttons rather than one button containing
       another: nested buttons are invalid HTML, and browsers repair them by
       splitting the markup, which drops the click handler on whichever half
       they decide to move. */
    $("historyList").innerHTML = j.sessions.map(s => {
      const score = s.overall_score;
      const col = score == null ? "text-white/40"
                : score >= 80 ? "text-volt" : score >= 60 ? "text-warn" : "text-crit";
      const name = esc(s.exercise_names || "—");
      return `<div class="relative rounded-2xl bg-panel2 border border-white/[0.06] hover:bg-white/[0.04] transition">
        <button data-sid="${esc(s.session_id)}" class="hist w-full text-left p-3.5 pr-12">
          <div class="flex items-baseline gap-2">
            <span class="text-[12px] font-bold truncate">${name}</span>
            <span class="ml-auto text-[15px] font-black tabular-nums ${col}">${score ?? "—"}</span>
          </div>
          <div class="text-[10px] text-white/30 mt-1">
            ${esc(fmtDate(s.timestamp))} · ${fmtTime(s.duration)} ·
            ${s.total_exercises} exercise${s.total_exercises === 1 ? "" : "s"} ·
            ${s.total_reps ?? 0} reps${s.analysis_mode === "local-fallback" ? " · local only" : ""}
          </div>
        </button>
        <button data-del="${esc(s.session_id)}" data-name="${name}" title="Delete this workout"
          class="del absolute top-2.5 right-2.5 w-7 h-7 rounded-lg text-white/20 hover:text-crit hover:bg-crit/10 flex items-center justify-center transition active:scale-90">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
          </svg>
        </button>
      </div>`;
    }).join("");
    $("historyList").querySelectorAll(".hist").forEach(b => {
      b.onclick = () => showHistorySession(b.dataset.sid);
    });
    $("historyList").querySelectorAll(".del").forEach(b => {
      b.onclick = () => deleteHistorySession(b.dataset.del, b.dataset.name);
    });
  } catch (e) {
    setExportEnabled(false);
    $("historyList").innerHTML =
      `<div class="text-[11px] text-crit p-4">${esc(e.message)}</div>`;
  }
}

function closeHistory() {
  $("historyDrawer").classList.add("translate-x-full");
  $("historyScrim").classList.add("hidden");
}

/* An empty export is a valid file and a confusing download, so the link is
   inert until there is something in it. pointer-events, not just opacity: a
   link that looks disabled and still downloads is worse than no state at all. */
function setExportEnabled(on) {
  const a = $("exportBtn");
  if (!a) return;
  a.style.opacity = on ? "1" : "0.3";
  a.style.pointerEvents = on ? "auto" : "none";
  a.title = on ? "Download every stored workout as JSON"
               : "Nothing stored to export yet";
}

/* Deletion is permanent and takes the proof keyframes with it (ON DELETE
   CASCADE in workouts.db), so it is confirmed and the confirmation says what
   actually goes. The list is reloaded from the server rather than the row
   removed locally: the count in the header and the export's enabled state both
   have to follow, and re-reading is one source of truth instead of three. */
/* Two requests on purpose: the first is a dry run, so the number in the
   confirmation is measured rather than estimated, and the user is told what
   the rule actually matched before anything is destroyed. */
async function purgeTestData() {
  const note = $("purgeNote");
  try {
    const dry = await (await fetch(API + "/api/v1/history/clear-test-data",
                                   { method: "POST" })).json();
    if (!dry.matched) {
      if (note) note.textContent =
        `Nothing to clear — all ${dry.total_sessions} stored session(s) have visual proof.`;
      return;
    }
    const names = dry.sessions.slice(0, 6)
      .map(s => `  • ${s.exercise_names || "—"}  (${fmtDate(s.timestamp)})`).join("\n");
    const more = dry.matched > 6 ? `\n  …and ${dry.matched - 6} more` : "";
    if (!confirm(
      `Delete ${dry.matched} of ${dry.total_sessions} stored session(s)?\n\n` +
      `${names}${more}\n\n` +
      `These stored no proof keyframe, so they can only ever show numbers.\n` +
      `Vault uploads are not touched. This cannot be undone.`)) return;

    const done = await (await fetch(
      API + "/api/v1/history/clear-test-data?confirm=1",
      { method: "POST" })).json();
    if (note) note.textContent = `Cleared ${done.deleted} session(s).`;
    openHistory();
  } catch (e) {
    if (note) note.textContent = `Failed: ${e.message}`;
  }
}

async function deleteHistorySession(sid, name) {
  const label = name && name !== "—" ? `\n\n${name}` : "";
  if (!confirm(`Delete this workout permanently?${label}\n\n` +
               `Its scores, cues and stored proof frames are removed and ` +
               `cannot be recovered.`)) return;
  try {
    const r = await fetch(`${API}/api/v1/history/${encodeURIComponent(sid)}`,
                          { method: "DELETE" });
    // 404 means it is already gone, which is the state we were asking for.
    if (!r.ok && r.status !== 404) throw new Error(`HTTP ${r.status}`);
    openHistory();
  } catch (e) {
    $("historyList").insertAdjacentHTML("afterbegin",
      `<div class="text-[11px] text-crit p-3">Delete failed: ${esc(e.message)}</div>`);
  }
}

function fmtDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric",
                                         hour: "2-digit", minute: "2-digit" });
  } catch { return iso; }
}

/* A stored session, rendered from the DB. The video is gone -- chunks are
   deleted when a session closes -- so the proof KEYFRAME stands in for it,
   with the stored bbox drawn over the image instead of over a player. */
async function showHistorySession(sid) {
  $("historyList").innerHTML = `<div class="text-[11px] text-white/30 p-4">Loading…</div>`;
  try {
    const r = await fetch(`${API}/api/v1/history/${encodeURIComponent(sid)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    /* Keyed by chapter AND kind. A flat chapter_index key silently kept
       whichever row came last, which happened to be 'worst' only because
       'best' sorts before it -- correct by alphabet, not by intent. */
    const frames = {};
    for (const f of s.proof_frames || []) {
      (frames[f.chapter_index] ||= {})[f.kind || "worst"] = f;
    }
    $("historyList").innerHTML = `
      <button id="histBack" class="text-[11px] font-bold text-volt mb-3">← All workouts</button>
      <div class="text-[11px] text-white/35 mb-3">
        ${esc(fmtDate(s.timestamp))} · ${fmtTime(s.duration)} · ${esc(s.analysis_mode || "")}
      </div>
      ${(s.timeline || []).map(c => {
        const pair = frames[c.chapter_index] || {};
        const fWorst = pair.worst, fBest = pair.best;
        const vp = c.visual_proof || {};
        const shape = JSON.stringify({
          contour: vp.normalized_contour || null,
          box: vp.normalized_bbox || null,
          centre: vp.target_joint_center || null,
        });
        return `<div class="rounded-2xl bg-panel2 border border-white/[0.06] p-3.5 mb-2.5">
          <div class="flex items-baseline gap-2 mb-1">
            <span class="text-[12px] font-bold capitalize">${esc(c.exercise_name)}</span>
            <span class="ml-auto text-[13px] font-black tabular-nums">${c.form_score ?? "—"}</span>
          </div>
          <div class="text-[10px] text-white/30 mb-2">
            ${fmtTime(c.time_range?.[0])}–${fmtTime(c.time_range?.[1])} · ${c.total_reps ?? 0} reps
          </div>
          ${fWorst ? `<div class="grid ${fBest ? "grid-cols-2" : "grid-cols-1"} gap-1.5 mb-2">
                 ${fBest ? `<figure class="m-0">
                   <figcaption class="text-[9px] font-black tracking-wider text-volt uppercase mb-1">&#10003; Best rep</figcaption>
                   <div class="rounded-lg overflow-hidden bg-black"><img src="${esc(fBest.url)}" alt="" class="w-full block"></div>
                 </figure>` : ""}
                 <figure class="m-0">
                   <figcaption class="text-[9px] font-black tracking-wider text-crit uppercase mb-1">&#9888; Flaw</figcaption>
                   <div class="relative rounded-lg overflow-hidden bg-black">
                     <img src="${esc(fWorst.url)}" alt="" class="w-full block"
                          data-shape="${esc(shape)}" onload="drawHistShape(this)">
                   </div>
                 </figure>
               </div>` : ""}
          ${c.actionable_cue ? `<div class="text-[11px] text-volt/90 leading-snug">${esc(c.actionable_cue)}</div>` : ""}
        </div>`;
      }).join("")}`;
    $("histBack").onclick = openHistory;
  } catch (e) {
    $("historyList").innerHTML = `<div class="text-[11px] text-crit p-4">${esc(e.message)}</div>`;
  }
}

/* Overlay the stored SHAPE on a stored keyframe -- the SAM contour where there
   is one, the bounding box only as a fallback. This used to draw a dashed
   amber rectangle unconditionally, so the history drawer kept showing the
   crude box for sessions whose report showed a proper outline: the same
   moment, described two different ways depending on which screen you were on.

   An inline SVG rather than a canvas. The shape is static here, so there is
   nothing to animate, and an SVG scales with the <img> for free -- a canvas
   would need re-rasterising every time the drawer is resized. */
window.drawHistShape = function (img) {
  let sh;
  try { sh = JSON.parse(img.dataset.shape || "{}"); } catch { return; }

  const pts = Array.isArray(sh.contour) && sh.contour.length >= 3
    ? sh.contour.map(pt => `${clamp01(pt[0]) * 100},${clamp01(pt[1]) * 100}`).join(" ")
    : null;
  let body;
  if (pts) {
    body = `<polygon points="${pts}" fill="${FLAW_FILL}" stroke="${FLAW_LINE}"
             stroke-width="0.7" stroke-linejoin="round"
             vector-effect="non-scaling-stroke"/>`;
  } else if (Array.isArray(sh.box) && sh.box.length === 4) {
    const [ymin, xmin, ymax, xmax] = sh.box.map(clamp01);
    body = `<rect x="${xmin * 100}" y="${ymin * 100}"
             width="${(xmax - xmin) * 100}" height="${(ymax - ymin) * 100}"
             rx="1" fill="${FLAW_FILL}" stroke="${FLAW_LINE}" stroke-width="0.7"
             vector-effect="non-scaling-stroke"/>`;
  } else {
    return;
  }
  if (Array.isArray(sh.centre) && sh.centre.length === 2) {
    body += `<circle cx="${clamp01(sh.centre[0]) * 100}" cy="${clamp01(sh.centre[1]) * 100}"
              r="1.1" fill="${FLAW_LINE}"/>`;
  }

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("style",
    "position:absolute;inset:0;width:100%;height:100%;pointer-events:none");
  svg.innerHTML = body;
  img.parentElement.appendChild(svg);
};

/* ───────────────────────────── boot ───────────────────────────── */
function boot() {
  wireTabs();
  wireVault();
  pingHealth();
  detectCameras();
  /* Every entry point into a workout primes the speech engine, because each
     one is a real tap and any of them can be the first. Priming is idempotent,
     so doing it in four places costs one throwaway utterance in total. */
  $("camStart").onclick = () => { primeSpeech(); openCamera(); };
  $("camFlip").onclick = () => {
    facing = facing === "environment" ? "user" : "environment";
    openCamera();
  };
  $("recBtn").onclick = () => {
    primeSpeech();
    recording ? stopRec() : startRec();
  };
  setMuted(muted);
  $("muteBtn").onclick = () => { primeSpeech(); setMuted(!muted); };
  $("testAudioBtn").onclick = testAudio;
  $("historyBtn").onclick = openHistory;
  $("purgeBtn").onclick = purgeTestData;
  $("historyClose").onclick = closeHistory;
  $("historyScrim").onclick = closeHistory;
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeHistory();
  });
  // Releasing the camera on unload stops the recording indicator lingering in
  // the browser chrome after the tab is closed.
  window.addEventListener("pagehide", stopStream);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
