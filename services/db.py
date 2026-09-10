"""SQLite persistence for completed sessions.

WHY SQLITE AND NOT JSON FILES
    user_analytics.json already showed the limits of the file approach: every
    append rewrites the whole document, two writers interleave and lose an
    entry, and a crash mid-write truncates the lot. A session payload is 15 KB
    and there will eventually be thousands, so the history list needs to read
    summaries WITHOUT parsing every stored timeline. That is a database.

WHAT IS STORED, AND THE ONE THING THAT IS NOT
    The timeline survives in full -- chapters, scores, cues, proof timestamps
    and their normalised bounding boxes. The VIDEO does not: recording chunks
    are deleted when the session closes, and keeping minutes of footage per
    workout would fill the disk within a week.

    So the proof KEYFRAMES are stored instead, as JPEGs, one per chapter that
    had one. That is what makes past visual proof reviewable at all -- without
    them the history drawer could show a box's coordinates but have nothing to
    draw them on. A frame is ~40 KB against a chunk's several megabytes.

THREADING
    FastAPI runs handlers in a thread pool, so connections are opened per call
    rather than shared: an sqlite3 connection is not safe to move between
    threads, and a module-level one would fail intermittently under load in a
    way that is painful to reproduce. WAL mode makes the concurrent
    reader/writer case cheap enough that per-call connections cost nothing
    worth optimising.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "workouts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,          -- ISO-8601 UTC
    created_at       REAL NOT NULL,          -- epoch, for ordering
    duration         REAL NOT NULL,
    total_exercises  INTEGER NOT NULL,
    overall_score    INTEGER,
    total_reps       INTEGER,
    exercise_names   TEXT,                   -- comma-joined, for the list view
    analysis_mode    TEXT,
    analysis_source  TEXT,
    actionable_cue   TEXT,
    timeline_json    TEXT NOT NULL
);

-- Ordering column for the history list. Without it every drawer open is a
-- full scan once this table is large.
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);

-- TWO frames per chapter, not one: 'worst' is the flagged moment the coaching
-- is about, 'best' is the rep that went well. The comparison is the product --
-- "here is the fault" is an accusation, "here is the fault and here is you
-- doing it right, forty seconds earlier" is coaching. `kind` is part of the
-- primary key so both can coexist for one chapter.
CREATE TABLE IF NOT EXISTS proof_frames (
    session_id     TEXT NOT NULL,
    chapter_index  INTEGER NOT NULL,
    kind           TEXT NOT NULL DEFAULT 'worst',
    timestamp_s    REAL,
    jpeg           BLOB NOT NULL,
    PRIMARY KEY (session_id, chapter_index, kind),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""


def _migrate(conn):
    """Add proof_frames.kind to a database created before best-rep frames.

    Rebuild rather than ALTER: the column has to join the PRIMARY KEY, and
    SQLite's ALTER TABLE ADD COLUMN cannot change a key. Every pre-existing
    row is a worst-moment frame, which is what the old table stored, so the
    default backfills them correctly and no image is lost.

    Guarded by a cheap PRAGMA so a warm database pays nothing on boot.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(proof_frames)")}
    if not cols or "kind" in cols:
        return False
    conn.executescript("""
        PRAGMA foreign_keys=OFF;
        BEGIN;
        CREATE TABLE proof_frames_new (
            session_id     TEXT NOT NULL,
            chapter_index  INTEGER NOT NULL,
            kind           TEXT NOT NULL DEFAULT 'worst',
            timestamp_s    REAL,
            jpeg           BLOB NOT NULL,
            PRIMARY KEY (session_id, chapter_index, kind),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                ON DELETE CASCADE
        );
        INSERT INTO proof_frames_new
              (session_id, chapter_index, kind, timestamp_s, jpeg)
        SELECT session_id, chapter_index, 'worst', timestamp_s, jpeg
          FROM proof_frames;
        DROP TABLE proof_frames;
        ALTER TABLE proof_frames_new RENAME TO proof_frames;
        COMMIT;
        PRAGMA foreign_keys=ON;
    """)
    return True


def connect(path=None):
    """A configured connection. Caller closes it."""
    conn = sqlite3.connect(str(path or DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL lets the history drawer read while a finalize is writing. Without it
    # a read during a write returns SQLITE_BUSY and the drawer looks broken.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(path=None):
    """Create tables if absent. Safe to call on every boot."""
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        migrated = _migrate(conn)
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {"path": str(path or DB_PATH), "sessions": n,
                "migrated_proof_frames": migrated}
    finally:
        conn.close()


def save_session(payload, proof_frames=None, path=None):
    """Insert or update one finished session. Returns the session_id.

    A session is written TWICE: once when finalize returns the local report,
    and again when the Gemini upgrade lands with better names and cues. The
    second write is the same workout, not a new one.

    Upsert via ON CONFLICT, deliberately NOT `INSERT OR REPLACE`. REPLACE is
    implemented as DELETE + INSERT, so it fires proof_frames' ON DELETE
    CASCADE and silently destroys the stored keyframes -- the upgrade write
    would erase the very images the history drawer exists to show, and the
    row would look perfectly healthy afterwards.
    """
    sid = payload.get("session_id")
    if not sid:
        return None

    timeline = payload.get("timeline") or []
    names = ", ".join(str(b.get("exercise_name") or "?") for b in timeline)

    conn = connect(path)
    try:
        conn.execute(
            """INSERT INTO sessions
               (session_id, timestamp, created_at, duration, total_exercises,
                overall_score, total_reps, exercise_names, analysis_mode,
                analysis_source, actionable_cue, timeline_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                 duration        = excluded.duration,
                 total_exercises = excluded.total_exercises,
                 overall_score   = excluded.overall_score,
                 total_reps      = excluded.total_reps,
                 exercise_names  = excluded.exercise_names,
                 analysis_mode   = excluded.analysis_mode,
                 analysis_source = excluded.analysis_source,
                 actionable_cue  = excluded.actionable_cue,
                 timeline_json   = excluded.timeline_json""",
            (sid,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             time.time(),
             float(payload.get("session_duration_s") or 0.0),
             int(payload.get("total_exercises_detected") or len(timeline)),
             payload.get("form_score"),
             payload.get("total_reps"),
             names,
             payload.get("analysis_mode"),
             payload.get("analysis_source"),
             payload.get("actionable_cue"),
             json.dumps(timeline)))

        # Rows are (chapter, timestamp, jpeg) or (chapter, kind, timestamp,
        # jpeg). The 3-tuple form is kept working because callers that predate
        # best-rep frames still use it, and every one of those frames IS a
        # worst-moment frame -- the default is not a guess.
        for row in (proof_frames or []):
            if len(row) == 4:
                ch_idx, kind, ts, jpg = row
            else:
                ch_idx, ts, jpg = row
                kind = "worst"
            if not jpg:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO proof_frames
                   (session_id, chapter_index, kind, timestamp_s, jpeg)
                   VALUES (?,?,?,?,?)""",
                (sid, int(ch_idx), str(kind), float(ts), sqlite3.Binary(jpg)))
        conn.commit()
        return sid
    finally:
        conn.close()


def list_sessions(limit=50, offset=0, path=None):
    """Summaries, newest first. Deliberately never reads timeline_json.

    The drawer shows a date, the exercises and a score. Parsing every stored
    timeline to render that would make opening the list scale with the size of
    the entire history.
    """
    conn = connect(path)
    try:
        rows = conn.execute(
            """SELECT session_id, timestamp, duration, total_exercises,
                      overall_score, total_reps, exercise_names, analysis_mode
                 FROM sessions
             ORDER BY created_at DESC
                LIMIT ? OFFSET ?""", (int(limit), int(offset))).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {"total": total, "sessions": [dict(r) for r in rows]}
    finally:
        conn.close()


def get_session(sid, path=None):
    """One session with its full timeline, or None."""
    conn = connect(path)
    try:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?",
                           (sid,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["timeline"] = json.loads(out.pop("timeline_json") or "[]")
        frames = conn.execute(
            """SELECT chapter_index, kind, timestamp_s FROM proof_frames
                WHERE session_id = ? ORDER BY chapter_index, kind""",
            (sid,)).fetchall()
        # URLs, not bytes: a JSON body carrying several base64 JPEGs is large
        # and uncacheable, while an <img src> per frame is neither.
        out["proof_frames"] = [
            {"chapter_index": f["chapter_index"],
             "kind": f["kind"],
             "timestamp_s": f["timestamp_s"],
             "url": f"/api/v1/session/{sid}/frame/{f['kind']}/{f['chapter_index']}"}
            for f in frames]
        return out
    finally:
        conn.close()


def get_frame(sid, chapter_index, kind="worst", path=None):
    """Raw JPEG bytes for one stored proof frame, or None."""
    conn = connect(path)
    try:
        row = conn.execute(
            """SELECT jpeg FROM proof_frames
                WHERE session_id = ? AND chapter_index = ? AND kind = ?""",
            (sid, int(chapter_index), str(kind))).fetchone()
        return bytes(row["jpeg"]) if row else None
    finally:
        conn.close()


def export_all(path=None):
    """Every stored session in one document, for download.

    Includes the full timelines -- scores, cues, chapter ranges, bounding boxes
    -- and METADATA for each proof keyframe, not the keyframe itself. Inlining
    the JPEGs would base64 them into the same file at a third again their size,
    and a dozen sessions of a real user's history is tens of megabytes of
    images to carry a few kilobytes of numbers. The `url` on each frame is the
    live endpoint, so an export is a portable record of what was measured, not
    a standalone archive of the pictures. That distinction is stated in the
    file itself so nobody discovers it by restoring one.
    """
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        # One query for every frame rather than one per session: a 200-session
        # export would otherwise be 201 round trips to assemble one file.
        frames = {}
        for f in conn.execute(
                """SELECT session_id, chapter_index, kind, timestamp_s,
                          LENGTH(jpeg) AS bytes
                     FROM proof_frames
                 ORDER BY session_id, chapter_index, kind""").fetchall():
            frames.setdefault(f["session_id"], []).append({
                "chapter_index": f["chapter_index"],
                "kind": f["kind"],
                "timestamp_s": f["timestamp_s"],
                "bytes": f["bytes"],
                "url": (f"/api/v1/session/{f['session_id']}"
                        f"/frame/{f['kind']}/{f['chapter_index']}"),
            })

        out = []
        for r in rows:
            s = dict(r)
            s["timeline"] = json.loads(s.pop("timeline_json") or "[]")
            s["proof_frames"] = frames.get(s["session_id"], [])
            out.append(s)
        return {
            "format": "have-hit.workout-export",
            "version": 1,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "Proof keyframes are referenced by URL, not embedded. "
                    "They live in workouts.db and are not recoverable from "
                    "this file alone.",
            "total_sessions": len(out),
            "sessions": out,
        }
    finally:
        conn.close()


def clear_test_data(dry_run=True, path=None):
    """Remove sessions that stored no usable visual proof. -> report dict.

    WHAT COUNTS AS TEST DATA
        A recorded session with zero rows in proof_frames, or whose rows are
        all empty or not JPEG. Those are the early synthetic runs: nothing was
        segmented, so there is nothing a stored bounding box could ever be
        drawn on and the history row can only show numbers.

    THE EXCLUSION THAT MATTERS
        Vault uploads are skipped, by their `upload_` id prefix. They go
        through /api/v1/analyze-workout, which saves the report WITHOUT proof
        frames -- so by the rule above every genuine uploaded workout in the
        database looks exactly like test data and would be deleted. That is a
        real user's file, analysed once, gone. Excluding them by id is the
        narrow fix; the broad one is to make that endpoint persist its
        keyframes too, which is a change to run_analysis's return shape and is
        not made here.

    dry_run defaults to TRUE. This deletes rows and cascades to their frames,
    so the caller has to ask for it explicitly rather than get it by calling
    the function.
    """
    conn = connect(path)
    try:
        rows = conn.execute(
            """SELECT s.session_id, s.timestamp, s.exercise_names,
                      s.analysis_mode,
                      (SELECT COUNT(*) FROM proof_frames f
                        WHERE f.session_id = s.session_id
                          AND f.jpeg IS NOT NULL
                          AND LENGTH(f.jpeg) > 2
                          AND SUBSTR(f.jpeg, 1, 2) = X'FFD8') AS good_frames
                 FROM sessions s
             ORDER BY s.created_at DESC""").fetchall()

        candidates, kept_uploads = [], 0
        for r in rows:
            if r["good_frames"]:
                continue
            if str(r["session_id"]).startswith("upload_"):
                kept_uploads += 1
                continue
            candidates.append({"session_id": r["session_id"],
                               "timestamp": r["timestamp"],
                               "exercise_names": r["exercise_names"],
                               "analysis_mode": r["analysis_mode"]})

        report = {
            "dry_run": bool(dry_run),
            "total_sessions": len(rows),
            "matched": len(candidates),
            "sessions": candidates,
            "skipped_uploads": kept_uploads,
            "criterion": "no stored proof keyframe that is a valid JPEG; "
                         "vault uploads (upload_*) excluded",
            "deleted": 0,
        }
        if dry_run or not candidates:
            return report

        ids = [c["session_id"] for c in candidates]
        marks = ",".join("?" * len(ids))
        cur = conn.execute(
            f"DELETE FROM sessions WHERE session_id IN ({marks})", ids)
        conn.execute(
            f"DELETE FROM proof_frames WHERE session_id IN ({marks})", ids)
        conn.commit()
        report["deleted"] = cur.rowcount
        return report
    finally:
        conn.close()


def delete_session(sid, path=None):
    """Remove a session and its frames. Returns True if one was deleted."""
    conn = connect(path)
    try:
        cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM proof_frames WHERE session_id = ?", (sid,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Form health and fatigue
# ---------------------------------------------------------------------------

# Half-life of the rolling average, in days. A score from this morning should
# count roughly twice what one from three and a half days ago counts: recent
# enough to reflect where the athlete IS, long enough that one bad set does
# not redraw the whole picture.
HEALTH_HALFLIFE_DAYS = 3.5

# How far back the window reaches. Sessions older than this contribute nothing,
# which is what makes it a HEALTH index rather than a lifetime average -- a
# lifetime average stops moving after a few weeks and stops being information.
HEALTH_WINDOW_DAYS = 7

# Fewer scored sessions than this in the window and there is no trend, only
# points. Reporting an index off one session would present a single score as a
# summary of a week.
HEALTH_MIN_SESSIONS = 2

# Percentage drop from the first set to the last, within one session, above
# which fatigue is called. 15 points on a 0-100 form score is the figure in
# the brief; it is a judgement, not a measurement, and is named here so it can
# be argued with rather than buried in a comparison.
FATIGUE_DROP_PCT = 15.0

# Sets needed before a slope means anything. Two points define a line through
# any two numbers, so a "degradation rate" from two sets is just their
# difference wearing a regression's clothes.
FATIGUE_MIN_SETS = 3


def _decay_weight(age_days, halflife=HEALTH_HALFLIFE_DAYS):
    return 0.5 ** (max(0.0, float(age_days)) / float(halflife))


def _slope(xs, ys):
    """Least-squares slope of ys against xs. None when it is undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def session_fatigue(timeline):
    """Score decline ACROSS THE SETS of one session.

    The sets are the exercise chapters, in the order they were performed. The
    slope is points of form score lost per set, from a least-squares fit --
    not last-minus-first, which one anomalous final set can dominate entirely.

    Both are reported because they answer different questions: the slope is
    the trend, the drop is what actually happened between the first set and
    the last. They disagree when the decline is not linear, and that
    disagreement is information rather than an error to hide.
    """
    sets = [c for c in (timeline or [])
            if isinstance(c.get("form_score"), (int, float))]
    sets.sort(key=lambda c: (c.get("time_range") or [0])[0])
    scores = [float(c["form_score"]) for c in sets]
    out = {
        "sets": len(scores),
        "scores": scores,
        "slope_per_set": None,
        "drop_pct": None,
        "status": "insufficient data",
    }
    if len(scores) < FATIGUE_MIN_SETS:
        return out

    slope = _slope(list(range(len(scores))), scores)
    first, last = scores[0], scores[-1]
    drop = ((first - last) / first * 100.0) if first > 0 else 0.0
    out["slope_per_set"] = round(slope, 3) if slope is not None else None
    out["drop_pct"] = round(drop, 1)
    out["status"] = ("High Fatigue Degradation" if drop > FATIGUE_DROP_PCT
                     else "Low Fatigue Risk")
    return out


def form_health(now=None, window_days=HEALTH_WINDOW_DAYS, path=None):
    """Rolling exponentially-weighted form score, plus the fatigue picture.

    WHY WEIGHTED AND NOT A FLAT MEAN
        A flat 7-day mean treats a session from six days ago as equally
        current as one from an hour ago, so a good week hides a bad today and
        an athlete who has just started improving sees no movement for most of
        a week. The exponential weight makes the number respond while still
        being an average rather than the last score repeated.

    Reads timeline_json only for the sessions inside the window, because
    fatigue needs the per-set scores and the summary columns do not carry
    them. Outside the window nothing is parsed.
    """
    now = float(now if now is not None else time.time())
    cutoff = now - window_days * 86400.0

    conn = connect(path)
    try:
        rows = conn.execute(
            """SELECT session_id, created_at, overall_score, exercise_names,
                      timeline_json
                 FROM sessions
                WHERE created_at >= ? AND overall_score IS NOT NULL
             ORDER BY created_at DESC""",
            (cutoff,)).fetchall()
    finally:
        conn.close()

    num = den = 0.0
    per_exercise = {}
    fatigued = []
    sessions = []
    for r in rows:
        age = (now - float(r["created_at"])) / 86400.0
        w = _decay_weight(age)
        score = float(r["overall_score"])
        num += w * score
        den += w

        try:
            timeline = json.loads(r["timeline_json"] or "[]")
        except (ValueError, TypeError):
            timeline = []
        for c in timeline:
            name = (c.get("exercise_name") or "").strip().lower()
            s = c.get("form_score")
            if name and isinstance(s, (int, float)):
                per_exercise.setdefault(name, []).append(float(s))

        fat = session_fatigue(timeline)
        if fat["status"] == "High Fatigue Degradation":
            fatigued.append({"session_id": r["session_id"],
                             "drop_pct": fat["drop_pct"],
                             "sets": fat["sets"]})
        sessions.append({
            "session_id": r["session_id"],
            "created_at": float(r["created_at"]),
            "age_days": round(age, 2),
            "weight": round(w, 4),
            "score": score,
            "fatigue": fat,
        })

    scored = len(rows)
    index = round(num / den, 1) if den > 0 and scored >= HEALTH_MIN_SESSIONS else None

    # The worst recent session decides the headline, not an average of drops:
    # "did you degrade" is answered by whether it happened, and averaging a
    # heavy fade with two clean sessions reports neither.
    worst = max((f["drop_pct"] for f in fatigued), default=None)
    if not any(s["fatigue"]["sets"] >= FATIGUE_MIN_SETS for s in sessions):
        status = "Not enough sets to judge"
    elif fatigued:
        status = "High Fatigue Degradation"
    else:
        status = "Low Fatigue Risk"

    return {
        "form_health_index": index,
        "window_days": window_days,
        "halflife_days": HEALTH_HALFLIFE_DAYS,
        "sessions_in_window": scored,
        "min_sessions": HEALTH_MIN_SESSIONS,
        # Says WHY the index is null rather than leaving the UI to guess.
        "note": (None if index is not None else
                 f"needs {HEALTH_MIN_SESSIONS}+ scored sessions in "
                 f"{window_days} days; found {scored}"),
        "fatigue_status": status,
        "fatigue_threshold_pct": FATIGUE_DROP_PCT,
        "worst_drop_pct": worst,
        "fatigued_sessions": fatigued,
        "recent_exercise_averages": sorted(
            ({"exercise": k,
              "mean_score": round(sum(v) / len(v), 1),
              "sets": len(v)} for k, v in per_exercise.items()),
            key=lambda d: -d["sets"]),
        "sessions": sessions[:20],
    }
