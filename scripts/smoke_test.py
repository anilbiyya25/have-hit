"""End-to-end check of POST /api/v1/analyze-workout.

    .\\run.ps1 scripts\\smoke_test.py
    .\\run.ps1 scripts\\smoke_test.py --clip path\\to\\video.mp4 --json

Verifies three things and says which one broke:

  1. the response carries every field the contract promises, with the right
     types -- not merely HTTP 200;
  2. the cross-checks that make the report trustworthy hold: the proof
     timestamp is the MEASURED worst moment and its box is a verbatim SAM
     candidate, not coordinates the language model invented;
  3. the run did not leak. Commit is sampled on a background thread because
     the peak lands mid-request and is gone by the time it returns -- a
     before/after pair would miss exactly the spike that kills long runs.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.health import commit_snapshot, gpu_status      # noqa: E402

DEFAULT_CLIP = ROOT / "gym_dataset" / "barbell_squat" / "-eO_VydErV0.mp4"
URL = "http://127.0.0.1:8000/api/v1/analyze-workout"

OK, BAD = "PASS", "FAIL"


class CommitSampler(threading.Thread):
    """Polls the commit charge until stopped, keeping the maximum seen."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        # NOT self._stop: threading.Thread defines a private _stop() that
        # join() calls internally, so shadowing it with an Event makes join()
        # raise "'Event' object is not callable".
        self.interval, self._done = interval, threading.Event()
        self.peak_used = 0.0
        self.peak_limit = 0.0
        self.samples = 0

    def run(self):
        while not self._done.is_set():
            used, limit = commit_snapshot()
            if used is not None:
                self.samples += 1
                if used > self.peak_used:
                    self.peak_used, self.peak_limit = used, limit
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()
        self.join(timeout=2)


def post_video(path, url, timeout=420):
    """Multipart POST, hand-rolled to keep this script dependency-free."""
    raw = Path(path).read_bytes()
    b = uuid.uuid4().hex
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{Path(path).name}\"\r\nContent-Type: video/mp4\r\n\r\n"
    ).encode() + raw + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def check_schema(d):
    """[(name, passed, detail)] for the contract's required shape."""
    want = [
        ("activity_detected", str), ("form_score", int), ("total_reps", int),
        ("actionable_cue", str), ("visual_proof", dict), ("best_rep", dict),
        ("worst_rep", dict),
    ]
    rows = []
    for key, typ in want:
        v = d.get(key)
        rows.append((f"{key} is {typ.__name__}", isinstance(v, typ),
                     f"{v!r}"[:64]))
    vp = d.get("visual_proof") or {}
    for key in ("timestamp", "target_object", "normalized_bbox",
                "issue_description"):
        rows.append((f"visual_proof.{key}", key in vp, f"{vp.get(key)!r}"[:64]))

    box = vp.get("normalized_bbox")
    rows.append((
        "normalized_bbox is 4 numbers in 0-1",
        isinstance(box, list) and len(box) == 4
        and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in box),
        f"{box}"))
    return rows


def check_integrity(d):
    """The cross-checks that separate a measured report from a plausible one."""
    lm = d.get("local_measurements") or {}
    dyn = lm.get("latent_dynamics") or {}
    seg = lm.get("segmentation") or {}
    vp = d.get("visual_proof") or {}
    flags = dyn.get("flagged_frames") or []
    worst = dyn.get("worst_moment") or {}
    sam_boxes = {tuple(c["normalized_bbox"])
                 for k in seg.get("keyframes", []) for c in k["candidates"]}

    rows = []
    if flags:
        errs = [f["kinetic_error"] for f in flags]
        rank1 = [f for f in flags if f["rank"] == 1]
        rows.append(("rank 1 is the maximum kinetic error",
                     len(rank1) == 1 and rank1[0]["kinetic_error"] == max(errs),
                     f"rank1={rank1[0]['kinetic_error'] if rank1 else None} "
                     f"max={max(errs)}"))
        rows.append(("flag list is time-ordered",
                     [f["timestamp"] for f in flags]
                     == sorted(f["timestamp"] for f in flags),
                     str([f["timestamp"] for f in flags])))
        rows.append(("worst_moment == rank 1",
                     worst == (rank1[0] if rank1 else None),
                     f"t={worst.get('timestamp')}s"))
    rows.append(("proof timestamp is the measured worst moment",
                 vp.get("timestamp_s") == worst.get("timestamp"),
                 f"proof={vp.get('timestamp_s')} measured={worst.get('timestamp')}"))
    rows.append(("proof bbox is a verbatim SAM candidate",
                 tuple(vp.get("normalized_bbox") or ()) in sam_boxes,
                 f"{len(sam_boxes)} candidates offered"))
    rows.append(("Stage C segmented exactly 1 keyframe",
                 len(seg.get("keyframes", [])) == 1,
                 f"{len(seg.get('keyframes', []))}"))
    rows.append(("SAM reported no error", not seg.get("error"),
                 str(seg.get("error"))[:64]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=str(DEFAULT_CLIP))
    ap.add_argument("--url", default=URL)
    ap.add_argument("--json", action="store_true",
                    help="dump the full response body at the end")
    args = ap.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"clip not found: {clip}")
        return 2

    size_mb = clip.stat().st_size / 1024 / 1024
    g0 = gpu_status()
    c0_used, c0_limit = commit_snapshot()
    print(f"clip      {clip.name}  ({size_mb:.1f} MB)")
    print(f"before    commit {c0_used:.2f}/{c0_limit:.2f} GB   "
          f"VRAM {g0['vram_used_gb']}/{g0['vram_total_gb']} GB")

    sampler = CommitSampler()
    sampler.start()
    t0 = time.time()
    try:
        d = post_video(clip, args.url)
    except urllib.error.HTTPError as e:
        sampler.stop()
        print(f"\nHTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        return 1
    except urllib.error.URLError as e:
        sampler.stop()
        print(f"\ncannot reach {args.url}: {e.reason}")
        return 1
    finally:
        elapsed = time.time() - t0
    sampler.stop()

    g1 = gpu_status()
    c1_used, _ = commit_snapshot()
    ru = d.get("resource_usage") or {}
    vram = ru.get("vram_mb") or {}

    print(f"after     commit {c1_used:.2f} GB              "
          f"VRAM {g1['vram_used_gb']}/{g1['vram_total_gb']} GB")
    print(f"PEAK      commit {sampler.peak_used:.2f}/{sampler.peak_limit:.2f} GB "
          f"({sampler.peak_used / sampler.peak_limit * 100:.1f}%, "
          f"{sampler.samples} samples)")
    print(f"elapsed   {elapsed:.1f}s      source: {d.get('analysis_source')}")
    print(f"VRAM MB   start={vram.get('start')} afterA={vram.get('after_stage_a')} "
          f"afterC={vram.get('after_stage_c')} end={vram.get('end')}  "
          f"delta={_delta(vram)}")

    print("\n--- timings ---")
    for k, v in (d.get("timing_ms") or {}).items():
        print(f"  {k:20} {v:>9.1f} ms")

    failures = 0
    for title, rows in (("schema", check_schema(d)),
                        ("integrity", check_integrity(d))):
        print(f"\n--- {title} ---")
        for name, passed, detail in rows:
            failures += (not passed)
            print(f"  [{OK if passed else BAD}] {name:44} {detail}")

    print("\n--- report ---")
    print(f"  activity  {d.get('activity_detected')}")
    print(f"  score     {d.get('form_score')}/100")
    print(f"  reps      {d.get('total_reps')}   "
          f"best={_rep(d.get('best_rep'))}  worst={_rep(d.get('worst_rep'))}")
    print(f"  cue       {d.get('actionable_cue')}")
    vp = d.get("visual_proof") or {}
    print(f"  proof     {vp.get('timestamp')}  {vp.get('target_object')}  "
          f"{vp.get('normalized_bbox')}")
    print(f"            {str(vp.get('issue_description'))[:150]}")

    leaked = (vram.get("end") or 0) - (vram.get("start") or 0)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}"
          f"   VRAM delta {leaked:+.1f} MB")

    if args.json:
        print("\n--- full response ---")
        print(json.dumps(d, indent=2)[:6000])
    return 1 if failures else 0


def _delta(v):
    s, e = v.get("start"), v.get("end")
    return f"{e - s:+.1f} MB" if s is not None and e is not None else "n/a"


def _rep(r):
    return f"#{r.get('rep_number')}@{r.get('timestamp')}" if r else "none"


if __name__ == "__main__":
    sys.exit(main())
