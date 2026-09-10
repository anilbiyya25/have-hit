"""Scrape short gym exercise clips from YouTube into gym_dataset/.

Runs headless and unattended: no prompts, no interactive format selection.
Blocked, private, age-restricted, and geo-restricted clips are skipped
automatically rather than aborting the run.

Resumable -- already-downloaded clips are skipped, so re-running tops up
categories that came back thin instead of starting over.

Usage:
    .\\run.ps1 gym_scraper.py                      # all categories, 20 each
    .\\run.ps1 gym_scraper.py --limit 5            # quick test
    .\\run.ps1 gym_scraper.py --category squats    # one category
    .\\run.ps1 gym_scraper.py --dry-run            # curate links, download nothing
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("gym_dataset")

# Categories and the search phrasing that actually surfaces demonstration
# footage. Bare exercise names return vlogs, ads, and compilations; adding
# "form"/"technique"/"how to" biases toward instructional clips.
# Several phrasings per category. One YouTube search returns a few hundred
# results at most and they repeat heavily, so a single query cannot reach
# 100+ clips. Different phrasings surface largely disjoint result sets.
CATEGORIES = {
    "squats": [
        "bodyweight squat proper form technique",
        "how to do a squat correctly beginner",
        "air squat demonstration tutorial",
        "bodyweight squat exercise guide",
        "squat form check tutorial",
    ],
    "pushups": [
        "push up proper form technique",
        "how to do push ups correctly beginner",
        "pushup demonstration tutorial",
        "perfect push up form guide",
        "push up exercise tutorial",
    ],
    "barbell_squat": [
        "barbell back squat form technique",
        "how to barbell squat properly",
        "back squat tutorial demonstration",
        "barbell squat form check",
        "low bar back squat technique",
    ],
    "bench_press": [
        "barbell bench press form technique",
        "how to bench press properly beginner",
        "flat bench press tutorial demonstration",
        "bench press form check",
        "barbell bench press exercise guide",
    ],
    # The three thin categories carry extra phrasings. One YouTube search
    # returns a few hundred heavily-repeating results, so reaching more clips
    # depends on wording that surfaces a different result set, not on asking
    # the same query for more.
    "barbell_row": [
        "barbell bent over row form technique",
        "how to do bent over rows properly",
        "pendlay row tutorial demonstration",
        "barbell row form check",
        "bent over barbell row exercise guide",
        "strict barbell row form execution",
    ],
    "bicep_curl": [
        "dumbbell bicep curl form technique",
        "how to do bicep curls properly",
        "barbell curl tutorial demonstration",
        "bicep curl form check",
        "dumbbell curl exercise guide",
        "heavy dumbbell bicep curl set",
    ],
    "shoulder_press": [
        "overhead shoulder press form technique",
        "how to do overhead press properly",
        "dumbbell shoulder press tutorial",
        "military press form demonstration",
        "standing overhead press exercise guide",
        "seated shoulder press tutorial",
        "seated dumbbell shoulder press technique",
    ],

    # ---- expansion to 27 classes -------------------------------------------
    # Phrasing is deliberately technical: "execution", "biomechanics",
    # "demonstration", "form check", "strict". Bare exercise names return
    # vlogs, ads and talking-head compilations, which are noise for a motion
    # model -- MAX_DURATION_S culls the worst of it, but wording is what keeps
    # the candidate pool clean in the first place.
    #
    # Six phrasings each, because one YouTube search returns a few hundred
    # heavily-repeating results and 120 clips cannot come from a single query.
    # Where two classes are visually close (pull_ups/chin_ups,
    # push_ups/diamond_pushups), the queries name the distinguishing detail --
    # grip, hand position -- rather than the shared movement.

    "pull_ups": [
        "strict bodyweight pull ups execution form",
        "pull up overhand grip technique demonstration",
        "how to do proper pull ups biomechanics",
        "dead hang pull up form check",
        "strict pull up tutorial demonstration",
        "pull ups exercise guide proper execution",
    ],
    "chin_ups": [
        "strict chin up underhand grip execution",
        "chin ups supinated grip form technique",
        "how to do chin ups properly demonstration",
        "chin up vs pull up form tutorial",
        "bodyweight chin up biomechanics guide",
        "strict chin ups form check",
    ],
    "dips": [
        "parallel bar dips execution form technique",
        "tricep dips biomechanics tutorial",
        "how to do dips properly demonstration",
        "chest dip form check strict",
        "bodyweight dips exercise guide",
        "ring dips strict execution tutorial",
    ],
    "diamond_pushups": [
        "diamond push ups close grip execution",
        "triangle pushup form technique demonstration",
        "diamond pushup tricep biomechanics tutorial",
        "close hand pushup proper form guide",
        "how to do diamond push ups correctly",
        "narrow grip pushup form check",
    ],
    "deadlift": [
        "conventional deadlift execution form technique",
        "deadlift biomechanics tutorial demonstration",
        "how to deadlift properly setup and pull",
        "sumo deadlift form check",
        "barbell deadlift strict execution guide",
        "romanian deadlift form technique tutorial",
    ],
    "lunges": [
        "walking lunges execution form technique",
        "forward lunge biomechanics demonstration",
        "how to do lunges properly tutorial",
        "reverse lunge form check",
        "bodyweight lunge exercise guide",
        "dumbbell lunges strict form demonstration",
    ],
    "lat_pulldown": [
        "lat pulldown execution form technique",
        "cable lat pulldown biomechanics tutorial",
        "how to do lat pulldowns properly",
        "wide grip lat pulldown demonstration",
        "lat pulldown form check machine setup",
        "lat pulldown exercise guide proper execution",
    ],
    "lat_raise": [
        "lateral raise execution form technique",
        "dumbbell lateral raise biomechanics tutorial",
        "how to do lateral raises properly",
        "side delt raise form check",
        "strict lateral raise demonstration",
        "dumbbell side raise exercise guide",
    ],
    "tricep_extension": [
        "tricep extension execution form technique",
        "overhead tricep extension biomechanics tutorial",
        "how to do tricep extensions properly",
        "cable tricep pushdown form demonstration",
        "skull crusher tricep form check",
        "dumbbell tricep extension exercise guide",
    ],
    "plank": [
        "plank hold execution proper form technique",
        "forearm plank biomechanics demonstration",
        "how to hold a plank correctly tutorial",
        "plank form check core bracing",
        "front plank exercise guide proper position",
        "high plank hold form demonstration",
    ],
    "sit_ups": [
        "sit ups execution form technique",
        "full sit up biomechanics demonstration",
        "how to do sit ups properly tutorial",
        "sit up form check core",
        "bodyweight sit ups exercise guide",
        "strict sit up demonstration tutorial",
    ],
    "leg_raises": [
        "lying leg raises execution form technique",
        "hanging leg raise biomechanics tutorial",
        "how to do leg raises properly",
        "leg raise form check lower abs",
        "captains chair leg raise demonstration",
        "strict hanging leg raises exercise guide",
    ],
    "russian_twist": [
        "russian twist execution form technique",
        "seated russian twist biomechanics tutorial",
        "how to do russian twists properly",
        "russian twist form check oblique",
        "weighted russian twist demonstration",
        "russian twist exercise guide proper execution",
    ],
    "burpees": [
        "burpee execution form technique breakdown",
        "how to do burpees properly tutorial",
        "burpee biomechanics demonstration",
        "strict burpee form check",
        "full burpee exercise guide proper execution",
        "burpee tutorial step by step demonstration",
    ],
    "jumping_jacks": [
        "jumping jacks execution proper form technique",
        "how to do jumping jacks correctly tutorial",
        "jumping jack demonstration exercise guide",
        "jumping jacks form check warm up",
        "star jump proper execution demonstration",
        "jumping jacks tutorial technique breakdown",
    ],
    "jump_rope": [
        "jump rope basic bounce technique tutorial",
        "how to jump rope properly form",
        "skipping rope execution demonstration",
        "jump rope form check footwork",
        "jump rope tutorial for beginners technique",
        "speed rope basic jump demonstration",
    ],
    "mountain_climbers": [
        "mountain climbers execution form technique",
        "how to do mountain climbers properly",
        "mountain climber biomechanics demonstration",
        "mountain climbers form check core",
        "mountain climber exercise guide proper execution",
        "mountain climbers tutorial technique breakdown",
    ],
    "high_knees": [
        "high knees execution proper form technique",
        "how to do high knees correctly tutorial",
        "high knee running drill demonstration",
        "high knees form check cardio drill",
        "high knees exercise guide proper execution",
        "high knee drill technique breakdown",
    ],
    "kettlebell_swing": [
        "clean kettlebell swing biomechanics tutorial",
        "russian kettlebell swing execution form",
        "how to kettlebell swing properly hip hinge",
        "kettlebell swing form check",
        "american kettlebell swing demonstration",
        "kettlebell swing technique breakdown tutorial",
    ],
    "box_jumps": [
        "box jump execution form technique",
        "how to do box jumps properly tutorial",
        "box jump biomechanics landing demonstration",
        "box jump form check plyometric",
        "plyo box jump exercise guide execution",
        "box jump technique breakdown demonstration",
    ],
}

# Short clips only. Long videos are mostly a person talking to camera, which
# is useless as training signal for a motion model.
MIN_DURATION_S = 5
MAX_DURATION_S = 120

# ffmpeg is installed now, so merged video+audio streams are usable. Falling
# back through single-file formats keeps this working if ffmpeg goes missing.
# Merged formats matter: several clips previously failed because no
# single-file mp4 existed for them (shoulder_press lost 10 of 19 that way).
FORMAT = (
    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
    "best[ext=mp4][height<=720]/"
    "bestvideo[height<=720]+bestaudio/"
    "best[ext=mp4]/best[height<=720]/best"
)

# What counts as a finished clip on disk. Globbing "*.*" instead would also
# count yt-dlp's half-written ".part" files, which inflates the "already have"
# number and makes a resumed run stop short of the target.
VIDEO_EXTS = {".mp4", ".mkv", ".webm"}


def have_clip_ids(out_dir):
    """Video ids already fully downloaded in out_dir."""
    if not out_dir.exists():
        return set()
    return {p.stem for p in out_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS}


def yt_dlp(*args, timeout=1800, capture=True):
    """Invoke yt-dlp as a module so it uses this venv's copy.

    Returns None if the call timed out or the subprocess could not be run.
    A single hung download must not abort an unattended run, so the failure is
    swallowed here rather than raised -- an earlier run died 200 clips in when
    one bench_press download hung and TimeoutExpired propagated to the top.
    """
    cmd = [sys.executable, "-m", "yt_dlp", *args]
    try:
        return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None


def curate_one(query, want, seen):
    """Search one query. Returns new clips not already in `seen`."""
    result = yt_dlp(
        f"ytsearch{want}:{query}",
        "--skip-download",
        "--dump-json",
        "--no-playlist",
        "--no-warnings",
        "--ignore-errors",
        "--match-filter", f"duration > {MIN_DURATION_S} & duration < {MAX_DURATION_S}",
        "--socket-timeout", "30",
        timeout=900,
    )

    if result is None:
        print(f"    !!! search timed out: {query[:46]}")
        return []

    clips = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            meta = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = meta.get("id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        clips.append({
            "id": vid,
            "title": (meta.get("title") or "")[:70],
            "duration": meta.get("duration"),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return clips


def curate(queries, limit, already_have):
    """Run every query for a category, deduplicating across them.

    Stops early once enough new candidates are found, so topping up a nearly
    full category does not re-run every query.
    """
    seen = set(already_have)
    clips = []
    # Over-fetch per query: the duration filter culls most results.
    per_query = max(30, (limit * 3) // len(queries))

    for q in queries:
        if len(clips) >= limit:
            break
        found = curate_one(q, per_query, seen)
        clips += found
        print(f"    +{len(found):>3} from: {q[:46]}")
    return clips[:limit]


def download(clips, out_dir):
    """Download clips into out_dir, skipping any that fail. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = have_clip_ids(out_dir)

    n_timeout = 0
    for clip in clips:
        if clip["id"] in before:
            continue
        ok = yt_dlp(
            clip["url"],
            "-f", FORMAT,
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            "--no-playlist",
            "--no-warnings",
            "--ignore-errors",     # a blocked clip must not abort the batch
            "--no-overwrites",
            "--socket-timeout", "30",
            "--retries", "2",
            "--quiet",
            timeout=600,
        )
        if ok is None:
            n_timeout += 1
            print(f"    !!! timed out, skipped: {clip['id']}")

    if n_timeout:
        print(f"    {n_timeout} clip(s) timed out and were skipped")

    return len(have_clip_ids(out_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20,
                    help="TARGET total clips per category, counting what is "
                         "already on disk")
    ap.add_argument("--add", type=int,
                    help="download this many MORE than are already on disk, "
                         "per category (overrides --limit)")
    ap.add_argument("--category", action="append",
                    help="restrict to a category; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="curate and write links, download nothing")
    args = ap.parse_args()

    unknown = [c for c in (args.category or []) if c not in CATEGORIES]
    if unknown:
        print(f"unknown category: {', '.join(unknown)}")
        print(f"choose from: {', '.join(CATEGORIES)}")
        return 1

    targets = list(args.category) if args.category else list(CATEGORIES)

    DATASET_DIR.mkdir(exist_ok=True)
    for name in CATEGORIES:
        (DATASET_DIR / name).mkdir(exist_ok=True)

    print("=" * 60)
    print("GYM DATASET SCRAPER")
    print("=" * 60)
    print(f"  dataset   : {DATASET_DIR.resolve()}")
    print(f"  categories: {', '.join(targets)}")
    print(f"  per cat   : " + (f"+{args.add} more than on disk" if args.add
                               else f"target total {args.limit}"))
    print(f"  duration  : {MIN_DURATION_S}-{MAX_DURATION_S}s")
    print(f"  mode      : {'DRY RUN (no download)' if args.dry_run else 'download'}")

    all_links = []
    summary = {}

    for name in targets:
        queries = CATEGORIES[name]
        out_dir = DATASET_DIR / name
        have_ids = have_clip_ids(out_dir)

        print(f"\n{name}  (already have {len(have_ids)})")
        # --add is "this many more", --limit is "up to this total". Asking for
        # --limit 40 on a folder holding 97 clips is a no-op, which is not what
        # anyone means when they ask for 40 fresh clips.
        target = len(have_ids) + args.add if args.add else args.limit
        need = max(0, target - len(have_ids))
        if need == 0:
            print(f"  already at target of {target}, skipping")
            summary[name] = (0, len(have_ids), len(have_ids))
            continue

        print(f"  searching {len(queries)} queries for {need} more...")
        clips = curate(queries, need, have_ids)
        print(f"  found: {len(clips)} new candidates after duration filter")
        all_links += [c["url"] for c in clips]

        if args.dry_run:
            summary[name] = (len(clips), len(have_ids), len(have_ids))
            continue

        print("  downloading (blocked clips skipped automatically)...")
        have = download(clips, out_dir)
        print(f"  on disk: {have}  (+{have - len(have_ids)})")
        summary[name] = (len(clips), len(have_ids), have)

    links_file = DATASET_DIR / "curated_links.txt"
    links_file.write_text("\n".join(all_links) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'category':<18}{'found':>8}{'before':>9}{'after':>8}{'gained':>9}")
    for name, (found, before, have) in summary.items():
        gained = have - before
        flag = ""
        if args.add and not args.dry_run and gained < args.add:
            flag = f"  <- wanted {args.add}, search exhausted"
        print(f"  {name:<18}{found:>8}{before:>9}{have:>8}{gained:>9}{flag}")

    total = sum(h for _, _, h in summary.values())
    print(f"\n  links written : {links_file}  ({len(all_links)} urls)")
    if not args.dry_run:
        print(f"  clips on disk : {total}")
        print("\nSearch results are noisy. Review the clips and delete any that")
        print("show the wrong exercise before using them for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
