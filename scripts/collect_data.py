"""Download gym exercise clips from YouTube for training.

Design note: we deliberately do NOT pre-convert videos to 384x384. The encoder
wrapper already does resize -> center-crop -> normalize at load time, so
pre-converting would only add a lossy re-encode and discard resolution. Videos
are stored at native resolution and sampled on demand.

Usage:
    .\\run.ps1 scripts\\collect_data.py --check          # verify setup only
    .\\run.ps1 scripts\\collect_data.py --exercise squats --limit 5
    .\\run.ps1 scripts\\collect_data.py                  # all exercises
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
import config

# Long tutorials are mostly a person talking to camera. Cap duration so we get
# demonstration clips rather than lectures.
MAX_DURATION_S = 180
MIN_DURATION_S = 5

# Search phrasing matters a lot for label quality. "form" and "technique"
# surface demonstration footage; plain exercise names surface vlogs and ads.
SEARCH_SUFFIX = "exercise form demonstration"


def check_setup():
    """Verify yt-dlp is importable and report ffmpeg status."""
    ok = True
    try:
        import yt_dlp
        print(f"  yt-dlp    : {yt_dlp.version.__version__}")
    except ImportError:
        print("  yt-dlp    : MISSING -> pip install yt-dlp")
        ok = False

    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, shell=True)
    if ff.returncode == 0:
        print("  ffmpeg    : present (enables format merging)")
    else:
        print("  ffmpeg    : absent (fine - we request single-file mp4)")

    return ok


def download(exercise, limit):
    """Download up to `limit` clips for one exercise. Returns count on disk."""
    out_dir = Path(config.CLIP_DIR) / exercise
    out_dir.mkdir(parents=True, exist_ok=True)
    before = len(list(out_dir.glob("*.mp4")))

    query = f"{exercise.replace('_', ' ')} {SEARCH_SUFFIX}"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"ytsearch{limit}:{query}",
        # Single-file mp4 avoids needing ffmpeg to merge separate streams.
        "-f", "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        "--match-filter", f"duration > {MIN_DURATION_S} & duration < {MAX_DURATION_S}",
        "--no-playlist",
        "--no-warnings",
        "--ignore-errors",
        "--no-overwrites",
        "--socket-timeout", "30",
        "--retries", "3",
        "--quiet", "--progress",
    ]

    print(f"\n{exercise}")
    print(f"  query: {query}")
    subprocess.run(cmd, timeout=1800)

    after = len(list(out_dir.glob("*.mp4")))
    print(f"  got {after - before} new ({after} total in {out_dir})")
    return after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify setup, download nothing")
    ap.add_argument("--exercise", help="single exercise (default: all)")
    ap.add_argument("--limit", type=int, default=30, help="clips per exercise")
    args = ap.parse_args()

    print("=" * 58)
    print("SETUP")
    print("=" * 58)
    if not check_setup():
        return 1
    print(f"  output    : {config.CLIP_DIR}/")
    print(f"  duration  : {MIN_DURATION_S}-{MAX_DURATION_S}s")

    if args.check:
        print("\n--check given, nothing downloaded.")
        return 0

    targets = [args.exercise] if args.exercise else config.EXERCISES
    if args.exercise and args.exercise not in config.EXERCISES:
        print(f"\nunknown exercise: {args.exercise}")
        print(f"choose from: {', '.join(config.EXERCISES)}")
        return 1

    print("\n" + "=" * 58)
    print(f"DOWNLOADING  ({len(targets)} exercises x up to {args.limit})")
    print("=" * 58)

    totals = {ex: download(ex, args.limit) for ex in targets}

    print("\n" + "=" * 58)
    print("SUMMARY")
    print("=" * 58)
    for ex, n in totals.items():
        flag = "" if n >= 20 else "   <- thin, needs more"
        print(f"  {ex:<18} {n:>4} clips{flag}")
    print(f"\n  total: {sum(totals.values())} clips")
    print("\nNext: review the clips by eye and delete bad ones before training.")
    print("Search results are noisy -- expect to throw some away.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
