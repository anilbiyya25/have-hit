"""Download a single YouTube video to the project folder.

Handles format-selection failures automatically by walking a fallback chain,
since the "best" format for a given video may not exist or may be split into
separate video/audio streams that need ffmpeg to merge.

Usage:
    .\\run.ps1 scripts\\download_video.py https://www.youtube.com/watch?v=VIDEO_ID
    .\\run.ps1 scripts\\download_video.py <url> --out my_clip.mp4
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Tried in order. Each entry is (format_string, description).
# Earlier entries are single-file (no ffmpeg needed); later ones may require
# merging, which we only attempt if ffmpeg is present.
FORMAT_CHAIN = [
    ("best[ext=mp4][height<=720]", "single-file mp4, <=720p"),
    ("best[ext=mp4]", "single-file mp4, any resolution"),
    ("best[height<=720]", "single-file any container, <=720p"),
    ("best", "single-file, best available"),
    ("bestvideo[ext=mp4]+bestaudio[ext=m4a]", "merged mp4 (needs ffmpeg)"),
    ("bestvideo+bestaudio", "merged, any container (needs ffmpeg)"),
]

VIDEO_URL_RE = re.compile(
    r"(youtube\.com/(watch\?|shorts/|embed/|live/)|youtu\.be/)", re.I
)


def has_ffmpeg():
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, shell=True)
        return r.returncode == 0
    except Exception:
        return False


def validate_url(url):
    """Reject URLs that point at no specific video."""
    if not url.startswith(("http://", "https://")):
        return False, "not an http(s) URL"
    if not VIDEO_URL_RE.search(url):
        return False, (
            "no video id found. This looks like a channel, playlist, or the\n"
            "    homepage rather than a single video. Expected something like:\n"
            "      https://www.youtube.com/watch?v=VIDEO_ID\n"
            "      https://youtu.be/VIDEO_ID"
        )
    return True, ""


def probe(url):
    """Fetch title/duration without downloading. Confirms the URL resolves."""
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download",
           "--print", "%(title)s|%(duration)s|%(resolution)s", "--no-warnings", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    parts = r.stdout.strip().splitlines()[0].split("|")
    return {
        "title": parts[0] if parts else "?",
        "duration": parts[1] if len(parts) > 1 else "?",
        "resolution": parts[2] if len(parts) > 2 else "?",
    }


def attempt(url, out_path, fmt):
    """One download attempt. Returns (ok, stderr_tail)."""
    # yt-dlp appends the real extension; give it a template without one so it
    # cannot produce e.g. test_input.mp4.webm
    stem = out_path.with_suffix("")
    cmd = [
        sys.executable, "-m", "yt_dlp", url,
        "-f", fmt,
        "-o", f"{stem}.%(ext)s",
        "--no-playlist",
        "--no-warnings",
        "--force-overwrites",
        "--socket-timeout", "30",
        "--retries", "3",
        "--quiet", "--progress",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return r.returncode == 0, (r.stderr or "")[-400:]


def find_output(out_path):
    """yt-dlp may have written a different extension than requested."""
    stem = out_path.with_suffix("")
    matches = sorted(stem.parent.glob(stem.name + ".*"))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube video URL")
    ap.add_argument("--out", default="test_input.mp4", help="output filename")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print("YOUTUBE VIDEO DOWNLOADER")
    print("=" * 58)
    print(f"url    : {args.url}")
    print(f"output : {out_path}")

    ok, why = validate_url(args.url)
    if not ok:
        print(f"\nINVALID URL: {why}")
        return 2

    ffmpeg = has_ffmpeg()
    print(f"ffmpeg : {'present' if ffmpeg else 'absent (merged formats skipped)'}")

    print("\nprobing...")
    info = probe(args.url)
    if info is None:
        print("  could not resolve this URL. It may be private, removed,")
        print("  region-blocked, or age-restricted.")
        return 3
    print(f"  title      : {info['title'][:60]}")
    print(f"  duration   : {info['duration']}s")
    print(f"  resolution : {info['resolution']}")

    print("\ndownloading...")
    for fmt, desc in FORMAT_CHAIN:
        needs_merge = "+" in fmt
        if needs_merge and not ffmpeg:
            print(f"  skip  : {desc}")
            continue

        print(f"  try   : {desc}")
        ok, err = attempt(args.url, out_path, fmt)
        if ok:
            written = find_output(out_path)
            if written and written.stat().st_size > 0:
                size_mb = written.stat().st_size / 1024**2
                print(f"\nOK  saved {written.name}  ({size_mb:.1f} MB)")
                if written.suffix.lower() != out_path.suffix.lower():
                    print(f"    note: container is {written.suffix}, not"
                          f" {out_path.suffix} -- no mp4 stream was available.")
                print(f"    path: {written}")
                return 0
            print("         reported success but no file appeared")
        else:
            first = err.strip().splitlines()
            print(f"         failed: {first[-1][:90] if first else 'unknown error'}")

    print("\nAll formats failed. The video may be DRM-protected or unavailable.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
