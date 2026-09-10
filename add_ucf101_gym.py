"""Add six gym exercises from UCF101 to gym_dataset/.

    .\\run.ps1 add_ucf101_gym.py                 # download, extract, install
    .\\run.ps1 add_ucf101_gym.py --check         # probe the server, do nothing
    .\\run.ps1 add_ucf101_gym.py --cleanup       # drop the archive when done

There is no per-class download. CRCV publishes UCF101 as ONE 6.46 GB RAR
covering all 101 action classes; the server exposes no per-class slices and no
index that would let a range request pull a single class out. So the whole
archive comes down once, and only the six wanted class folders are unpacked
from it. The archive is kept by default -- re-downloading 6.46 GB to recover
from a bad unpack is a far worse trade than 6.46 GB of disk on a drive with
180 GB free.

Resuming, not restarting: the server sends Accept-Ranges: bytes, so a dropped
connection continues from the byte already on disk rather than starting over.
That matters at this size -- a 90% complete transfer dying at hour two must
not throw away the first 5.8 GB.

Extraction uses 7-Zip, which reads RAR natively. Python has no stdlib RAR
support and `rarfile` would still need an external unrar binary, so shelling
out to the 7z already installed here removes a dependency rather than adding
one.

UCF101 clips are .avi (Xvid, 320x240, ~7 s). They are NOT transcoded to mp4:
cv2 decodes Xvid directly, and re-encoding would burn hours to make the
footage strictly worse. config.VIDEO_EXTS is what teaches the pipeline to see
them.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

# crcv.ucf.edu serves an incomplete certificate chain -- it omits the
# intermediate. Windows fetches the missing intermediate on its own, so
# PowerShell reaches the host fine, but OpenSSL against certifi's bundle
# cannot and fails with "unable to get local issuer certificate". Routing
# verification through the OS trust store fixes it properly; the alternative
# people reach for, verify=False, would leave 6.5 GB of binary unauthenticated.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    print("!!! truststore not installed -- crcv.ucf.edu will likely fail TLS.")
    print("    pip install truststore")

sys.path.insert(0, ".")
import config

DATASET_DIR = Path("gym_dataset")
ARCHIVE = Path("D:/ucf101_download/UCF101.rar")

MIRRORS = [
    "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
    "https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar",
]

# UCF101 folder name -> our snake_case label. The archive's top level is
# "UCF-101/<ClassName>/v_<ClassName>_g##_c##.avi".
UCF_CLASSES = {
    "PullUps": "pull_ups",
    "CleanAndJerk": "clean_and_jerk",
    "HandstandPushups": "handstand_pushups",
    "WallPushups": "wall_pushups",
    "JumpRope": "jump_rope",
    "JumpingJack": "jumping_jack",
}

CHUNK = 1 << 20          # 1 MiB
MAX_ATTEMPTS = 40        # a long transfer on flaky wifi needs a lot of retries
BACKOFF_CAP = 60.0


def find_7z():
    """7z.exe, or None. Checks PATH then the usual install locations."""
    found = shutil.which("7z") or shutil.which("7za")
    if found:
        return found
    for p in (r"C:\Program Files\7-Zip\7z.exe",
              r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(p).exists():
            return p
    return None


def probe():
    """First reachable mirror as (url, size_bytes, resumable)."""
    for url in MIRRORS:
        try:
            r = requests.head(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            size = int(r.headers.get("Content-Length", 0))
            resumable = r.headers.get("Accept-Ranges", "").lower() == "bytes"
            print(f"  reachable : {url}")
            print(f"  size      : {size / 1e9:.2f} GB")
            print(f"  resumable : {resumable}")
            return url, size, resumable
        except requests.RequestException as exc:
            print(f"  unreachable: {url}  ({exc.__class__.__name__})")
    return None, 0, False


def download(url, total, dest):
    """Fetch `url` to `dest`, resuming across dropouts. True if complete.

    Every failure mode here is transient by assumption: the loop re-probes
    what is already on disk and asks for the remainder, so a reset connection
    costs one retry rather than the whole transfer.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and dest.stat().st_size == total:
        print(f"  already complete: {dest}")
        return True

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        have = part.stat().st_size if part.exists() else 0
        if have >= total > 0:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if have and r.status_code not in (206, 416):
                    # Server ignored the range; restarting is the only option.
                    print(f"  !!! resume refused (HTTP {r.status_code}), restarting")
                    part.unlink(missing_ok=True)
                    attempt += 1
                    continue
                r.raise_for_status()

                t0, last, got = time.time(), time.time(), have
                with open(part, "ab") as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 15:
                            pct = 100 * got / total if total else 0
                            rate = (got - have) / max(time.time() - t0, 1e-6) / 1e6
                            eta = (total - got) / max((got - have) /
                                                      max(time.time() - t0, 1e-6), 1)
                            print(f"    {got / 1e9:5.2f}/{total / 1e9:.2f} GB "
                                  f"({pct:5.1f}%)  {rate:5.1f} MB/s  "
                                  f"eta {eta / 60:4.1f} min")
                            last = time.time()
        except (requests.RequestException, OSError) as exc:
            attempt += 1
            wait = min(2 ** min(attempt, 6), BACKOFF_CAP)
            done = part.stat().st_size if part.exists() else 0
            print(f"  !!! {exc.__class__.__name__} at {done / 1e9:.2f} GB "
                  f"-- retry {attempt}/{MAX_ATTEMPTS} in {wait:.0f}s")
            time.sleep(wait)
            continue

        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        attempt += 1
        print(f"  stream ended early at {have / 1e9:.2f} GB, resuming "
              f"({attempt}/{MAX_ATTEMPTS})")
        time.sleep(2)

    final = part.stat().st_size if part.exists() else 0
    if total and final != total:
        print(f"  !!! incomplete: {final / 1e9:.2f} of {total / 1e9:.2f} GB")
        return False

    part.replace(dest)
    print(f"  downloaded -> {dest}  ({final / 1e9:.2f} GB)")
    return True


def extract(seven_zip, archive, staging):
    """Unpack only the wanted class folders. Returns {ucf_name: n_files}."""
    staging.mkdir(parents=True, exist_ok=True)
    counts = {}

    for ucf_name in UCF_CLASSES:
        target = staging / "UCF-101" / ucf_name
        if target.is_dir() and any(target.iterdir()):
            counts[ucf_name] = len(list(target.glob("*.avi")))
            print(f"  {ucf_name:<20} already staged ({counts[ucf_name]})")
            continue

        print(f"  {ucf_name:<20} extracting...")
        proc = subprocess.run(
            [seven_zip, "x", str(archive), f"-o{staging}",
             f"UCF-101/{ucf_name}/*", "-r", "-y"],
            capture_output=True, text=True, timeout=3600,
        )
        n = len(list(target.glob("*.avi"))) if target.is_dir() else 0
        counts[ucf_name] = n
        if n == 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            print(f"  !!! nothing extracted for {ucf_name}: {' | '.join(tail)}")

    return counts


def install(staging):
    """Move staged classes into gym_dataset/ under our snake_case names."""
    installed = {}
    for ucf_name, label in UCF_CLASSES.items():
        src = staging / "UCF-101" / ucf_name
        dst = DATASET_DIR / label
        dst.mkdir(parents=True, exist_ok=True)

        moved = 0
        if src.is_dir():
            for f in src.glob("*.avi"):
                target = dst / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
                    moved += 1
        installed[label] = (moved, len(list(dst.glob("*.avi"))))
    return installed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="probe the server and exit")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the archive and staging after a good install")
    ap.add_argument("--archive", default=str(ARCHIVE))
    args = ap.parse_args()

    archive = Path(args.archive)
    staging = archive.parent / "staging"

    print("=" * 66)
    print("UCF101 -> gym_dataset")
    print("=" * 66)
    print(f"  classes wanted: {', '.join(UCF_CLASSES.values())}")

    seven_zip = find_7z()
    print(f"  7-Zip         : {seven_zip or 'NOT FOUND'}")
    if not seven_zip:
        print("\n  Cannot unpack RAR without 7-Zip. Install it, then re-run:")
        print("    scoop install 7zip     (or https://www.7-zip.org)")
        return 1

    print("\n-- server ---------------------------------------------------")
    url, total, resumable = probe()
    if not url:
        print("\n  No mirror reachable. Check the network and re-run;")
        print("  the download resumes from whatever is already on disk.")
        return 1
    if args.check:
        print("\n  --check only, stopping here.")
        return 0

    free = shutil.disk_usage(archive.parent.anchor).free
    need = total * 2                      # archive + extracted copy
    print(f"  disk free     : {free / 1e9:.1f} GB (need ~{need / 1e9:.1f} GB)")
    if free < need:
        print("  !!! not enough free space")
        return 1

    print("\n-- download -------------------------------------------------")
    if not download(url, total, archive):
        print("\n  Download did not complete. Re-run to resume.")
        return 1

    print("\n-- extract --------------------------------------------------")
    counts = extract(seven_zip, archive, staging)

    print("\n-- install --------------------------------------------------")
    installed = install(staging)

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  {'label':<22}{'moved':>8}{'on disk':>10}")
    total_new = 0
    for label, (moved, on_disk) in installed.items():
        flag = "" if on_disk else "   <- EMPTY"
        print(f"  {label:<22}{moved:>8}{on_disk:>10}{flag}")
        total_new += on_disk
    print(f"  {'TOTAL':<22}{'':>8}{total_new:>10}")

    if args.cleanup and total_new:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
        print(f"\n  cleaned up {archive} and staging")

    print("\n  These are .avi. config.VIDEO_EXTS must include '.avi' or the")
    print("  extractor will not see a single one of them.")
    return 0 if total_new else 1


if __name__ == "__main__":
    sys.exit(main())
