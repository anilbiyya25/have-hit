"""Production launcher for the Have Hit API.

    python run_server.py                 # 0.0.0.0:8000
    python run_server.py --port 8080
    python run_server.py --host 127.0.0.1 --no-preflight

WHY A LAUNCHER AND NOT JUST `uvicorn app:app`
    Three things have to be true before the first request, and none of them
    happen on their own:

    1. STDOUT MUST BE UNBUFFERED. Python block-buffers stdout when it is a
       pipe or a file rather than a terminal, so `python -m uvicorn ... > log`
       holds every print in an 8 KB buffer until the process exits. That is
       how the startup memory diagnostics, the naming discrepancy warnings and
       the health check all went missing from a redirected log while appearing
       perfectly on a terminal -- the logging worked, the reading of it did
       not. A warning nobody can read while the server runs is not a warning.

    2. THE CACHES MUST POINT AT D:. C: holds the pagefile and has very little
       room; HuggingFace, torch.hub and the temp files are gigabytes. run.ps1
       sets these, so anything started through it is fine and anything started
       directly is not -- which is a trap, because both look identical until
       C: fills up.

    3. THE API KEY MUST BE FOUND. It is stored in the User environment block,
       and a shell that was already running when it was set does not have it.
       This re-reads the registry rather than reporting gemini=false at a user
       who set the key correctly ten minutes ago.

    The preflight snapshot prints BEFORE the app is imported. That ordering is
    the point of it: if loading V-JEPA dies of commit pressure, the numbers
    that explain why are already on screen.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def unbuffer():
    """Make stdout/stderr line-buffered for the rest of this process.

    PYTHONUNBUFFERED is set as well, but on its own it would do nothing here:
    the interpreter reads it at startup, and by the time this module runs the
    streams already exist with their buffering decided. The env var is for
    anything this process spawns; reconfigure() is for this process.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass


def set_cache_paths():
    """Point every cache at D:, matching run.ps1.

    UNCONDITIONALLY, and that is the whole point of the function. The obvious
    version keeps any value the environment already has -- and TMP always
    already has one, the Windows default under C:\\Users\\...\\AppData\\Local.
    Inheriting it sends recording chunks, temp uploads and HF downloads to the
    drive with 8 GB free that also holds the pagefile, which is the exact
    failure this project keeps hitting. An inherited default is not a user's
    choice, so it is not treated as one.
    """
    cache = HERE / ".cache"
    for name, path in {
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "TMP": cache / "tmp",
        "TEMP": cache / "tmp",
        "PIP_CACHE_DIR": cache / "pip",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }.items():
        os.environ[name] = str(path)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    Path(os.environ["TMP"]).mkdir(parents=True, exist_ok=True)


def recover_api_key():
    """Pull the key from the User environment block if this process lacks it.

    The exact failure this prevents, which happened once already: the key was
    set with SetEnvironmentVariable(..., "User") while the parent shell was
    already running, so the server inherited a stale block, started with no
    key, and reported gemini=false while the key sat correctly in the
    registry. Nothing about that is visible from the server's side.
    """
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                stored, _ = winreg.QueryValueEx(k, name)
        except OSError:
            stored = None
        live = os.environ.get(name)

        if live and stored and live != stored:
            # The process value wins -- an explicitly exported key should beat
            # the stored one -- but SILENTLY preferring it is the trap. This
            # exact case cost a debugging round: a key was set in the registry
            # while the launching shell still carried the previous one, so the
            # server booted on the old, quota-exhausted key and reported
            # gemini=true the whole time. Same symptom either way; only the
            # tail of the key tells them apart.
            return (f"{name} (process copy ...{live[-4:]} IN USE, but the "
                    f"registry holds a DIFFERENT key ...{stored[-4:]} -- "
                    f"open a new shell to pick up the stored one)")
        if live:
            return f"{name} ...{live[-4:]} (already in this process)"
        if stored:
            os.environ[name] = stored
            return f"{name} ...{stored[-4:]} (recovered from the registry)"
    return None


def preflight():
    """Machine state before the app -- and the model -- is loaded.

    The memory block itself comes from services/health.py rather than being
    formatted again here: it is the same report /health serves and the same
    one the lifespan logs, and a second copy of that formatting would drift
    from the first. This adds only what the LAUNCHER knows and the app cannot
    -- which interpreter, which caches, and whether the key was found.

    This does load torch, via gpu_status. It does not load the 1.2 GB of
    V-JEPA weights, which is the part that fails when commit is tight, so the
    ordering still does its job: if the load dies, these numbers are already
    on screen and name the reason.
    """
    from services import health

    print("=" * 66)
    print("  HAVE HIT :: PREFLIGHT")
    print(f"    python     {sys.version.split()[0]}")
    print(f"    exe        {sys.executable}")
    print(f"    workdir    {HERE}")
    print(f"    HF_HOME    {os.environ.get('HF_HOME')}")
    print(f"    TMP        {os.environ.get('TMP')}")
    key = recover_api_key()
    print(f"    gemini key {key or 'NOT SET -- reports will be local-only'}")

    rep = health.log_startup_diagnostics(workspace=HERE)
    # The lifespan runs the same check and prints it too. One block is
    # diagnostics; two identical blocks thirty lines apart is noise that
    # trains people to skip past the warnings in it. This tells the app the
    # numbers are already on screen -- it still MEASURES them, because
    # /health serves that snapshot, it just does not print them again.
    os.environ["HH_PREFLIGHT_DONE"] = "1"
    return rep


def main():
    ap = argparse.ArgumentParser(description="Launch the Have Hit API server.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the memory snapshot")
    args = ap.parse_args()

    unbuffer()
    set_cache_paths()
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))

    if not args.no_preflight:
        try:
            preflight()
        except Exception as exc:                              # noqa: BLE001
            # A broken diagnostic must never stop the server it was meant to
            # help diagnose.
            print(f"  preflight unavailable: {exc.__class__.__name__}: {exc}")
    else:
        recover_api_key()

    import uvicorn

    print(f"starting uvicorn on {args.host}:{args.port} (1 worker)")
    # ONE worker, not a pool. Each worker is a separate process that would
    # load its own V-JEPA and its own SAM; two of those do not fit in 4 GB of
    # VRAM, and the pipeline is already serialised behind a semaphore inside
    # the process, so a second worker would buy queueing bugs and no
    # throughput. reload is off for the same reason -- a reload reloads the
    # backbones.
    uvicorn.run("app:app", host=args.host, port=args.port, workers=1,
                reload=False, log_level=args.log_level, access_log=True)


if __name__ == "__main__":
    main()
