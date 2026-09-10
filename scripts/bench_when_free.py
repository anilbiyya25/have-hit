"""Wait for RAM to free up, then benchmark V-JEPA 2.1 ViT-L.

Runs detached so it survives the IDE (and this Claude session) closing.
Loading the 4.8 GB ViT-L checkpoint needs ~6 GB free; this polls until that is
available, then runs and writes results to vitl_result.txt.

Launch:
    Start-Process -WindowStyle Hidden D:\\have-hit\\venv\\Scripts\\python.exe `
        -ArgumentList '-u','scripts\\bench_when_free.py'
"""

import os
import subprocess
import sys
import time
from datetime import datetime

import psutil

NEED_GB = 6.0
POLL_SECONDS = 10
TIMEOUT_MINUTES = 30

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "vitl_result.txt")
PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RESULT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    open(RESULT, "w", encoding="utf-8").close()
    log(f"waiting for {NEED_GB} GB free RAM (timeout {TIMEOUT_MINUTES} min)")
    log(f"currently free: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    deadline = time.time() + TIMEOUT_MINUTES * 60
    while time.time() < deadline:
        avail = psutil.virtual_memory().available / 1024**3
        if avail >= NEED_GB:
            log(f"RAM available: {avail:.1f} GB -- starting benchmark")
            break
        time.sleep(POLL_SECONDS)
    else:
        log(f"TIMEOUT -- never saw {NEED_GB} GB free. Benchmark not run.")
        return 1

    env = dict(os.environ, TORCH_HOME="D:\\torch-hub", HF_HOME="D:\\hf-cache",
               PYTHONUNBUFFERED="1")
    proc = subprocess.run(
        [PYTHON, "-u", "scripts/bench_variant.py", "vjepa2_1_vit_large_384"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )

    with open(RESULT, "a", encoding="utf-8") as f:
        f.write("\n===== STDOUT =====\n")
        f.write(proc.stdout or "(none)")
        if proc.returncode != 0:
            f.write("\n===== STDERR =====\n")
            f.write((proc.stderr or "(none)")[-4000:])
        f.write(f"\n\nexit code: {proc.returncode}\n")

    log(f"done, exit {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
