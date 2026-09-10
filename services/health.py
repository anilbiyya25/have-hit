"""System health diagnostics: memory, GPU and disk headroom.

WHY THIS EXISTS
    Every hard failure this project has hit traced back to Windows COMMIT
    pressure, not to a bug in the model code:

      - torch.load dying with WinError 1455 (paging file too small)
      - the dataset scraper dying with MemoryError inside subprocess reads
      - the HF downloader failing to allocate 21 MB, as a Rust panic
      - a plain segmentation fault when several models loaded at once

    All of them are the same thing wearing different masks, and none of them
    say "you are out of commit". The failure surfaces wherever the next
    allocation happened to land, which is why they looked unrelated for so
    long. Commit was measured at 26.47 GB of a 27.45 GB limit -- 96% full --
    while the pagefile sat on a nearly-full C: drive.

    So the numbers get printed at startup. A refusal to allocate that arrives
    with the commit figure next to it is a five-minute diagnosis; the same
    refusal without it has cost this project days.

WHAT "COMMIT" MEANS HERE
    Windows' commit limit is physical RAM plus the pagefile. It caps how much
    memory all processes may RESERVE, whether or not they touch it. PyTorch
    reserves aggressively, so this ceiling is hit long before RAM looks full --
    which is why "I still have 2 GB free" and "allocation failed" are both true
    at the same time and neither explains the other.
"""

import ctypes
import shutil
import sys
from pathlib import Path

# A pagefile that leaves the commit limit under this cannot hold this project's
# working set: V-JEPA, SAM and a decoded video together reserve well past what
# 16 GB of RAM alone provides.
MIN_COMMIT_LIMIT_GB = 20.0

# Below this on either drive, the temp video writes and HF checkpoint pulls
# start failing in ways that look like corruption rather than a full disk.
MIN_FREE_DISK_GB = 5.0

GB = 1024 ** 3


class _MemoryStatusEx(ctypes.Structure):
    """Windows MEMORYSTATUSEX. Mirrors the Win32 struct field for field."""
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def commit_snapshot():
    """(used_gb, limit_gb) for the system commit charge.

    Read straight from GlobalMemoryStatusEx rather than through psutil.
    psutil.swap_memory() on Windows reports the pagefile as
    (total_pagefile - total_phys), which is not the commit limit and drifts
    from what Task Manager and perfmon show. ullTotalPageFile IS the commit
    limit, so this matches the number used to diagnose the failures above.

    Returns (None, None) off Windows, where the concept does not apply.
    """
    if not sys.platform.startswith("win"):
        try:
            import psutil
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            limit = (vm.total + sw.total) / GB
            return (limit - (vm.available + sw.free) / GB), limit
        except Exception:                                     # noqa: BLE001
            return None, None
    st = _MemoryStatusEx()
    st.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None, None
    limit = st.ullTotalPageFile / GB
    return limit - st.ullAvailPageFile / GB, limit


def memory_status():
    """Physical RAM and commit charge."""
    out = {"ram_total_gb": None, "ram_available_gb": None, "ram_percent": None,
           "commit_used_gb": None, "commit_limit_gb": None,
           "commit_percent": None, "pagefile_gb": None}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(vm.total / GB, 2)
        out["ram_available_gb"] = round(vm.available / GB, 2)
        out["ram_percent"] = vm.percent
    except Exception:                                         # noqa: BLE001
        pass

    used, limit = commit_snapshot()
    if limit:
        out["commit_used_gb"] = round(used, 2)
        out["commit_limit_gb"] = round(limit, 2)
        out["commit_percent"] = round(used / limit * 100, 1)
        # Commit limit minus RAM is what the pagefile contributes.
        if out["ram_total_gb"]:
            out["pagefile_gb"] = round(limit - out["ram_total_gb"], 2)
    return out


def gpu_status():
    """CUDA availability and free/total VRAM, without allocating anything."""
    out = {"cuda_available": False, "device": None, "vram_total_gb": None,
           "vram_free_gb": None, "vram_used_gb": None, "torch_version": None}
    try:
        import torch
        out["torch_version"] = torch.__version__
        if not torch.cuda.is_available():
            return out
        out["cuda_available"] = True
        out["device"] = torch.cuda.get_device_name(0)
        # mem_get_info reports the DRIVER's view, so it counts other processes
        # on the card too -- which is the number that decides whether the next
        # allocation succeeds. torch.cuda.memory_allocated() sees only us.
        free, total = torch.cuda.mem_get_info()
        out["vram_total_gb"] = round(total / GB, 2)
        out["vram_free_gb"] = round(free / GB, 2)
        out["vram_used_gb"] = round((total - free) / GB, 2)
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{exc.__class__.__name__}: {exc}"
    return out


def disk_status(workspace=None):
    """Free space on C: and on the workspace drive.

    Both matter and for different reasons: C: holds the pagefile (so a full C:
    caps the commit limit), while the workspace drive takes the temp video
    writes and the model cache.
    """
    workspace = Path(workspace or Path(__file__).resolve().parent.parent)
    out = {"drives": {}, "workspace": str(workspace)}
    seen = set()
    for label, path in (("system", "C:\\" if sys.platform.startswith("win") else "/"),
                        ("workspace", str(workspace))):
        try:
            anchor = Path(path).anchor or path
            if anchor in seen and label == "workspace":
                out["drives"][label] = {"same_as_system": True, "path": anchor}
                continue
            seen.add(anchor)
            u = shutil.disk_usage(path)
            out["drives"][label] = {
                "path": anchor,
                "total_gb": round(u.total / GB, 2),
                "free_gb": round(u.free / GB, 2),
                "percent_used": round((u.total - u.free) / u.total * 100, 1),
            }
        except Exception as exc:                              # noqa: BLE001
            out["drives"][label] = {"path": path, "error": str(exc)}
    return out


def system_report(workspace=None):
    """Everything at once, plus any warnings worth acting on."""
    mem, gpu, disk = memory_status(), gpu_status(), disk_status(workspace)
    warnings = []

    limit = mem.get("commit_limit_gb")
    if limit is not None and limit < MIN_COMMIT_LIMIT_GB:
        pf = mem.get("pagefile_gb")
        warnings.append(
            f"commit limit is {limit} GB, below the {MIN_COMMIT_LIMIT_GB} GB "
            f"this workload needs (pagefile contributes {pf} GB). Long runs "
            "may fail as WinError 1455, MemoryError, or a bare segfault -- "
            "none of which name the real cause. Raise the pagefile, "
            "preferably on the workspace drive.")

    pct = mem.get("commit_percent")
    if pct is not None and pct >= 85:
        warnings.append(
            f"commit is already {pct}% full at startup "
            f"({mem['commit_used_gb']}/{mem['commit_limit_gb']} GB). "
            "Close other applications before a long run.")

    if not gpu["cuda_available"]:
        warnings.append("CUDA is unavailable; inference will fall back to CPU "
                        "and take minutes per clip rather than seconds.")
    elif gpu.get("vram_free_gb") is not None and gpu["vram_free_gb"] < 1.5:
        warnings.append(
            f"only {gpu['vram_free_gb']} GB VRAM free of "
            f"{gpu['vram_total_gb']} GB. Stage A and Stage C load backbones "
            "sequentially and need roughly 1.5 GB between them.")

    for label, d in disk["drives"].items():
        free = d.get("free_gb")
        if free is not None and free < MIN_FREE_DISK_GB:
            extra = (" This drive holds the pagefile, so filling it also "
                     "lowers the commit limit." if label == "system" else "")
            warnings.append(f"{label} drive {d.get('path')} has {free} GB "
                            f"free.{extra}")

    return {"memory": mem, "gpu": gpu, "disk": disk, "warnings": warnings,
            "healthy": not warnings}


def log_startup_diagnostics(workspace=None, printer=print):
    """Print the report as a compact block. Returns it for /health to reuse."""
    r = system_report(workspace)
    m, g, d = r["memory"], r["gpu"], r["disk"]

    printer("-" * 66)
    printer("  system check")
    printer(f"    RAM        {m['ram_available_gb']} GB free of "
            f"{m['ram_total_gb']} GB ({m['ram_percent']}% used)")
    printer(f"    commit     {m['commit_used_gb']} / {m['commit_limit_gb']} GB "
            f"({m['commit_percent']}% used, pagefile {m['pagefile_gb']} GB)")
    if g["cuda_available"]:
        printer(f"    GPU        {g['device']}")
        printer(f"               {g['vram_free_gb']} GB VRAM free of "
                f"{g['vram_total_gb']} GB  (torch {g['torch_version']})")
    else:
        printer(f"    GPU        UNAVAILABLE (torch {g['torch_version']})")
    for label, info in d["drives"].items():
        if info.get("same_as_system"):
            printer(f"    {label:10} same volume as system")
        elif "error" not in info:
            printer(f"    {label:10} {info['path']} {info['free_gb']} GB free "
                    f"of {info['total_gb']} GB")

    for w in r["warnings"]:
        printer(f"    WARNING    {w}")
    if not r["warnings"]:
        printer("    status     all clear")
    printer("-" * 66)
    return r
