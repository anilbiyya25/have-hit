"""Benchmark any V-JEPA 2.1 variant from Meta's official checkpoints.

Usage:
    .\\run.ps1 scripts\\bench_variant.py vjepa2_1_vit_base_384
    .\\run.ps1 scripts\\bench_variant.py vjepa2_1_vit_large_384

Same clip length and measurement protocol as scripts/check_env.py, so numbers
are comparable across variants and against V-JEPA 2.0.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
import config

FRAMES = config.NUM_FRAMES
RES = 384


def load_encoder(entrypoint):
    torch.hub.list(config.HUB_REPO, trust_repo=True)
    repo_dir = os.path.join(
        torch.hub.get_dir(), config.HUB_REPO.replace("/", "_") + "_main"
    )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from src.hub import backbones

    if "localhost" in backbones.VJEPA_BASE_URL:
        backbones.VJEPA_BASE_URL = config.VJEPA_BASE_URL

    # NOTE: the ViT-L checkpoint is 4.8 GB because Meta ships optimizer and
    # scaler state alongside the weights, and torch.load deserialises the whole
    # archive before anything can be discarded. It needs ~6 GB free RAM.
    # torch.load(mmap=True) is the obvious fix but segfaults (0xC0000005) on
    # Windows for files over 4 GB, so free the RAM instead -- see
    # scripts/bench_when_free.py.
    encoder, _ = getattr(backbones, entrypoint)(pretrained=True)
    return encoder


def main():
    entrypoint = sys.argv[1] if len(sys.argv) > 1 else config.HUB_ENTRYPOINT
    print(f"variant        : {entrypoint}")

    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / 1024**2
    print(f"gpu            : {props.name}  ({total_mb:.0f} MB)")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    encoder = load_encoder(entrypoint)
    print(f"loaded in      : {time.time() - t0:.1f}s")

    encoder = encoder.to("cuda", dtype=torch.float32).eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    weights_mb = torch.cuda.memory_allocated() / 1024**2

    clip = torch.randn(1, 3, FRAMES, RES, RES, device="cuda", dtype=torch.float32)
    amp = torch.autocast("cuda", dtype=torch.float16)

    with torch.no_grad(), amp:
        out = encoder(clip)
    if isinstance(out, (list, tuple)):
        out = out[-1]

    for _ in range(2):
        with torch.no_grad(), amp:
            encoder(clip)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad(), amp:
            encoder(clip)
        torch.cuda.synchronize()
        times.append((time.time() - t) * 1000)

    reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    med = np.median(times)

    print(f"\n{'=' * 46}")
    print(f"params         : {n_params / 1e6:.1f}M")
    print(f"tokens/clip    : {out.shape[1]}")
    print(f"embed dim      : {out.shape[-1]}")
    print(f"weights        : {weights_mb:.0f} MB")
    print(f"peak reserved  : {reserved_mb:.0f} MB  (of {total_mb:.0f} MB)")
    print(f"latency median : {med:.0f} ms")
    print(f"latency min/max: {min(times):.0f} / {max(times):.0f} ms")
    print(f"fits in 4GB    : {'yes' if reserved_mb < total_mb * 0.9 else 'NO'}")
    print(f"{'=' * 46}")


if __name__ == "__main__":
    main()
