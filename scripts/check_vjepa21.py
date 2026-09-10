"""Benchmark V-JEPA 2.1 ViT-B/384 from Meta's official checkpoint.

Deliberately mirrors scripts/check_env.py so numbers are comparable against
V-JEPA 2.0 ViT-L/256 measured on the same GPU.

Upstream bug workaround: facebookresearch/vjepa2 @ main ships
    VJEPA_BASE_URL = "http://localhost:8300"
with the real fbaipublicfiles URL commented out just above it -- a local dev
override someone committed. Every pretrained torch.hub entrypoint in that repo
is broken by it. We restore the real URL at runtime rather than editing the
cached clone, so a re-clone can't silently reintroduce it.

Weights still come from Meta's own CDN. No third-party conversions.

Run:  .\\run.ps1 scripts\\check_vjepa21.py
"""

import os
import sys
import time

import numpy as np
import torch

REAL_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

FRAMES = 16
RES = 384
TUBELET = 2
PATCH = 16
TOKENS = (FRAMES // TUBELET) * (RES // PATCH) ** 2


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_encoder():
    """Clone the repo via hub, patch the bad URL, then build the encoder."""
    # Ensures the repo is cloned/cached; we ignore the returned entrypoint list.
    torch.hub.list("facebookresearch/vjepa2", trust_repo=True)

    repo_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_vjepa2_main")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from src.hub import backbones

    if "localhost" in backbones.VJEPA_BASE_URL:
        print(f"  patching base url: {backbones.VJEPA_BASE_URL} -> {REAL_BASE_URL}")
        backbones.VJEPA_BASE_URL = REAL_BASE_URL

    # Returns (encoder, predictor). We only want the encoder -- the predictor
    # exists for the JEPA pretraining objective.
    encoder, _predictor = backbones.vjepa2_1_vit_base_384(pretrained=True)
    return encoder


def main():
    header("ENVIRONMENT")
    print(f"torch          : {torch.__version__}")
    print(f"cuda           : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / 1024**2
    print(f"gpu            : {props.name}")
    print(f"vram total     : {total_mb:.0f} MB")

    header("MODEL: V-JEPA 2.1 ViT-B/384 (official Meta checkpoint)")
    print(f"clip           : {FRAMES} frames @ {RES}px")
    print(f"tokens/clip    : {TOKENS}")
    print("\nloading (first run downloads 1.55 GB)...")

    t0 = time.time()
    encoder = load_encoder()
    print(f"loaded in      : {time.time() - t0:.1f}s")

    # Keep weights in fp32 and use autocast rather than a hard .half().
    # The 2.1 RoPE path computes query/key in fp32 while value stays in the
    # input dtype, so a hard cast makes SDPA fail on mismatched dtypes.
    # autocast promotes correctly and still gets fp16 matmuls.
    encoder = encoder.to("cuda", dtype=torch.float32).eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    weights_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"encoder params : {n_params / 1e6:.1f}M")
    print(f"weights on gpu : {weights_mb:.0f} MB  (fp32 + autocast)")

    header("INFERENCE BENCHMARK")
    clip = torch.randn(1, 3, FRAMES, RES, RES, device="cuda", dtype=torch.float32)
    print(f"input shape    : {tuple(clip.shape)}")

    amp = torch.autocast("cuda", dtype=torch.float16)
    with torch.no_grad(), amp:
        out = encoder(clip)
    if isinstance(out, (list, tuple)):
        print(f"output         : {type(out).__name__} of {len(out)}")
        out = out[-1]
    print(f"output shape   : {tuple(out.shape)}")
    print(f"embed dim      : {out.shape[-1]}")

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

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    med = np.median(times)

    print(f"latency median : {med:.0f} ms")
    print(f"latency min/max: {min(times):.0f} / {max(times):.0f} ms")
    print(f"peak allocated : {peak_mb:.0f} MB")
    print(f"peak reserved  : {reserved_mb:.0f} MB  (of {total_mb:.0f} MB)")

    header("HEAD TO HEAD (same GPU, same 16-frame clip)")
    print(f"{'':16}{'2.0 ViT-L/256':>16}{'2.1 ViT-B/384':>16}")
    print(f"{'-' * 48}")
    print(f"{'params':16}{'326M':>16}{f'{n_params / 1e6:.0f}M':>16}")
    print(f"{'tokens/clip':16}{'2048':>16}{TOKENS:>16}")
    print(f"{'embed dim':16}{'1024':>16}{out.shape[-1]:>16}")
    print(f"{'weights':16}{'624 MB':>16}{f'{weights_mb:.0f} MB':>16}")
    print(f"{'peak reserved':16}{'736 MB':>16}{f'{reserved_mb:.0f} MB':>16}")
    print(f"{'latency':16}{'193 ms':>16}{f'{med:.0f} ms':>16}")
    print(f"{'fits in 4GB':16}{'yes':>16}{'yes' if reserved_mb < total_mb * 0.9 else 'NO':>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
