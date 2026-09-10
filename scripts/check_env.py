"""Environment + VRAM reality check.

Answers the one question that decides the architecture: does V-JEPA 2 ViT-L
actually fit and run fast enough on this 4GB laptop GPU?

Run:  D:\\have-hit\\venv\\Scripts\\python.exe scripts\\check_env.py
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
import config


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    header("ENVIRONMENT")
    print(f"python           : {sys.version.split()[0]}")
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\nNo CUDA. Everything below would run on CPU and be far too slow.")
        return 1

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    print(f"gpu              : {props.name}")
    print(f"vram total       : {total_gb:.2f} GB")
    print(f"compute cap      : {props.major}.{props.minor}")

    header("MODEL")
    print(f"model id         : {config.MODEL_ID}")
    print(f"clip shape       : {config.NUM_FRAMES} frames @ {config.CROP_SIZE}px")
    print(f"tokens per clip  : {config.NUM_TOKENS}")
    print(f"dtype            : {config.DTYPE}")
    print("\nloading (first run downloads ~1.2GB)...")

    t0 = time.time()
    from models.world_model_wrapper import VJEPAEncoder
    enc = VJEPAEncoder()
    load_s = time.time() - t0

    n_params = sum(p.numel() for p in enc.model.parameters())
    weights_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"loaded in        : {load_s:.1f}s")
    print(f"parameters       : {n_params / 1e6:.0f}M")
    print(f"weights on gpu   : {weights_mb:.0f} MB")

    header("INFERENCE BENCHMARK")
    # Random clip standing in for a real camera window.
    clip = np.random.randint(
        0, 255, (config.NUM_FRAMES, config.CROP_SIZE, config.CROP_SIZE, 3), dtype=np.uint8
    )

    print("warmup...")
    for _ in range(2):
        enc.embed(clip)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.time()
        vec = enc.embed(clip)
        torch.cuda.synchronize()
        times.append((time.time() - t) * 1000)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    print(f"embedding shape  : {vec.shape}")
    print(f"latency median   : {np.median(times):.0f} ms")
    print(f"latency min/max  : {min(times):.0f} / {max(times):.0f} ms")
    print(f"peak allocated   : {peak_mb:.0f} MB")
    print(f"peak reserved    : {reserved_mb:.0f} MB  (of {total_gb * 1024:.0f} MB)")
    print(f"headroom         : {total_gb * 1024 - reserved_mb:.0f} MB")

    header("VERDICT")
    fits = reserved_mb < total_gb * 1024 * 0.9
    fast = np.median(times) < 500
    print(f"fits in VRAM     : {'YES' if fits else 'NO - too tight'}")
    print(f"usable latency   : {'YES' if fast else 'NO - too slow'}  (<500ms target)")
    print(f"throughput       : {1000 / np.median(times):.1f} clips/sec")
    return 0 if (fits and fast) else 1


if __name__ == "__main__":
    sys.exit(main())
