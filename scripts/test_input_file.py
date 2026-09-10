"""Verify a downloaded file is usable by the V-JEPA pipeline."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import config
from models.vjepa_wrapper import VJEPAEncoder
from pipeline.clip_sampler import sample_clip_from_video

path = Path(sys.argv[1] if len(sys.argv) > 1 else "test_input.mp4")
if not path.exists():
    print(f"not found: {path}")
    sys.exit(1)

print(f"file  : {path.name}  ({path.stat().st_size / 1024**2:.1f} MB)")

clip = sample_clip_from_video(path)
if clip is None:
    print("FAILED: could not sample frames -- file is not a readable video")
    sys.exit(1)

print(f"clip  : {clip.shape} {clip.dtype}")
assert clip.shape[0] == config.NUM_FRAMES

spread = float(np.mean([np.abs(clip[i].astype(float) - clip[0].astype(float)).mean()
                        for i in range(1, len(clip))]))
print(f"motion: {spread:.1f}", "  <- WARNING: frames identical" if spread < 1.0 else "")

enc = VJEPAEncoder()
vec = enc.embed(clip)
print(f"embed : {vec.shape}  finite={np.isfinite(vec).all()}")

print("\nOK - file is usable by the pipeline")
