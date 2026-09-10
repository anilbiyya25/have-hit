"""Smoke-test the V-JEPA 2.1 wrapper end to end on a synthetic clip."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
import config
from models.world_model import VJEPAEncoder

print("loading encoder...")
enc = VJEPAEncoder()
print("params        :", round(sum(p.numel() for p in enc.model.parameters()) / 1e6, 1), "M")

# Non-square 720p frames on purpose, to exercise resize + center-crop.
frames = np.random.randint(0, 255, (config.NUM_FRAMES, 720, 1280, 3), dtype=np.uint8)

x = enc.preprocess(frames)
print("preprocessed  :", tuple(x.shape), x.dtype)
assert tuple(x.shape) == (1, 3, config.NUM_FRAMES, config.CROP_SIZE, config.CROP_SIZE), x.shape

v = enc.embed(frames)
print("embedding     :", v.shape, v.dtype)
assert v.shape == (config.HIDDEN_SIZE,), v.shape
assert np.isfinite(v).all(), "embedding contains nan/inf"

t = time.time()
for _ in range(5):
    enc.embed(frames)
elapsed = (time.time() - t) / 5 * 1000

print("latency       :", round(elapsed), "ms  (includes preprocessing)")
print("peak VRAM     :", round(torch.cuda.max_memory_reserved() / 1024**2), "MB")

# Two noise clips SHOULD land near 1.0 -- they are draws from the same
# distribution, and mean-pooling 4608 tokens averages them to the same place.
# This is not evidence of collapse. scripts/test_discriminates.py checks
# separation properly, using structured motion.
other = np.random.randint(0, 255, (config.NUM_FRAMES, 720, 1280, 3), dtype=np.uint8)
v2 = enc.embed(other)
cos = float(v @ v2 / (np.linalg.norm(v) * np.linalg.norm(v2)))
print("cos(noise,noise):", round(cos, 4), "(~1.0 expected; see test_discriminates.py)")

print("\nOK")
