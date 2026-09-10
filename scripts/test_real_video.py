"""Run real downloaded videos through the full pipeline.

clip_sampler.py and the encoder have only ever been tested on synthetic
arrays. This is the first contact with actual mp4 files off YouTube.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import config
from models.vjepa_wrapper import VJEPAEncoder
from pipeline.clip_sampler import sample_clip_from_video

videos = sorted(Path(config.CLIP_DIR).rglob("*.mp4"))
if not videos:
    print(f"No videos in {config.CLIP_DIR}/ -- run collect_data.py first")
    sys.exit(1)

print(f"found {len(videos)} video(s)\n")

print("loading encoder...")
enc = VJEPAEncoder()
print()

embeddings = {}
for path in videos:
    size_mb = path.stat().st_size / 1024**2
    print(f"{path.parent.name}/{path.name}  ({size_mb:.1f} MB)")

    t0 = time.time()
    clip = sample_clip_from_video(path)
    sample_ms = (time.time() - t0) * 1000

    if clip is None:
        print("  FAILED to sample -- unreadable\n")
        continue

    print(f"  sampled : {clip.shape} {clip.dtype}  ({sample_ms:.0f} ms)")
    assert clip.shape[0] == config.NUM_FRAMES, clip.shape
    assert clip.dtype == np.uint8, clip.dtype

    # A clip of identical frames means seeking failed and we got one frame N times.
    spread = float(np.mean([np.abs(clip[i].astype(float) - clip[0].astype(float)).mean()
                            for i in range(1, len(clip))]))
    print(f"  motion  : mean abs diff vs frame0 = {spread:.1f}", end="")
    print("   <- WARNING: frames look identical" if spread < 1.0 else "")

    t0 = time.time()
    vec = enc.embed(clip)
    embed_ms = (time.time() - t0) * 1000
    print(f"  embedded: {vec.shape}  ({embed_ms:.0f} ms)")
    assert np.isfinite(vec).all(), "non-finite values in embedding"

    embeddings[f"{path.parent.name}/{path.name}"] = vec
    print()

if len(embeddings) >= 2:
    keys = list(embeddings)
    print("cosine similarity between real clips:")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va, vb = embeddings[a], embeddings[b]
            c = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
            print(f"  {c:.4f}  {a[:28]:<30} vs {b[:28]}")

print(f"\nOK - {len(embeddings)}/{len(videos)} videos processed end to end")
