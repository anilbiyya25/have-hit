# Have Hit

Point a camera at someone and identify what exercise they're doing.
Phase 1: 7 exercises, mobile-first.

## Setup status

Backend environment installed, model wired up, pipeline verified. No training
data yet.

## Quick start

```powershell
cd D:\have-hit
.\run.ps1 scripts\test_wrapper.py        # smoke-test the encoder
.\run.ps1 scripts\test_discriminates.py  # confirm it separates motions
.\run.ps1 scripts\check_vjepa21.py       # full benchmark
.\run.ps1                                # or just activate the venv
```

Always go through `run.ps1`. C: is nearly full, so it points `HF_HOME`,
`TORCH_HOME`, `TMP`, and the pip cache at D:. Running `python` directly sends
1.5 GB of model cache to C: and fails.

## Model choice, measured not assumed

We use **V-JEPA 2.1 ViT-B/384** (Meta, March 2026). Benchmarked head to head
against V-JEPA 2.0 ViT-L/256 on this GPU with the same 16-frame clip:

| | 2.0 ViT-L/256 | 2.1 ViT-B/384 |
|---|---|---|
| params | 326 M | **87 M** |
| tokens/clip | 2048 | 4608 |
| embed dim | 1024 | 768 |
| weights on GPU | 624 MB | **331 MB** |
| peak VRAM | 736 MB | 794 MB |
| latency | 193 ms | 197 ms |

**Latency is a wash.** 2.1's 3.7x parameter saving is exactly cancelled by
384px producing 2.25x more tokens — do not assume fewer params means faster.
We chose 2.1 for the better representations and because 87M is a far more
plausible on-device model if the mobile app ever runs locally.

Both fit the 4 GB card with room to spare.

## Two upstream landmines

**1. Meta's repo has a broken checkpoint URL.** `facebookresearch/vjepa2` @
main ships:

```python
# VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"   <- real, commented out
VJEPA_BASE_URL = "http://localhost:8300"                      <- committed dev override
```

This breaks every pretrained `torch.hub` entrypoint in the repo with
`ConnectionRefusedError`. `models/vjepa_wrapper.py` restores the real URL at
runtime rather than editing the cached clone, so a re-clone can't silently
reintroduce it. Weights still come from Meta's CDN.

**2. fp16 breaks the 2.1 RoPE path.** A hard `.half()` gives:

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: float key.dtype: float and value.dtype: struct c10::Half
```

Query/key are computed in fp32 while value keeps the input dtype. Keep weights
in fp32 and wrap inference in `torch.autocast` instead.

## Also worth knowing

**V-JEPA is a video model, not an image model.** Input is `(B, C, T, H, W)`.
Classifying single still frames would discard the motion that separates a
squat from a deadlift. Training data must be **short video clips**, not
extracted JPEGs.

**V-JEPA 2.1 is not in `transformers`.** Support is
[open](https://github.com/huggingface/transformers/issues/45496), and Meta has
[not published 2.1](https://github.com/facebookresearch/vjepa2/issues/137) to
their HF org. The HF repos named `vjepa2.1-*` are community conversions. We use
`torch.hub` against Meta's own CDN instead. When transformers adds support this
wrapper can be simplified a lot.

## Architecture

The encoder stays frozen. Only the small head is trained:

```
clip (16 frames)
  -> resize 438 short side, center crop 384, ImageNet normalize
  -> V-JEPA 2.1 ViT-B (frozen)    -> 4608 tokens x 768
  -> mean-pool over tokens        -> 768-dim embedding
  -> ExerciseClassifier (trained) -> 7 logits
```

Because the encoder is frozen, embeddings are cached once and the head
retrains in minutes. Adding an 8th exercise later means retraining the head
only.

Mean-pooling was verified to preserve signal (`scripts/test_discriminates.py`):
self-consistency 1.000000, noise vs structured motion 0.72, distinct motions
0.965–0.989.

## Layout

| Path | What |
|---|---|
| `run.ps1` | env vars + venv; use this, not bare `python` |
| `config.py` | model, clip shape, preprocessing constants, exercise list |
| `models/vjepa_wrapper.py` | clip -> 768-dim embedding |
| `models/classifier.py` | embedding -> exercise logits |
| `pipeline/clip_sampler.py` | video file or live camera -> 16-frame clip |
| `scripts/check_vjepa21.py` | 2.1 benchmark, head to head vs 2.0 |
| `scripts/check_env.py` | 2.0 baseline benchmark (kept for comparison) |
| `scripts/test_wrapper.py` | wrapper smoke test |
| `scripts/test_discriminates.py` | proves pooling preserves motion signal |
| `data/clips/` | training clips, one folder per exercise (empty) |
| `data/features/` | cached embeddings (empty) |

## Caches (all on D:)

| Path | Size |
|---|---|
| `D:\torch-hub` | 1.6 GB — V-JEPA 2.1 checkpoint + Meta repo |
| `D:\hf-cache` | 1.2 GB — V-JEPA 2.0 weights, only needed for `check_env.py` |
| `D:\pip-cache`, `D:\pip-tmp` | pip scratch |

## Exercises

squats, push_ups, barbell_squat, bench_press, barbell_row, bicep_curl,
shoulder_press

## Not built yet

- data collection scripts
- feature caching + training loop
- FastAPI inference server
- mobile app
