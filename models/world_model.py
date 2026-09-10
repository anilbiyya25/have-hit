"""V-JEPA 2.1 encoder wrapper: video clip -> single embedding vector.

V-JEPA is a video model. It takes a clip of shape (B, C, T, H, W) and returns
per-token hidden states (B, num_tokens, 768). We mean-pool the tokens into one
vector per clip, which is what the exercise classifier consumes.

Loading goes through Meta's torch.hub repo rather than transformers, because
transformers does not support 2.1 yet. Two upstream quirks are handled here:

  1. the repo hardcodes a localhost checkpoint URL (see config.VJEPA_BASE_URL)
  2. its RoPE path breaks under a hard .half(), so we use autocast instead
"""

import os
import sys
import types

import cv2
import numpy as np
import torch

import config


def _load_hub_encoder():
    """Clone/cache Meta's repo, fix the bad checkpoint URL, build the encoder."""
    torch.hub.list(config.HUB_REPO, trust_repo=True)

    repo_dir = os.path.join(
        torch.hub.get_dir(), config.HUB_REPO.replace("/", "_") + "_main"
    )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from src.hub import backbones

    if "localhost" in backbones.VJEPA_BASE_URL:
        backbones.VJEPA_BASE_URL = config.VJEPA_BASE_URL

    # Meta's loader does `from app.vjepa_2_1.models import ...`, meaning the
    # app/ package inside its own repo. Our FastAPI entrypoint is also called
    # app.py, and uvicorn imports it as the module "app", so that name is
    # already taken and the import dies with "'app' is not a package".
    #
    # Just deleting the name is not enough: the import system would then search
    # sys.path, where the working directory precedes the hub repo, and find
    # app.py again. Bind the name directly to the repo's app/ directory
    # instead, which does not depend on sys.path order at all. It has no
    # __init__.py, hence a namespace-style spec.
    # A bare ModuleType with __path__ set, rather than module_from_spec: the
    # latter leaves __file__ set to None, and torch's op registration walks
    # sys.modules through inspect.getfile(), which rejects a module whose
    # __file__ is None with "is a built-in module". A real namespace package
    # has no __file__ attribute at all, which is what this reproduces.
    shadowed = sys.modules.get("app")
    pkg = types.ModuleType("app")
    pkg.__path__ = [os.path.join(repo_dir, "app")]
    sys.modules["app"] = pkg
    try:
        # Entrypoint returns (encoder, predictor). The predictor only serves
        # the JEPA pretraining objective, so we drop it.
        encoder, _predictor = getattr(backbones, config.HUB_ENTRYPOINT)(pretrained=True)
    finally:
        # Hand the name back so uvicorn's "app:app" still resolves to ours.
        if shadowed is not None:
            sys.modules["app"] = shadowed
        else:
            sys.modules.pop("app", None)
    return encoder


class VJEPAEncoder:
    def __init__(self, device=config.DEVICE):
        self.device = device
        self.model = _load_hub_encoder().to(device, dtype=torch.float32).eval()
        self._amp = torch.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            dtype=config.AUTOCAST_DTYPE,
            enabled=(device == "cuda"),
        )
        # Shaped (1, 3, 1, 1, 1) to broadcast over (B, C, T, H, W).
        self._mean = torch.tensor(config.NORM_MEAN, device=device,
                                  dtype=torch.float32).view(1, 3, 1, 1, 1)
        self._std = torch.tensor(config.NORM_STD, device=device,
                                 dtype=torch.float32).view(1, 3, 1, 1, 1)

    def preprocess(self, frames):
        """frames: uint8 (T, H, W, 3) RGB -> float32 tensor (1, 3, T, H, W).

        Mirrors the repo's eval transform: resize short side, center crop,
        scale to [0,1], ImageNet normalize.
        """
        out = []
        for f in frames:
            h, w = f.shape[:2]
            scale = config.RESIZE_SHORT_SIDE / min(h, w)
            resized = cv2.resize(
                f, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR
            )
            rh, rw = resized.shape[:2]
            top = (rh - config.CROP_SIZE) // 2
            left = (rw - config.CROP_SIZE) // 2
            out.append(resized[top:top + config.CROP_SIZE, left:left + config.CROP_SIZE])

        # Scale and normalise on the device, in float32. Doing it in numpy
        # promoted the whole clip to float64 -- NORM_MEAN/NORM_STD are Python
        # floats, so `clip - np.array(mean)` upcast a 27 MB float32 array into
        # a 56 MB float64 one and then threw it away on the next line.
        clip = torch.from_numpy(np.ascontiguousarray(np.stack(out)))  # uint8
        clip = clip.to(self.device).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)
        clip = clip.float().div_(255.0)
        return clip.sub_(self._mean).div_(self._std)

    @staticmethod
    def pool(out):
        """(B, num_tokens, 768) -> (B, 768): spatial MAX, then temporal mean.

        Tokens arrive temporal-major -- 8 temporal positions of 256 spatial
        tokens each at 256px -- so they reshape to (B, 8, 256, 768).

        Max over the spatial axis, not mean. A barbell covers a few patches out
        of 256; averaging divides its response across the ~250 patches of wall
        and floor that surround it, which is why squats and barbell_squat stayed
        the top confusion through three runs. A max keeps, per channel, the
        strongest response anywhere in the frame, which is what a small
        high-contrast object needs to survive pooling.

        Time is still averaged: max over time as well would reduce the clip to
        a single peak and throw away the movement entirely.
        """
        b, n, d = out.shape
        n_t, n_s = config.NUM_TEMPORAL_POSITIONS, config.NUM_SPATIAL_TOKENS
        assert n == n_t * n_s, (
            f"token layout changed: got {n} tokens, expected {n_t}x{n_s}. "
            "Pooling assumes temporal-major ordering."
        )
        per_step = out.view(b, n_t, n_s, d).amax(dim=2)   # (B, T', D)
        return per_step.mean(dim=1)                       # (B, D)

    @torch.no_grad()
    def embed_preprocessed(self, x):
        """x: (1, 3, T, H, W) already preprocessed -> float32 numpy (2304,).

        Split out from embed() so a caller holding a preprocessed clip can run
        variants of it (a mirrored copy, say) without paying for preprocessing
        twice or allocating another full-size uint8 clip on the host.
        """
        with self._amp:
            out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return self.pool(out.float()).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed(self, frames):
        """frames: uint8 (T, H, W, 3) RGB -> float32 numpy vector (768,)."""
        return self.embed_preprocessed(self.preprocess(frames))

    @torch.no_grad()
    def embed_batch(self, clips):
        """clips: list of (T, H, W, 3) arrays -> (B, 768) float32 numpy.

        Sequential. Kept for callers that hold clips of DIFFERING resolution --
        preprocess centre-crops to a fixed size, but a caller mixing sources
        can still hand in ragged input, and torch.cat would raise on that.
        """
        return np.stack([self.embed(c) for c in clips])

    @torch.no_grad()
    def embed_many(self, clips):
        """clips: list of (T, H, W, 3) arrays -> (B, 768). ONE forward pass.

        Worth roughly 1.3x over embed_batch on this card (measured: 119 ms per
        window at B=1 against 92 ms at B=4). The speedup is modest because a
        single 16-frame 256px clip already keeps the GPU fairly busy -- there
        is not much idle capacity for a second clip to fill, so this recovers
        per-call launch overhead rather than unlocking parallelism.

        B beyond ~4 is not worth taking. B=8 measured 90 ms/window, 2% better
        than B=4, while doubling both the activation peak and the host-side
        uint8 frames the caller must hold. On a 4 GB card that already keeps
        SAM resident, that is a bad trade for 2%.
        """
        x = torch.cat([self.preprocess(c) for c in clips], dim=0)
        with self._amp:
            out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return self.pool(out.float()).cpu().numpy()
