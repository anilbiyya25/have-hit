"""Prove the checkpoint on disk is Meta's real V-JEPA 2.1 ViT-B file."""

import hashlib
import os

import torch

# get_dir() already ends in "hub"
PATH = os.path.join(torch.hub.get_dir(), "checkpoints",
                    "vjepa2_1_vitb_dist_vitG_384.pt")

print("file :", PATH)
print("size :", round(os.path.getsize(PATH) / 1024**2, 1), "MB")

h = hashlib.sha256()
with open(PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print("sha256:", h.hexdigest())

ckpt = torch.load(PATH, map_location="cpu", weights_only=False)
print("\ntop-level keys:", list(ckpt.keys()))

enc = ckpt["ema_encoder"]
print(f"\nema_encoder: {len(enc)} tensors")
n = sum(v.numel() for v in enc.values() if hasattr(v, "numel"))
print(f"encoder params: {n / 1e6:.1f}M")

print("\nfirst 8 tensor names:")
for k in list(enc)[:8]:
    print(f"  {k:<55} {tuple(enc[k].shape)}")

# ViT-B is 12 layers @ 768 hidden. Confirm from the weights themselves.
depths = set()
for k in enc:
    parts = k.split(".")
    for i, p in enumerate(parts):
        if p == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
            depths.add(int(parts[i + 1]))
if depths:
    print(f"\nblocks found : {min(depths)}..{max(depths)}  ({len(depths)} layers)")

for k in enc:
    if "blocks.0" in k and "qkv.weight" in k:
        print(f"qkv shape    : {tuple(enc[k].shape)}  -> hidden = {enc[k].shape[1]}")
        break
