"""Does the encoder actually separate different motions?

The noise-vs-noise smoke test gave cos=0.9997, which is expected for i.i.d.
noise but would also be what a broken/washed-out pooling looks like. This uses
structured synthetic motion instead: if embeddings of genuinely different
movements are still ~1.0 apart, mean-pooling over 4608 tokens is destroying
the signal and the pooling strategy needs to change before we train anything.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
import config
from models.vjepa_wrapper import VJEPAEncoder

H, W = 480, 640
T = config.NUM_FRAMES


def blank():
    return np.full((T, H, W, 3), 30, dtype=np.uint8)


def vertical_bar():
    """A bright block moving top to bottom -- stands in for a squat."""
    v = blank()
    for t in range(T):
        y = int(40 + (H - 200) * t / (T - 1))
        v[t, y:y + 120, W // 2 - 60:W // 2 + 60] = 240
    return v


def horizontal_bar():
    """Same block moving left to right -- different motion, same texture."""
    v = blank()
    for t in range(T):
        x = int(40 + (W - 200) * t / (T - 1))
        v[t, H // 2 - 60:H // 2 + 60, x:x + 120] = 240
    return v


def vertical_bar_reversed():
    """Bottom to top: same path as vertical_bar, opposite time direction."""
    return vertical_bar()[::-1].copy()


def static_bar():
    """No motion at all."""
    v = blank()
    v[:, H // 2 - 60:H // 2 + 60, W // 2 - 60:W // 2 + 60] = 240
    return v


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("loading encoder...")
enc = VJEPAEncoder()

clips = {
    "vertical_down": vertical_bar(),
    "vertical_up": vertical_bar_reversed(),
    "horizontal": horizontal_bar(),
    "static": static_bar(),
    "noise": np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8),
}

emb = {k: enc.embed(v) for k, v in clips.items()}

names = list(emb)
print(f"\n{'':16}" + "".join(f"{n:>15}" for n in names))
for a in names:
    row = "".join(f"{cos(emb[a], emb[b]):>15.4f}" for b in names)
    print(f"{a:16}{row}")

# Same clip twice must be identical -- proves the pipeline is deterministic.
repeat = cos(emb["vertical_down"], enc.embed(clips["vertical_down"]))
print(f"\nself-consistency (same clip twice): {repeat:.6f}  (want ~1.0)")

pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
worst = max(cos(emb[a], emb[b]) for a, b in pairs)
best = min(cos(emb[a], emb[b]) for a, b in pairs)
print(f"most similar different pair : {worst:.4f}")
print(f"least similar pair          : {best:.4f}")
print(f"spread                      : {worst - best:.4f}")

print("\nVERDICT")
if worst > 0.999:
    print("  BAD - different motions are indistinguishable. Mean-pooling over")
    print("  all tokens is washing out the signal; change the pooling.")
else:
    print("  OK - the encoder separates different motions. Mean-pooling holds.")
