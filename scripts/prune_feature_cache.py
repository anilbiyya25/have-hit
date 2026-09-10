"""Drop cached embeddings whose source clip no longer exists on disk.

dedupe_dataset.py deletes videos, but their vectors stay in the feature cache
and would still be trained on under the old label -- the deletion would have
achieved nothing. The script's own advice is to re-extract everything, which
costs ~80 minutes to reproduce vectors that are byte-for-byte identical for
every clip that was NOT deleted (the encoder is frozen and preprocessing is
deterministic).

Pruning by source filename is exact and takes seconds. A vector is kept if and
only if its clip is still in the folder for its label.

Deliberately does NOT import config: config imports torch, and this has to run
on a machine where torch cannot always be loaded. The one alias below is the
only project knowledge needed.

Usage:
    .\\run.ps1 scripts\\prune_feature_cache.py            # dry run
    .\\run.ps1 scripts\\prune_feature_cache.py --apply    # rewrite the cache
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

DATASET_DIR = Path("gym_dataset")
FEATURES_PATH = Path("data/features/embeddings.npz")
LABELS_PATH = Path("data/features/labels.json")

# config.DIR_ALIASES: the scraper made "pushups/" before the label settled as
# "push_ups". Keep in step with config.py if another alias is ever added.
DIR_ALIASES = {"push_ups": "pushups"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the cache (default: dry run)")
    args = ap.parse_args()

    if not FEATURES_PATH.exists():
        print(f"no cache at {FEATURES_PATH}")
        return 1

    data = np.load(FEATURES_PATH)
    X, y = data["X"], data["y"]
    meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    classes, sources = meta["classes"], np.array(meta["sources"])

    print("=" * 66)
    print("PRUNE FEATURE CACHE")
    print("=" * 66)
    print(f"  vectors : {len(X)} of dim {X.shape[1]}")
    print(f"  videos  : {len(set(sources.tolist()))}")
    print(f"  classes : {len(classes)}")
    print(f"  mode    : {'APPLY' if args.apply else 'DRY RUN'}")

    keep = np.ones(len(X), dtype=bool)
    missing = Counter()
    for i, (label_id, src) in enumerate(zip(y, sources)):
        label = classes[int(label_id)]
        folder = DIR_ALIASES.get(label, label)
        if not (DATASET_DIR / folder / str(src)).exists():
            keep[i] = False
            missing[label] += 1

    n_drop = int((~keep).sum())
    print(f"\n  vectors whose clip is gone: {n_drop}")
    if missing:
        print(f"  {'class':<20}{'vectors dropped':>16}")
        for label, n in missing.most_common():
            print(f"  {label:<20}{n:>16}")

    if not n_drop:
        print("\n  cache already matches the dataset, nothing to do.")
        return 0

    print(f"\n  {len(X)} -> {int(keep.sum())} vectors "
          f"({len(set(sources[keep].tolist()))} videos)")

    if not args.apply:
        print("\n  Dry run -- nothing written. Re-run with --apply.")
        return 0

    backup = FEATURES_PATH.with_suffix(".prebackup.npz")
    if not backup.exists():
        shutil.copy2(FEATURES_PATH, backup)
        print(f"  backed up -> {backup}")

    np.savez_compressed(FEATURES_PATH, X=X[keep], y=y[keep])
    LABELS_PATH.write_text(
        json.dumps({"classes": classes,
                    "sources": [str(s) for s in sources[keep]]}, indent=2),
        encoding="utf-8")
    print(f"  wrote {FEATURES_PATH}")
    print("\n  Remaining vectors are unchanged and still valid: the encoder is")
    print("  frozen, so a kept clip embeds identically. Run train_classifier.py")
    print("  WITH --reuse-features to top up only the new classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
