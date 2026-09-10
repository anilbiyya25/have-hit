"""Train an exercise classifier on frozen V-JEPA 2.1 features.

Two phases:
  1. Extract 768-dim motion vectors for every clip in gym_dataset/ and cache
     them to disk. The encoder is frozen, so this only ever runs once per
     dataset -- retraining the head afterwards takes seconds.
  2. Train a small trainable head on those cached vectors.

The V-JEPA encoder is frozen with requires_grad = False and never sees a
gradient. Only the head is trained.

Model provenance: V-JEPA 2.1 ViT-B/384 from Meta's own CDN via torch.hub
(see models/world_model.py), not a third-party HuggingFace conversion.

Usage:
    .\\run.ps1 train_classifier.py                    # extract + train
    .\\run.ps1 train_classifier.py --epochs 30
    .\\run.ps1 train_classifier.py --reuse-features   # skip extraction
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
import config
from models.classifier import ExerciseClassifier
from pipeline.clip_sampler import sample_windows_from_video

DATASET_DIR = Path("gym_dataset")
FEATURES_PATH = Path(config.FEATURE_DIR) / "embeddings.npz"
LABELS_PATH = Path(config.FEATURE_DIR) / "labels.json"


# --------------------------------------------------------------------------
# Phase 1: feature extraction
# --------------------------------------------------------------------------

def discover_classes():
    """Label id -> clip directory, ordered by config.EXERCISES.

    Ids come from config.EXERCISES, NOT from sorted directory order. Those two
    orders differ, so deriving ids from the filesystem silently mislabelled
    every prediction made through config.ID_TO_LABEL. The head always has
    config.NUM_CLASSES outputs, so a checkpoint stays loadable even if a class
    folder is empty or missing.
    """
    dirs = {}
    for label in config.EXERCISES:
        d = DATASET_DIR / config.dir_for_label(label)
        if d.is_dir() and config.find_videos(d):
            dirs[config.LABEL_TO_ID[label]] = d
    return dirs


def extract_features(windows_per_video, existing=None):
    """Embed clips in gym_dataset/. Returns (X, y, class_names, sources).

    `existing` is an already label-aligned (X, y, sources) triple to keep.
    Any video whose filename is already in it is skipped, so adding a class
    costs only the new clips: re-encoding the 678 videos already cached would
    burn 80 minutes to reproduce vectors that are byte-for-byte identical,
    because the encoder is frozen and preprocessing is deterministic.
    """
    from models.world_model import VJEPAEncoder

    classes = list(config.EXERCISES)
    dirs = discover_classes()
    if not dirs:
        print(f"No videos found under {DATASET_DIR}/")
        return None, None, None, None

    missing = [c for i, c in enumerate(classes) if i not in dirs]
    print(f"classes: {len(classes)}  ->  {', '.join(classes)}")
    print("label ids come from config.EXERCISES (checkpoint-stable order)")
    if missing:
        print(f"  !!! no clips for: {', '.join(missing)} -- these stay untrained")

    print("\nloading V-JEPA 2.1 encoder...")
    enc = VJEPAEncoder()

    # Freeze the backbone explicitly. embed() already runs under no_grad, but
    # this makes the intent unambiguous and guarantees no head-training step
    # can ever touch these weights.
    frozen = 0
    for p in enc.model.parameters():
        p.requires_grad = False
        frozen += p.numel()
    trainable = sum(p.numel() for p in enc.model.parameters() if p.requires_grad)
    print(f"  frozen params    : {frozen / 1e6:.1f}M")
    print(f"  trainable params : {trainable}  (must be 0)")
    assert trainable == 0, "encoder is not fully frozen"

    X, y, sources = [], [], []
    t_start = time.time()

    cached_sources = set()
    last_flush = 0
    if existing is not None:
        cached_sources = set(existing[2].tolist())
        print(f"\nreusing {len(existing[0])} cached vectors from "
              f"{len(cached_sources)} videos -- only new clips are encoded")
        print(f"flushing to disk every {SAVE_EVERY} new vectors "
              "(a crash resumes, it does not restart)")

    for label_id, clip_dir in sorted(dirs.items()):
        cls = config.ID_TO_LABEL[label_id]
        videos = config.find_videos(clip_dir)
        todo = [p for p in videos if p.name not in cached_sources]
        print(f"\n[id {label_id}] {cls}  ({len(videos)} videos, from "
              f"{clip_dir.name}/, {len(todo)} to encode)")
        if not todo:
            print("  all cached, skipping")
            continue

        n_ok = 0
        for i, path in enumerate(todo, 1):
            windows = sample_windows_from_video(path, windows_per_video)
            if not windows:
                print(f"  [{i}/{len(todo)}] {path.name[:20]} -- unreadable, skipped")
                continue
            for w in windows:
                X.append(enc.embed(w))
                y.append(label_id)
                sources.append(path.name)
            n_ok += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {len(X)} new vectors so far")

            if len(X) - last_flush >= SAVE_EVERY:
                mX, my, ms = merge(existing, X, y, sources)
                save_cache(mX, my, classes, ms)
                last_flush = len(X)
                print(f"       ...flushed {len(mX)} vectors to disk")

        print(f"  usable videos: {n_ok}/{len(todo)}")

    n_new = len(X)
    print(f"\nencoded {n_new} new vectors in {time.time() - t_start:.0f}s")

    if existing is None and not n_new:
        print("nothing encoded and no cache to fall back on")
        return None, None, None, None

    X, y, sources = merge(existing, X, y, sources)
    print(f"total: {X.shape[0]} vectors of dim {X.shape[1]} "
          f"from {len(set(sources.tolist()))} videos")

    assert len(sources) == len(X) == len(y), "feature/label/source length mismatch"

    save_cache(X, y, classes, sources)
    print(f"cached -> {FEATURES_PATH}")
    return X, y, classes, np.asarray(sources)


# Flush the cache this often during extraction. A crash mid-run then costs
# only the clips since the last flush: re-running skips everything already
# cached, so an interrupted 40-minute encode resumes instead of restarting.
# This machine runs the encode with ~2.5 GB of RAM free, which is the regime
# where allocations have failed before, so assume it can die at any point.
SAVE_EVERY = 150


def save_cache(X, y, classes, sources):
    """Write features + labels atomically enough to survive a mid-run crash."""
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES_PATH, X=X, y=y)
    LABELS_PATH.write_text(
        json.dumps({"classes": list(classes), "sources": list(map(str, sources))},
                   indent=2), encoding="utf-8")


def merge(existing, X, y, sources):
    """Combine cached arrays with freshly encoded ones."""
    if not len(X):
        return existing[0], existing[1], existing[2]
    Xn = np.stack(X).astype(np.float32) if isinstance(X, list) else X
    yn = np.array(y, dtype=np.int64) if isinstance(y, list) else y
    sn = np.array(sources) if isinstance(sources, list) else sources
    if existing is None:
        return Xn, yn, sn
    return (np.concatenate([existing[0], Xn]),
            np.concatenate([existing[1], yn]),
            np.concatenate([existing[2], sn]))


def align_cached_labels(y, classes):
    """Remap cached label ids into config.EXERCISES order.

    Caches written before label ids were pinned to config.EXERCISES numbered
    the classes by sorted directory name, so their y values mean something
    different now. The embeddings themselves are unaffected -- only the label
    ids need renumbering -- so this is an exact relabelling, not a reason to
    spend 80 minutes re-encoding every video.

    Returns (y, classes) aligned to config, or (None, None) if the cache holds
    a class this config does not know about.
    """
    if list(classes) == list(config.EXERCISES):
        return y, list(classes)

    remap = {}
    for old_id, old_name in enumerate(classes):
        canonical = config.label_for_dir(old_name)
        if canonical not in config.LABEL_TO_ID:
            print(f"\n  !!! cached class {old_name!r} is not in config.EXERCISES.")
            print("      Re-extract without --reuse-features.")
            return None, None
        remap[old_id] = config.LABEL_TO_ID[canonical]

    if all(old == new for old, new in remap.items()):
        # Appending classes to config.EXERCISES leaves existing ids untouched,
        # so the cache is already correct -- only the class list grew.
        print(f"  cache labels unchanged; class list grew "
              f"{len(classes)} -> {len(config.EXERCISES)}")
        return y, list(config.EXERCISES)

    print("\n  remapping cached label ids to config.EXERCISES order:")
    for old_id, new_id in sorted(remap.items()):
        print(f"    {old_id} {classes[old_id]:<16} ->  {new_id} "
              f"{config.ID_TO_LABEL[new_id]}")

    return np.array([remap[int(v)] for v in y], dtype=np.int64), list(config.EXERCISES)


def load_cached_features():
    if not FEATURES_PATH.exists():
        return None, None, None, None
    data = np.load(FEATURES_PATH)
    meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return data["X"], data["y"], meta["classes"], np.array(meta["sources"])


# --------------------------------------------------------------------------
# Phase 2: train the head
# --------------------------------------------------------------------------

def train(X, y, classes, sources, epochs, lr, seed, batch_size, patience):
    device = config.DEVICE
    counts = Counter(y.tolist())

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"  {'id':<4}{'class':<18}{'vectors':>9}{'videos':>9}")
    for i, c in enumerate(classes):
        n_vid = len(set(sources[y == i]))
        print(f"  {i:<4}{c:<18}{counts.get(i, 0):>9}{n_vid:>9}")
    print(f"  {'':<4}{'TOTAL':<18}{len(y):>9}{len(set(sources)):>9}")

    # Split by VIDEO, not by window. Three windows cut from one video are
    # near-duplicates; letting them straddle the split lets the model see the
    # answer at training time and inflates validation accuracy.
    # StratifiedGroupKFold keeps whole videos on one side while still
    # balancing classes across the split.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=sources))

    X_tr, y_tr_np = X[train_idx], y[train_idx]
    X_va, y_va_np = X[val_idx], y[val_idx]

    # Prove no video leaked across the split.
    overlap = set(sources[train_idx]) & set(sources[val_idx])
    print(f"\n  train {len(y_tr_np)} vectors / {len(set(sources[train_idx]))} videos")
    print(f"  val   {len(y_va_np)} vectors / {len(set(sources[val_idx]))} videos")
    print(f"  videos in both splits: {len(overlap)}  (must be 0)")
    assert not overlap, f"video leakage: {overlap}"

    X_tr = torch.tensor(X_tr, device=device)
    y_tr = torch.tensor(y_tr_np, device=device)
    X_va = torch.tensor(X_va, device=device)
    y_va = torch.tensor(y_va_np, device=device)

    model = ExerciseClassifier(
        input_dim=X.shape[1], num_classes=len(classes)
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  head params: {n_train / 1e3:.1f}K trainable")

    # Class weights counteract the imbalance (shoulder_press has ~half the
    # clips of the others), so the head cannot win by ignoring rare classes.
    weights = torch.tensor(
        [len(y) / (len(classes) * max(counts.get(i, 1), 1)) for i in range(len(classes))],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    n_batches = max(1, (len(y_tr) + batch_size - 1) // batch_size)
    print("\n" + "=" * 60)
    print(f"TRAINING  ({epochs} epochs on {device})")
    print("=" * 60)
    print(f"  batch size {batch_size}  ->  {n_batches} steps/epoch, "
          f"{n_batches * epochs} total")
    print(f"\n  {'epoch':>6}{'loss':>10}{'train acc':>12}{'val acc':>10}")

    # Early stopping. The 30-epoch run hit 100% training accuracy by epoch 20
    # while validation flatlined from epoch 3 -- every epoch after that was
    # pure memorisation. Stop once validation has not improved for `patience`
    # consecutive epochs and keep the best weights seen.
    print(f"  early stopping: patience {patience} epochs (0 disables)")

    best_val, best_state, best_epoch = 0.0, None, 0
    stale, stopped_early = 0, False
    for epoch in range(1, epochs + 1):
        # Mini-batches rather than one full-batch step per epoch. The previous
        # version took 15 gradient steps in total, which is nowhere near
        # enough for the head to converge.
        model.train()
        perm = torch.randperm(len(y_tr), device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) < 2:
                continue
            optimiser.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= n_batches

        model.eval()
        with torch.no_grad():
            tr_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
            va_acc = (model(X_va).argmax(1) == y_va).float().mean().item()

        if va_acc > best_val:
            best_val, best_epoch, stale = va_acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch <= 3 or epoch % 5 == 0 or epoch == epochs or stale == patience:
            print(f"  {epoch:>6}{epoch_loss:>10.4f}{tr_acc:>12.3f}{va_acc:>10.3f}")

        if patience and stale >= patience:
            print(f"\n  stopped early at epoch {epoch}: no validation gain in "
                  f"{patience} epochs (best {best_val * 100:.1f}% @ epoch {best_epoch})")
            stopped_early = True
            break

    if best_state:
        model.load_state_dict(best_state)
    if not stopped_early:
        print(f"\n  ran all {epochs} epochs (best {best_val * 100:.1f}% "
              f"@ epoch {best_epoch})")

    # ---- final report -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X_tr), y_tr).item()
        tr_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
        va_pred = model(X_va).argmax(1)
        va_acc = (va_pred == y_va).float().mean().item()

    print("\n" + "=" * 60)
    print("FINAL")
    print("=" * 60)
    print(f"  training loss     : {final_loss:.4f}")
    print(f"  training accuracy : {tr_acc * 100:.1f}%")
    print(f"  VALIDATION acc    : {va_acc * 100:.1f}%   <- the number that matters")
    print(f"  random baseline   : {100 / len(classes):.1f}%")

    print("\n  per-class validation accuracy:")
    y_va_np, va_pred_np = y_va.cpu().numpy(), va_pred.cpu().numpy()
    for i, c in enumerate(classes):
        mask = y_va_np == i
        if mask.sum() == 0:
            print(f"    {c:<18}    n/a  (none in val split)")
            continue
        acc = (va_pred_np[mask] == i).mean()
        print(f"    {c:<18} {acc * 100:>5.1f}%  ({int(mask.sum())} samples)")

    print("\n  confusion matrix (rows = true, cols = predicted):")
    print("    " + "".join(f"{c[:6]:>8}" for c in classes))
    for i, c in enumerate(classes):
        row = [(int(((y_va_np == i) & (va_pred_np == j)).sum())) for j in range(len(classes))]
        print(f"    {c[:6]:<6}" + "".join(f"{v:>8}" for v in row))

    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    torch.save(model.state_dict(), config.CLASSIFIER_PATH)

    # Sidecar rather than a dict inside the .pt, so the file stays a plain
    # state_dict that models.classifier.load_classifier() can read unchanged.
    # Anything decoding logits should check this matches its config.EXERCISES.
    labels_out = Path(config.CHECKPOINT_DIR) / "classes.json"
    labels_out.write_text(json.dumps({
        "classes": list(config.EXERCISES),
        "num_classes": config.NUM_CLASSES,
        "val_accuracy": round(va_acc, 4),
    }, indent=2), encoding="utf-8")

    print(f"\n  saved -> {config.CLASSIFIER_PATH}")
    print(f"  saved -> {labels_out}  (label order for inference)")

    return va_acc, len(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--windows", type=int, default=3,
                    help="temporal windows sampled per video")
    ap.add_argument("--reuse-features", action="store_true",
                    help="use cached embeddings instead of re-extracting")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many epochs with no validation gain "
                         "(0 disables early stopping)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("PHASE 1: FEATURE EXTRACTION (frozen encoder)")
    print("=" * 60)

    X = y = classes = sources = None

    # Load whatever is cached and pull its labels into config order first, so
    # it can be reused whether we are skipping extraction entirely or only
    # topping it up with newly added clips.
    cached = load_cached_features()
    if cached[0] is not None and cached[0].shape[1] != config.EMBED_DIM:
        # Pooling changed shape, so every cached vector means something else.
        # Reusing them would train a head on features the encoder no longer
        # produces, and the mismatch would only surface at inference.
        print(f"  !!! cache is {cached[0].shape[1]}-dim but the encoder now "
              f"emits {config.EMBED_DIM}-dim.")
        print("      Discarding it; every clip must be re-encoded.")
        cached = (None, None, None, None)

    if cached[0] is not None:
        c_X, c_y, c_classes, c_sources = cached
        c_y, c_classes = align_cached_labels(c_y, c_classes)
        if c_y is None:
            return 1
        print(f"cache: {c_X.shape[0]} vectors, dim {c_X.shape[1]}, "
              f"{len(set(c_sources.tolist()))} videos")
        cached = (c_X, c_y, c_classes, c_sources)

    if args.reuse_features:
        if cached[0] is None:
            print("no cache found, extracting from scratch")
        else:
            X, y, classes, sources = cached
            absent = [c for i, c in enumerate(classes) if not (y == i).any()]
            if absent:
                print(f"  !!! no cached vectors for: {', '.join(absent)}")
                print("      Those classes cannot be learned. Drop "
                      "--reuse-features to encode them.")

    if X is None:
        base = None if cached[0] is None else (cached[0], cached[1], cached[3])
        X, y, classes, sources = extract_features(args.windows, existing=base)
        if X is None:
            return 1

    va_acc, n_classes = train(X, y, classes, sources, args.epochs, args.lr,
                              args.seed, args.batch_size, args.patience)

    # Honest read on whether this result means anything yet.
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    baseline = 1.0 / n_classes
    if va_acc < baseline * 1.5:
        print("  Validation accuracy is near random. The features are not")
        print("  separating these classes, or there is far too little data.")
    elif va_acc < 0.7:
        print("  Above random but weak. Most likely cause is dataset size --")
        print("  a few hundred clips per class is the usual minimum.")
    else:
        print("  Promising. Treat with caution: the validation split is small,")
        print("  so this number carries a wide error bar.")
    print("\n  This split is video-level, so no clip from a validation video")
    print("  was seen in training -- the number is not inflated by leakage.")
    print("  It is still computed on ~20 videos, so the error bar is wide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
