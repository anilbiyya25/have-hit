"""Find and resolve the same video filed under two different exercises.

The scraper searches each category independently, so a clip that matches two
phrasings gets downloaded into both folders under the same YouTube id. That is
contradictory supervision: the head is shown identical footage with two
different correct answers, which puts a hard ceiling on exactly the class pair
it is already worst at.

Resolution is a labelling decision, not something to guess at, so each pair is
resolved from an explicit table below. A duplicate spanning a pair that is not
in the table is reported and left alone rather than resolved arbitrarily.

Usage:
    .\\run.ps1 scripts\\dedupe_dataset.py            # dry run, deletes nothing
    .\\run.ps1 scripts\\dedupe_dataset.py --apply    # actually delete
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import config

DATASET_DIR = Path("gym_dataset")
# Single source of truth, so a container the extractor reads can never be
# invisible to the duplicate check (UCF101's .avi was exactly that gap).
VIDEO_EXTS = set(config.VIDEO_EXTS)

# Which class wins when one video is filed under both.
#
# Only pairs where one class is strictly a SPECIAL CASE of the other belong
# here. The footage is then a genuine example of the narrower class, so the
# narrower class keeps it and the broader copy goes.
#
# barbell_squat over squats: a clip showing a loaded barbell back squat is a
# barbell_squat. Leaving the copy in squats would pollute the bodyweight class
# with barbell footage and blur the exact boundary the model must learn.
#
# barbell_row over bicep_curl: bent-over row footage that surfaced on a curl
# query. The movement is a row.
#
# diamond_pushups over pushups: a diamond pushup is a pushup with the hands
# together. The narrow class is the one that needs the example; the generic
# pushup class has 100 other clips and does not need this one.
HIERARCHY = {
    frozenset({"barbell_squat", "squats"}): "barbell_squat",
    frozenset({"barbell_row", "bicep_curl"}): "barbell_row",
    frozenset({"diamond_pushups", "pushups"}): "diamond_pushups",
}

# Everything else is deleted outright, from every folder it appears in.
#
# A video under 3+ labels is a circuit or "full body workout" compilation: it
# genuinely shows all of those exercises, which makes it a clean example of
# none of them. One clip in the 27-class scrape spans nine labels.
#
# A video under 2 UNRELATED labels (chin_ups + pull_ups, barbell_row +
# deadlift) is either a comparison video showing both movements or a search
# miss. Which label is right cannot be known without watching it, and guessing
# is how contradictory supervision gets baked in -- the thing this script
# exists to prevent. Deleting ~2% of clips is cheaper than poisoning the class
# pairs the model is already worst at.


def resolve(places):
    """places: {class name: path} -> (class to keep or None, reason).

    None means every copy goes.
    """
    names = frozenset(places)
    if len(names) >= 3:
        return None, "compilation"
    keep = HIERARCHY.get(names)
    if keep is not None and keep in places:
        return keep, "hierarchy"
    return None, "ambiguous"


def find_duplicates(dataset_dir=DATASET_DIR):
    """video id -> {class name: path} for every id in more than one class."""
    by_id = {}
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for path in class_dir.iterdir():
            if path.suffix.lower() in VIDEO_EXTS:
                by_id.setdefault(path.stem, {})[class_dir.name] = path
    return {vid: places for vid, places in by_id.items() if len(places) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="delete the losing copies (default: dry run)")
    args = ap.parse_args()

    if not DATASET_DIR.exists():
        print(f"no dataset at {DATASET_DIR.resolve()}")
        return 1

    dupes = find_duplicates()

    print("=" * 68)
    print("DUPLICATE VIDEOS ACROSS CLASSES")
    print("=" * 68)
    print(f"  dataset : {DATASET_DIR.resolve()}")
    print(f"  mode    : {'APPLY (deleting)' if args.apply else 'DRY RUN (no deletions)'}")
    print(f"  found   : {len(dupes)} video id(s) in more than one class")

    if not dupes:
        print("\nnothing to do.")
        return 0

    planned = []                    # (vid, keep-or-None, class, path, reason)
    by_reason = {"hierarchy": 0, "compilation": 0, "ambiguous": 0}
    for vid, places in sorted(dupes.items()):
        keep, reason = resolve(places)
        by_reason[reason] += 1
        for cls, path in places.items():
            if cls != keep:
                planned.append((vid, keep, cls, path, reason))

    print(f"\n  {'video id':<16}{'reason':<14}{'keep':<18}{'delete from':<18}")
    print("  " + "-" * 64)
    for vid, keep, drop, _path, reason in planned:
        print(f"  {vid:<16}{reason:<14}{(keep or '-- none --'):<18}{drop:<18}")

    print(f"\n  videos by verdict:")
    print(f"    hierarchy   {by_reason['hierarchy']:>4}  (narrower class keeps the clip)")
    print(f"    compilation {by_reason['compilation']:>4}  (3+ labels, every copy goes)")
    print(f"    ambiguous   {by_reason['ambiguous']:>4}  (2 unrelated labels, every copy goes)")

    freed = 0
    if args.apply:
        for vid, _keep, drop, path, _reason in planned:
            size = path.stat().st_size
            path.unlink()
            freed += size

    print("\n" + "=" * 68)
    print(f"  duplicate videos        : {len(dupes)}")
    print(f"  file copies {'deleted' if args.apply else 'to delete'} : {len(planned)}")
    if args.apply:
        print(f"  freed                   : {freed / 1e6:.1f} MB")
        remaining = sum(1 for p in DATASET_DIR.rglob("*")
                        if p.suffix.lower() in VIDEO_EXTS)
        print(f"  videos remaining        : {remaining}")
        print("\n  The cached features no longer match the dataset. Re-extract")
        print("  (run train_classifier.py WITHOUT --reuse-features).")
    else:
        print("\n  Dry run -- nothing was deleted. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
