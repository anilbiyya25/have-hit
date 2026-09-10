"""Central config for Have Hit.

Model facts verified by measurement on this machine (Aug 2026), not assumed.

We use V-JEPA 2.1 ViT-B/384, Meta's March 2026 release. Benchmarked head to
head against V-JEPA 2.0 ViT-L/256 on the same GPU and clip:

                  2.0 ViT-L/256   2.1 ViT-B/384
    params                 326M             87M
    tokens/clip            2048            4608
    weights              624 MB          331 MB
    peak VRAM            736 MB          794 MB
    latency              193 ms          197 ms

Latency is a wash -- 2.1's 3.7x parameter saving is cancelled by 384px giving
2.25x more tokens. We pick 2.1 for the better representations and because 87M
is a far more plausible on-device model if the mobile app ever runs locally.
"""

import torch

# --- Model ---------------------------------------------------------------
# V-JEPA 2.1 is NOT in transformers yet (huggingface/transformers#45496) and
# Meta has not published it to their HF org (facebookresearch/vjepa2#137), so
# we load it from Meta's own CDN through their torch.hub repo.
HUB_REPO = "facebookresearch/vjepa2"
HUB_ENTRYPOINT = "vjepa2_1_vit_base_384"

# Upstream bug: that repo @ main hardcodes
#     VJEPA_BASE_URL = "http://localhost:8300"
# with the real URL commented out just above. We restore this at runtime.
# Re-check on repo updates; once fixed upstream the patch becomes a no-op.
VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

NUM_FRAMES = 16
# 256, the best configuration measured. Both resolutions were run end to end
# with identical data, split and pooling:
#
#   256px  2048 tokens  validation 79.6%  API 160 ms p50
#   384px  4608 tokens  validation 78.0%  API 275 ms p50
#
# 384 is the resolution the weights were distilled at, so the expectation was
# that it would win; it did not. The 1.6-point gap is inside the +/-2 point
# split noise, so treat the two as equal on accuracy -- at which point 256
# takes it on latency, being 1.7x faster. The encoder uses RoPE, which is why
# a resolution it was not trained at works at all.
CROP_SIZE = 256
TUBELET_SIZE = 2
PATCH_SIZE = 16
HIDDEN_SIZE = 768

NUM_TOKENS = (NUM_FRAMES // TUBELET_SIZE) * (CROP_SIZE // PATCH_SIZE) ** 2

# The encoder emits tokens laid out temporal-major: NUM_FRAMES // TUBELET_SIZE
# temporal positions, each holding (CROP_SIZE // PATCH_SIZE)^2 spatial tokens.
NUM_TEMPORAL_POSITIONS = NUM_FRAMES // TUBELET_SIZE      # 8
NUM_SPATIAL_TOKENS = (CROP_SIZE // PATCH_SIZE) ** 2      # 576

# Back to one 768-d vector. Three-chunk temporal concatenation was tried and
# measured no better: 74.9% against 76.6% for a plain mean on the same split,
# at 3x the width, both undertrained and converged.
#
# Pooling is now MAX over the spatial tokens, then mean over time. Averaging
# spatially is what buries a barbell: one bar occupies a handful of the 256
# patches, and its activation is divided away by the ~250 patches of wall and
# floor around it. A max keeps the strongest response for each of the 768
# channels, which is the behaviour wanted for a small high-contrast object
# that decides squats vs barbell_squat.
EMBED_DIM = HIDDEN_SIZE                                  # 768

# Official eval preprocessing, read out of the repo's make_transforms():
# resize short side -> center crop -> /255 -> ImageNet normalize.
RESIZE_SHORT_SIDE = int(CROP_SIZE * 256 / 224)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# --- Runtime -------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Weights stay fp32 and we wrap inference in autocast. A hard .half() breaks:
# the 2.1 RoPE path computes query/key in fp32 while value keeps the input
# dtype, and scaled_dot_product_attention rejects the mismatch.
AUTOCAST_DTYPE = torch.float16

# --- Exercises -----------------------------------------------------------

# EXERCISES is the single source of truth for label order. A checkpoint's
# output index i ALWAYS means ID_TO_LABEL[i] -- training assigns ids from this
# list, never from directory iteration order. Appending a new exercise here is
# safe; reordering or inserting invalidates every existing checkpoint.
EXERCISES = [
    # --- original 7, ids 0-6. Do not reorder: every checkpoint depends on it.
    "squats",
    "push_ups",
    "barbell_squat",
    "bench_press",
    "barbell_row",
    "bicep_curl",
    "shoulder_press",
    # --- expansion to 27, ids 7-26, appended so ids 0-6 keep their meaning
    # and the existing feature cache stays valid.
    "pull_ups",
    "chin_ups",
    "dips",
    "diamond_pushups",
    "deadlift",
    "lunges",
    "lat_pulldown",
    "lat_raise",
    "tricep_extension",
    "plank",
    "sit_ups",
    "leg_raises",
    "russian_twist",
    "burpees",
    "jumping_jacks",
    "jump_rope",
    "mountain_climbers",
    "high_knees",
    "kettlebell_swing",
    "box_jumps",
]

# Six UCF101 classes (pull_ups, clean_and_jerk, handstand_pushups,
# wall_pushups, jump_rope, jumping_jack) were added here and then removed.
# They scored 100.0% each while only 1 of 99 validation errors ever crossed
# between UCF101 and the YouTube classes, against 50% expected by chance --
# the head was separating 320x240 Xvid from 720p H.264, not exercises. That
# lifted the headline from 76.6% to 88.3% while YouTube-only accuracy went
# 76.6% -> 75.6%.
#
# pull_ups and jump_rope are now BACK in the list above, but as YouTube
# classes, not UCF101 ones. Their UCF101 folders were renamed to the
# _ucf101_ prefix below so find_videos() cannot reach them -- had they kept
# their old names, adding the labels would have silently re-armed exactly the
# artifact this comment exists to warn about. jumping_jacks (plural) is a new
# folder; the UCF101 jumping_jack (singular) is a different name and stays
# unreferenced. clean_and_jerk, handstand_pushups and wall_pushups are still
# unnamed here, so the extractor ignores them.
#
# Anything reintroducing a _ucf101_ folder under a canonical label must expect
# that class to score near 100% for reasons that have nothing to do with the
# exercise.
NUM_CLASSES = len(EXERCISES)
LABEL_TO_ID = {name: i for i, name in enumerate(EXERCISES)}
ID_TO_LABEL = {i: name for i, name in enumerate(EXERCISES)}

# gym_dataset/ subfolders whose name differs from the canonical label above.
# The scraper created "pushups/" before the label was settled as "push_ups".
# Mapping it here avoids renaming 100 files on disk; delete the entry if the
# folder is ever renamed to match.
DIR_ALIASES = {"pushups": "push_ups"}

# Clip containers the extractor will read. UCF101 ships .avi (Xvid) and is
# left that way on purpose -- cv2 decodes it directly, and transcoding ~740
# clips to mp4 would cost hours to produce strictly worse footage. Anything
# globbing "*.mp4" instead of this set silently ignores every UCF101 clip.
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm")



def find_videos(directory):
    """Every clip in `directory`, sorted, across all supported containers."""
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in VIDEO_EXTS)


def label_for_dir(dir_name):
    """gym_dataset/ subfolder name -> canonical label from EXERCISES."""
    return DIR_ALIASES.get(dir_name, dir_name)


def dir_for_label(label):
    """Canonical label -> the subfolder name that holds its clips."""
    for dir_name, canonical in DIR_ALIASES.items():
        if canonical == label:
            return dir_name
    return label

# --- Paths ---------------------------------------------------------------
DATA_DIR = "data"
CLIP_DIR = f"{DATA_DIR}/clips"          # raw training clips, one dir per class
FEATURE_DIR = f"{DATA_DIR}/features"    # cached V-JEPA embeddings
CHECKPOINT_DIR = "checkpoints"
CLASSIFIER_PATH = f"{CHECKPOINT_DIR}/classifier.pt"
