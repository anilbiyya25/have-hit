"""Exercise classifier: V-JEPA embedding (2304,) -> exercise logits.

The V-JEPA encoder stays frozen; only this head is trained. That means
training runs on cached embeddings and takes minutes, not hours, and adding
an eighth exercise later only requires retraining this ~1.4M-param head.

Input width is config.EMBED_DIM, which is HIDDEN_SIZE x TEMPORAL_CHUNKS --
the encoder pools into beginning/middle/end chunks rather than one global
mean, so a checkpoint trained before that change will not load here.
"""

import torch
import torch.nn as nn

import config


class ExerciseClassifier(nn.Module):
    """Small head over frozen V-JEPA embeddings.

    Uses LayerNorm rather than BatchNorm deliberately. BatchNorm keeps running
    mean/var estimates that only converge after many optimiser steps; on a
    dataset this small it produced train/eval disagreement large enough to
    make the reported loss meaningless (0.41 in train mode vs 1.55 in eval on
    identical data). LayerNorm normalises per-sample, has no running state,
    and so behaves identically in both modes at any batch size.
    """

    def __init__(self, input_dim=config.EMBED_DIM, num_classes=config.NUM_CLASSES,
                 hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_classifier(path=config.CLASSIFIER_PATH, device=config.DEVICE):
    model = ExerciseClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
