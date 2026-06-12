"""Shared utilities used across train/eval/inference scripts.

Kept deliberately small: only duplication that exists in 3+ call sites
(or where a bug would need fixing in multiple files) belongs here.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms

# Class labels in torchvision ImageFolder order (alphabetical by directory
# name), which is the index order the trained model's outputs map to.
CLASS_NAMES = ['amd', 'cataract', 'diabetic_retinopathy', 'normal']


def get_device():
    """Return the best available torch device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_inference_transform():
    """Standard inference preprocessing: 224x224 resize + ImageNet normalization.

    Used by evaluate_model.py, test_external.py, api/main.py, and matches
    the 'val' transform used during training. Training's augmentation
    pipeline (RandomHorizontalFlip, RandomRotation) is intentionally kept
    separate in train.py since it evolves independently.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_resnet18(num_classes):
    """Build a ResNet18 with the final FC layer replaced for num_classes output.

    Weights are left at their default (None). Callers are responsible for
    loading trained weights (the error-handling story differs per caller:
    the API warns-and-continues, the evaluator prints-and-returns, and
    test_external lets errors propagate).
    """
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
