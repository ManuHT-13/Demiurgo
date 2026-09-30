"""
Automatically computes the mean and standard deviation for each channel (R, G, B)
of the training dataset

Normalizing images helps the training process converge faster and
more stably
"""

import json
from pathlib import Path

import numpy as np


def compute_mean_std(images_path):
    """
    Computes the RGB channel-wise (mean, std) over an image array
    of shape (N, H, W, 3) with uint8 values in the range [0, 255]

    Returns two tuples of 3 floats each, in the format expected by
    transforms.Normalize(mean, std)
    """
    images = np.load(images_path)
    assert images.ndim == 4 and images.shape[-1] == 3, (
        f"Expected images with shape (N, H, W, 3), but found {images.shape}"
    )

    # Use float64 for higher precision during computation
    # Divide by 255 to work in the [0, 1] range, matching what
    # the Dataset will later produce using ToTensor()
    imgs = images.astype(np.float64) / 255.0

    # axis=(0,1,2) averages across the image batch as well as
    # height and width, leaving one value per color channel
    # (the last axis, axis 3)
    mean = imgs.mean(axis=(0, 1, 2))
    std = imgs.std(axis=(0, 1, 2))

    return tuple(mean.tolist()), tuple(std.tolist())


def compute_or_load_mean_std(images_path, cache_path):
    """
    Same as compute_mean_std, but stores/loads the result from a
    cached .json file to avoid recomputing it every time
    training is started
    """
    images_path_str = str(Path(images_path).resolve())
    cache_path = Path(cache_path)

    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)
        if cache.get("images_path") == images_path_str:
            print(f"[stats.py] Using cached mean/std from {cache_path}")
            print(f"[stats.py] mean={tuple(cache['mean'])} std={tuple(cache['std'])}")
            return tuple(cache["mean"]), tuple(cache["std"])

    print(f"[stats.py] Computing mean/std for {images_path_str} ...")
    mean, std = compute_mean_std(images_path)
    print(f"[stats.py] mean={mean} std={std}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {"images_path": images_path_str, "mean": list(mean), "std": list(std)},
            f,
            indent=2,
        )

    return mean, std