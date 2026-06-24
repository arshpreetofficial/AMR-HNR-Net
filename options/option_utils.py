import os
import json
import random
from datetime import datetime

import numpy as np
import torch


def str2bool(value):
    """
    Convert command-line string to boolean.
    """
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ("yes", "true", "t", "1", "y"):
        return True

    if value in ("no", "false", "f", "0", "n"):
        return False

    raise ValueError("Boolean value expected.")


def parse_tuple(value):
    """
    Convert string tuple into integer tuple.

    Example:
        "96,96,96" -> (96, 96, 96)
    """
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    return tuple(int(v.strip()) for v in value.split(","))


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def save_options(options, save_path):
    """
    Save parsed options as JSON file.
    """
    ensure_dir(os.path.dirname(save_path))

    options_dict = vars(options).copy()

    for key, value in options_dict.items():
        if isinstance(value, tuple):
            options_dict[key] = list(value)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(options_dict, f, indent=4)


def load_options(json_path):
    """
    Load options from JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_random_seed(seed=42):
    """
    Fix random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id=0):
    """
    Select CUDA device if available.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")

    return torch.device("cpu")


def get_timestamp():
    """
    Return readable timestamp for experiment naming.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_inverse_class_weights(class_counts):
    """
    Compute inverse-frequency class weights for focal loss.

    Example:
        class_counts = [320, 380, 250]
        classes = CN, MCI, AD
    """
    counts = np.array(class_counts, dtype=np.float32)

    if np.any(counts <= 0):
        raise ValueError("Class counts must be positive.")

    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()

    return weights.tolist()


def validate_existing_file(path, name):
    """
    Check that a required file exists.
    """
    if path is None or path == "":
        raise ValueError(f"{name} path is empty.")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} file not found: {path}")


def validate_existing_dir(path, name):
    """
    Check that a required directory exists.
    """
    if path is None or path == "":
        raise ValueError(f"{name} path is empty.")

    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} directory not found: {path}")


def create_experiment_name(prefix="AMR_HNR_Net"):
    """
    Create a clean experiment name.
    """
    return f"{prefix}_{get_timestamp()}"
