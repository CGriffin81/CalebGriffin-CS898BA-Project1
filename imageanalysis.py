import cv2
import numpy as np
import argparse
from collections import Counter
from typing import Any, Dict, Tuple
import argparse
import os

try:
    from scipy import stats  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def load_image(image_path: str) -> np.ndarray:
    """Load a color image from disk using OpenCV.

    Inputs:
        image_path (str): Path to the input image file.

    Outputs:
        np.ndarray: Loaded BGR image array.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image from path: {image_path}")
    return image


def split_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a color image into its BGR channels.

    Inputs:
        image (np.ndarray): Input BGR image.

    Outputs:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Blue, green, and red channels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a color image with 3 channels.")
    return cv2.split(image)


def compute_mode(channel: np.ndarray) -> Tuple[int, int]:
    """Compute the mode value and count for a channel.

    Inputs:
        channel (np.ndarray): Single-channel image array.

    Outputs:
        Tuple[int, int]: The mode pixel value and its count.
    """
    flat = channel.flatten()
    if _HAS_SCIPY:
        result = stats.mode(flat, axis=None, keepdims=False)
        mode_value = int(result.mode) if hasattr(result, "mode") else int(result[0])
        count = int(result.count) if hasattr(result, "count") else int(result[1])
        return mode_value, count

    counts = Counter(int(value) for value in flat)
    mode_value, count = counts.most_common(1)[0]
    return mode_value, count


def compute_skew(channel: np.ndarray) -> float:
    """Compute skewness for a single channel.

    Inputs:
        channel (np.ndarray): Single-channel image array.

    Outputs:
        float: Skewness of the channel distribution.
    """
    flat = channel.flatten().astype(np.float64)
    if _HAS_SCIPY:
        return float(stats.skew(flat, bias=False))

    mean = np.mean(flat)
    std = np.std(flat)
    if std == 0:
        return 0.0
    return float(np.mean((flat - mean) ** 3) / (std ** 3))


def compute_channel_stats(channel: np.ndarray, channel_name: str = "channel") -> Dict[str, Any]:
    """Compute a set of statistics for a single image channel.

    Inputs:
        channel (np.ndarray): Single-channel image array.
        channel_name (str): Human-readable channel name.

    Outputs:
        Dict[str, Any]: Statistics including min, max, mean, median, range,
            variance, std, mode, mode_count, and skew.
    """
    flat = channel.flatten().astype(np.float64)
    stats_dict: Dict[str, Any] = {
        "channel": channel_name,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "range": float(np.max(flat) - np.min(flat)),
        "variance": float(np.var(flat)),
        "std": float(np.std(flat)),
    }
    mode_value, mode_count = compute_mode(channel)
    stats_dict["mode"] = mode_value
    stats_dict["mode_count"] = mode_count
    stats_dict["skew"] = compute_skew(channel)
    return stats_dict


def analyze_image_channels(image_path: str) -> Dict[str, Dict[str, Any]]:
    """Analyze the BGR channels of an image and return per-channel statistics.

    Inputs:
        image_path (str): Path to the input image file.

    Outputs:
        Dict[str, Dict[str, Any]]: Statistics keyed by channel name.
    """
    image = load_image(image_path)
    blue, green, red = split_channels(image)
    return {
        "blue": compute_channel_stats(blue, "blue"),
        "green": compute_channel_stats(green, "green"),
        "red": compute_channel_stats(red, "red"),
    }


def format_channel_report(channel_stats: Dict[str, Dict[str, Any]]) -> str:
    """Format channel statistics into a human-readable report.

    Inputs:
        channel_stats (Dict[str, Dict[str, Any]]): Per-channel statistics.

    Outputs:
        str: Formatted multi-line report.
    """
    lines = ["Image channel statistics report:"]
    for channel_name, stats_dict in channel_stats.items():
        lines.append(f"\n{channel_name.title()} channel:")
        for key, value in stats_dict.items():
            if key == "channel":
                continue
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    DEFAULT_IMAGE_NAME = "HW1_IMG_CS898BA.png"

    parser = argparse.ArgumentParser(description="Analyze BGR image channels.")
    parser.add_argument("--image_path", "-i", default=None,
                        help=("Path to the input image file. If omitted, the script will look "
                              "for HW1_IMG_CS898BA.png in the script directory and the current working directory."))
    args = parser.parse_args()

    image_path = args.image_path
    if image_path is None:
        # Search common locations for the default image name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, DEFAULT_IMAGE_NAME),
            os.path.join(os.getcwd(), DEFAULT_IMAGE_NAME),
        ]
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                f"No image_path provided and {DEFAULT_IMAGE_NAME} not found in: {candidates}")
        image_path = found

    stats = analyze_image_channels(image_path)
    print(format_channel_report(stats))
