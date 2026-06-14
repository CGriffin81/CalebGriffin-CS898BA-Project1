import argparse
import os
import random
from collections import Counter
from typing import Any, Dict, Tuple

import cv2
import numpy as np

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
    """Split a 3-channel image into separate channels.

    Inputs:
        image (np.ndarray): Input image with three channels.

    Outputs:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The three image channels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel image.")
    return cv2.split(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Inputs:
        image (np.ndarray): Input BGR image.

    Outputs:
        np.ndarray: Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_binary_otsu(gray_image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to binary using Otsu's thresholding.

    Inputs:
        gray_image (np.ndarray): Input grayscale image.

    Outputs:
        np.ndarray: Binary image with values 0 or 255.
    """
    if gray_image.ndim != 2:
        raise ValueError("Expected a grayscale image for Otsu thresholding.")
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the HSV color space.

    Inputs:
        image (np.ndarray): Input BGR image.

    Outputs:
        np.ndarray: Image in HSV color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def to_lab(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the LAB color space.

    Inputs:
        image (np.ndarray): Input BGR image.

    Outputs:
        np.ndarray: Image in LAB color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def to_hls(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the HLS color space.

    Inputs:
        image (np.ndarray): Input BGR image.

    Outputs:
        np.ndarray: Image in HLS color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)


def equalize_hsv_value(hsv_image: np.ndarray) -> np.ndarray:
    """Equalize the V channel of an HSV image and return the adjusted HSV image.

    Inputs:
        hsv_image (np.ndarray): Input HSV image.

    Outputs:
        np.ndarray: HSV image with equalized V channel.
    """
    if hsv_image.ndim != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Expected an HSV image with 3 channels.")
    h, s, v = split_channels(hsv_image)
    v_eq = cv2.equalizeHist(v)
    return cv2.merge([h, s, v_eq])


def save_transformation(image: np.ndarray, filename_template: str, transformation_dir: str) -> str:
    """Save a transformed image using a consistent filename template.

    Inputs:
        image (np.ndarray): Image to save.
        filename_template (str): Template for the output filename, without extension.
        transformation_dir (str): Directory where the image will be saved.

    Outputs:
        str: The saved image path.
    """
    os.makedirs(transformation_dir, exist_ok=True)
    output_path = os.path.join(transformation_dir, f"{filename_template}.png")
    cv2.imwrite(output_path, image)
    return output_path


def generate_random_affine_matrix(transform_type: str, image_shape: Tuple[int, int]) -> np.ndarray:
    """Generate a random affine transformation matrix for a specific type.

    Inputs:
        transform_type (str): One of ['rotation', 'translation', 'scale', 'shear'].
        image_shape (Tuple[int, int]): The image height and width.

    Outputs:
        np.ndarray: 2x3 affine transformation matrix.
    """
    height, width = image_shape
    center = (width / 2.0, height / 2.0)

    if transform_type == "rotation":
        angle = 0.0
        while abs(angle) < 2.0:
            angle = random.uniform(-30.0, 30.0)
        return cv2.getRotationMatrix2D(center, angle, 1.0)

    if transform_type == "translation":
        tx = 0.0
        ty = 0.0
        while abs(tx) < 0.05 * width and abs(ty) < 0.05 * height:
            tx = random.uniform(-0.15, 0.15) * width
            ty = random.uniform(-0.15, 0.15) * height
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)

    if transform_type == "scale":
        sx = 1.0
        sy = 1.0
        while abs(sx - 1.0) < 0.05 or abs(sy - 1.0) < 0.05:
            sx = random.uniform(0.8, 1.2)
            sy = random.uniform(0.8, 1.2)
        tx = (1 - sx) * width / 2.0
        ty = (1 - sy) * height / 2.0
        return np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)

    if transform_type == "shear":
        shear = 0.0
        while abs(shear) < 0.05:
            shear = random.uniform(-0.25, 0.25)
        return np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    raise ValueError(f"Unsupported affine transform type: {transform_type}")


def apply_affine_transform(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transformation matrix to an image.

    Inputs:
        image (np.ndarray): Input image.
        matrix (np.ndarray): 2x3 affine transform matrix.

    Outputs:
        np.ndarray: Transformed image.
    """
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def images_are_equal(image_a: np.ndarray, image_b: np.ndarray) -> bool:
    """Return True if two images are pixel-identical.

    Inputs:
        image_a (np.ndarray): First image.
        image_b (np.ndarray): Second image.

    Outputs:
        bool: True if the images match exactly.
    """
    return image_a.shape == image_b.shape and np.array_equal(image_a, image_b)


def apply_affine_transformations(image: np.ndarray, base_filename: str, output_dir: str) -> None:
    """Apply two distinct random affine transformations and save the results.

    Inputs:
        image (np.ndarray): Input image.
        base_filename (str): Base filename without extension.
        output_dir (str): Directory where affine images will be saved.

    Outputs:
        None: Saves two uniquely-typed affine images to disk.
    """
    transform_types = ["rotation", "translation", "scale", "shear"]
    chosen_types = random.sample(transform_types, 2)
    assert len(chosen_types) == 2 and chosen_types[0] != chosen_types[1], \
        "Affine transform types must be unique for each image."

    transformed_images = []
    for transform_type in chosen_types:
        transformed = None
        for attempt in range(10):
            matrix = generate_random_affine_matrix(transform_type, image.shape[:2])
            candidate = apply_affine_transform(image, matrix)
            if not any(images_are_equal(candidate, existing) for existing in transformed_images):
                transformed = candidate
                break
        if transformed is None:
            raise RuntimeError(
                f"Unable to generate a unique affine output for transform type '{transform_type}' after 10 attempts."
            )
        save_transformation(transformed, f"{base_filename}_affine_{transform_type}", output_dir)
        transformed_images.append(transformed)


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

    image = load_image(image_path)
    transformation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Image Transformations")
    os.makedirs(transformation_dir, exist_ok=True)

    grayscale = to_grayscale(image)
    binary = to_binary_otsu(grayscale)
    hsv = to_hsv(image)
    lab = to_lab(image)
    hls = to_hls(image)

    save_transformation(grayscale, "HW1_IMG_CS898BA_grayscale", transformation_dir)
    save_transformation(binary, "HW1_IMG_CS898BA_binary_otsu", transformation_dir)
    save_transformation(hsv, "HW1_IMG_CS898BA_hsv", transformation_dir)
    save_transformation(lab, "HW1_IMG_CS898BA_lab", transformation_dir)
    save_transformation(hls, "HW1_IMG_CS898BA_hls", transformation_dir)

    hsv_equalized = equalize_hsv_value(hsv)
    bgr_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

    save_transformation(hsv_equalized, "HW1_IMG_CS898BA_hsv_equalized", transformation_dir)
    save_transformation(bgr_equalized, "HW1_IMG_CS898BA_hsv_equalized_bgr", transformation_dir)

    print(f"Saved transformed images to: {transformation_dir}")

    affine_dir = os.path.join(transformation_dir, "Affine Transformations")
    source_images = [
        f for f in os.listdir(transformation_dir)
        if f.lower().endswith(".png") and "_affine_" not in f
    ]
    for source_image in source_images:
        source_path = os.path.join(transformation_dir, source_image)
        image = cv2.imread(source_path)
        if image is None:
            continue
        base_name, _ = os.path.splitext(source_image)
        apply_affine_transformations(image, base_name, affine_dir)

    print(f"Saved affine images to: {affine_dir}")
