import argparse
import os
import random
import shutil
from collections import Counter
from typing import Any, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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


def generate_random_affine_matrix(transform_type: str, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a random affine transformation matrix and metadata for a specific type.

    Inputs:
        transform_type (str): One of ['rotation', 'translation', 'scale', 'shear'].
        image_shape (Tuple[int, int]): The image height and width.

    Outputs:
        Tuple[np.ndarray, Dict[str, Any]]: 2x3 affine transformation matrix and metadata.
    """
    height, width = image_shape
    center = (width / 2.0, height / 2.0)

    if transform_type == "rotation":
        angle = 0.0
        while abs(angle) < 2.0:
            angle = random.uniform(-30.0, 30.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return matrix, {"type": "rotation", "angle": float(angle)}

    if transform_type == "translation":
        tx = 0.0
        ty = 0.0
        while abs(tx) < 0.05 * width and abs(ty) < 0.05 * height:
            tx = random.uniform(-0.15, 0.15) * width
            ty = random.uniform(-0.15, 0.15) * height
        matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        return matrix, {"type": "translation", "tx": float(tx), "ty": float(ty)}

    if transform_type == "scale":
        sx = 1.0
        sy = 1.0
        while abs(sx - 1.0) < 0.05 or abs(sy - 1.0) < 0.05:
            sx = random.uniform(0.8, 1.2)
            sy = random.uniform(0.8, 1.2)
        tx = (1 - sx) * width / 2.0
        ty = (1 - sy) * height / 2.0
        matrix = np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)
        return matrix, {"type": "scale", "sx": float(sx), "sy": float(sy)}

    if transform_type == "shear":
        shear = 0.0
        while abs(shear) < 0.05:
            shear = random.uniform(-0.25, 0.25)
        matrix = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return matrix, {"type": "shear", "shear": float(shear)}

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
        transform_metadata: Dict[str, Any] = {"type": transform_type}
        for attempt in range(10):
            matrix, metadata = generate_random_affine_matrix(transform_type, image.shape[:2])
            candidate = apply_affine_transform(image, matrix)
            if not any(images_are_equal(candidate, existing) for existing in transformed_images):
                transformed = candidate
                transform_metadata = metadata
                break
        if transformed is None:
            raise RuntimeError(
                f"Unable to generate a unique affine output for transform type '{transform_type}' after 10 attempts."
            )

        filename = f"{base_filename}_affine_{transform_type}"
        if transform_type == "rotation":
            filename += f"_angle_{transform_metadata['angle']:.1f}"
        elif transform_type == "translation":
            filename += f"_tx_{transform_metadata['tx']:.1f}_ty_{transform_metadata['ty']:.1f}"
        elif transform_type == "scale":
            filename += f"_sx_{transform_metadata['sx']:.2f}_sy_{transform_metadata['sy']:.2f}"
        elif transform_type == "shear":
            filename += f"_shear_{transform_metadata['shear']:.2f}"

        save_transformation(transformed, filename, output_dir)
        transformed_images.append(transformed)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to an image using the given sigma.

    Inputs:
        image (np.ndarray): Input image.
        sigma (float): Standard deviation for the Gaussian kernel.

    Outputs:
        np.ndarray: Blurred image.
    """
    kernel_radius = max(1, int(round(sigma * 3)))
    kernel_size = kernel_radius * 2 + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def blur_images_in_directories(transformation_dir: str, affine_dir: str, output_dir: str, sigma_values: Tuple[float, ...]) -> None:
    """Apply Gaussian blur to every image in the transformation directories.

    Inputs:
        transformation_dir (str): Root Image Transformations directory.
        affine_dir (str): Affine Transformations subdirectory.
        output_dir (str): Directory where blurred images will be saved.
        sigma_values (Tuple[float, ...]): Sigma values to use for blur.

    Outputs:
        None: Saves blurred images to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for dir_path in (transformation_dir, affine_dir):
        if not os.path.isdir(dir_path):
            continue
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".png"):
                image_paths.append(os.path.join(dir_path, filename))

    for image_path in image_paths:
        image = load_image(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for sigma in sigma_values:
            blurred = apply_gaussian_blur(image, sigma)
            save_transformation(blurred, f"{base_name}_gaussian_sigma_{sigma:.1f}", output_dir)


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


def count_png_images(directory: str) -> int:
    """Count the number of PNG image files in a directory.

    Inputs:
        directory (str): Path to the directory to count.

    Outputs:
        int: Number of PNG files in the directory. Returns 0 if directory doesn't exist.
    """
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(".png"))


def clear_directory(directory: str) -> None:
    """Delete all PNG image files from a directory.

    Inputs:
        directory (str): Path to the directory to clear.

    Outputs:
        None: Deletes all PNG files in the directory.
    """
    if not os.path.isdir(directory):
        return
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(directory, filename)
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Warning: could not delete {filepath}: {e}")


def validate_and_clear_directory(directory: str, expected_count: int) -> bool:
    """Validate that a directory has the expected number of PNG images.
    If not, delete all PNG images in the directory.

    Inputs:
        directory (str): Path to the directory to validate.
        expected_count (int): Expected number of PNG files.

    Outputs:
        bool: True if the directory already has the expected count, False otherwise.
    """
    actual_count = count_png_images(directory)
    if actual_count != expected_count:
        print(f"Directory {directory} has {actual_count} images (expected {expected_count}). Clearing directory.")
        clear_directory(directory)
        return False
    return True


def validate_edge_detection_directory(edge_dir: str, subset_dir: str, expected_count: int = 168) -> bool:
    """Validate that the edge detection directory matches the current subset images.

    Inputs:
        edge_dir (str): Path to the Edge Detection directory.
        subset_dir (str): Path to the subset directory containing source PNGs.
        expected_count (int): Expected number of PNG files.

    Outputs:
        bool: True if the edge directory is valid for the subset, False otherwise.
    """
    if not os.path.isdir(edge_dir):
        return False
    if count_png_images(edge_dir) != expected_count:
        print(f"Edge Detection directory {edge_dir} has unexpected PNG count; clearing.")
        clear_directory(edge_dir)
        return False

    source_images = sorted(
        f for f in os.listdir(subset_dir)
        if f.lower().endswith(".png")
    )
    if len(source_images) != expected_count // 4:
        print(
            f"Subset {subset_dir} expected {expected_count // 4} source images, "
            f"but found {len(source_images)}."
        )
        return False

    for source_image in source_images:
        base_name, _ = os.path.splitext(source_image)
        for suffix in ("sobel", "laplacian", "canny", "prewitt"):
            expected_path = os.path.join(edge_dir, f"{base_name}_{suffix}.png")
            if not os.path.isfile(expected_path):
                print(f"Missing edge file for source image: {expected_path}")
                clear_directory(edge_dir)
                return False

    return True


def gather_transformation_images(transformation_dir: str, excluded_dirs: Tuple[str, ...]) -> list[str]:
    """Gather all PNG image paths under a transformation directory while excluding specified subdirectories."""
    image_paths: list[str] = []
    for root, dirs, files in os.walk(transformation_dir):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for filename in files:
            if filename.lower().endswith(".png"):
                image_paths.append(os.path.join(root, filename))
    return image_paths


def create_random_subsets(transformation_dir: str, subset_names: Tuple[str, ...], expected_total: int) -> None:
    """Create equal random subsets of transformation images."""
    excluded_dirs = tuple(subset_names) + ("Plots",)
    image_paths = gather_transformation_images(transformation_dir, excluded_dirs)
    total_images = len(image_paths)
    if total_images != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} source images for subset creation, but found {total_images}."
        )

    random.shuffle(image_paths)
    subset_size = total_images // len(subset_names)
    for index, subset_name in enumerate(subset_names, start=1):
        subset_dir = os.path.join(transformation_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)
        clear_directory(subset_dir)
        start = (index - 1) * subset_size
        end = start + subset_size
        for image_path in image_paths[start:end]:
            shutil.copy2(image_path, os.path.join(subset_dir, os.path.basename(image_path)))
    print(f"Created {len(subset_names)} random subsets in {transformation_dir}: {', '.join(subset_names)}")


def verify_subset_counts(transformation_dir: str, subset_names: Tuple[str, ...], expected_count: int) -> bool:
    """Verify that each subset contains the expected number of images."""
    all_good = True
    for subset_name in subset_names:
        subset_dir = os.path.join(transformation_dir, subset_name)
        count = count_png_images(subset_dir)
        print(f"Subset '{subset_name}' contains {count} PNG images.")
        if count != expected_count:
            all_good = False
    return all_good


def normalize_edge_image(image: np.ndarray) -> np.ndarray:
    """Normalize a floating-point edge image into an 8-bit grayscale image."""
    image = np.absolute(image)
    image = np.clip(image, 0, None)
    max_val = image.max()
    if max_val == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = 255 * image / max_val
    return normalized.astype(np.uint8)


def perform_edge_detection_on_random_subset(
    transformation_dir: str,
    subset_names: Tuple[str, ...],
    chosen_subset: str | None = None,
) -> str:
    """Perform edge detection on all images in a selected subset.

    Inputs:
        transformation_dir (str): Root transformations directory.
        subset_names (Tuple[str, ...]): Available subset folder names.
        chosen_subset (str | None): Specific subset name to process.
            If omitted, a random subset is chosen.

    Outputs:
        str: Path to the generated edge detection directory.
    """
    if chosen_subset is None:
        chosen_subset = random.choice(subset_names)

    subset_dir = os.path.join(transformation_dir, chosen_subset)
    image_paths = [
        os.path.join(subset_dir, f)
        for f in os.listdir(subset_dir)
        if f.lower().endswith('.png')
    ]
    if not image_paths:
        raise RuntimeError(f"No PNG images found in subset '{chosen_subset}' for edge detection.")

    edge_dir = os.path.join(subset_dir, 'Edge Detection')
    os.makedirs(edge_dir, exist_ok=True)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    for image_path in image_paths:
        image = load_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = normalize_edge_image(np.hypot(sobel_x, sobel_y))

        laplacian = normalize_edge_image(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
        canny = cv2.Canny(gray, 100, 200)
        prewitt = normalize_edge_image(np.hypot(
            cv2.filter2D(gray, cv2.CV_64F, prewitt_x),
            cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
        ))

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(edge_dir, f"{base_name}_sobel.png"), sobel)
        cv2.imwrite(os.path.join(edge_dir, f"{base_name}_laplacian.png"), laplacian)
        cv2.imwrite(os.path.join(edge_dir, f"{base_name}_canny.png"), canny)
        cv2.imwrite(os.path.join(edge_dir, f"{base_name}_prewitt.png"), prewitt)

    print(f"Performed edge detection on subset '{chosen_subset}' and saved results to {edge_dir}")
    return edge_dir


def describe_transformation_pipeline(base_name: str) -> str:
    """Build a human-readable description of the transformation pipeline from the image filename."""
    prefix = "HW1_IMG_CS898BA_"
    remainder = base_name[len(prefix):] if base_name.startswith(prefix) else base_name
    color_desc = "Original"
    affine_desc = "No affine transform"
    gaussian_desc = "No Gaussian blur"

    affine_part = ""
    if "_affine_" in remainder:
        color_part, remainder = remainder.split("_affine_", 1)
        affine_part = remainder
    elif "_gaussian_sigma_" in remainder:
        color_part, remainder = remainder.split("_gaussian_sigma_", 1)
        affine_part = ""
    else:
        color_part = remainder
        remainder = ""

    color_map = {
        "grayscale": "Grayscale",
        "binary_otsu": "Binary (Otsu)",
        "hsv": "HSV",
        "lab": "LAB",
        "hls": "HLS",
        "hsv_equalized": "HSV equalized",
        "hsv_equalized_bgr": "HSV equalized -> BGR",
    }
    color_desc = color_map.get(color_part, color_part.replace("_", " ").title())

    if affine_part:
        gaussian_split = affine_part.split("_gaussian_sigma_", 1)
        affine_type_part = gaussian_split[0]
        remainder = gaussian_split[1] if len(gaussian_split) > 1 else ""
        parts = affine_type_part.split("_")
        if parts[0] in {"rotation", "translation", "scale", "shear"}:
            affine_type = parts[0].title()
            if affine_type == "Rotation" and len(parts) >= 3 and parts[1] == "angle":
                affine_desc = f"Rotation ({parts[2]}°)"
            elif affine_type == "Translation" and "tx" in parts and "ty" in parts:
                tx = parts[parts.index("tx") + 1]
                ty = parts[parts.index("ty") + 1]
                affine_desc = f"Translation (tx={tx}, ty={ty})"
            elif affine_type == "Scale" and "sx" in parts and "sy" in parts:
                sx = parts[parts.index("sx") + 1]
                sy = parts[parts.index("sy") + 1]
                affine_desc = f"Scale (sx={sx}, sy={sy})"
            elif affine_type == "Shear" and len(parts) >= 2 and parts[1] == "shear":
                affine_desc = f"Shear (s={parts[2]})"
            else:
                affine_desc = affine_type
        else:
            affine_desc = affine_type_part.replace("_", " ").title()
    elif remainder.startswith("gaussian_sigma_"):
        affine_desc = "No affine transform"
        remainder = remainder

    if "_gaussian_sigma_" in base_name:
        gaussian_desc = f"Gaussian blur sigma {base_name.split('_gaussian_sigma_')[1]}"

    return f"Original -> {color_desc} -> {affine_desc} -> {gaussian_desc}"


def build_edge_detection_plots(subset_dir: str, edge_dir: str, plot_dir: str) -> list[str]:
    """Build one comparison plot for each subset image and its edge-detected outputs.

    Inputs:
        subset_dir (str): Path to the selected subset directory.
        edge_dir (str): Path to the corresponding Edge Detection directory.
        plot_dir (str): Directory where generated plots will be saved.

    Outputs:
        list[str]: Paths to the generated plot image files.
    """
    if not os.path.isdir(edge_dir):
        raise FileNotFoundError(f"Edge detection directory not found: {edge_dir}")

    os.makedirs(plot_dir, exist_ok=True)
    source_images = sorted(
        f for f in os.listdir(subset_dir) if f.lower().endswith(".png")
    )
    if len(source_images) != 42:
        raise RuntimeError(
            f"Expected 42 source images in subset '{subset_dir}', but found {len(source_images)}."
        )

    plot_paths: list[str] = []
    for index, source_image in enumerate(source_images, start=1):
        base_name, _ = os.path.splitext(source_image)
        source_path = os.path.join(subset_dir, source_image)
        original_bgr = load_image(source_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        edge_names = ["sobel", "laplacian", "canny", "prewitt"]
        edge_images: list[np.ndarray] = []
        for name in edge_names:
            edge_path = os.path.join(edge_dir, f"{base_name}_{name}.png")
            if not os.path.exists(edge_path):
                raise FileNotFoundError(f"Expected edge image not found: {edge_path}")
            edge_image = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            if edge_image is None:
                raise FileNotFoundError(f"Unable to load edge image: {edge_path}")
            edge_images.append(edge_image)

        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        titles = ["Original", "Sobel", "Laplacian", "Canny", "Prewitt"]
        images = [original_rgb] + edge_images

        for ax, img, title in zip(axes, images, titles):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=12, pad=10)
            ax.axis("off")

        process_description = describe_transformation_pipeline(base_name)
        fig.suptitle(
            f"Sample {index}: {base_name}\nOriginal and Edge Detection Transformations",
            fontsize=18,
            y=0.99,
        )
        fig.text(
            0.5,
            0.02,
            process_description,
            ha="center",
            fontsize=11,
            wrap=True,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])

        plot_path = os.path.join(plot_dir, f"{base_name}_edge_comparison.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        plot_paths.append(plot_path)

    print(f"Created {len(plot_paths)} edge comparison plots in {plot_dir}")
    return plot_paths


def select_random_plot_examples(plot_paths: list[str], sample_count: int = 6) -> list[str]:
    """Select a random sample of plot images from a generated plot list."""
    if len(plot_paths) < sample_count:
        raise ValueError(
            f"Cannot select {sample_count} plots from {len(plot_paths)} available plots."
        )
    return random.sample(plot_paths, sample_count)


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

    transformations_valid = validate_and_clear_directory(transformation_dir, 7)
    if not transformations_valid:
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
    else:
        print(f"Image Transformations already has the expected 7 files; skipping transformation regeneration.")

    print(f"Saved transformed images to: {transformation_dir}")

    affine_dir = os.path.join(transformation_dir, "Affine Transformations")
    os.makedirs(affine_dir, exist_ok=True)
    affine_valid = validate_and_clear_directory(affine_dir, 14)

    if not affine_valid:
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
    else:
        print(f"Affine Transformations already has the expected 14 files; skipping affine generation.")

    print(f"Saved affine images to: {affine_dir}")

    gaussian_dir = os.path.join(transformation_dir, "Gaussian Blur")
    os.makedirs(gaussian_dir, exist_ok=True)
    gaussian_valid = validate_and_clear_directory(gaussian_dir, 147)

    if not gaussian_valid:
        sigma_values = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
        blur_images_in_directories(transformation_dir, affine_dir, gaussian_dir, sigma_values)
    else:
        print(f"Gaussian Blur already has the expected 147 files; skipping blur generation.")

    print(f"Saved Gaussian blurred images to: {gaussian_dir}")

    subset_names = ("Subset 1", "Subset 2", "Subset 3", "Subset 4")
    create_random_subsets(transformation_dir, subset_names, 168)
    verify_subset_counts(transformation_dir, subset_names, 42)

    chosen_subset = random.choice(subset_names)
    subset_dir = os.path.join(transformation_dir, chosen_subset)
    edge_dir = os.path.join(subset_dir, "Edge Detection")
    edge_valid = validate_edge_detection_directory(edge_dir, 168)
    if edge_valid:
        print(f"Edge Detection directory already exists with 168 images; skipping edge detection for '{chosen_subset}'.")
    else:
        edge_dir = perform_edge_detection_on_random_subset(transformation_dir, subset_names, chosen_subset)

    plot_dir = os.path.join(transformation_dir, "Plots")
    plot_paths = build_edge_detection_plots(subset_dir, edge_dir, plot_dir)
    selected_examples = select_random_plot_examples(plot_paths, 6)
    print("Selected plot examples for README:")
    for example_path in selected_examples:
        print(f" - {example_path}")
