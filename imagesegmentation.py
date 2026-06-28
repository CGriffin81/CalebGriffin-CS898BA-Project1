"""Feature Segmentation Module.

This branch starts by loading the renamed base image, base_image.png,
splitting it into its B, G, and R channels, and saving those channel images
into the Image_Transformations directory.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_IMAGE_NAME = "base_image.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_image_path(image_name: str) -> str:
    """Find an image in the working directory, then fall back to the script directory."""
    working_directory_path = os.path.join(SCRIPT_DIR, image_name)
    if os.path.exists(working_directory_path):
        return working_directory_path

    script_directory_path = os.path.join(SCRIPT_DIR, image_name)
    if os.path.exists(script_directory_path):
        return script_directory_path

    raise FileNotFoundError(f"Image not found in working directory or script directory: {image_name}")


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Loaded BGR image.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def split_color_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a BGR image into its individual channels.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Blue, green, and red channels.
    """
    blue_channel, green_channel, red_channel = cv2.split(image)
    return blue_channel, green_channel, red_channel


def save_channel_images(base_image_name: str, channels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Save the separated color channels to the Image_Transformations directory.

    Args:
        base_image_name (str): Name of the source image without extension.
        channels (tuple[np.ndarray, np.ndarray, np.ndarray]): Blue, green, and red channels.
    """
    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")
    os.makedirs(transformations_dir, exist_ok=True)

    channel_names = ("blue", "green", "red")
    for channel_name, channel_image in zip(channel_names, channels):
        output_path = os.path.join(transformations_dir, f"{base_image_name}_{channel_name}_channel.png")
        cv2.imwrite(output_path, channel_image)


def equalize_color_channels(channels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply histogram equalization to each color channel.

    Args:
        channels (tuple[np.ndarray, np.ndarray, np.ndarray]): Blue, green, and red channels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Equalized blue, green, and red channels.
    """
    return tuple(cv2.equalizeHist(channel) for channel in channels)


def save_equalized_image(equalized_channels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Merge equalized channels and save the result as equalized_image.png."""
    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")
    os.makedirs(transformations_dir, exist_ok=True)

    equalized_image = cv2.merge(equalized_channels)
    output_path = os.path.join(transformations_dir, "equalized_image.png")
    cv2.imwrite(output_path, equalized_image)


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def save_hsv_image(hsv_image: np.ndarray) -> None:
    """Save the HSV image as hsv_image.png."""
    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")
    os.makedirs(transformations_dir, exist_ok=True)

    output_path = os.path.join(transformations_dir, "hsv_image.png")
    cv2.imwrite(output_path, hsv_image)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_foreground_background(gray_image: np.ndarray, binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create foreground and background images from a binary mask."""
    foreground = cv2.bitwise_and(gray_image, gray_image, mask=binary_mask)
    inverted_mask = cv2.bitwise_not(binary_mask)
    background = cv2.bitwise_and(gray_image, gray_image, mask=inverted_mask)
    return foreground, background


def save_threshold_outputs(prefix: str, gray_image: np.ndarray, binary_mask: np.ndarray, foreground: np.ndarray, background: np.ndarray) -> None:
    """Save grayscale, binary mask, foreground, and background images for a thresholding method."""
    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")
    os.makedirs(transformations_dir, exist_ok=True)

    cv2.imwrite(os.path.join(transformations_dir, f"{prefix}_grayscale.png"), gray_image)
    cv2.imwrite(os.path.join(transformations_dir, f"{prefix}_mask.png"), binary_mask)
    cv2.imwrite(os.path.join(transformations_dir, f"{prefix}_foreground.png"), foreground)
    cv2.imwrite(os.path.join(transformations_dir, f"{prefix}_background.png"), background)


def apply_otsu_threshold(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Otsu's automatic thresholding to a grayscale image."""
    _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground, background = create_foreground_background(gray_image, binary_mask)
    return binary_mask, foreground, background


def apply_adaptive_threshold(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply adaptive Gaussian thresholding to a grayscale image."""
    binary_mask = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    foreground, background = create_foreground_background(gray_image, binary_mask)
    return binary_mask, foreground, background


def apply_kmeans_clustering(hsv_image: np.ndarray, cluster_count: int) -> np.ndarray:
    """Cluster HSV pixels with K-means and return the label map."""
    pixel_values = hsv_image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(pixel_values, cluster_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.flatten()


def save_kmeans_outputs(hsv_image: np.ndarray, labels: np.ndarray, cluster_count: int) -> None:
    """Save binary masks and foregrounds for each K-means cluster."""
    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")
    os.makedirs(transformations_dir, exist_ok=True)

    height, width = hsv_image.shape[:2]
    for cluster_index in range(cluster_count):
        cluster_mask = np.where(labels == cluster_index, 255, 0).astype(np.uint8).reshape((height, width))
        foreground_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=cluster_mask)
        foreground_bgr = cv2.cvtColor(foreground_hsv, cv2.COLOR_HSV2BGR)

        mask_output = os.path.join(transformations_dir, f"hsv_image_k{cluster_count}_cluster{cluster_index + 1}_mask.png")
        foreground_output = os.path.join(transformations_dir, f"hsv_image_k{cluster_count}_cluster{cluster_index + 1}_foreground.png")

        cv2.imwrite(mask_output, cluster_mask)
        cv2.imwrite(foreground_output, foreground_bgr)


def load_binary_mask(mask_path: str) -> np.ndarray:
    """Load an image as a binary mask where nonzero pixels are foreground."""
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    return (mask_image > 0).astype(np.uint8)


def compute_overlap_metrics(ground_truth_mask: np.ndarray, predicted_mask: np.ndarray) -> tuple[float, float]:
    """Compute IoU/Jaccard and Dice coefficients for two binary masks."""
    intersection = float(np.logical_and(ground_truth_mask, predicted_mask).sum())
    union = float(np.logical_or(ground_truth_mask, predicted_mask).sum())
    ground_truth_sum = float(ground_truth_mask.sum())
    predicted_sum = float(predicted_mask.sum())

    iou = intersection / union if union else 0.0
    dice = (2.0 * intersection) / (ground_truth_sum + predicted_sum) if (ground_truth_sum + predicted_sum) else 0.0
    return iou, dice


def report_ground_truth_metrics() -> None:
    """Calculate and print IoU and Dice metrics against the manual ground truth mask."""
    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")
    ground_truth_path = os.path.join(transformations_dir, "ground_truth_mask.png")
    comparison_files = {
        "Adaptive foreground": "equalized_image_adaptive_foreground.png",
        "Otsu foreground": "equalized_image_otsu_foreground.png",
        "K-means k=4 cluster 4 foreground": "hsv_image_k4_cluster4_foreground.png",
    }

    ground_truth_mask = load_binary_mask(ground_truth_path)
    print("Ground truth metrics against ground_truth_mask.png")
    print("Result                           IoU / Jaccard   Dice")
    print("--------------------------------------------------------")

    for label, filename in comparison_files.items():
        predicted_mask = load_binary_mask(os.path.join(transformations_dir, filename))
        iou, dice = compute_overlap_metrics(ground_truth_mask, predicted_mask)
        print(f"{label:<31} {iou:>12.4f} {dice:>8.4f}")

def generate_segmentation_comparison_plot() -> None:
    """Generate a comparison figure of the original image and the final
    segmentation masks."""

    transformations_dir = os.path.join(SCRIPT_DIR, "Image_Transformations")

    image_files = [
        ("Original", resolve_image_path(BASE_IMAGE_NAME), False),
        ("Equalized", os.path.join(transformations_dir, "equalized_image.png"), False),
        ("Ground Truth", os.path.join(transformations_dir, "ground_truth_mask.png"), True),
        ("Adaptive", os.path.join(transformations_dir, "equalized_image_adaptive_mask.png"), True),
        ("Otsu", os.path.join(transformations_dir, "equalized_image_otsu_mask.png"), True),
        ("K-Means (K=4)", os.path.join(transformations_dir, "hsv_image_k4_cluster4_mask.png"), True),
    ]

    figure, axes = plt.subplots(
        1,
        len(image_files),
        figsize=(24, 5),
        constrained_layout=True,
    )

    for axis, (title, filename, grayscale) in zip(axes, image_files):
        if grayscale:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            axis.imshow(image, cmap="gray")
        else:
            image = cv2.imread(filename)
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        axis.set_title(title)
        axis.axis("off")

    output_path = os.path.join(
        transformations_dir,
        "segmentation_comparison.png",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

def main() -> None:
    """Load the base image, split its channels, and save the channel images."""
    base_image_path = resolve_image_path(BASE_IMAGE_NAME)
    base_image = load_image(base_image_path)
    channels = split_color_channels(base_image)
    save_channel_images(os.path.splitext(os.path.basename(base_image_path))[0], channels)
    equalized_channels = equalize_color_channels(channels)
    save_equalized_image(equalized_channels)
    equalized_image = cv2.merge(equalized_channels)
    grayscale_image = convert_to_grayscale(equalized_image)
    otsu_mask, otsu_foreground, otsu_background = apply_otsu_threshold(grayscale_image)
    adaptive_mask, adaptive_foreground, adaptive_background = apply_adaptive_threshold(grayscale_image)
    save_threshold_outputs("equalized_image_otsu", grayscale_image, otsu_mask, otsu_foreground, otsu_background)
    save_threshold_outputs("equalized_image_adaptive", grayscale_image, adaptive_mask, adaptive_foreground, adaptive_background)
    hsv_image = convert_to_hsv(equalized_image)
    save_hsv_image(hsv_image)
    for cluster_count in (3, 4, 5):
        labels = apply_kmeans_clustering(hsv_image, cluster_count)
        save_kmeans_outputs(hsv_image, labels, cluster_count)
    report_ground_truth_metrics()
    print(f"Saved channel images for {BASE_IMAGE_NAME} in {os.path.join(SCRIPT_DIR, 'Image_Transformations')}")
    print(f"Saved equalized merged image as equalized_image.png in {os.path.join(SCRIPT_DIR, 'Image_Transformations')}")
    print(f"Saved Otsu and adaptive threshold outputs in {os.path.join(SCRIPT_DIR, 'Image_Transformations')}")
    print(f"Saved HSV conversion and K-means outputs in {os.path.join(SCRIPT_DIR, 'Image_Transformations')}")
    generate_segmentation_comparison_plot()

if __name__ == "__main__":
    main()

