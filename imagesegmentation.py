"""Feature Segmentation Module.

This branch starts by loading the renamed base image, base_image.png,
splitting it into its B, G, and R channels, and saving those channel images
into the Image_Transformations directory.
"""

import os

import cv2
import numpy as np


BASE_IMAGE_NAME = "base_image.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_image_path(image_name: str) -> str:
    """Find an image in the working directory, then fall back to the script directory."""
    working_directory_path = os.path.join(os.getcwd(), image_name)
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
    transformations_dir = os.path.join(os.getcwd(), "Image_Transformations")
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
    transformations_dir = os.path.join(os.getcwd(), "Image_Transformations")
    os.makedirs(transformations_dir, exist_ok=True)

    equalized_image = cv2.merge(equalized_channels)
    output_path = os.path.join(transformations_dir, "equalized_image.png")
    cv2.imwrite(output_path, equalized_image)


def main() -> None:
    """Load the base image, split its channels, and save the channel images."""
    base_image_path = resolve_image_path(BASE_IMAGE_NAME)
    base_image = load_image(base_image_path)
    channels = split_color_channels(base_image)
    save_channel_images(os.path.splitext(os.path.basename(base_image_path))[0], channels)
    equalized_channels = equalize_color_channels(channels)
    save_equalized_image(equalized_channels)
    print(f"Saved channel images for {BASE_IMAGE_NAME} in {os.path.join(os.getcwd(), 'Image_Transformations')}")
    print(f"Saved equalized merged image as equalized_image.png in {os.path.join(os.getcwd(), 'Image_Transformations')}")


if __name__ == "__main__":
    main()

