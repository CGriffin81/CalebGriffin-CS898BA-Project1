"""
Feature Segmentation Module

This module contains functions for image segmentation and ground truth analysis.
Base image: HW1_IMG_CS898BA.png
"""

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

def load_image(image_path):
    """
    Load an image from file path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Image array in BGR format, or None if load fails
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

