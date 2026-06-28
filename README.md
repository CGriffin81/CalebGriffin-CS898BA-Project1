# CS898BA Image Segmentation Project

## Purpose

This branch is for a new segmentation and base-truth analysis pass on `base_image.png`.

The current roadmap is:
- load and inspect the base image
- build preprocessing and normalization helpers
- compare threshold-based segmentation methods
- compare classical segmentation methods such as K-Means
- evaluate and document the results clearly as the script evolves

## Current Files

- [imagesegmentation.py](imagesegmentation.py) is the active script for this branch.
- [AI_Log.md](AI_Log.md) records prompts, responses, and code changes as we go.

## Dependencies

- Python 3.8+
- numpy
- opencv-python
- matplotlib
- scipy is optional and only needed for statistics helpers that use `mode` or `skew`

## Setup

Create a virtual environment and install the libraries used by the script:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install numpy opencv-python matplotlib
pip install scipy
```

## Run

Use the default image lookup:

```bash
python imagesegmentation.py
```

Or provide an image path explicitly:

```bash
python imagesegmentation.py --image_path path\to\image.png
```

The current branch uses `base_image.png` as the renamed starter image.
The script looks for that file in the working directory first and then falls back to the script directory if needed.
Running the script saves the separated channel images into `Image_Transformations` as `base_image_blue_channel.png`, `base_image_green_channel.png`, and `base_image_red_channel.png`.
It also applies histogram equalization to each channel, merges the equalized channels, and saves the result as `equalized_image.png` in the same folder.
The equalized image is then converted to grayscale and processed with Otsu and adaptive Gaussian thresholding. The script saves the grayscale image, binary mask, foreground image, and background image for each method in `Image_Transformations`.
The equalized image is also converted to HSV and saved as `hsv_image.png`, then clustered with K-means for `k=3`, `k=4`, and `k=5` with one binary mask and one foreground image saved for each cluster.

## Notes

- Keep this README synchronized with the actual script behavior.
- Add results, outputs, and segmentation-specific details after the next implementation pass.

## Discussion
- Otsu
  - Did a pretty poor job of being able to identify the figure as being in the foreground, opting to assume that things bright and in the center of the image were meant to be identified as foreground objects.
- Adaptive
  - Also did a poor job of identifying the figure as being in the foreground. The resulting foreground image looks like it mostly brought brightness values in the image to a more normalized range of values, but if I were given the foreground image that the Adaptive technique provided, I wouldn't be able to tell that it is supposed to be identifying a foreground.
- K-Means
  - K-Means where k=4 did a pretty good job of identifying the figure as being in the foreground in cluster 4, with some minor noise in the back coming from the houses, because the colors were fairly similar. Before getting the K-Means code and results, I predicted this would be the case, because the color spaces are fairly well divided into four categories.
- Color Normalization
  - The edge detection from homework one was very shoddy and did not produce the results I was hoping for.
  - Implementing color normalization on the image before performing edge detection made the methods used more fruitful, with K-means being the best at identifying the figure as being in the foreground.

## Ground Truth Metrics

Using `ground_truth_mask.png` as the reference mask and treating nonzero pixels in the saved foreground images as predicted foreground, the overlap metrics are:

| Result | IoU / Jaccard Index | Dice Coefficient |
| --- | ---: | ---: |
| Adaptive foreground | 0.0610 | 0.1150 |
| Otsu foreground | 0.0365 | 0.0705 |
| K-means `k=4` cluster 4 foreground | 0.0631 | 0.1187 |

The `k=4` cluster 4 foreground performed best on both metrics, with adaptive foreground close behind and Otsu trailing the other two.

## Ran out of CoPilot credits after making it write the metrics gathering into the script.
