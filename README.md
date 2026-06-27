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

## Notes

- Keep this README synchronized with the actual script behavior.
- Add results, outputs, and segmentation-specific details after the next implementation pass.
