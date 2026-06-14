# AI Interaction Log - CS898BA Image Analysis Project

---

## Entry 1: Establish AI log process for future prompts

**Date & Time:** June 13, 2026 | 8:24 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** OpenCV overview and documentation guidance

### Full Prompt
```
The image file titled HW1_IMG_CS898BA.png will be the starting point. We will use OpenCV to analyze the photo. I have never used OpenCV before, so I'll need you to give me an overview of what functions are available to suit my needs. As we generate code, I want you to create and maintain a readme.md with code explanations, setup and execution steps, and the results of the code, along with relevant discussion. I also want you to recommend docstrings to add into the python scripts to document my functions with the following criteria: Inputs, Outputs, and an overview of what the function is meant to do.
```

### Result Synopsis
- Confirmed the project will use OpenCV and that `HW1_IMG_CS898BA.png` is the starting image
- Provided an overview of OpenCV capabilities and task categories
- Clarified how `README.md` can be maintained with setup, execution, results, and discussion as code is generated
- Recommended docstring structure for future Python functions: purpose, inputs, outputs

### Design/Code Changes
- None. No code or file changes were made beyond logging this prompt.

---

**Date & Time:** June 13, 2026 | 8:56 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Function recommendations for channel statistics

### Full Prompt
```
Show me the functions you recommend to analyze the channels in the image I mentioned.
```

### Result Synopsis
- Recommended a set of functions to load the image, split channels, compute per-channel statistics, and optionally compute skew/mode
- Suggested using OpenCV for image I/O, NumPy for core statistics, and SciPy only if mode/skew is required
- Emphasized that these functions should be descriptive, reusable, and return structured results without modifying the image

### Design/Code Changes
- None. Logged prompt and response design guidance only.

---

**Date & Time:** June 13, 2026 | 9:28 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Code update

### Full Prompt
```
Because we will always start with HW1_IMG_CS898BA.png, let's add locating that file to the script, rather than passing an argument on runtime.
```

### Result Synopsis
- Updated `imageanalysis.py` to search for `HW1_IMG_CS898BA.png` in the script directory and current working directory when no `--image_path` is provided
- Added CLI option `--image_path / -i` to override the default lookup behavior

### Design/Code Changes
- Modified `imageanalysis.py`: now auto-locates default image and raises a clear `FileNotFoundError` if missing

---

**Date & Time:** June 13, 2026 | 10:00 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Documentation

### Full Prompt
```
Generate a Readme.md file with code explanations, setup and execution steps, and results from my script. Write updates to this file as we make changes. Include all changes and steps we have already made so far.
```

### Result Synopsis
- Created `README.md` describing the project purpose, dependencies, script functions, CLI usage, docstring recommendations, and a summary of changes implemented so far.
- Noted next steps and suggested actions (run script, add requirements, tests).

### Design/Code Changes
- Added `README.md` to repository root.

---

**Date & Time:** June 13, 2026 | 10:32 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Execution & documentation

### Full Prompt
```
Run imageanalysis.py to include the relevant output in the Readme.
```

### Result Synopsis
- Executed `imageanalysis.py` using the default `HW1_IMG_CS898BA.png` found in the project folder.
- Captured the printed per-channel statistics and appended them to `README.md` under a `Results` section.

### Design/Code Changes
- Updated `README.md` to include the analysis output.

---

**Date & Time:** June 13, 2026 | 11:04 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Function recommendations

### Full Prompt
```
What functions can I use to convert the original image into greyscale, binary, and different color spaces?
```

### Result Synopsis
- Listed OpenCV functions and common usage patterns for converting images to grayscale, producing binary masks (global, adaptive, and Otsu thresholding), and converting between BGR and other color spaces (HSV, LAB, YCrCb, HLS, RGB, YUV, XYZ).
- Implemented helper functions in `imageanalysis.py` for grayscale conversion, Otsu thresholding, and BGR conversions to HSV, LAB, and HLS.

### Design/Code Changes
- Added helper functions to `imageanalysis.py`: `to_grayscale`, `to_binary_otsu`, `to_hsv`, `to_lab`, and `to_hls`.
- Added script logic to create an `Image Transformations` directory and save transformed images there with descriptive filenames.
- Added `save_transformation()` helper to centralize image saving and enforce a consistent filename template.
- Added affine transformation helpers and logic to process existing images in `Image Transformations`, saving outputs in `Image Transformations/Affine Transformations`.
- Added explicit uniqueness verification that each image receives two different affine transform types.

---

**Date & Time:** June 13, 2026 | 11:20 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Code implementation

### Full Prompt
```
We're going to apply gaussian blur to each image in Image Transformations and Affine Transformations in sigma increments from .5 to 3.5, in steps of .5.
```

### Result Synopsis
- Implemented Gaussian blur generation for all images in `Image Transformations` and `Affine Transformations`.
- Applied blur at sigma values: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 (7 values per source image).
- Generated 147 total blurred images (21 sources × 7 sigma values).

### Design/Code Changes
- Added `apply_gaussian_blur(image, sigma)` function to apply Gaussian blur using OpenCV.
- Added `blur_images_in_directories(transformation_dir, affine_dir, output_dir, sigma_values)` to process all images.
- Created `Image Transformations/Gaussian Blur` output directory with consistent naming: `{source}_gaussian_sigma_{sigma:.1f}.png`.
- Added main script logic to generate blur pipeline on startup.

---

**Date & Time:** June 13, 2026 | 11:52 PM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Code implementation

### Full Prompt
```
We now need to create four equal subsets... randomly drafting the 168 images into four different directories inside Image Transformations: Subset 1, Subset 2, Subset 3, and Subset 4.
```

### Result Synopsis
- Implemented random subset creation to divide all 168 transformation+blur images evenly.
- Each subset contains 42 images (168 ÷ 4).
- Images are randomly shuffled before distribution to ensure unbiased subsets.

### Design/Code Changes
- Added `gather_transformation_images(transformation_dir, excluded_dirs)` to collect images while excluding subdirectories.
- Added `create_random_subsets(transformation_dir, subset_names, expected_total)` to perform random distribution.
- Added `verify_subset_counts(transformation_dir, subset_names, expected_count)` to confirm each subset has 42 images.
- Created four subset directories: `Subset 1`, `Subset 2`, `Subset 3`, `Subset 4` inside `Image Transformations`.

---

**Date & Time:** June 14, 2026 | 12:24 AM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Code implementation

### Full Prompt
```
We need to add validation to detect if the Edge Detection subdirectory already exists and has 168 images in it.
```

### Result Synopsis
- Added specialized validation for the edge detection output directory.
- Validates both count (168 images) and name matching with source images.
- Clears directory if validation fails.

### Design/Code Changes
- Added `validate_edge_detection_directory(edge_dir, subset_dir, expected_count=168)` to check count and file name consistency.
- Verifies that for each source image, all four edge variants exist.
- Integrated into main script to skip edge detection if already complete.

---

**Date & Time:** June 14, 2026 | 12:54 AM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Code update and documentation

### Full Prompt
```
The plots need to contain the following information: What sample number it is. The transformation process in full: Original -> Color Space Change Type (Greyscale, binary, HSV, LAB, or HLS) -> The details of the Affine Transformation (What type of transformation, and the metrics of the transformation) -> The degree of Gaussian Blue applied. The plot should also include headings for each image selected. It should select a base image and its four edge transformations.
```

### Result Synopsis
- Enhanced plot captions to include sample number, full transformation pipeline, and explicit headings.
- Each subplot is clearly labeled: Original, Sobel, Laplacian, Canny, Prewitt.
- Parsed affine metadata from filenames to display exact transformation parameters.

### Design/Code Changes
- Updated `generate_random_affine_matrix()` to return metadata (angle, tx/ty, sx/sy, shear values).
- Updated `apply_affine_transformations()` to append metadata to filenames for later parsing.
- Enhanced `describe_transformation_pipeline()` to extract and display affine parameters.
- Updated `build_edge_detection_plots()` to include sample index, detailed pipeline description, and clear subplot titles.
- Improved plot layout with larger figsize and refined spacing for readability.

---

**Date & Time:** June 14, 2026 | 1:24 AM CST
**AI Tool:** GitHub Copilot (Raptor mini Preview)
**Prompt Type:** Documentation update

### Full Prompt
```
Excellent. Update the AI Log and the readme.
```

### Result Synopsis
- Documented all plot generation features and improvements.
- Added matplotlib dependency to README.
- Included example plot filenames and transformation details.
- Updated README function list with new plotting helpers.

### Design/Code Changes
- Added matplotlib to project dependencies in README.
- Documented `Image Transformations/Plots` output directory.
- Added six selected plot examples to README.
- Created comprehensive AI Log entry documenting the feature.

---

