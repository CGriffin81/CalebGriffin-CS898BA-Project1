# AI Interaction Log - Feature-Segmentation Branch

This log documents all interactions for the Feature-Segmentation work on image segmentation and ground truth analysis.

---

## Entry 8: Add metric collection to the script

**Date & Time:** June 28, 2026 | 10:16 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Code implementation update

### Full Prompt
```
It's great that you did that, but I want the procedure included in the imagesegmentation.py script. Add the code to collect these metrics in there, but do not make it add the results to readme.
```

### Result Synopsis
- Added a script procedure to compute IoU/Jaccard and Dice metrics against `ground_truth_mask.png`.
- Kept the metric collection inside `imagesegmentation.py` and limited the output to the console.
- Left the README unchanged for this step.

### Design/Code Changes
- Added `load_binary_mask()`, `compute_overlap_metrics()`, and `report_ground_truth_metrics()` to `imagesegmentation.py`.
- Integrated the metric reporting step into `main()` after the K-means outputs are generated.
- Did not add any new README content for this step.

---

## Entry 1: Start the Feature-Segmentation branch log

**Date & Time:** June 27, 2026 | 9:58 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Branch setup and documentation

### Full Prompt
```
Following the rules as previously described, write entries in AI_Log as we go through. Also update the README to keep it in line with imagesegmentation.py. Use the Purpose section of README as a roadmap to what we are expecting to accomplish. If you need me to re-explain the rules for AI_Log entries, ask. Include this prompt as entry 1 for this project.
```

### Result Synopsis
- Confirmed the Feature-Segmentation branch should start with a fresh log and branch-specific documentation.
- Began aligning README content with the current `imagesegmentation.py` roadmap.
- No analysis pipeline changes were made yet.

### Design/Code Changes
- Reset the README to describe the segmentation and base-truth analysis plan for this branch.
- Established the first AI_Log entry for the new project phase.

---

## Entry 2: Load base image and split color channels

**Date & Time:** June 27, 2026 | 10:26 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Code implementation and documentation update

### Full Prompt
```
Let's move on. Write code to load the base image (base_image.png) and split it into its three separate color channels. Save those images in the Image Transformations directory. Also update the readme to reflect the renamed image file.
```

### Result Synopsis
- Added code to load `base_image.png`, split it into B, G, and R channels, and save the channel images in `Image Transformations`.
- Updated the README to reference the renamed base image.
- Set the branch up for the next segmentation steps.

### Design/Code Changes
- Replaced the starter script body with base-image loading and color-channel export logic.
- Added `split_color_channels()` and `save_channel_images()` helpers.
- Updated the README purpose text to mention `base_image.png`.

---

## Entry 3: Resolve image paths from the working directory

**Date & Time:** June 27, 2026 | 10:33 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Code implementation and documentation update

### Full Prompt
```
Add code to find the image and paths from the working directory using os.
```

### Result Synopsis
- Updated the script to resolve `base_image.png` from the working directory first, with a script-directory fallback.
- Switched output path handling to `os.path` and `os.makedirs`.
- Updated the README to describe the working-directory lookup behavior.

### Design/Code Changes
- Added `resolve_image_path()` to locate the base image using `os.getcwd()`.
- Replaced `pathlib` usage with `os.path` and `os.makedirs` for channel output paths.
- Added a README note explaining the working-directory search order.

---

## Entry 4: Equalize the channel images and merge them back together

**Date & Time:** June 27, 2026 | 10:42 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Code implementation and documentation update

### Full Prompt
```
On the channel images, apply histogram equalization to normalize illumination and maximize contrast in each image, then merge the three images back to a single BGR image, as equalized_image.png.
```

### Result Synopsis
- Added histogram equalization for each color channel.
- Merged the equalized channels back into a single BGR image and saved it as `equalized_image.png`.
- Updated the README to document the new equalized output.

### Design/Code Changes
- Added `equalize_color_channels()` to equalize each channel independently.
- Added `save_equalized_image()` to merge the equalized channels and write `equalized_image.png`.
- Updated the README to mention the equalized merged image output.

---

## Entry 5: Threshold the equalized grayscale image

**Date & Time:** June 27, 2026 | 11:08 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Code implementation and documentation update

### Full Prompt
```
Take equalized_image and convert it to grayscale, then apply Otsu's automatic thresholding and adaptive thresholding using Gaussian window to the grayscale image to separate foreground from background. Save the binary masks, foreground images, and background images for both.
```

### Result Synopsis
- Added grayscale conversion for `equalized_image.png`.
- Added Otsu and adaptive Gaussian thresholding on the grayscale image.
- Saved the grayscale, binary mask, foreground, and background images for both thresholding methods.

### Design/Code Changes
- Added `convert_to_grayscale()` to create a grayscale version of the equalized image.
- Added `apply_otsu_threshold()` and `apply_adaptive_threshold()` for foreground/background separation.
- Added `create_foreground_background()` and `save_threshold_outputs()` helpers to write the derived images.
- Updated the README to document the new thresholding outputs.

---

## Entry 6: Convert to HSV and apply K-means clustering

**Date & Time:** June 28, 2026 | 9:33 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Code implementation and documentation update

### Full Prompt
```
Use equalized_image and convert it to HSV color space. Save the output from the conversion as hsv_image. Apply K-means clustering to hsv_image using k= 3, 4, and 5. Save the binary masks and foregrounds.
```

### Result Synopsis
- Converted `equalized_image.png` to HSV and saved the output as `hsv_image.png`.
- Applied K-means clustering for `k=3`, `k=4`, and `k=5` on the HSV image.
- Saved one binary mask and one foreground image for each cluster in each K-means run.

### Design/Code Changes
- Added `convert_to_hsv()` and `save_hsv_image()` to handle the HSV conversion output.
- Added `apply_kmeans_clustering()` and `save_kmeans_outputs()` to segment the HSV image into cluster masks and foregrounds.
- Updated the README to document the HSV conversion and K-means outputs.

---

## Entry 7: Measure overlap against the ground truth mask

**Date & Time:** June 28, 2026 | 10:14 AM CST
**AI Tool:** GitHub Copilot (GPT-5.4 mini)
**Prompt Type:** Metrics calculation and documentation update

### Full Prompt
```
Using the manually generated ground_truth_mask.png as the ground truth, calculate the Intersection over Union/Jaccard Index and Dice Coefficient for adaptive foreground, otsu foreground, and k4 cluster 4 foreground, entering the results in the readme file.
```

### Result Synopsis
- Calculated IoU / Jaccard Index and Dice Coefficient against `ground_truth_mask.png`.
- Compared adaptive foreground, Otsu foreground, and K-means `k=4` cluster 4 foreground.
- Added the measured values to the README in a dedicated ground truth metrics section.

### Design/Code Changes
- No script changes were required for this measurement step.
- Updated the README with a metrics table and a short interpretation of the results.

