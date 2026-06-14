# CS898BA Image Analysis Project

**Overview:**
- **Purpose:** Analyze `HW1_IMG_CS898BA.png` using OpenCV to compute per-channel (B, G, R) statistics and produce human-readable reports.
- **Primary script:** [imageanalysis.py](imageanalysis.py)
- **Log file:** [AI_Log.md](AI_Log.md) — every prompt, response, and code change is recorded there.

**Dependencies:**
- Python 3.8+
- numpy
- opencv-python
- scipy (optional — used for `mode` and `skew`; the script falls back to NumPy when SciPy isn't installed)

Install dependencies (recommended venv):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install numpy opencv-python
# Optional
pip install scipy
```

**Files of interest:**
- [imageanalysis.py](imageanalysis.py) — module that implements:
  - `load_image(image_path)` — load image via OpenCV
  - `split_channels(image)` — split into B, G, R
  - `compute_channel_stats(channel, channel_name)` — min/max/mean/median/range/variance/std/mode/skew
  - `compute_mode(channel)` — mode calculation (SciPy or histogram fallback)
  - `compute_skew(channel)` — skewness (SciPy or manual calculation)
  - `to_grayscale(image)` — convert BGR to grayscale
  - `to_binary_otsu(gray_image)` — convert a grayscale image to binary using Otsu's method
  - `to_hsv(image)` — convert BGR to HSV
  - `to_lab(image)` — convert BGR to LAB
  - `to_hls(image)` — convert BGR to HLS
  - `analyze_image_channels(image_path)` — orchestrator returning a dict of per-channel stats
  - `format_channel_report(channel_stats)` — returns a formatted text report
  - CLI entry point: accepts `--image_path / -i` or will auto-locate `HW1_IMG_CS898BA.png` (script dir, then cwd)

- [AI_Log.md](AI_Log.md) — project interaction log (prompts, responses, and design/code changes).

**How to run**

Default behavior (auto-locates `HW1_IMG_CS898BA.png`):

```bash
python imageanalysis.py
```

Specify an image path explicitly:

```bash
python imageanalysis.py --image_path path/to/image.png
# or
python imageanalysis.py -i path/to/image.png
```

**Output**
- The script prints a multi-line channel statistics report to stdout. The report contains per-channel keys such as `min`, `max`, `mean`, `median`, `range`, `variance`, `std`, `mode`, `mode_count`, and `skew`.

**Results**
The following output was produced by running `imageanalysis.py` with the default image `HW1_IMG_CS898BA.png`:

```
Image channel statistics report:

Blue channel:
  min: 0.0
  max: 255.0
  mean: 21.831687002471
  median: 10.0
  range: 255.0
  variance: 687.9921505729162
  std: 26.229604468480193
  mode: 4
  mode_count: 241924
  skew: 1.6796477233450815

Green channel:
  min: 0.0
  max: 255.0
  mean: 24.640292543353457
  median: 16.0
  range: 255.0
  variance: 493.9638283375659
  std: 22.22529703598055
  mode: 10
  mode_count: 194198
  skew: 1.7618096642654713

Red channel:
  min: 0.0
  max: 255.0
  mean: 20.60784912109375
  median: 12.0
  range: 255.0
  variance: 504.2560132758861
  std: 22.45564546558139
  mode: 4
  mode_count: 222718
  skew: 2.105449369974975
```

**Docstring recommendations**
- Each function in `imageanalysis.py` uses a docstring pattern including:
  - Purpose: a one-line summary
  - Inputs: parameter names and expected types
  - Outputs: return values and types
  - Notes: any additional behavior or side-effects

Example:

```python
def load_image(image_path: str) -> np.ndarray:
    """Load a color image from disk using OpenCV.

    Inputs:
        image_path (str): Path to the input image file.

    Outputs:
        np.ndarray: Loaded BGR image array.
    """
```

**Changes implemented so far**
- Created `imageanalysis.py` implementing channel analysis functions and CLI.
- Added optional SciPy support for `mode` and `skew` with NumPy-based fallbacks.
- Added CLI option `--image_path / -i` and default lookup for `HW1_IMG_CS898BA.png` in both the script directory and the current working directory.
- Moved `argparse` and `os` imports to module top-level for style/clarity.
- Maintained activity and decisions in [AI_Log.md](AI_Log.md).

**Contact / Notes**
- I will update this `README.md` every time we make code changes or generate results. If you prefer a different layout or additional sections (visualizations, CSV export, sample outputs), tell me and I'll add them.
