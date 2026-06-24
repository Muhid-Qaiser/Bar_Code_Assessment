# Barcode Orient and Decode

A classical computer vision pipeline for orienting and decoding 1D barcodes from images. No model training is required; the approach relies on gradient morphology, geometric deskewing, and [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar).

## Main Entry Point

The primary workflow is `**main.ipynb**`. It processes manually cropped barcode images from `manual_cropping/` through orientation and decoding. This notebook uses `barcode_orient_decode.py` only and does not perform automatic detection.

Place pre-cropped images (any rotation) in `manual_cropping/`, then run the notebook to produce upright, decoded results.

## Project Structure


| File                       | Role                                                     |
| -------------------------- | -------------------------------------------------------- |
| `main.ipynb`               | Orient and decode manually cropped images                |
| `auto_crop_pipeline.ipynb` | Full pipeline including automatic detection and cropping |
| `barcode_orient_decode.py` | Orientation and decoding logic                           |
| `barcode_detect.py`        | Barcode detection and cropping from full photos          |
| `manual_cropping/`         | Input folder for manually cropped images                 |
| `tagss/`                   | Input folder for full images (auto pipeline)             |


## Orientation Logic

Orientation is handled in `barcode_orient_decode.py` via `orient_and_decode()`.

1. **Find barcode stripes** — A gradient mask isolates parallel stripe regions inside the crop using Scharr gradients and morphological closing.
2. **Deskew** — The largest stripe contour is passed to `minAreaRect`. The estimated angle straightens tilted barcodes. Near-square contours also test 90 degree alternatives to resolve ambiguity.
3. **Cardinal rotation** — The deskewed image is tested at 0, 90, 180, and 270 degrees so pyzbar can read axis-aligned barcodes regardless of initial orientation.
4. **Visual upright correction** — pyzbar reports an orientation flag (`UP`, `DOWN`, `LEFT`, `RIGHT`). An additional rotation is applied so the output image is visually upright with bars vertical and text at the bottom.
5. **Fallback** — If contour deskew fails, a brute-force sweep from -90 to 90 degrees is used. The result with the smallest total rotation is preferred.

## Decoding Logic

Decoding is performed with **pyzbar** in `decode_barcode()`.

1. **Preprocessing variants** — Each image is tried as raw grayscale, CLAHE-enhanced, Otsu-binarised, and upscaled (1.5x and 2x) to handle low contrast, blur, and small barcodes.
2. **First successful decode wins** — Variants are passed to `pyzbar.decode()` in order until a barcode is found.
3. **Output** — A `DecodeResult` is returned containing the decoded text, barcode type, total rotation applied, and the upright oriented image.

In `main.ipynb`, `orient_and_decode()` is called directly on each manual crop. In the auto pipeline, a fast path attempts direct decode on the crop first; full orientation runs only when needed.

## Automatic Cropping

`**auto_crop_pipeline.ipynb`** extends the workflow by automating the cropping step. It reads full images from `tagss/`, detects barcode regions with gradient morphology (`barcode_detect.py`), crops each detection with a rotated axis-aligned slice, then orients and decodes the result.

Outputs are saved to `cropped/` with optional debug visualisations in `debug/`.

## Setup

```bash
pip install -r requirements.txt
```

**Note:** pyzbar requires the ZBar library on your system. On Windows, install the [ZBar binaries](https://sourceforge.net/projects/zbar/files/) and ensure they are on your PATH.

## Requirements

- Python 3.10+
- OpenCV, NumPy, matplotlib, pandas, pyzbar, Pillow

