# Barcode Orient and Decode

A classical computer vision pipeline for orienting and decoding 1D barcodes from images. No model training is required; the approach relies on edge-based deskewing and [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar).

## Main Entry Point

The primary workflow is `main.ipynb`. It processes manually cropped barcode images from `manual_cropping/` through orientation and decoding. This notebook uses `barcode_orient_decode.py` only and does not perform automatic detection.

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

1. **Binary threshold and edges** — The crop is converted to grayscale, binarised with Otsu, then passed through Canny edge detection to highlight barcode bar edges.
2. **Stripe zone vs text below** — The image is split into a top stripe zone (barcode bars) and a lower zone (human-readable text). This keeps deskew focused on bar edges, not text below.
3. **Edge-based deskew** — Hough line segments are counted in each zone. Straight bar edges are nearly horizontal or vertical. The image is rotated over a range of angles and the best angle is chosen where aligned stripe edges outnumber edges in the text zone below.
4. **Cardinal rotation** — The deskewed image is tested at 0, 90, 180, and 270 degrees so pyzbar can read axis-aligned barcodes regardless of initial orientation.
5. **Visual upright correction** — pyzbar reports an orientation flag (`UP`, `DOWN`, `LEFT`, `RIGHT`). An additional rotation is applied so the output image is visually upright with bars vertical and text at the bottom.
6. **Fallback** — If edge deskew plus decode still fails, a brute-force sweep from -90 to 90 degrees is used. The result with the smallest total rotation is preferred.

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

## Proposed Solution and Thought Processes

My goal was to take already-cropped barcode images with unknown rotation and return two things: the correct decoded value, and an image that looks properly upright. I did not want to train a model. Barcodes are a good fit for classic image processing because they are just repeating parallel lines with strong light-dark contrast in one direction.

I built my main workflow in `main.ipynb` around manually cropped images in `manual_cropping/`. Each image may be tilted, sideways, or upside down. The job is to straighten it, read it, and save a clean upright version.

### High-Level Approach

At a high level, I split the problem into two steps.

**Orienting** means figuring out how much to rotate the image so the barcode lines are straight and the label reads the right way up, not just readable.

**Decoding** means passing the image to pyzbar and reading the barcode value.

The important insight is that these two steps are linked. pyzbar can often decode a barcode even when it is rotated, so getting the text right does not automatically mean the image looks correct. I needed both: a valid decode and a visually upright output.

### How the Solution Evolved

I started with the most direct idea: find the barcode shape in the crop, rotate it to straighten the lines, then send it to pyzbar. That worked for decoding on many images, but several still looked wrong even when the text was correct. I improved the pipeline step by step rather than replacing it entirely.

#### **Orientation**

**What I tried first**

I used a gradient mask to find a barcode stripe blob, then `minAreaRect` on that contour to estimate tilt and deskew with `warpAffine`. After straightening, I tested rotations of 0, 90, 180, and 270 degrees and used pyzbar orientation metadata to correct the final upright view.

**Why I picked this**

Fitting a rotated rectangle around the stripe region is a standard, lightweight way to measure tilt without training a model. pyzbar orientation metadata seemed like a clean way to fix the last 90 degree ambiguity for visual upright output.

**What went wrong**

Decoding was often correct, but orientation was not reliable enough. The contour included noise from text below the bars or extra background, so `minAreaRect` gave the wrong tilt on several images. Even after narrower morphological kernels, deskew candidates, and pyzbar upright correction, some crops still did not look properly straight. The blob approach was approximating the barcode angle instead of measuring the bar edges directly.

**What I changed (final approach)**

I moved to edge-based deskewing, which worked better on my manual crops.

1. Otsu binary threshold on the crop
2. Canny edge detection
3. Split the crop into a top stripe zone (bars) and a lower text zone
4. Hough lines in each zone; count edges that are nearly horizontal or vertical
5. Search rotation angles and pick the one where straight bar edges in the stripe zone outnumber edges in the text zone below
6. Keep the same decode step: cardinal rotations plus pyzbar orientation correction for visual upright output

This aligns rotation with the actual barcode line edges rather than a loose bounding contour, which is why the results looked better.

**What I tried in between but moved away from**

I briefly tried faster angle search with Hough during the contour-based fallback, and extra preprocessing variants inside orientation. Those changes did not improve results on my dataset, so I dropped them. The edge-based deskew above is what I kept.

#### **Decoding**

**What I tried first**

I passed the deskewed grayscale image directly to `pyzbar.decode()`.

**Why I picked pyzbar**

It is a mature barcode library, works well on classical 1D formats, and exposes orientation metadata that I could reuse for upright correction. That made it a better fit than treating decode and orient as fully separate problems.

**What went wrong**

Direct decode was enough on clean crops, but it failed on images with low contrast, uneven lighting, blur, or very small barcodes.

**What I changed**

Before giving up on an image, I now try a short list of preprocessing variants in order:

- raw grayscale
- CLAHE for local contrast
- Otsu binarisation for noisy or washed-out images
- upscaling at 1.5x and 2.0x for small barcodes

The pipeline stops at the first variant pyzbar can read. This improved recall without changing the orientation logic itself.

### Results

On my manual crop set in `manual_cropping/` (27 images), the final edge-based orient and decode pipeline decodes every image and produces visually upright barcodes with vertical bars. The contour-based deskew was a useful starting point, but edge-based straightening gave the most consistent visual results. The main lesson for me was that decode success and visual correctness are not the same problem, and measuring bar edges directly was the key to getting both right.

