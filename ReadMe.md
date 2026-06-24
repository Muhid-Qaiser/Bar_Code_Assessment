# Barcode Orient and Decode

A classical computer vision pipeline for orienting and decoding 1D barcodes from images. No model training is required; the approach relies on gradient morphology, geometric deskewing, and [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar).

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

I used OpenCV `minAreaRect` on a barcode contour to estimate the tilt angle, then `warpAffine` to deskew the crop. After straightening, I tested rotations of 0, 90, 180, and 270 degrees and kept the first decode that pyzbar accepted.

**Why I picked this**

`minAreaRect` is a standard way to measure rotated rectangles. 1D barcodes only need to be axis-aligned for reading, so deskew plus a few right-angle rotations felt like the right lightweight approach.

**What went wrong**

Two problems showed up quickly.

First, the contour I used for deskewing was not reliable. A single wide morphological kernel `(21, 7)` bridged the barcode stripes with human-readable text below the bars. The detected shape was too large and the deskew angle was off.

Second, pyzbar does not care about visual upright. It can return the correct string while the saved image is still sideways or upside down. Stopping at the first successful decode was not enough.

**What I changed**

To isolate the stripes more accurately, I built a gradient mask with Scharr filters and used two narrower kernels, `(25, 5)` and `(5, 25)`, then merged them. Horizontal and vertical bar layouts are both covered, but nearby text is less likely to be pulled into the mask.

`minAreaRect` also has a known 90 degree ambiguity on near-square regions. I added multiple deskew candidates and tested them instead of trusting the first angle only.

For visual upright output, I used pyzbar's own `orientation` metadata (`UP`, `DOWN`, `LEFT`, `RIGHT`) to apply one final correction rotation after decode. I tested all four cardinal angles on the deskewed image and chose the result that needed the smallest total rotation.

When contour-based deskew still failed, I kept a fallback brute-force search from -90 to 90 degrees in 5 degree steps, again preferring the least rotation.

**What I tried later but reverted**

I experimented with Hough line detection to estimate barcode angle faster during fallback search, and with extra preprocessing such as adaptive thresholding. On my 27 manual crops, these changes made orientation worse, not better. I reverted to the simpler contour deskew plus cardinal testing plus pyzbar orientation correction, which gave the best visual results.

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

On my manual crop set in `manual_cropping/` (27 images), the final orient and decode pipeline decodes every image and produces visually upright barcodes with vertical bars. The main lesson for me was that decode success and visual correctness are not the same problem, and the final solution needed both geometric deskewing and pyzbar's orientation signal to get consistent upright outputs.

