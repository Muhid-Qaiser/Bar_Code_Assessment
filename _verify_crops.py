"""Verify cropping quality: print size, std-dev, mean for every crop."""
from pathlib import Path
import cv2
import numpy as np
from barcode_detect import detect_barcode_regions, crop_barcode

IMG_DIR = Path(r"d:\D-Documents\WOrk\Interviews\Bar_Code_Assessment\tagss")

total = valid = blank = skipped = 0
for img_path in sorted(IMG_DIR.glob("*")):
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    regions, _ = detect_barcode_regions(image)
    total += len(regions)
    for r in regions:
        crop = crop_barcode(image, r)
        if crop is None:
            skipped += 1
            print(f"  SKIP (None)  {img_path.name}")
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        mean = float(np.mean(gray))
        h, w = crop.shape[:2]
        is_blank = std < 10 or mean > 242
        if is_blank:
            blank += 1
            print(f"  BLANK  {img_path.name}  {w}x{h}  std={std:.1f}  mean={mean:.1f}")
        else:
            valid += 1

print()
print(f"Regions detected : {total}")
print(f"  None (skipped) : {skipped}")
print(f"  Blank/uniform  : {blank}")
print(f"  Valid crops    : {valid}")
