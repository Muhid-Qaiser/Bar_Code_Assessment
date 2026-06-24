"""Phase 2 (orientation) and Phase 3 (decoding) for cropped barcode images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyzbar.pyzbar import decode


@dataclass
class DecodeResult:
    text: str
    barcode_type: str
    rotation_angle: float
    deskew_angle: float
    oriented_image: np.ndarray
    raw_image: np.ndarray
    contour: np.ndarray | None = None


def _rotate_image(image: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]
        size = (new_width, new_height)
    else:
        size = (width, height)

    return cv2.warpAffine(
        image,
        matrix,
        size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _barcode_mask(gray: np.ndarray) -> np.ndarray:
    """Build a binary mask of barcode-like parallel line regions."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 1, 0))
    grad_y = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 0, 1))
    gradient = cv2.max(cv2.subtract(grad_x, grad_y), cv2.subtract(grad_y, grad_x))

    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def find_barcode_contour(image: np.ndarray) -> np.ndarray | None:
    """Find the largest barcode-like contour inside a cropped ROI."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = _barcode_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _correct_min_area_angle(angle: float) -> float:
    """Convert minAreaRect angle into a deskew rotation."""
    if angle < -45:
        return -(90 + angle)
    return -angle


def straighten_barcode(cropped_image: np.ndarray, contour: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Phase 2: straighten a cropped barcode using minAreaRect + warpAffine.

    1D barcodes only need to be axis-aligned (horizontal or vertical bars).
    Pyzbar can read them upside down; they do not need text at the bottom.
    """
    if contour is None:
        contour = find_barcode_contour(cropped_image)
    if contour is None:
        return cropped_image.copy(), 0.0, None

    rect = cv2.minAreaRect(contour)
    angle = _correct_min_area_angle(rect[-1])
    straightened = _rotate_image(cropped_image, angle, expand=True)
    return straightened, angle, contour


def _preprocess_variants(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variants = [gray]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    for scale in (1.5, 2.0):
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)

    return variants


def decode_barcode(image: np.ndarray) -> list:
    """Phase 3: decode a straightened barcode image with pyzbar."""
    results = []
    for variant in _preprocess_variants(image):
        results.extend(decode(variant))
        if results:
            return results
    return results


def _decode_with_cardinals(image: np.ndarray, base_angle: float) -> DecodeResult | None:
    """Try straightened image at 0/90/180/270 degrees (axis-aligned is enough)."""
    for cardinal in (0, 90, 180, 270):
        oriented = image if cardinal == 0 else _rotate_image(image, cardinal)
        decoded = decode_barcode(oriented)
        if decoded:
            item = decoded[0]
            return DecodeResult(
                text=item.data.decode("utf-8"),
                barcode_type=item.type,
                rotation_angle=base_angle + cardinal,
                deskew_angle=base_angle,
                oriented_image=oriented,
                raw_image=image,
            )
    return None


def _fallback_angle_search(image: np.ndarray) -> DecodeResult | None:
    """Fallback if minAreaRect deskew alone is not enough."""
    for angle in range(-90, 91, 5):
        rotated = _rotate_image(image, angle)
        for flip in (0, 180):
            oriented = rotated if flip == 0 else _rotate_image(rotated, 180)
            decoded = decode_barcode(oriented)
            if decoded:
                item = decoded[0]
                return DecodeResult(
                    text=item.data.decode("utf-8"),
                    barcode_type=item.type,
                    rotation_angle=angle + flip,
                    deskew_angle=angle,
                    oriented_image=oriented,
                    raw_image=image,
                )
    return None


def orient_and_decode(image: np.ndarray) -> DecodeResult | None:
    """Run Phase 2 straightening, then Phase 3 decoding."""
    straightened, deskew_angle, contour = straighten_barcode(image)

    result = _decode_with_cardinals(straightened, deskew_angle)
    if result is not None:
        result.raw_image = image
        result.contour = contour
        return result

    fallback = _fallback_angle_search(image)
    if fallback is not None:
        fallback.contour = contour
    return fallback


def process_manual_crops(input_dir: str | Path) -> list[dict]:
    input_dir = Path(input_dir)
    rows = []

    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({"file": image_path.name, "error": "failed to read image"})
            continue

        result = orient_and_decode(image)
        if result is None:
            rows.append({"file": image_path.name, "error": "decode failed"})
            continue

        rows.append(
            {
                "file": image_path.name,
                "text": result.text,
                "type": result.barcode_type,
                "rotation_angle": result.rotation_angle,
                "deskew_angle": result.deskew_angle,
                "oriented_image": result.oriented_image,
                "raw_image": result.raw_image,
                "contour": result.contour,
            }
        )

    return rows


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    results = process_manual_crops(project_root / "manual_cropping")

    print(f"{'File':<40} {'Type':<10} {'Angle':>7}  Decoded")
    print("-" * 80)
    for row in results:
        if "error" in row:
            print(f"{row['file']:<40} ERROR: {row['error']}")
        else:
            print(
                f"{row['file']:<40} {row['type']:<10} {row['rotation_angle']:>7.1f}  {row['text']}"
            )
