"""Phase 2 (orientation) and Phase 3 (decoding) for cropped barcode images."""

from __future__ import annotations

from dataclasses import dataclass

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


# ZBar orientation to extra rotation for upright barcode.
_ROTATION_TO_UP = {
    "UP": 0.0,
    "RIGHT": -90.0,
    "DOWN": 180.0,
    "LEFT": 90.0,
}


def _rotate_image(image: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Expand canvas so rotated image is not clipped.
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
    """Binary mask of barcode stripes (narrow kernels avoid merging text below bars)."""
    # Gradient and threshold to isolate stripe regions.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 1, 0))
    grad_y = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 0, 1))
    gradient = cv2.max(cv2.subtract(grad_x, grad_y), cv2.subtract(grad_y, grad_x))

    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Close stripe gaps per orientation then merge.
    masks = []
    for kw, kh in ((25, 5), (5, 25)):  # horizontal and vertical bar layouts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        masks.append(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel))

    merged = cv2.bitwise_or(masks[0], masks[1])
    return cv2.morphologyEx(
        merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _bar_edge_counts(gray: np.ndarray) -> tuple[int, int]:
    """Count aligned bar edges in stripe zone vs edge lines in text zone below."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150)
    h = edges.shape[0]
    split = max(1, int(h * 0.7))
    stripe = edges[:split]
    below = edges[split:]

    def aligned_lines(region: np.ndarray) -> int:
        if region.size == 0:
            return 0
        min_len = max(8, region.shape[1] // 20)
        lines = cv2.HoughLinesP(
            region, 1, np.pi / 180, threshold=12,
            minLineLength=min_len, maxLineGap=6,
        )
        if lines is None:
            return 0
        count = 0
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            angle = min(angle, 180.0 - angle)
            if angle <= 15 or angle >= 75:
                count += 1
        return count

    return aligned_lines(stripe), aligned_lines(below)


def _stripe_vertical_deviation(gray: np.ndarray) -> tuple[float, int, int]:
    """Mean deviation of stripe edges from 90 degrees; requires stripe edges to exceed below."""
    stripe_n, below_n = _bar_edge_counts(gray)
    if stripe_n <= below_n:
        return float("inf"), stripe_n, below_n

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150)
    h = edges.shape[0]
    split = max(1, int(h * 0.7))
    stripe = edges[:split]
    min_len = max(8, stripe.shape[1] // 20)
    lines = cv2.HoughLinesP(
        stripe, 1, np.pi / 180, threshold=12,
        minLineLength=min_len, maxLineGap=6,
    )
    if lines is None:
        return float("inf"), stripe_n, below_n

    total_w = 0.0
    weighted_dev = 0.0
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        weighted_dev += abs(abs(angle) - 90.0) * length
        total_w += length

    if total_w < 1.0:
        return float("inf"), stripe_n, below_n
    return weighted_dev / total_w, stripe_n, below_n


def _edge_deskew_angle(image: np.ndarray) -> float:
    """Rotate so stripe edges are exactly vertical and outnumber edges below the bars."""
    gray = _to_gray(image)
    coarse = 0.0
    coarse_dev = float("inf")

    for angle in range(-45, 46, 2):
        rotated = _rotate_image(gray, float(angle), expand=True)
        dev, _, _ = _stripe_vertical_deviation(rotated)
        if dev < coarse_dev:
            coarse_dev = dev
            coarse = float(angle)

    if coarse_dev == float("inf"):
        best_key = (-1, -1)
        for angle in range(-45, 46, 2):
            rotated = _rotate_image(gray, float(angle), expand=True)
            stripe_n, below_n = _bar_edge_counts(rotated)
            key = (stripe_n - below_n, stripe_n)
            if stripe_n > below_n and key > best_key:
                best_key = key
                coarse = float(angle)

    best_angle = coarse
    best_dev = coarse_dev
    for step in range(-30, 31):
        angle = coarse + step / 10.0
        rotated = _rotate_image(gray, angle, expand=True)
        dev, _, _ = _stripe_vertical_deviation(rotated)
        if dev < best_dev:
            best_dev = dev
            best_angle = angle

    return best_angle


def find_barcode_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = _barcode_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def straighten_barcode(
    cropped_image: np.ndarray,
    contour: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    angle = _edge_deskew_angle(cropped_image)
    return _rotate_image(cropped_image, angle, expand=True), angle, contour


def _preprocess_variants(image: np.ndarray) -> list[np.ndarray]:
    """Build image variants to improve pyzbar decode on hard crops."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    variants = [gray]

    # Contrast boost binarisation and upscaling for small or blurry barcodes.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    for scale in (1.5, 2.0):
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)

    return variants


def decode_barcode(image: np.ndarray) -> list:
    # Try each preprocess variant until pyzbar succeeds.
    for variant in _preprocess_variants(image):
        results = decode(variant)
        if results:
            return results
    return []


def _make_upright(rotated: np.ndarray, item, deskew_angle: float, cardinal: float, raw_image: np.ndarray) -> DecodeResult:
    """Build result with image rotated to visual upright using pyzbar orientation."""
    orientation = getattr(item, "orientation", "UP")
    correction = _ROTATION_TO_UP.get(orientation, 0.0)
    upright = _rotate_image(rotated, correction) if correction else rotated

    return DecodeResult(
        text=item.data.decode("utf-8"),
        barcode_type=item.type,
        rotation_angle=deskew_angle + cardinal + correction,
        deskew_angle=deskew_angle,
        oriented_image=upright,
        raw_image=raw_image,
    )


def _decode_upright(image: np.ndarray, deskew_angle: float, raw_image: np.ndarray) -> DecodeResult | None:
    """Try cardinals on deskewed image; rotate each hit to upright via pyzbar orientation."""
    best: DecodeResult | None = None
    best_score = float("inf")

    # Test 0 90 180 270 degrees with pyzbar orientation correction.
    for cardinal in (0, 90, 180, 270):
        rotated = image if cardinal == 0 else _rotate_image(image, cardinal)
        decoded = decode_barcode(rotated)
        if not decoded:
            continue

        result = _make_upright(rotated, decoded[0], deskew_angle, cardinal, raw_image)
        orientation = getattr(decoded[0], "orientation", "UP")
        score = abs(_ROTATION_TO_UP.get(orientation, 0.0)) + abs(cardinal) * 0.1  # least rotation wins

        if score < best_score:
            best_score = score
            best = result

    return best


def _fallback_angle_search(image: np.ndarray) -> DecodeResult | None:
    """Brute force sweep from -90 to 90 degrees when contour deskew fails."""
    best: DecodeResult | None = None
    best_score = float("inf")

    for angle in range(-90, 91, 5):
        rotated = _rotate_image(image, angle)
        for flip in (0, 180):
            candidate = rotated if flip == 0 else _rotate_image(rotated, 180)
            decoded = decode_barcode(candidate)
            if not decoded:
                continue

            result = _make_upright(candidate, decoded[0], angle, flip, image)
            orientation = getattr(decoded[0], "orientation", "UP")
            score = abs(angle) + abs(flip) + abs(_ROTATION_TO_UP.get(orientation, 0.0))

            if score < best_score:
                best_score = score
                best = result

    return best


def orient_and_decode(image: np.ndarray) -> DecodeResult | None:
    """Deskew by edge alignment, decode, and return a visually upright barcode image."""
    deskew_angle = _edge_deskew_angle(image)
    straightened = _rotate_image(image, deskew_angle, expand=True)
    result = _decode_upright(straightened, deskew_angle, image)
    if result is not None:
        return result
    return _fallback_angle_search(image)
