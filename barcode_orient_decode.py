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


# ZBar orientation → rotation to make barcode upright (bars vertical, text at bottom).
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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 1, 0))
    grad_y = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 0, 1))
    gradient = cv2.max(cv2.subtract(grad_x, grad_y), cv2.subtract(grad_y, grad_x))

    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    masks = []
    for kw, kh in ((25, 5), (5, 25)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        masks.append(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel))

    merged = cv2.bitwise_or(masks[0], masks[1])
    return cv2.morphologyEx(
        merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def find_barcode_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = _barcode_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _correct_min_area_angle(angle: float) -> float:
    if angle < -45:
        return -(90 + angle)
    return -angle


def _deskew_candidates(contour: np.ndarray) -> list[float]:
    """Deskew angle candidates (handles minAreaRect 90° ambiguity)."""
    rect = cv2.minAreaRect(contour)
    angle = _correct_min_area_angle(rect[-1])
    w, h = rect[1]
    if max(w, h) < 1:
        return [angle]

    candidates = [angle]
    if min(w, h) / max(w, h) > 0.55:
        candidates.extend([angle + 90.0, angle - 90.0])
    return candidates


def straighten_barcode(
    cropped_image: np.ndarray,
    contour: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    if contour is None:
        contour = find_barcode_contour(cropped_image)
    if contour is None:
        return cropped_image.copy(), 0.0, None

    angle = _deskew_candidates(contour)[0]
    return _rotate_image(cropped_image, angle, expand=True), angle, contour


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

    for cardinal in (0, 90, 180, 270):
        rotated = image if cardinal == 0 else _rotate_image(image, cardinal)
        decoded = decode_barcode(rotated)
        if not decoded:
            continue

        result = _make_upright(rotated, decoded[0], deskew_angle, cardinal, raw_image)
        orientation = getattr(decoded[0], "orientation", "UP")
        score = abs(_ROTATION_TO_UP.get(orientation, 0.0)) + abs(cardinal) * 0.1

        if score < best_score:
            best_score = score
            best = result

    return best


def _fallback_angle_search(image: np.ndarray) -> DecodeResult | None:
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
    """Deskew, decode, and return a visually upright barcode image."""
    contour = find_barcode_contour(image)
    if contour is None:
        return _fallback_angle_search(image)

    best: DecodeResult | None = None
    best_score = float("inf")

    for deskew_angle in _deskew_candidates(contour):
        straightened = _rotate_image(image, deskew_angle, expand=True)
        result = _decode_upright(straightened, deskew_angle, image)
        if result is None:
            continue

        if abs(result.rotation_angle) < best_score:
            best_score = abs(result.rotation_angle)
            result.contour = contour
            best = result

    if best is not None:
        return best

    fallback = _fallback_angle_search(image)
    if fallback is not None:
        fallback.contour = contour
    return fallback
