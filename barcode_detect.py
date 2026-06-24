"""Phase 1: Barcode detection and cropping using gradient morphology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BarcodeRegion:
    bbox: tuple[int, int, int, int]
    min_area_rect: tuple
    box_points: np.ndarray
    angle: float
    score: float

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


def _adaptive_threshold(gradient: np.ndarray) -> np.ndarray:
    """Use Otsu when sparse; otherwise fall back to a high percentile."""
    _, otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.mean(otsu == 255) <= 0.05:
        return otsu

    threshold = max(float(np.percentile(gradient, 94)), 1.0)
    _, thresh = cv2.threshold(gradient, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def _build_barcode_mask(blurred: np.ndarray) -> np.ndarray:
    """Highlight barcode-like parallel line patterns in any orientation."""
    grad_x = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 1, 0))
    grad_y = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 0, 1))

    vertical_bars = cv2.subtract(grad_x, grad_y)
    horizontal_bars = cv2.subtract(grad_y, grad_x)
    combined = cv2.max(vertical_bars, horizontal_bars)

    thresh = _adaptive_threshold(combined)

    # Bridge gaps between parallel barcode lines for each orientation.
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21)),
    ]
    masks = []
    for kernel in kernels:
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        masks.append(closed)

    merged = cv2.bitwise_or(masks[0], masks[1])
    return cv2.morphologyEx(
        merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )


def _nms(regions: list[BarcodeRegion], overlap_thresh: float = 0.4) -> list[BarcodeRegion]:
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: r.score, reverse=True)
    kept: list[BarcodeRegion] = []

    for region in regions:
        x, y, w, h = region.bbox
        duplicate = False
        for kept_region in kept:
            kx, ky, kw, kh = kept_region.bbox
            ix1, iy1 = max(x, kx), max(y, ky)
            ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = w * h + kw * kh - inter
            if union > 0 and inter / union > overlap_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)

    return kept


def detect_barcode_regions(
    image: np.ndarray,
    *,
    blur_ksize: int = 5,
    min_area_ratio: float = 0.0003,
    max_area_ratio: float = 0.01,
    min_short_side: float = 60,
    min_aspect: float = 1.4,
    max_aspect: float = 6.0,
    min_extent: float = 0.30,
    padding: int = 15,
) -> tuple[list[BarcodeRegion], dict[str, np.ndarray]]:
    """
    Detect barcode regions in a BGR image.

    Returns cropped-region metadata plus intermediate debug images.
    """
    if image.ndim != 3:
        raise ValueError("Expected a BGR color image.")

    height, width = image.shape[:2]
    image_area = height * width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    grad_x = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 1, 0))
    grad_y = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 0, 1))
    gradient = cv2.max(cv2.subtract(grad_x, grad_y), cv2.subtract(grad_y, grad_x))

    mask = _build_barcode_mask(blurred)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[BarcodeRegion] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        rect = cv2.minAreaRect(contour)
        _, (rect_w, rect_h), angle = rect
        short_side = min(rect_w, rect_h)
        long_side = max(rect_w, rect_h)
        if short_side < min_short_side:
            continue

        aspect = long_side / short_side
        if aspect < min_aspect or aspect > max_aspect:
            continue

        rect_area = rect_w * rect_h
        extent = area / rect_area if rect_area else 0.0
        if extent < min_extent:
            continue

        # Require strong barcode-like gradients inside the contour.
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        mean_gradient = cv2.mean(gradient, mask=contour_mask)[0]
        if mean_gradient < 18:
            continue

        box_points = cv2.boxPoints(rect).astype(np.int32)
        x, y, box_w, box_h = cv2.boundingRect(box_points)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + box_w + padding)
        y2 = min(height, y + box_h + padding)

        regions.append(
            BarcodeRegion(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                min_area_rect=rect,
                box_points=box_points,
                angle=angle,
                score=area * extent,
            )
        )

    debug = {
        "gray": gray,
        "gradient": gradient,
        "mask": mask,
    }
    return _nms(regions), debug


def _order_box_points(box: np.ndarray) -> np.ndarray:
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
    box = box.astype(np.float32)
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(s)]
    ordered[2] = box[np.argmax(s)]
    ordered[1] = box[np.argmin(diff)]
    ordered[3] = box[np.argmax(diff)]
    return ordered


def crop_barcode_rotated(
    image: np.ndarray,
    region: BarcodeRegion,
    pad_ratio: float = 0.10,
) -> np.ndarray:
    """Tight crop via minAreaRect perspective warp (better for tilted barcodes)."""
    center, (rect_w, rect_h), angle = region.min_area_rect
    rect_w = max(rect_w, 1.0) * (1.0 + pad_ratio)
    rect_h = max(rect_h, 1.0) * (1.0 + pad_ratio)

    box = cv2.boxPoints(((center[0], center[1]), (rect_w, rect_h), angle))
    src = _order_box_points(box)
    dst_w, dst_h = int(rect_w), int(rect_h)
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_barcode(image: np.ndarray, region: BarcodeRegion) -> np.ndarray:
    """Crop a barcode ROI — prefers rotated perspective crop, falls back to bbox."""
    try:
        return crop_barcode_rotated(image, region)
    except cv2.error:
        x, y, w, h = region.bbox
        return image[y : y + h, x : x + w].copy()


def draw_detections(
    image: np.ndarray,
    regions: list[BarcodeRegion],
    labels: list[str] | None = None,
) -> np.ndarray:
    output = image.copy()
    for index, region in enumerate(regions):
        x, y, w, h = region.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.drawContours(output, [region.box_points], 0, (255, 0, 0), 2)
        if labels and index < len(labels):
            cv2.putText(
                output,
                labels[index][:24],
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
    return output


def process_image(
    image: np.ndarray,
    source_name: str,
    output_dir: Path,
    *,
    decode: bool = True,
    **detect_kwargs,
) -> dict:
    """Detect, crop (rotated), optionally decode a single image."""
    detect_kwargs.pop("decode", None)
    regions, debug = detect_barcode_regions(image, **detect_kwargs)
    barcodes = []
    stem = Path(source_name).stem

    for index, region in enumerate(regions):
        crop = crop_barcode(image, region)
        crop_path = output_dir / f"{stem}_barcode_{index:02d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        entry = {
            "index": index,
            "crop_path": str(crop_path),
            "crop": crop,
            "region": region,
            "decoded": None,
            "text": None,
            "type": None,
        }

        if decode:
            from barcode_orient_decode import DecodeResult, decode_barcode, orient_and_decode

            decoded_list = decode_barcode(crop)
            if decoded_list:
                item = decoded_list[0]
                result = DecodeResult(
                    text=item.data.decode("utf-8"),
                    barcode_type=item.type,
                    rotation_angle=0.0,
                    deskew_angle=0.0,
                    oriented_image=crop,
                    raw_image=crop,
                )
            else:
                result = orient_and_decode(crop)

            if result:
                entry["decoded"] = result
                entry["text"] = result.text
                entry["type"] = result.barcode_type
                oriented_path = output_dir / f"{stem}_barcode_{index:02d}_oriented.jpg"
                cv2.imwrite(str(oriented_path), result.oriented_image)
                entry["oriented_path"] = str(oriented_path)

        barcodes.append(entry)

    labels = [(b["text"] if b["text"] else "?") for b in barcodes]
    return {
        "source": source_name,
        "count": len(regions),
        "decoded_count": sum(1 for b in barcodes if b["text"]),
        "barcodes": barcodes,
        "regions": regions,
        "debug": debug,
        "annotated": draw_detections(image, regions, labels),
    }


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    debug_dir: str | Path | None = None,
    *,
    decode: bool = True,
    **detect_kwargs,
) -> list[dict]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detect_kwargs.pop("decode", None)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    results = []
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            results.append({"source": image_path.name, "error": "failed to read image"})
            continue

        item = process_image(
            image, image_path.name, output_dir, decode=decode, **detect_kwargs
        )

        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{image_path.stem}_mask.jpg"), item["debug"]["mask"])
            cv2.imwrite(
                str(debug_dir / f"{image_path.stem}_detect.jpg"),
                item["annotated"],
            )

        results.append(item)

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    summary = process_folder(
        project_root / "tagss",
        project_root / "cropped",
        project_root / "debug",
    )

    print(f"{'Source':<45} {'Found':>5}  {'Decoded':>7}")
    print("-" * 62)
    total_found = total_decoded = 0
    for item in summary:
        if "error" in item:
            print(f"{item['source']:<45} ERROR")
        else:
            total_found += item["count"]
            total_decoded += item["decoded_count"]
            print(f"{item['source']:<45} {item['count']:>5}  {item['decoded_count']:>7}")
    print("-" * 62)
    print(f"{'TOTAL':<45} {total_found:>5}  {total_decoded:>7}")
