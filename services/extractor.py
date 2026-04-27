import io
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from rembg import remove as rembg_remove
except Exception as exc:  # pragma: no cover - optional dependency path
    rembg_remove = None
    _REMBG_IMPORT_ERROR = str(exc)
else:
    _REMBG_IMPORT_ERROR = None


_extractor = None
_status: Dict[str, Optional[str]] = {
    "loaded": True,
    "error": _REMBG_IMPORT_ERROR,
    "backend": "rembg" if rembg_remove is not None else "opencv_grabcut",
}


class ReplacementExtractor:
    def extract(self, image: Image.Image) -> Image.Image:
        rgba = None
        if rembg_remove is not None:
            try:
                rgba = self._run_rembg(image)
            except Exception:
                rgba = None

        if rgba is None:
            rgba = self._run_grabcut(image)

        return self._postprocess(rgba)

    def _run_rembg(self, image: Image.Image) -> Image.Image:
        result = rembg_remove(image.convert("RGBA"))
        if isinstance(result, Image.Image):
            return result.convert("RGBA")
        return Image.open(io.BytesIO(result)).convert("RGBA")

    def _run_grabcut(self, image: Image.Image) -> Image.Image:
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]

        top = rgb[0, :, :]
        bottom = rgb[-1, :, :]
        left = rgb[:, 0, :]
        right = rgb[:, -1, :]
        border = np.concatenate([top, bottom, left, right], axis=0)
        background_color = np.median(border, axis=0)

        distance = np.linalg.norm(rgb.astype(np.float32) - background_color.astype(np.float32), axis=2)
        threshold = max(24.0, float(np.percentile(distance, 65)))

        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        mask[distance > threshold] = cv2.GC_PR_FGD

        margin_x = max(4, w // 20)
        margin_y = max(4, h // 20)
        rect = (margin_x, margin_y, max(1, w - margin_x * 2), max(1, h - margin_y * 2))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        alpha = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)

        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, mode="RGBA")

    def _postprocess(self, image: Image.Image) -> Image.Image:
        rgba = np.array(image.convert("RGBA"))
        alpha = rgba[:, :, 3]
        alpha[alpha < 8] = 0
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
        rgba[:, :, 3] = alpha
        return crop_to_alpha(Image.fromarray(rgba, mode="RGBA"))


def crop_to_alpha(image: Image.Image, padding: int = 6) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    points = np.argwhere(alpha > 8)
    if points.size == 0:
        return rgba

    y1, x1 = points.min(axis=0)
    y2, x2 = points.max(axis=0)
    left = max(0, int(x1) - padding)
    top = max(0, int(y1) - padding)
    right = min(rgba.width, int(x2) + padding + 1)
    bottom = min(rgba.height, int(y2) + padding + 1)
    return rgba.crop((left, top, right, bottom))


def get_extractor() -> ReplacementExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ReplacementExtractor()
    return _extractor


def get_status() -> Dict[str, Optional[str]]:
    return dict(_status)
