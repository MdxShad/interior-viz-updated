import base64
import io
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# The Segment Anything library may not be installed in all environments.
# Attempt to import it, but fall back gracefully if unavailable.  When the
# import fails, ``SamPredictor`` will be ``None`` and ``sam_model_registry``
# will be an empty dict.  This allows the rest of the module to load and
# DummySegmentor to be used as a fallback.
try:
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    SamPredictor = None  # type: ignore
    sam_model_registry = {}  # type: ignore
    _SAM_IMPORT_ERROR = str(exc)
else:
    _SAM_IMPORT_ERROR = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAM_CHECKPOINT = PROJECT_DIR / "models" / "sam_vit_b_01ec64.pth"

_segmentor = None
_status: Dict[str, Optional[str]] = {
    "loaded": False,
    "error": None,
    "checkpoint": str(DEFAULT_SAM_CHECKPOINT),
}


class DummySegmentor:
    """
    Fallback segmentation implementation used when the Segment Anything
    checkpoint is unavailable.  This class simply returns a circular mask
    around the clicked point.  It is intended to allow the application to
    continue functioning in a degraded manner when the SAM model cannot be
    loaded.  The mask radius is chosen relative to the shorter dimension of
    the input image so that the segmented region scales with the image size.
    """

    def __init__(self, radius_fraction: float = 0.1) -> None:
        # radius as a fraction of the smaller image dimension
        self.radius_fraction = radius_fraction

    def segment_with_point(self, image: Image.Image, x: int, y: int) -> Tuple[np.ndarray, str, str]:
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        # Clamp click coordinates to image bounds
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        # Determine radius relative to image size
        radius = int(max(1, self.radius_fraction * min(w, h)))
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        mask = (dist2 <= radius ** 2).astype(np.uint8)
        return mask, mask_to_base64(mask), mask_overlay_to_base64(mask)

    def segment_with_box(self, image: Image.Image, box: Sequence[float]) -> Tuple[np.ndarray, str, str]:
        """Fallback box segmentation simply fills the bounding box with ones."""
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        x1, y1, x2, y2 = clip_box(box, w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(y1) : int(y2), int(x1) : int(x2)] = 1
        return mask, mask_to_base64(mask), mask_overlay_to_base64(mask)


class SAMSegmentor:
    def __init__(self, checkpoint_path: Path = DEFAULT_SAM_CHECKPOINT, model_type: str = "vit_b") -> None:
        # Ensure the underlying SAM library is available
        if SamPredictor is None or not sam_model_registry:
            # If the import failed earlier, propagate a descriptive error
            raise ImportError(
                _SAM_IMPORT_ERROR or "segment_anything library is not installed"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at '{checkpoint_path}'. "
                "Download sam_vit_b_01ec64.pth and place it in interior-viz/models/."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)

        self.device = device
        self.predictor = SamPredictor(sam)

    def segment_with_point(self, image: Image.Image, x: int, y: int) -> Tuple[np.ndarray, str, str]:
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        input_point = np.array([[x, y]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)
        mask = self._predict_mask(image_np, point_coords=input_point, point_labels=input_label)
        return mask, mask_to_base64(mask), mask_overlay_to_base64(mask)

    def segment_with_box(
        self, image: Image.Image, box: Sequence[float]
    ) -> Tuple[np.ndarray, str, str]:
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        x1, y1, x2, y2 = clip_box(box, w, h)
        input_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        center_point = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
        point_label = np.array([1], dtype=np.int32)

        try:
            mask = self._predict_mask(
                image_np,
                point_coords=center_point,
                point_labels=point_label,
                box=input_box,
            )
        except Exception:
            mask = self._predict_mask(image_np, box=input_box)

        return mask, mask_to_base64(mask), mask_overlay_to_base64(mask)

    def _predict_mask(
        self,
        image_np: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.predictor.set_image(image_np)
        kwargs = {"multimask_output": True}
        if point_coords is not None and point_labels is not None:
            kwargs["point_coords"] = point_coords
            kwargs["point_labels"] = point_labels
        if box is not None:
            kwargs["box"] = box

        masks, scores, _ = self.predictor.predict(**kwargs)
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.uint8)


def clip_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = int(np.clip(min(x1, x2), 0, max(0, width - 1)))
    y1 = int(np.clip(min(y1, y2), 0, max(0, height - 1)))
    x2 = int(np.clip(max(x1 + 1, x2), 1, width))
    y2 = int(np.clip(max(y1 + 1, y2), 1, height))
    return x1, y1, x2, y2


def mask_to_base64(mask: np.ndarray) -> str:
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def mask_overlay_to_base64(mask: np.ndarray) -> str:
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask > 0] = [255, 0, 0, 128]
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_segmentor() -> object:
    """
    Return a segmentor instance.  If the SAM model is available and loaded
    successfully, this will be an instance of ``SAMSegmentor``.  Otherwise,
    a ``DummySegmentor`` will be returned to allow the application to
    continue operating without raising an exception.  In either case the
    ``_status`` dictionary will reflect whether the SAM model was loaded.
    """
    global _segmentor
    if _segmentor is not None:
        return _segmentor

    try:
        _segmentor = SAMSegmentor()
        _status["loaded"] = True
        _status["error"] = None
    except Exception as exc:  # pragma: no cover - runtime dependency error path
        # Failed to load SAM; fall back to dummy implementation
        _status["loaded"] = False
        _status["error"] = str(exc)
        _segmentor = DummySegmentor()

    return _segmentor


def get_status() -> Dict[str, Optional[str]]:
    return dict(_status)
