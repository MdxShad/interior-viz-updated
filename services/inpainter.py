from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from simple_lama_inpainting import SimpleLama  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    SimpleLama = None  # type: ignore
    _LAMA_IMPORT_ERROR = str(exc)
else:
    _LAMA_IMPORT_ERROR = None


_inpainter = None
_status: Dict[str, Optional[str]] = {"loaded": False, "error": None, "model": "simple-lama-inpainting"}


class DummyInpainter:
    """
    Fallback inpainter used when the LaMa model cannot be loaded.  This
    implementation uses OpenCV's Telea algorithm to fill the masked region.
    It first dilates the binary mask to reduce edge artifacts, then
    performs inpainting on a NumPy array and returns the result as a PIL
    image.  While the quality is generally lower than LaMa, it avoids
    dependency on the large PyTorch-based model and works on CPU only.
    """

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        # Dilate mask to include edge pixels
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(image_bgr, dilated_mask, 3, cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_rgb)


class LaMaInpainter:
    def __init__(self) -> None:
        if SimpleLama is None:
            raise ImportError(
                _LAMA_IMPORT_ERROR or "simple-lama-inpainting library is not installed"
            )
        self.model = SimpleLama()

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # Dilate to expand removal region and reduce object edge remnants.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

        image_rgb = image.convert("RGB")
        mask_img = Image.fromarray(dilated_mask, mode="L")
        return self.model(image_rgb, mask_img).convert("RGB")


def get_inpainter() -> object:
    """
    Return an initialized inpainter.  If the Simple LaMa model loads
    successfully, this will return an instance of ``LaMaInpainter``.  If
    loading fails (for example, due to missing Torch or model files), a
    ``DummyInpainter`` will be used instead.  The ``_status`` dictionary
    reflects whether LaMa was loaded.
    """
    global _inpainter
    if _inpainter is not None:
        return _inpainter

    try:
        _inpainter = LaMaInpainter()
        _status["loaded"] = True
        _status["error"] = None
    except Exception as exc:  # pragma: no cover - runtime dependency error path
        _status["loaded"] = False
        _status["error"] = str(exc)
        # Fallback to dummy implementation
        _inpainter = DummyInpainter()

    return _inpainter


def get_status() -> Dict[str, Optional[str]]:
    return dict(_status)
