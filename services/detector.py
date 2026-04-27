from pathlib import Path
from typing import Dict, List, Optional

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency path
    YOLO = None  # type: ignore
    _YOLO_IMPORT_ERROR = str(exc)
else:
    _YOLO_IMPORT_ERROR = None


_detector = None
_status: Dict[str, Optional[str]] = {"loaded": False, "error": None, "model": "yolov8n.pt"}


class DummyDetector:
    """
    Fallback detector used when the Ultralytics YOLO implementation is not available.
    This detector returns an empty list of detections for any input image.  It
    allows the application to run without throwing an ImportError when the
    ``ultralytics`` package is missing or fails to load.
    """

    def detect(self, image_path: Path) -> list:
        # No detections without a detection model
        return []


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        if YOLO is None:
            raise ImportError(
                _YOLO_IMPORT_ERROR or "ultralytics library is not installed"
            )
        self.model_name = model_name
        self.model = YOLO(model_name)

    def detect(self, image_path: Path) -> List[dict]:
        results = self.model(str(image_path), verbose=False)
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: List[dict] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            detections.append(
                {
                    "label": str(names.get(cls_id, cls_id)),
                    "confidence": float(box.conf[0].item()),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections


def get_detector() -> object:
    global _detector
    if _detector is not None:
        return _detector

    try:
        _detector = YOLODetector()
        _status["loaded"] = True
        _status["error"] = None
    except Exception as exc:  # pragma: no cover - runtime dependency error path
        # Fallback to dummy detector
        _status["loaded"] = False
        _status["error"] = str(exc)
        _detector = DummyDetector()

    return _detector


def get_status() -> Dict[str, Optional[str]]:
    return dict(_status)
