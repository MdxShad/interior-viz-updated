from typing import Dict, Optional

import cv2
import numpy as np
import torch


_estimator = None
_status: Dict[str, Optional[str]] = {"loaded": False, "error": None, "model": "MiDaS_small"}


class DummyDepthEstimator:
    """
    Fallback depth estimator used when the MiDaS model cannot be loaded.  This
    implementation approximates depth using the y-coordinate of a pixel.  The
    assumption is that objects closer to the bottom of the image are nearer to
    the viewer.  This provides a simple and fast heuristic for scaling objects
    in a scene when true depth data is unavailable.
    """

    def depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        # Generate a gradient depth map from top (far) to bottom (near)
        h, w = image_rgb.shape[:2]
        # Normalize y coordinates to [0,1], invert so bottom is 0 and top is 1
        y_coords = np.linspace(1.0, 0.0, h, dtype=np.float32)
        depth = np.tile(y_coords[:, None], (1, w))
        return depth

    def depth_at_point(self, image_rgb: np.ndarray, x: int, y: int) -> float:
        h, w = image_rgb.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        return float(1.0 - (y / max(1, h - 1)))


class DepthEstimator:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

    def depth_map(self, image_rgb: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        input_batch = self.transform(image_bgr).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        min_val = float(depth.min())
        max_val = float(depth.max())
        if max_val - min_val < 1e-8:
            return np.zeros_like(depth, dtype=np.float32)
        return ((depth - min_val) / (max_val - min_val)).astype(np.float32)

    def depth_at_point(self, image_rgb: np.ndarray, x: int, y: int) -> float:
        dmap = self.depth_map(image_rgb)
        h, w = dmap.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        return float(dmap[y, x])


def get_depth_estimator() -> object:
    """
    Return an initialized depth estimator.  If the MiDaS model is available
    and loads correctly, a ``DepthEstimator`` instance will be used.  Otherwise
    this function falls back to a ``DummyDepthEstimator`` which approximates
    depth using a simple vertical gradient.  The returned object always
    implements ``depth_map`` and ``depth_at_point``.
    """
    global _estimator
    if _estimator is not None:
        return _estimator

    try:
        _estimator = DepthEstimator()
        _status["loaded"] = True
        _status["error"] = None
    except Exception as exc:  # pragma: no cover - runtime dependency error path
        # Failed to load MiDaS; use fallback estimator
        _status["loaded"] = False
        _status["error"] = str(exc)
        _estimator = DummyDepthEstimator()

    return _estimator


def get_status() -> Dict[str, Optional[str]]:
    return dict(_status)
