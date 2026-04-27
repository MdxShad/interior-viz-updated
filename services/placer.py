from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from services.depth import DepthEstimator


def place_object(
    room_image: Image.Image,
    object_name: str,
    x: int,
    y: int,
    catalog_dir: Path,
    depth_estimator: DepthEstimator,
    scale: Optional[float] = None,
    rotation: float = 0.0,
) -> Image.Image:
    """
    Composite an object from the catalog onto the provided room image at the given
    (x, y) position.  Scaling is determined by depth unless a ``scale``
    parameter is provided.  If ``scale`` is provided, it is treated as a
    multiplier on the automatically computed size (1.0 means default size,
    values >1 enlarge, values <1 shrink).  Rotation in degrees (clockwise)
    can be specified to better align objects with scene perspective.
    """
    room_rgb = room_image.convert("RGB")
    room_rgba = room_rgb.convert("RGBA")
    room_np = np.array(room_rgb)

    object_path = catalog_dir / object_name
    if not object_path.exists():
        raise FileNotFoundError(f"Catalog object '{object_name}' not found.")

    obj = Image.open(object_path).convert("RGBA")
    obj_w, obj_h = obj.size
    if obj_w == 0 or obj_h == 0:
        raise ValueError(f"Catalog object '{object_name}' is invalid.")

    # Determine desired long edge size based on depth
    depth_value = depth_estimator.depth_at_point(room_np, x, y)
    base_scale = min(room_rgba.size) * 0.25
    # Default scale factor derived from depth; clamp minimal size
    target_long_edge = max(24, float(base_scale * (1 - depth_value * 0.5)))
    # Apply user‑provided relative scale if supplied
    if scale is not None:
        try:
            rel = float(scale)
            # Only accept positive scales to avoid inversion
            if rel > 0:
                target_long_edge *= rel
        except Exception:
            pass

    # Compute resizing ratio preserving aspect ratio
    ratio = target_long_edge / float(max(obj_w, obj_h))
    new_size = (max(1, int(obj_w * ratio)), max(1, int(obj_h * ratio)))
    obj_resized = obj.resize(new_size, Image.LANCZOS)

    # Optionally rotate the object; expand canvas to fit rotation
    rot_deg = float(rotation) if rotation is not None else 0.0
    if abs(rot_deg) > 0.01:
        obj_resized = obj_resized.rotate(-rot_deg, expand=True, resample=Image.BICUBIC)

    # Blur alpha for smoother edges
    r, g, b, a = obj_resized.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=3))
    obj_processed = Image.merge("RGBA", (r, g, b, a))

    # Compute paste coordinates to center object at (x, y)
    obj_w2, obj_h2 = obj_processed.size
    paste_x = int(x - obj_w2 / 2)
    paste_y = int(y - obj_h2 / 2)
    paste_x = max(-obj_w2 + 1, min(paste_x, room_rgba.width - 1))
    paste_y = max(-obj_h2 + 1, min(paste_y, room_rgba.height - 1))

    # Composite onto a transparent layer then blend
    layer = Image.new("RGBA", room_rgba.size, (0, 0, 0, 0))
    layer.paste(obj_processed, (paste_x, paste_y), obj_processed)
    result = Image.alpha_composite(room_rgba, layer).convert("RGB")
    return result
