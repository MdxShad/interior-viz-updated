import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


OBJECT_ALIASES = {
    "sofa": "couch",
    "couch": "couch",
    "loveseat": "couch",
    "settee": "couch",
    "chair": "chair",
    "chairs": "chair",
    "table": "dining table",
    "desk": "dining table",
    "lamp": "lamp",
    "light": "lamp",
    "tv": "tv",
    "television": "tv",
    "bed": "bed",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_label(label: str) -> str:
    normalized = label.lower().strip()
    return OBJECT_ALIASES.get(normalized, normalized)


def parse_replace_prompt(prompt: str) -> Dict[str, Optional[str]]:
    tokens = tokenize(prompt)
    target_label = None
    for token in tokens:
        if token in OBJECT_ALIASES:
            target_label = OBJECT_ALIASES[token]
            break

    position = None
    if "left" in tokens:
        position = "left"
    elif "right" in tokens:
        position = "right"
    elif "center" in tokens or "middle" in tokens:
        position = "center"

    return {
        "target_label": target_label,
        "position": position,
        "mentions_uploaded": "upload" in tokens or "uploaded" in tokens,
    }


def choose_target_detection(
    detections: Sequence[dict],
    target_label: str,
    position: Optional[str] = None,
    click: Optional[Sequence[int]] = None,
) -> Tuple[Optional[dict], List[dict]]:
    matches = [det for det in detections if normalize_label(str(det.get("label", ""))) == target_label]
    if not matches:
        return None, []

    if click is not None:
        x, y = int(click[0]), int(click[1])
        containing = [det for det in matches if point_in_box(x, y, det["bbox"])]
        if containing:
            containing.sort(key=lambda det: box_area(det["bbox"]))
            return containing[0], []
        nearest = min(matches, key=lambda det: distance_to_box_center(x, y, det["bbox"]))
        return nearest, []

    if len(matches) == 1:
        return matches[0], []

    if position == "left":
        return min(matches, key=lambda det: box_center(det["bbox"])[0]), []
    if position == "right":
        return max(matches, key=lambda det: box_center(det["bbox"])[0]), []
    if position == "center":
        center_x = sum(box_center(det["bbox"])[0] for det in matches) / float(len(matches))
        return min(matches, key=lambda det: abs(box_center(det["bbox"])[0] - center_x)), []

    return None, matches


def choose_replacement_asset(
    prompt: str,
    session_assets: Sequence[Path],
    catalog_assets: Sequence[Path],
    explicit_name: Optional[str] = None,
    explicit_source: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if explicit_name:
        if explicit_source == "catalog":
            return "catalog", explicit_name
        if explicit_source == "session":
            return "session", explicit_name
        if any(path.name == explicit_name for path in session_assets):
            return "session", explicit_name
        if any(path.name == explicit_name for path in catalog_assets):
            return "catalog", explicit_name

    tokens = tokenize(prompt)
    best_session = best_asset_match(tokens, session_assets)
    best_catalog = best_asset_match(tokens, catalog_assets)

    if best_session is not None:
        return "session", best_session.name
    if best_catalog is not None:
        return "catalog", best_catalog.name

    if len(session_assets) == 1:
        return "session", session_assets[0].name
    if "upload" in tokens or "uploaded" in tokens:
        if session_assets:
            newest = max(session_assets, key=lambda path: path.stat().st_mtime)
            return "session", newest.name

    return None, None


def best_asset_match(tokens: Sequence[str], assets: Sequence[Path]) -> Optional[Path]:
    if not assets:
        return None

    scored: List[Tuple[int, float, Path]] = []
    token_set = set(tokens)
    for asset in assets:
        stem_tokens = set(tokenize(asset.stem.replace("_", " ")))
        overlap = len(token_set & stem_tokens)
        if overlap == 0:
            continue
        scored.append((overlap, asset.stat().st_mtime, asset))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def point_in_box(x: int, y: int, bbox: Sequence[float]) -> bool:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(1.0, x2 - x1) * max(1.0, y2 - y1)


def distance_to_box_center(x: int, y: int, bbox: Sequence[float]) -> float:
    center_x, center_y = box_center(bbox)
    return ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
