import base64
import io
import uuid
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from typing import Optional

from services import depth, detector, inpainter, placer, segmentor


PROJECT_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = PROJECT_DIR / "uploads"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
STATIC_DIR = PROJECT_DIR / "static"
CATALOG_OBJECTS_DIR = PROJECT_DIR / "catalog" / "objects"

for folder in (UPLOADS_DIR, OUTPUTS_DIR, STATIC_DIR, CATALOG_OBJECTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="AI Interior Visualization Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount(
    "/catalog/objects",
    StaticFiles(directory=str(CATALOG_OBJECTS_DIR)),
    name="catalog_objects",
)


class SegmentRequest(BaseModel):
    image_id: str
    x: int
    y: int


class RemoveRequest(BaseModel):
    image_id: str
    mask_base64: str


class PlaceRequest(BaseModel):
    image_id: str
    catalog_object_name: str
    x: int
    y: int
    # Optional manual scale factor. 1.0 means default size; >1 enlarges; <1 shrinks.
    scale: Optional[float] = None
    # Optional rotation in degrees (clockwise positive).
    rotation: Optional[float] = 0.0


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_mask_base64(mask_base64: str) -> np.ndarray:
    payload = mask_base64.split(",", 1)[-1]
    raw = base64.b64decode(payload)
    mask = Image.open(io.BytesIO(raw)).convert("L")
    return (np.array(mask) > 127).astype(np.uint8)


def upload_image_path(image_id: str) -> Path:
    return UPLOADS_DIR / f"{image_id}.png"


def current_image_path(image_id: str) -> Path:
    return OUTPUTS_DIR / f"{image_id}_current.png"


def load_current_image(image_id: str) -> Image.Image:
    path = current_image_path(image_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' was not found.")
    return Image.open(path).convert("RGB")


@app.on_event("startup")
async def startup_event() -> None:
    # Warm-load all models once for better request latency.
    for loader in (
        detector.get_detector,
        segmentor.get_segmentor,
        inpainter.get_inpainter,
        depth.get_depth_estimator,
    ):
        try:
            loader()
        except Exception:
            # Keep app bootable so /health can expose which model failed.
            continue


@app.get("/")
async def root() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file static/index.html not found.")
    return FileResponse(index_path)


@app.get("/health")
async def health() -> dict:
    models = {
        "detector": detector.get_status(),
        "segmentor": segmentor.get_status(),
        "inpainter": inpainter.get_status(),
        "depth": depth.get_status(),
    }
    app_status = "ok" if all(model.get("loaded") for model in models.values()) else "degraded"

    return {
        "status": app_status,
        "models": models,
    }


@app.get("/catalog")
async def list_catalog() -> dict:
    objects: List[str] = sorted(
        [file.name for file in CATALOG_OBJECTS_DIR.iterdir() if file.suffix.lower() == ".png"]
    )
    return {"objects": objects}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    image_id = uuid.uuid4().hex
    source_path = upload_image_path(image_id)
    current_path = current_image_path(image_id)

    image.save(source_path, format="PNG")
    image.save(current_path, format="PNG")

    try:
        detections = detector.get_detector().detect(source_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {exc}") from exc

    return {
        "image_id": image_id,
        "image_base64": pil_to_base64(image),
        "objects": detections,
    }


@app.post("/segment")
async def segment(req: SegmentRequest) -> dict:
    try:
        image = load_current_image(req.image_id)
        mask, mask_base64, overlay_base64 = segmentor.get_segmentor().segment_with_point(
            image=image, x=req.x, y=req.y
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc

    return {
        "image_id": req.image_id,
        "mask_base64": mask_base64,
        "overlay_base64": overlay_base64,
        "mask_area_pixels": int(mask.sum()),
    }


@app.post("/remove")
async def remove(req: RemoveRequest) -> dict:
    try:
        image = load_current_image(req.image_id)
        mask = decode_mask_base64(req.mask_base64)
        result = inpainter.get_inpainter().inpaint(image, mask)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {exc}") from exc

    out_path = current_image_path(req.image_id)
    result.save(out_path, format="PNG")

    return {"image_id": req.image_id, "image_base64": pil_to_base64(result)}


@app.post("/place")
async def place(req: PlaceRequest) -> dict:
    try:
        image = load_current_image(req.image_id)
        result = placer.place_object(
            room_image=image,
            object_name=req.catalog_object_name,
            x=req.x,
            y=req.y,
            catalog_dir=CATALOG_OBJECTS_DIR,
            depth_estimator=depth.get_depth_estimator(),
            scale=req.scale,
            rotation=req.rotation,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Placement failed: {exc}") from exc

    out_path = current_image_path(req.image_id)
    result.save(out_path, format="PNG")

    return {"image_id": req.image_id, "image_base64": pil_to_base64(result)}
