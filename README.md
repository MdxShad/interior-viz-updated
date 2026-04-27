# AI Interior Visualization Tool (MVP Demo)

This prototype lets you:
1. Upload a room image
2. Click an object to segment it
3. Remove the selected object with inpainting
4. Place a new object from a small PNG catalog with depth-aware scaling

## Tech Stack
- Backend: FastAPI
- Frontend: single-file HTML + vanilla JS
- Models:
  - YOLOv8n (Ultralytics) for object detection
  - SAM (Segment Anything) for click-based mask generation
  - LaMa (simple-lama-inpainting) for object removal
  - MiDaS small (torch.hub) for depth estimation

## Project Structure
```
interior-viz/
├── main.py
├── services/
│   ├── detector.py
│   ├── segmentor.py
│   ├── inpainter.py
│   ├── depth.py
│   └── placer.py
├── catalog/
│   └── objects/
│       ├── chair.png
│       ├── table.png
│       └── lamp.png
├── static/
│   └── index.html
├── uploads/
├── outputs/
├── models/
│   └── sam_vit_b_01ec64.pth
└── requirements.txt
```

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > **Optional dependencies**: This project uses several heavy AI models
   > (YOLOv8, Segment Anything, LaMa, MiDaS).  These packages may fail to
   > install or load depending on your environment.  The backend will
   > automatically fall back to simpler CPU‑only implementations when a
   > dependency is missing.  For example, if Ultralytics YOLO cannot be
   > imported, detection simply returns no objects.  If Segment Anything
   > or its checkpoint cannot be loaded, a circular mask around the
   > clicked point is used instead.  You can still run and test the
   > prototype without installing every optional model.

## Download SAM Checkpoint
SAM requires `sam_vit_b_01ec64.pth` in `interior-viz/models/`.

PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path .\models | Out-Null
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
  -OutFile ".\models\sam_vit_b_01ec64.pth"
```

Notes:
- YOLOv8n downloads automatically on first run.
- MiDaS model files download automatically on first run via `torch.hub`.

If you prefer not to download these checkpoints, the application will
still function using heuristic fallbacks.  See the code in
`services/segmentor.py`, `services/depth.py` and `services/inpainter.py`
for details.

## Run
From `interior-viz/`:
```bash
uvicorn main:app --reload
```

Then open:
- `http://127.0.0.1:8000/`

## API Endpoints
- `GET /health` model load status
- `POST /upload` upload image + run detection
- `POST /segment` click-point segmentation
- `POST /remove` object removal via inpainting
- `POST /place` place catalog object with depth-aware scaling
  - Accepts JSON with keys:
    - `image_id` (string)
    - `catalog_object_name` (string)
    - `x` and `y` coordinates (integers) where the object should be centred
    - Optional `scale` (float) – multiplies the automatically computed object size; 1.0 means default depth-based size, values >1 enlarge, values <1 shrink
    - Optional `rotation` (float) – degrees clockwise to rotate the object before placement
  - Returns modified image as base64
- `GET /catalog` list available catalog objects

## Sample Catalog Objects

This repository includes three simple placeholder PNG assets in
`catalog/objects/` (`chair.png`, `table.png` and `lamp.png`).  These
images are basic colored shapes with transparent backgrounds so that you
can immediately test the placement functionality.  Replace them with
your own product cut‑outs for more realistic results.

## Add New Catalog Objects
1. Add RGBA PNG files into `catalog/objects/`
2. Restart server (or refresh frontend) so `/catalog` picks up new files
3. Keep transparent backgrounds for best compositing quality
