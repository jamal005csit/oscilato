"""
Image Sonification API — FastAPI Backend
POST /process  → full image analysis → returns all columns as JSON
GET  /column   → single column query (for streaming / incremental mode)
"""

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import io
import json

app = FastAPI(title="Image Sonification API")

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response Schema ───────────────────────────────────────────────────────────

class PixelEvent(BaseModel):
    frequency: float     # Hz — derived from Y position via semitone formula
    amplitude: float     # 0.0–1.0 — derived from HSV Value (brightness²)
    filter_cutoff: float # Hz — derived from Hue (200–4000 Hz)
    duration: float      # seconds — constant per column, configurable
    pan: float           # -1.0 to 1.0, reserved for stereo spread (left = low, right = high)

class ColumnData(BaseModel):
    column_index: int
    events: List[PixelEvent]   # one event per significant pixel in this column

class SonificationResult(BaseModel):
    width: int
    height: int
    columns: List[ColumnData]
    total_duration: float      # seconds


# ─── Core DSP: HSV → Audio Parameters ─────────────────────────────────────────

# Musical constants
A2_HZ = 110.0        # Bottom of range (image bottom)
SEMITONE_RANGE = 48  # A2 → A6 (4 octaves)
FILTER_MIN_HZ = 200.0
FILTER_MAX_HZ = 4000.0
MIN_AMPLITUDE = 0.03  # Threshold — skip near-black pixels for performance


def y_to_frequency(y: int, height: int) -> float:
    """
    Map vertical pixel position to musical frequency.
    Top of image = high pitch (A6), bottom = low pitch (A2).
    Uses equal temperament semitone formula: f = 110 * 2^(n/12)
    """
    # Invert Y so top = high pitch
    normalized = 1.0 - (y / height)
    # Map to semitone index across SEMITONE_RANGE
    n = normalized * SEMITONE_RANGE
    return A2_HZ * (2 ** (n / 12))


def value_to_amplitude(v: float) -> float:
    """
    Map HSV Value (brightness) to perceptual amplitude.
    Squared for perceptual linearity (matches how we hear volume).
    """
    return float(v ** 2)


def hue_to_filter_cutoff(h: float) -> float:
    """
    Map Hue (0–360°) to filter cutoff frequency (200–4000 Hz).
    Red (0°/360°) = lowest cutoff (darker, more muffled)
    Green/Cyan (120°–180°) = mid-range
    Blue/Violet (240°–300°) = highest cutoff (brighter, more open)
    """
    normalized = h / 360.0
    return FILTER_MIN_HZ + normalized * (FILTER_MAX_HZ - FILTER_MIN_HZ)


def saturation_to_timbre(s: float) -> float:
    """
    Map Saturation to oscillator type blend.
    0.0 = pure sine (mellow, monochrome)
    1.0 = sawtooth (harsh, vivid)
    Returned as 0.0–1.0; front-end interpolates between oscillator types.
    """
    return float(s)


# ─── Image Processing ──────────────────────────────────────────────────────────

def process_image(image_bytes: bytes, total_duration: float = 10.0) -> SonificationResult:
    """
    Convert image bytes → SonificationResult containing per-column audio events.

    Steps:
    1. Decode image via Pillow
    2. Convert RGB → HSV via NumPy
    3. For each column X: extract pixel rows, map HSV → audio params
    4. Filter out near-silent events (amplitude < MIN_AMPLITUDE)
    5. Return structured column data
    """

    # Step 1: Decode image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to max 256px wide for performance — keeps column count manageable
    MAX_WIDTH = 256
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)

    width, height = img.size

    # Step 2: RGB → HSV via NumPy
    rgb = np.array(img, dtype=np.float32) / 255.0  # shape: (H, W, 3)
    hsv = rgb_to_hsv_numpy(rgb)                     # shape: (H, W, 3) [H:0-360, S:0-1, V:0-1]

    columns = []
    col_duration = total_duration / width

    # Step 3: Column scan (left → right = time)
    for x in range(width):
        col_hsv = hsv[:, x, :]   # shape: (H, 3)
        events = []

        for y in range(height):
            h, s, v = float(col_hsv[y, 0]), float(col_hsv[y, 1]), float(col_hsv[y, 2])

            # Step 4: Skip near-dark pixels
            amp = value_to_amplitude(v)
            if amp < MIN_AMPLITUDE:
                continue

            events.append(PixelEvent(
                frequency=round(y_to_frequency(y, height), 2),
                amplitude=round(amp, 4),
                filter_cutoff=round(hue_to_filter_cutoff(h), 1),
                duration=round(col_duration * 0.9, 4),  # 90% of column time for slight gap
                pan=round((y / height) * 2 - 1, 3),     # low pitch = left, high = right
            ))

        columns.append(ColumnData(column_index=x, events=events))

    return SonificationResult(
        width=width,
        height=height,
        columns=columns,
        total_duration=total_duration,
    )


def rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """
    Pure NumPy RGB→HSV conversion. Shape: (H, W, 3).
    Avoids OpenCV dependency for portability; swap for cv2.cvtColor in production.
    Returns H in [0,360], S and V in [0,1].
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin + 1e-8  # avoid /0

    h = np.zeros_like(r)
    mask_r = (cmax == r) & (cmax != cmin)
    mask_g = (cmax == g) & (cmax != cmin)
    mask_b = (cmax == b) & (cmax != cmin)

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 360)
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240)

    s = np.where(cmax == 0, 0, diff / (cmax + 1e-8))
    v = cmax

    return np.stack([h, s, v], axis=-1)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.post("/process", response_model=SonificationResult)
async def process_endpoint(
    file: UploadFile = File(...),
    duration: float = 10.0
):
    """
    Upload an image; receive full sonification data.
    duration: total playback time in seconds (default 10s).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image (JPEG, PNG, WebP, etc.)")

    contents = await file.read()

    try:
        result = process_image(contents, total_duration=duration)
    except Exception as e:
        raise HTTPException(500, detail=f"Image processing failed: {str(e)}")

    return result


@app.post("/stream")
async def stream_endpoint(file: UploadFile = File(...), duration: float = 10.0):
    """
    Streaming variant: yields columns one by one as newline-delimited JSON.
    Useful for large images — React can start playing before full processing is done.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Not an image file")

    contents = await file.read()

    async def generate():
        result = process_image(contents, total_duration=duration)
        # Send header frame first
        yield json.dumps({"type": "header", "width": result.width,
                          "height": result.height, "total_duration": result.total_duration}) + "\n"
        for col in result.columns:
            yield col.model_dump_json() + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    return {"status": "ok"}
