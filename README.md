# oscilato
I was supposed to be studying because I have an operating system exam in 6 hours, but instead i made the oscilato that takes any picture and change it to a sound ( just be careful ) 


[README.md](https://github.com/user-attachments/files/26364921/README.md)
# Image Sonification — PoC

Converts images to sound by scanning left→right, mapping pixel data to audio parameters.

## Quick start

### Backend (FastAPI)
```bash
cd sonification_backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend (React + Tone.js)
```bash
# In your React project root
npm install tone
cp ImageSonifier.tsx src/components/
# Import and use: <ImageSonifier />
```

---

## Architecture overview

```
React UI  ──POST /process──►  FastAPI  ──►  Pillow/NumPy  ──►  HSV array
   ▲                                                                │
   │                                                                ▼
   └──── Tone.js PolySynth ◄────── column JSON ◄──── math mapper ◄┘
```

---

## Sonification mapping

| Image property | Audio parameter | Formula |
|---|---|---|
| Y position | Pitch (Hz) | `f = 110 × 2^(n/12)`, n ∈ [0,48] |
| HSV Value (brightness) | Amplitude | `amp = V²` (perceptual) |
| Hue | Filter cutoff | `fc = 200 + (H/360) × 3800` Hz |
| Saturation | Timbre blend | `0 = sine`, `1 = sawtooth` |
| X position | Time | `t = x × (duration / width)` |

---

## Extending to Method B (Semantic / PyTorch)

1. Install `torch` and `transformers`
2. Load CLIP: `model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`
3. Segment image into regions using a lightweight detector (e.g. YOLOv8-nano)
4. Map detected object classes → instrument presets:
   - Sky/water → pads / filter sweeps
   - Faces → melodic sine tones
   - Foliage → granular textures
   - Architecture → rhythmic percussive events
5. Blend with Method A output (statistical base layer + semantic accent layer)

---

## Performance notes

- Image is resized to max 256px wide before processing (configurable via `MAX_WIDTH`)
- Near-dark pixels (`V² < 0.03`) are skipped — reduces event count by ~40% on typical photos
- `/stream` endpoint yields NDJSON — React can begin playback before full response arrives
- Tone.js `PolySynth` is capped at 16 simultaneous voices; dense columns are automatically
  sorted by amplitude and trimmed to top-16 events (add this to `fireColumn` for production)

---

## Low-latency tips

- Keep `TICK_INTERVAL_MS` ≥ 16ms (one frame) — below this, AudioContext scheduling
  jitter becomes audible
- Use `Tone.Transport` + `Tone.Part` for precise scheduling instead of `setInterval`
  (upgrade path for production)
- Pre-schedule 2–3 columns ahead using `Tone.now() + lookahead` to smooth gaps

---

## File structure

```
sonification_backend/
  main.py          ← FastAPI app, image processing, math mapper
  requirements.txt

sonification_frontend/
  ImageSonifier.tsx  ← React component, Tone.js engine, playhead canvas
```
