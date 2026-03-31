/**
 * ImageSonifier — React PoC Component
 *
 * Handles:
 *  - Drag-and-drop / click-to-upload image
 *  - POST to FastAPI /process endpoint
 *  - Animated playhead canvas overlay
 *  - Tone.js polyphonic oscillator engine
 *
 * Install deps:
 *   npm install tone zustand
 */

import { useState, useRef, useCallback, useEffect } from "react";
import * as Tone from "tone";

// ─── Types ────────────────────────────────────────────────────────────────────

interface PixelEvent {
  frequency: number;
  amplitude: number;
  filter_cutoff: number;
  duration: number;
  pan: number;
}

interface ColumnData {
  column_index: number;
  events: PixelEvent[];
}

interface SonificationResult {
  width: number;
  height: number;
  columns: ColumnData[];
  total_duration: number;
}

// ─── Tone.js Engine ───────────────────────────────────────────────────────────

/**
 * Lightweight polyphonic synth pool.
 * Each column fires N simultaneous voices (one per significant pixel row).
 * We reuse a pool of PolySynth voices to avoid allocation overhead.
 */
class SonificationEngine {
  private synth: Tone.PolySynth;
  private filter: Tone.BiquadFilter;
  private panner: Tone.Panner;
  private reverb: Tone.Reverb;

  constructor() {
    // Reverb for spatial depth
    this.reverb = new Tone.Reverb({ decay: 1.2, wet: 0.15 });
    // Low-pass filter controlled by hue
    this.filter = new Tone.BiquadFilter(2000, "lowpass");
    // Stereo panner controlled by vertical position
    this.panner = new Tone.Panner(0);

    // Polyphonic synth — up to 16 simultaneous voices
    this.synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sine" },
      envelope: {
        attack: 0.005,
        decay: 0.05,
        sustain: 0.8,
        release: 0.1,
      },
    });

    // Signal chain: synth → filter → panner → reverb → output
    this.synth.connect(this.filter);
    this.filter.connect(this.panner);
    this.panner.connect(this.reverb);
    this.reverb.toDestination();
  }

  /**
   * Fire all pixel events for a single column.
   * Called once per column tick during playback loop.
   */
  fireColumn(events: PixelEvent[], time?: number) {
    if (events.length === 0) return;

    // Use the dominant (highest amplitude) event's filter cutoff and pan
    const dominant = events.reduce((a, b) =>
      a.amplitude > b.amplitude ? a : b
    );

    this.filter.frequency.setValueAtTime(
      dominant.filter_cutoff,
      time ?? Tone.now()
    );
    this.panner.pan.setValueAtTime(dominant.pan, time ?? Tone.now());

    // Trigger all frequencies simultaneously (polyphony)
    events.forEach((event) => {
      this.synth.triggerAttackRelease(
        event.frequency,
        event.duration,
        time ?? Tone.now(),
        event.amplitude
      );
    });
  }

  /**
   * Blend oscillator type based on saturation.
   * sat=0 → sine (pure), sat=1 → sawtooth (rich harmonics)
   * Tone.js doesn't interpolate oscillator types natively,
   * so we threshold at 0.5 for this PoC.
   */
  setTimbre(saturation: number) {
    const type = saturation > 0.5 ? "sawtooth" : "sine";
    this.synth.set({ oscillator: { type } });
  }

  dispose() {
    this.synth.dispose();
    this.filter.dispose();
    this.panner.dispose();
    this.reverb.dispose();
  }
}

// ─── Playhead Canvas ──────────────────────────────────────────────────────────

function PlayheadCanvas({
  imageUrl,
  progress, // 0.0 → 1.0
  isPlaying,
}: {
  imageUrl: string;
  progress: number;
  isPlaying: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
  }, [imageUrl]);

  useEffect(() => {
    draw();
  }, [progress]);

  const draw = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    // Draw image
    ctx.drawImage(img, 0, 0, w, h);

    if (isPlaying || progress > 0) {
      const x = progress * w;

      // Scanned region overlay (darkened)
      ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
      ctx.fillRect(0, 0, x, h);

      // Playhead line
      const grad = ctx.createLinearGradient(x - 2, 0, x + 3, 0);
      grad.addColorStop(0, "rgba(255,255,255,0.0)");
      grad.addColorStop(0.4, "rgba(255,255,255,0.9)");
      grad.addColorStop(0.6, "rgba(255,255,255,0.9)");
      grad.addColorStop(1, "rgba(255,255,255,0.0)");
      ctx.fillStyle = grad;
      ctx.fillRect(x - 2, 0, 5, h);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        height: "100%",
        display: "block",
        borderRadius: "8px",
      }}
    />
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";
const TICK_INTERVAL_MS = 16; // ~60fps column scan

export default function ImageSonifier() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(10);
  const [error, setError] = useState<string | null>(null);
  const [columnCount, setColumnCount] = useState(0);

  const sonificationData = useRef<SonificationResult | null>(null);
  const engine = useRef<SonificationEngine | null>(null);
  const playbackTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentColumn = useRef(0);

  // ── File handling ────────────────────────────────────────────────────────────

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, WebP).");
      return;
    }

    setError(null);
    setIsLoading(true);
    setProgress(0);
    setImageUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);
    formData.append("duration", String(duration));

    try {
      const res = await fetch(`${API_BASE}/process`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Processing failed");
      }

      const data: SonificationResult = await res.json();
      sonificationData.current = data;
      setColumnCount(data.width);
    } catch (e: any) {
      setError(e.message ?? "Could not connect to processing server.");
      setImageUrl(null);
    } finally {
      setIsLoading(false);
    }
  }, [duration]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  // ── Playback engine ──────────────────────────────────────────────────────────

  const startPlayback = async () => {
    if (!sonificationData.current) return;

    // Resume AudioContext on user gesture (browser requirement)
    await Tone.start();

    if (!engine.current) {
      engine.current = new SonificationEngine();
    }

    const data = sonificationData.current;
    currentColumn.current = 0;
    setIsPlaying(true);

    const columnDuration = (data.total_duration / data.width) * 1000; // ms

    playbackTimer.current = setInterval(() => {
      const col = currentColumn.current;

      if (col >= data.columns.length) {
        stopPlayback();
        return;
      }

      const column = data.columns[col];
      engine.current?.fireColumn(column.events);
      currentColumn.current++;
      setProgress(col / (data.width - 1));
    }, columnDuration);
  };

  const stopPlayback = () => {
    if (playbackTimer.current) {
      clearInterval(playbackTimer.current);
      playbackTimer.current = null;
    }
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (isPlaying) {
      stopPlayback();
    } else {
      startPlayback();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPlayback();
      engine.current?.dispose();
    };
  }, []);

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div style={{ maxWidth: 640, margin: "0 auto", padding: "2rem 1rem" }}>
      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onClick={() => document.getElementById("file-input")?.click()}
        style={{
          border: `1.5px dashed ${isDragging ? "#7F77DD" : "#ccc"}`,
          borderRadius: 12,
          padding: "2rem",
          textAlign: "center",
          cursor: "pointer",
          background: isDragging ? "rgba(127,119,221,0.05)" : "transparent",
          transition: "all 0.15s",
          marginBottom: "1.5rem",
        }}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={onFileInput}
        />
        {isLoading ? (
          <p style={{ margin: 0, color: "#888", fontSize: 14 }}>
            Processing image...
          </p>
        ) : (
          <p style={{ margin: 0, color: "#888", fontSize: 14 }}>
            Drop an image here, or click to upload
          </p>
        )}
      </div>

      {error && (
        <p style={{ color: "red", fontSize: 13, marginBottom: "1rem" }}>
          {error}
        </p>
      )}

      {/* Image + playhead */}
      {imageUrl && (
        <div
          style={{
            position: "relative",
            aspectRatio: "16/9",
            background: "#111",
            borderRadius: 8,
            overflow: "hidden",
            marginBottom: "1.25rem",
          }}
        >
          <PlayheadCanvas
            imageUrl={imageUrl}
            progress={progress}
            isPlaying={isPlaying}
          />
        </div>
      )}

      {/* Controls */}
      {sonificationData.current && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {/* Duration slider */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 13, color: "#666" }}>
            <span>Duration</span>
            <input
              type="range"
              min={3}
              max={30}
              value={duration}
              disabled={isPlaying}
              onChange={(e) => setDuration(Number(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{ minWidth: 32 }}>{duration}s</span>
          </div>

          {/* Info */}
          <p style={{ margin: 0, fontSize: 12, color: "#999" }}>
            {columnCount} columns · {sonificationData.current.height}px tall ·{" "}
            {sonificationData.current.columns.reduce(
              (acc, col) => acc + col.events.length,
              0
            ).toLocaleString()}{" "}
            audio events
          </p>

          {/* Play / Stop */}
          <button
            onClick={togglePlayback}
            style={{
              padding: "10px 24px",
              fontSize: 15,
              fontWeight: 500,
              border: "1.5px solid #7F77DD",
              borderRadius: 8,
              background: isPlaying ? "#7F77DD" : "transparent",
              color: isPlaying ? "#fff" : "#7F77DD",
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            {isPlaying ? "Stop" : "Play"}
          </button>
        </div>
      )}
    </div>
  );
}
