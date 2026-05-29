# myWhisper

Browser-based audio transcription powered by OpenAI's **Whisper** model, running
locally via [Transformers.js](https://github.com/huggingface/transformers.js).

Drop or select audio files (M4A, MP3, WAV, OGG, FLAC, …), pick a language, model
size, and backend, then transcribe — **entirely in your browser**. No audio is
uploaded; nothing is sent to any server. Queue several files and they transcribe
one after another. Each finished transcript is shown as **timestamped segments**:
click a timestamp to re-listen to that part of the audio. Copy or download any
transcript (or all of them), optionally with `[m:ss]` timestamps inline.

## How it works

1. Each audio file is decoded and resampled to 16 kHz mono using the Web Audio
   API, on the main thread (off the UI thread, so it stays responsive).
2. The Whisper model is loaded from local files (see "Pre-downloading models"),
   never from the network.
3. Transcription runs in a single **Web Worker** so the UI stays responsive and
   live progress (audio-time processed, ETA, streaming text) is reported back.
4. Files are processed **sequentially** by the one warm worker — the right model
   for the hardware, since parallel workers would each reload the model and just
   contend for the same CPU/GPU. While one file transcribes, the **next file's
   audio is decoded ahead**, hiding decode latency.
5. The queue, transcripts, and original audio are **persisted to IndexedDB**, so
   a reload restores everything. A file interrupted mid-transcription resumes
   from its last completed ~30s window rather than starting over.

### Persistence & resume

- Completed transcripts (as timestamped segments), queue order, settings, and
  the original audio blobs are stored in IndexedDB; the app calls
  `navigator.storage.persist()` to request durable storage.
- During a run, progress is checkpointed at each 30s window boundary. After a
  refresh/crash, an interrupted job reopens as "pending" and, when run, only the
  unfinished tail is re-transcribed (its timestamps offset to stay absolute).
- The audio bytes are persisted so resume and re-listen work after a reload.
  This costs storage (tens of MB per file); if the quota is exceeded the
  transcript still saves, but that file won't survive a reload.
- A `beforeunload` prompt warns before leaving while a transcription is running.

### Backends

| Backend | When | Notes |
| ------- | ---- | ----- |
| **WebGPU** | GPU available | Much faster. Uses fp32 encoder + q4 decoder. |
| **WASM** | fallback / everywhere | CPU, multi-threaded. Uses q8 weights. |

The **Auto** setting (default) uses WebGPU when the browser exposes a GPU adapter
and falls back to WASM otherwise. If WebGPU fails to initialize at load time, it
falls back to WASM automatically.

## Develop

```bash
npm install
npm run download-models   # fetch Whisper models to ~/.mywhisper (required, once)
npm run dev               # dev server at http://localhost:5173
npm run build             # self-contained static site in dist/
npm run preview           # preview the production build
```

## Pre-downloading models to disk

**This app loads models locally only — it never contacts the Hugging Face hub
at runtime.** Download them once before using the app:

```bash
npm run download-models                       # all models, both backends
npm run download-models Xenova/whisper-base   # just one
npm run download-models -- --backend=wasm     # WASM (q8) weights only
npm run download-models -- --backend=webgpu   # WebGPU weights only
```

Files are saved under `~/.mywhisper/models/<model>/`. From there:

- **`npm run dev`** serves them at `/models/` from `~/.mywhisper` via a Vite
  plugin — no copy needed while developing.
- **`npm run build`** copies them into `dist/models/`, so the built site is
  fully self-contained.

Details:

- ONNX precision per backend lives in [src/config.js](src/config.js) (a single
  source of truth shared by the worker and the download script), so the files on
  disk always match what the app requests. By default both backends' weights are
  downloaded; use `--backend=` to fetch only one.
- The app loads the model directory using an absolute URL derived from the
  page's location, so it works at a domain root or in a subdirectory.
- A model that hasn't been downloaded produces a clear error telling you to run
  `download-models` — there is no silent network fallback.

## Deploy

`npm run build` outputs a fully self-contained static site in `dist/` —
including the model files under `dist/models/`. Drop it on any static host
(GitHub Pages, Netlify, Cloudflare Pages, S3, …). Asset paths are relative, so it
works from a subdirectory too. Nothing is fetched from any third party at runtime.

> **Required: cross-origin isolation.** onnxruntime-web's multi-threaded WASM
> backend needs `SharedArrayBuffer`, which only exists when the page is
> cross-origin isolated. The site **will not load models** without these two
> response headers:
>
> ```
> Cross-Origin-Opener-Policy: same-origin
> Cross-Origin-Embedder-Policy: require-corp
> ```
>
> The dev and preview servers set them automatically. On a static host, send
> them via the host's config (e.g. a Netlify/Cloudflare `_headers` file). Hosts
> that can't set headers (GitHub Pages) need the
> [`coi-serviceworker`](https://github.com/gzuidhof/coi-serviceworker) shim.

> Heads up: bundling models makes `dist/` large (tens to hundreds of MB).
> Download only the model sizes and backends you intend to ship.

## Model sizes

Approximate on-disk size of the WASM (q8) weights. WebGPU weights (fp32 encoder
+ q4 decoder) add more, so downloading both backends roughly doubles this.

| Model | q8 size | Speed | Accuracy |
| ----- | ------- | ----- | -------- |
| Tiny  | ~40 MB  | fastest | lowest |
| Base  | ~80 MB  | balanced | good |
| Small | ~250 MB | slowest | best |

Loaded from disk, so there's no per-use download — the only wait is the one-time
model load into memory (warmed as soon as you add files).

## Ideas for later

- English-only (`.en`) models when the language is English — smaller and more
  accurate for English.
- Subtitle export (`.srt`, `.vtt`) — the timestamped segments are already there.
- Microphone recording as an input source.
- A storage cap / "clear stored audio" control to bound IndexedDB usage.
