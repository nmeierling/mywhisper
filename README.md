# myWhisper

Browser-based audio transcription powered by OpenAI's **Whisper** model, running
locally via [Transformers.js](https://github.com/huggingface/transformers.js).

Drop or select any audio file (M4A, MP3, WAV, OGG, FLAC, …), pick a language and
model size, and get a text transcript — **entirely in your browser**. No audio
is uploaded; nothing is sent to any server. After transcribing you can edit,
copy, or download the text as a `.txt` file.

## How it works

1. The audio file is decoded and resampled to 16 kHz mono using the Web Audio API.
2. The Whisper model is loaded from local files (see "Pre-downloading models"),
   never from the network.
3. Transcription runs in a **Web Worker** (off the main thread) so the UI stays
   responsive. Inference uses the WASM backend.

## Develop

```bash
npm install
npm run download-models   # fetch Whisper models to ~/.mywhisper (required, once)
npm run dev               # start the dev server at http://localhost:5173
npm run build             # produce a self-contained static site in dist/
npm run preview           # preview the production build locally
```

## Pre-downloading models to disk

**This app loads models locally only — it never contacts the Hugging Face hub
at runtime.** You must download the models once before using the app:

```bash
npm run download-models                       # tiny + base + small
npm run download-models Xenova/whisper-base    # just one
```

Files are saved under `~/.mywhisper/models/<model>/`. From there:

- **`npm run dev`** serves them at `/models/` from `~/.mywhisper` via a Vite
  plugin — no copy needed while developing.
- **`npm run build`** copies them into `dist/models/`, so the built site is
  fully self-contained.

Details:

- The download grabs the `q8` (quantized) ONNX weights to match the precision
  pinned in [src/worker.js](src/worker.js). Keep `ONNX_KEEP` in the script in
  sync if you change `DTYPE`.
- The app loads the model directory using an absolute URL derived from the
  page's location, so it works at a domain root or in a subdirectory.
- A model that hasn't been downloaded produces a clear error telling you which
  `download-models` command to run — there is no silent network fallback.

## Deploy

`npm run build` outputs a fully self-contained static site in `dist/` —
including the model files under `dist/models/`. Drop it on any static host
(GitHub Pages, Netlify, Cloudflare Pages, S3, …). Asset paths are relative, so
it works from a subdirectory too. Nothing is fetched from any third party at
runtime.

> Heads up: bundling models makes `dist/` large (~40 MB for tiny, up to a few
> hundred MB with all three). Only download the model sizes you intend to ship.

## Model sizes

Sizes are for the `q8` (quantized) weights actually downloaded.

| Model | Download | Speed | Accuracy |
| ----- | -------- | ----- | -------- |
| Tiny  | ~40 MB   | fastest | lowest |
| Base  | ~80 MB   | balanced | good |
| Small | ~250 MB  | slowest | best |

These are loaded from disk (see above), so there is no per-use download — the
only wait is the initial model load into memory.

## Notes & ideas for later

- Runs on the WASM backend for broad compatibility. A WebGPU backend would be
  significantly faster on supported browsers — a natural next enhancement.
- Timestamps / subtitle (`.srt`, `.vtt`) export.
- Microphone recording as an input source.
