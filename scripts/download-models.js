// Downloads Whisper model files from the Hugging Face hub to ~/.mywhisper/models
// so they can be served locally (Vite plugin) and bundled into builds.
//
//   npm run download-models                          # all models, both backends
//   npm run download-models Xenova/whisper-base      # one model
//   npm run download-models -- --backend=wasm        # only WASM (q8) weights
//   npm run download-models -- --backend=webgpu      # only WebGPU weights
//
// ONNX precision is read from src/config.js, so the files on disk always match
// what the app requests at runtime.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline as streamPipeline } from 'node:stream/promises';
import { MODELS, onnxFilesFor } from '../src/config.js';

const MODELS_DIR = path.join(os.homedir(), '.mywhisper', 'models');
const HF = 'https://huggingface.co';

// Repo bookkeeping files we never need at runtime.
const SKIP = new Set(['.gitattributes', 'README.md']);

// --- Parse args -------------------------------------------------------------
const argv = process.argv.slice(2);
let backend = 'both';
const requested = [];
for (const arg of argv) {
  const m = /^--backend=(wasm|webgpu|both)$/.exec(arg);
  if (m) backend = m[1];
  else if (!arg.startsWith('--')) requested.push(arg);
}
const models = requested.length ? requested : MODELS.map((m) => m.id);

// ONNX weight files to keep, as full repo paths (e.g. onnx/encoder_model.onnx).
const keepOnnx = new Set(
  (backend === 'both' ? ['wasm', 'webgpu'] : [backend]).flatMap(onnxFilesFor),
);

function shouldDownload(rfilename) {
  if (SKIP.has(rfilename)) return false;
  if (rfilename.startsWith('onnx/')) {
    // Keep the chosen weights plus any external-data sidecar (`*.onnx_data`).
    for (const keep of keepOnnx) {
      if (rfilename === keep || rfilename === `${keep}_data`) return true;
    }
    return false;
  }
  return true; // config / tokenizer / preprocessor metadata — always keep
}

function human(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function listFiles(model) {
  const res = await fetch(`${HF}/api/models/${model}`);
  if (!res.ok) throw new Error(`Could not list ${model}: HTTP ${res.status}`);
  const info = await res.json();
  return (info.siblings ?? []).map((s) => s.rfilename);
}

async function downloadFile(model, rfilename) {
  const dest = path.join(MODELS_DIR, model, rfilename);
  fs.mkdirSync(path.dirname(dest), { recursive: true });

  const res = await fetch(`${HF}/${model}/resolve/main/${rfilename}`);
  if (!res.ok) throw new Error(`  ✗ ${rfilename}: HTTP ${res.status}`);

  const expected = Number(res.headers.get('content-length')) || 0;
  // Skip re-download if a complete copy already exists.
  if (fs.existsSync(dest) && expected && fs.statSync(dest).size === expected) {
    console.log(`  • ${rfilename} (cached, ${human(expected)})`);
    return;
  }

  await streamPipeline(Readable.fromWeb(res.body), fs.createWriteStream(dest));
  console.log(`  ✓ ${rfilename} (${human(fs.statSync(dest).size)})`);
}

async function main() {
  console.log(`Saving models to: ${MODELS_DIR}`);
  console.log(`Backend weights: ${backend}\n`);
  for (const model of models) {
    console.log(`${model}`);
    const files = (await listFiles(model)).filter(shouldDownload);
    if (!files.length) {
      console.log('  (no matching files found)');
      continue;
    }
    for (const f of files) {
      await downloadFile(model, f);
    }
    console.log('');
  }
  console.log('Done. Start the app with `npm run dev` to serve them locally.');
}

main().catch((err) => {
  console.error(`\nError: ${err.message}`);
  process.exit(1);
});
