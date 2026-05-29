// Downloads Whisper model files from the Hugging Face hub to ~/.mywhisper/models
// so they can be served locally (see the serveLocalModels plugin in
// vite.config.js). Run: `npm run download-models [model-id ...]`.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline as streamPipeline } from 'node:stream/promises';

const MODELS_DIR = path.join(os.homedir(), '.mywhisper', 'models');
const HF = 'https://huggingface.co';

// Which ONNX weight files to keep. Must match the dtype pinned in
// src/worker.js: `q8` → the "_quantized" variants. Other ONNX precisions are
// skipped to save disk space.
const ONNX_KEEP = ['encoder_model_quantized', 'decoder_model_merged_quantized'];

// Repo bookkeeping files we never need at runtime.
const SKIP = new Set(['.gitattributes', 'README.md']);

const DEFAULT_MODELS = [
  'Xenova/whisper-tiny',
  'Xenova/whisper-base',
  'Xenova/whisper-small',
];

const models = process.argv.slice(2).length
  ? process.argv.slice(2)
  : DEFAULT_MODELS;

function shouldDownload(rfilename) {
  if (SKIP.has(rfilename)) return false;
  if (rfilename.startsWith('onnx/')) {
    const base = rfilename.slice('onnx/'.length);
    return ONNX_KEEP.some((keep) => base.startsWith(keep));
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
  if (!res.ok) {
    throw new Error(`Could not list ${model}: HTTP ${res.status}`);
  }
  const info = await res.json();
  return (info.siblings ?? []).map((s) => s.rfilename);
}

async function downloadFile(model, rfilename) {
  const dest = path.join(MODELS_DIR, model, rfilename);
  fs.mkdirSync(path.dirname(dest), { recursive: true });

  const res = await fetch(`${HF}/${model}/resolve/main/${rfilename}`);
  if (!res.ok) {
    throw new Error(`  ✗ ${rfilename}: HTTP ${res.status}`);
  }

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
  console.log(`Saving models to: ${MODELS_DIR}\n`);
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
