// Web Worker: loads the Whisper model and runs transcription off the main
// thread so the UI stays responsive during inference. Processes one job at a
// time (the model isn't reentrant); the main thread queues the rest.
import { env, pipeline, WhisperTextStreamer } from '@huggingface/transformers';
import { DTYPE } from './config.js';

// Resolve models ONLY from local files — never the network. In dev they are
// served at /models/ from ~/.mywhisper (Vite plugin); in a build they are
// bundled into dist/models/. `localModelPath` is set per request from the
// page's base URL (a worker resolves relative URLs against its own script
// location, not the page, so we need an absolute base for subdirectory deploys).
env.allowLocalModels = true;
env.allowRemoteModels = false;

// Audio arrives already resampled to this rate (see decodeAudio in main.js).
const SAMPLE_RATE = 16000;

// Whisper processes fixed 30s windows; chunking with overlap (stride) handles
// audio of any length. Each new window advances by (chunk - stride) seconds of
// fresh audio — used to turn per-window timestamps into an absolute position.
const CHUNK_LENGTH_S = 30;
const STRIDE_LENGTH_S = 5;

// --- Backend selection ------------------------------------------------------
let webgpuSupported = null; // cached result of the adapter probe

async function hasWebGPU() {
  if (webgpuSupported !== null) return webgpuSupported;
  try {
    webgpuSupported = !!navigator.gpu && !!(await navigator.gpu.requestAdapter());
  } catch {
    webgpuSupported = false;
  }
  return webgpuSupported;
}

// Resolve a backend preference ('auto' | 'webgpu' | 'wasm') to a concrete device.
async function resolveDevice(preference) {
  if (preference === 'wasm') return 'wasm';
  if (preference === 'webgpu') return (await hasWebGPU()) ? 'webgpu' : 'wasm';
  return (await hasWebGPU()) ? 'webgpu' : 'wasm'; // auto
}

// --- Pipeline cache ---------------------------------------------------------
// One instance per (model, device). Switching model or backend lazily builds a
// new pipeline; previously loaded ones stay cached for instant reuse.
const instances = new Map();

async function getPipeline(model, device, progress_callback) {
  const key = `${model}|${device}`;
  if (!instances.has(key)) {
    const promise = pipeline('automatic-speech-recognition', model, {
      device,
      dtype: DTYPE[device],
      progress_callback,
    }).catch((err) => {
      instances.delete(key); // allow a retry after a failed load
      throw err;
    });
    instances.set(key, promise);
  }
  return instances.get(key);
}

// Load a model, transparently falling back from WebGPU to WASM if it fails.
async function loadModel(model, preference, progress_callback) {
  const device = await resolveDevice(preference);
  try {
    const t = await getPipeline(model, device, progress_callback);
    return { transcriber: t, device };
  } catch (err) {
    if (device === 'webgpu') {
      webgpuSupported = false; // don't try WebGPU again this session
      const t = await getPipeline(model, 'wasm', progress_callback);
      return { transcriber: t, device: 'wasm' };
    }
    throw err;
  }
}

// --- Message handling -------------------------------------------------------
let currentJobId = null;
let abortRequested = false;

self.addEventListener('message', (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'abort':
      // Cooperative cancel: a flag the streamer checks between tokens.
      if (msg.jobId === currentJobId) abortRequested = true;
      break;
    case 'load':
      handleLoad(msg); // fire-and-forget so the listener stays responsive
      break;
    case 'transcribe':
      handleTranscribe(msg);
      break;
  }
});

async function handleLoad({ model, backend, modelBase }) {
  if (modelBase) env.localModelPath = modelBase;
  try {
    const { device } = await loadModel(model, backend, (data) =>
      self.postMessage({ scope: 'model', status: 'progress', data }),
    );
    self.postMessage({ scope: 'model', status: 'ready', model, device });
  } catch (error) {
    self.postMessage({ scope: 'model', status: 'error', message: errMessage(error, model) });
  }
}

async function handleTranscribe({ jobId, audio, model, language, backend, modelBase }) {
  if (modelBase) env.localModelPath = modelBase;
  currentJobId = jobId;
  abortRequested = false;

  try {
    const { transcriber } = await loadModel(model, backend, (data) =>
      self.postMessage({ scope: 'model', status: 'progress', data }),
    );

    const totalSeconds = audio.length / SAMPLE_RATE;
    self.postMessage({ scope: 'job', jobId, status: 'transcribing', total: totalSeconds });

    // All Whisper models use a 0.02s timestamp resolution; derive it when
    // possible, otherwise fall back to that value.
    const chunkLen = transcriber.processor?.feature_extractor?.config?.chunk_length;
    const maxPos = transcriber.model?.config?.max_source_positions;
    const timePrecision = chunkLen && maxPos ? chunkLen / maxPos : 0.02;

    let chunkCount = 0; // completed 30s windows
    let segmentEnd = 0; // latest segment-end time within the current window
    let partialText = '';

    const sendUpdate = () => {
      const base = (CHUNK_LENGTH_S - STRIDE_LENGTH_S) * chunkCount;
      const processed = Math.min(totalSeconds, base + segmentEnd);
      self.postMessage({ scope: 'job', jobId, status: 'update', processed, total: totalSeconds, text: partialText });
    };

    const checkAbort = () => {
      if (abortRequested) throw new Error('ABORTED');
    };

    const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
      time_precision: timePrecision,
      on_chunk_end: (time) => {
        segmentEnd = time;
        sendUpdate();
        checkAbort();
      },
      callback_function: (text) => {
        partialText += text;
        sendUpdate();
        checkAbort();
      },
      on_finalize: () => {
        chunkCount += 1;
        segmentEnd = 0;
      },
    });

    const options = {
      chunk_length_s: CHUNK_LENGTH_S,
      stride_length_s: STRIDE_LENGTH_S,
      return_timestamps: true, // required for the streamer's timestamp tokens
      streamer,
    };
    if (language && language !== 'auto') {
      options.language = language;
      options.task = 'transcribe';
    }

    const output = await transcriber(audio, options);

    // Streamed text can have overlap artifacts at window boundaries; the
    // pipeline's returned text + chunks are the cleanly merged version.
    // chunks: [{ text, timestamp: [start, end] }] → segments for the UI.
    const segments = (output.chunks ?? [])
      .filter((c) => Array.isArray(c.timestamp))
      .map((c) => ({ start: c.timestamp[0] ?? 0, end: c.timestamp[1] ?? null, text: c.text }));

    self.postMessage({
      scope: 'job',
      jobId,
      status: 'complete',
      text: output.text.trim(),
      segments,
    });
  } catch (error) {
    if (error?.message === 'ABORTED') {
      self.postMessage({ scope: 'job', jobId, status: 'canceled' });
    } else {
      self.postMessage({ scope: 'job', jobId, status: 'error', message: errMessage(error, model) });
    }
  } finally {
    if (currentJobId === jobId) currentJobId = null;
  }
}

function errMessage(error, model) {
  const message = error?.message ?? String(error);
  if (/\b(404|not.*found|could not locate|failed to fetch)\b/i.test(message)) {
    return `Model "${model}" is not available locally. Run \`npm run download-models\` and reload.`;
  }
  return message;
}
