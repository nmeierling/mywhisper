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

async function handleTranscribe({
  jobId,
  audio,
  model,
  language,
  backend,
  modelBase,
  resumeFrom = 0, // seconds of audio already transcribed in a previous run
  priorSegments = [], // segments produced before the interruption
}) {
  if (modelBase) env.localModelPath = modelBase;
  currentJobId = jobId;
  abortRequested = false;

  try {
    const { transcriber } = await loadModel(model, backend, (data) =>
      self.postMessage({ scope: 'model', status: 'progress', data }),
    );

    const fullSeconds = audio.length / SAMPLE_RATE;
    // On resume, only transcribe the tail; its timestamps are offset by
    // resumeFrom to stay absolute, and prior segments are prepended.
    const work = resumeFrom > 0 ? audio.slice(Math.floor(resumeFrom * SAMPLE_RATE)) : audio;
    self.postMessage({ scope: 'job', jobId, status: 'transcribing', total: fullSeconds });

    // All Whisper models use a 0.02s timestamp resolution; derive it when
    // possible, otherwise fall back to that value.
    const chunkLen = transcriber.processor?.feature_extractor?.config?.chunk_length;
    const maxPos = transcriber.model?.config?.max_source_positions;
    const timePrecision = chunkLen && maxPos ? chunkLen / maxPos : 0.02;

    let chunkCount = 0; // completed 30s windows within this run
    let segmentEnd = 0; // latest segment-end time within the current window
    let liveSegments = []; // segments seen so far in this run (approximate)
    let curSeg = null; // segment currently being decoded
    let partialText = ''; // text streamed in this run (for live display)

    const priorText = priorSegments.map((s) => s.text).join('');

    // Absolute time of the current window's start (relative to the whole file).
    const windowBase = () => resumeFrom + (CHUNK_LENGTH_S - STRIDE_LENGTH_S) * chunkCount;

    const sendUpdate = () => {
      const processed = Math.min(fullSeconds, windowBase() + segmentEnd);
      self.postMessage({
        scope: 'job',
        jobId,
        status: 'update',
        processed,
        total: fullSeconds,
        text: priorText + partialText,
      });
    };

    const checkpoint = () => {
      // Persist progress at a window boundary so a refresh can resume here.
      self.postMessage({
        scope: 'job',
        jobId,
        status: 'checkpoint',
        segments: priorSegments.concat(liveSegments),
        processedUpTo: resumeFrom + (CHUNK_LENGTH_S - STRIDE_LENGTH_S) * chunkCount,
      });
    };

    const checkAbort = () => {
      if (abortRequested) throw new Error('ABORTED');
    };

    const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
      time_precision: timePrecision,
      on_chunk_start: (time) => {
        curSeg = { start: windowBase() + time, end: windowBase() + time, text: '' };
      },
      on_chunk_end: (time) => {
        segmentEnd = time;
        if (curSeg) {
          curSeg.end = windowBase() + time;
          liveSegments.push(curSeg);
          curSeg = null;
        }
        sendUpdate();
        checkAbort();
      },
      callback_function: (text) => {
        if (!curSeg) curSeg = { start: windowBase() + segmentEnd, end: windowBase() + segmentEnd, text: '' };
        curSeg.text += text;
        partialText += text;
        sendUpdate();
        checkAbort();
      },
      on_finalize: () => {
        chunkCount += 1;
        segmentEnd = 0;
        checkpoint();
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

    const output = await transcriber(work, options);

    // The pipeline's returned chunks are the cleanly merged version of this
    // run; offset by resumeFrom and prepend the prior segments.
    const tail = (output.chunks ?? [])
      .filter((c) => Array.isArray(c.timestamp))
      .map((c) => ({
        start: (c.timestamp[0] ?? 0) + resumeFrom,
        end: c.timestamp[1] == null ? null : c.timestamp[1] + resumeFrom,
        text: c.text,
      }));
    const segments = priorSegments.concat(tail);
    const text = segments.map((s) => s.text).join('').trim();

    self.postMessage({ scope: 'job', jobId, status: 'complete', text, segments });
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
