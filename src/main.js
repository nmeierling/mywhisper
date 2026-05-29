import './style.css';
import { LANGUAGES } from './languages.js';

// Whisper expects mono PCM audio at 16 kHz.
const WHISPER_SAMPLE_RATE = 16000;

// Absolute URL of the local /models/ directory, resolved from the page's base
// so it's correct whether deployed at a domain root or a subdirectory. Passed
// to the worker, which can't resolve page-relative paths itself.
const MODEL_BASE = new URL('./models/', document.baseURI).href;

// --- Element refs -----------------------------------------------------------
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileNameEl = document.getElementById('file-name');
const modelSelect = document.getElementById('model-select');
const languageSelect = document.getElementById('language-select');
const transcribeBtn = document.getElementById('transcribe-btn');
const statusEl = document.getElementById('status');
const statusMessage = document.getElementById('status-message');
const statusBar = statusEl.querySelector('.status__bar');
const statusBarFill = document.getElementById('status-bar-fill');
const resultEl = document.getElementById('result');
const transcriptEl = document.getElementById('transcript');
const copyBtn = document.getElementById('copy-btn');
const downloadBtn = document.getElementById('download-btn');

// --- State -------------------------------------------------------------------
let selectedFile = null;
let isWorking = false;
let worker = null;

// --- Populate language dropdown ---------------------------------------------
for (const [value, label] of LANGUAGES) {
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = label;
  languageSelect.appendChild(opt);
}

// --- Worker setup ------------------------------------------------------------
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.js', import.meta.url), {
    type: 'module',
  });
  worker.addEventListener('message', onWorkerMessage);
  worker.addEventListener('error', (e) => {
    setError(`Worker error: ${e.message}`);
    finishWorking();
  });
  return worker;
}

function onWorkerMessage(event) {
  const data = event.data;
  switch (data.status) {
    case 'progress': {
      // `data.data` is a Transformers.js progress event. We only get useful
      // percentages during file downloads.
      const p = data.data;
      if (p?.status === 'progress' && typeof p.progress === 'number') {
        showProgress(`Downloading model: ${p.file ?? ''}`, p.progress);
      } else if (p?.status === 'initiate') {
        showStatus('Loading model…');
      } else if (p?.status === 'ready') {
        showStatus('Model ready');
      }
      break;
    }
    case 'transcribing':
      showStatus('Transcribing… this can take a while for long files.');
      hideProgressBar();
      break;
    case 'complete':
      showTranscript(data.text);
      showStatus('Done.');
      finishWorking();
      break;
    case 'error':
      setError(data.message);
      finishWorking();
      break;
  }
}

// --- File handling -----------------------------------------------------------
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(m4a|mp3|wav|ogg|flac|aac|webm)$/i.test(file.name)) {
    setError('That does not look like an audio file.');
    return;
  }
  selectedFile = file;
  fileNameEl.textContent = `${file.name} (${formatBytes(file.size)})`;
  fileNameEl.hidden = false;
  dropzone.classList.add('dropzone--has-file');
  hideError();
  updateButtonState();
}

dropzone.addEventListener('click', () => !isWorking && fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && !isWorking) {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    if (!isWorking) dropzone.classList.add('dropzone--dragover');
  }),
);
['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dropzone--dragover');
  }),
);
dropzone.addEventListener('drop', (e) => {
  if (isWorking) return;
  handleFile(e.dataTransfer.files[0]);
});

// --- Audio decoding ----------------------------------------------------------
// Decode any browser-supported audio file into a mono Float32Array at 16 kHz.
async function decodeAudio(file) {
  const arrayBuffer = await file.arrayBuffer();

  // Use a one-shot AudioContext just to decode the compressed file.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const decodeCtx = new AudioCtx();
  let decoded;
  try {
    decoded = await decodeCtx.decodeAudioData(arrayBuffer);
  } finally {
    decodeCtx.close();
  }

  // Resample to 16 kHz mono via an OfflineAudioContext.
  const duration = decoded.duration;
  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(duration * WHISPER_SAMPLE_RATE),
    WHISPER_SAMPLE_RATE,
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = decoded;
  source.connect(offlineCtx.destination);
  source.start();
  const rendered = await offlineCtx.startRendering();

  return rendered.getChannelData(0);
}

// --- Transcription flow ------------------------------------------------------
transcribeBtn.addEventListener('click', async () => {
  if (!selectedFile || isWorking) return;

  startWorking();
  hideError();
  resultEl.hidden = true;

  let audio;
  try {
    showStatus('Decoding audio…');
    audio = await decodeAudio(selectedFile);
  } catch (err) {
    setError(`Could not decode that audio file: ${err.message}`);
    finishWorking();
    return;
  }

  showStatus('Loading model…');
  getWorker().postMessage({
    audio,
    model: modelSelect.value,
    language: languageSelect.value,
    modelBase: MODEL_BASE,
  });
});

// --- Result actions ----------------------------------------------------------
copyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(transcriptEl.value);
    flashButton(copyBtn, 'Copied!');
  } catch {
    flashButton(copyBtn, 'Copy failed');
  }
});

downloadBtn.addEventListener('click', () => {
  const blob = new Blob([transcriptEl.value], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const base = selectedFile?.name.replace(/\.[^.]+$/, '') || 'transcript';
  a.href = url;
  a.download = `${base}.txt`;
  a.click();
  URL.revokeObjectURL(url);
});

// --- UI helpers --------------------------------------------------------------
function updateButtonState() {
  transcribeBtn.disabled = !selectedFile || isWorking;
}

function startWorking() {
  isWorking = true;
  transcribeBtn.textContent = 'Working…';
  modelSelect.disabled = true;
  languageSelect.disabled = true;
  updateButtonState();
  statusEl.hidden = false;
}

function finishWorking() {
  isWorking = false;
  transcribeBtn.textContent = 'Transcribe';
  modelSelect.disabled = false;
  languageSelect.disabled = false;
  hideProgressBar();
  updateButtonState();
}

function showStatus(msg) {
  statusEl.hidden = false;
  statusEl.classList.remove('status--error');
  statusMessage.textContent = msg;
}

function showProgress(msg, percent) {
  showStatus(msg);
  statusBar.hidden = false;
  statusBarFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

function hideProgressBar() {
  statusBar.hidden = true;
  statusBarFill.style.width = '0%';
}

function setError(msg) {
  statusEl.hidden = false;
  statusEl.classList.add('status--error');
  statusMessage.textContent = `⚠️ ${msg}`;
  hideProgressBar();
}

function hideError() {
  statusEl.classList.remove('status--error');
}

function showTranscript(text) {
  resultEl.hidden = false;
  transcriptEl.value = text;
  resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function flashButton(btn, msg) {
  const original = btn.textContent;
  btn.textContent = msg;
  btn.disabled = true;
  setTimeout(() => {
    btn.textContent = original;
    btn.disabled = false;
  }, 1500);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
