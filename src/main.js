import './style.css';
import { LANGUAGES } from './languages.js';
import { MODELS } from './config.js';
import {
  putJobMeta,
  getAllJobMeta,
  putAudio,
  getAudio,
  deleteJob,
  requestPersistentStorage,
} from './db.js';

// Whisper expects mono PCM audio at 16 kHz.
const WHISPER_SAMPLE_RATE = 16000;

// Absolute URL of the local /models/ directory, resolved from the page's base
// so it's correct at a domain root or a subdirectory. Passed to the worker,
// which can't resolve page-relative paths itself.
const MODEL_BASE = new URL('./models/', document.baseURI).href;

// --- Element refs -----------------------------------------------------------
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const modelSelect = document.getElementById('model-select');
const languageSelect = document.getElementById('language-select');
const backendSelect = document.getElementById('backend-select');
const transcribeBtn = document.getElementById('transcribe-btn');
const statusEl = document.getElementById('status');
const statusMessage = document.getElementById('status-message');
const statusBar = statusEl.querySelector('.status__bar');
const statusBarFill = document.getElementById('status-bar-fill');
const queueSection = document.getElementById('queue-section');
const queueEl = document.getElementById('queue');
const queueSummary = document.getElementById('queue-summary');
const copyAllBtn = document.getElementById('copy-all-btn');
const downloadAllBtn = document.getElementById('download-all-btn');
const clearDoneBtn = document.getElementById('clear-done-btn');
const timestampsToggle = document.getElementById('timestamps-toggle');
const timestampsToggleLabel = document.getElementById('timestamps-toggle-label');

// --- State -------------------------------------------------------------------
// jobs: { id, order, file, status, processed, total, text, segments,
//         processedUpTo, settings, error, startTime, audioUrl, activeSeg, els }
// status: pending | decoding | transcribing | done | error | canceled
const jobs = [];
let orderSeq = 0; // monotonic ordering for the queue (persisted per job)
let running = false; // is the queue currently being processed?
let worker = null;
let lastWarmKey = null; // model|backend last asked to warm-load
let persistRequested = false;

let currentResolve = null; // resolves the in-flight transcribeOnWorker promise
let currentJob = null;

// Decoded audio cached by job id (Promise<Float32Array>) — enables decoding the
// next file while the current one transcribes.
const decoded = new Map();

const AUDIO_RE = /\.(m4a|mp3|wav|ogg|flac|aac|webm|opus|mp4|mpga|mpeg)$/i;
const isAudio = (file) => file.type.startsWith('audio/') || AUDIO_RE.test(file.name);

// --- Populate dropdowns -----------------------------------------------------
for (const m of MODELS) {
  const opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = `${m.label}, ${m.size}`;
  modelSelect.appendChild(opt);
}
modelSelect.value = 'Xenova/whisper-base';

for (const [value, label] of LANGUAGES) {
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = label;
  languageSelect.appendChild(opt);
}

// --- Worker setup ------------------------------------------------------------
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.addEventListener('message', onWorkerMessage);
  worker.addEventListener('error', (e) => {
    setError(`Worker error: ${e.message}`);
    if (currentJob) {
      currentJob.status = 'error';
      currentJob.error = e.message;
      updateJobRow(currentJob);
      resolveCurrent();
    }
  });
  return worker;
}

function onWorkerMessage(event) {
  const data = event.data;
  if (data.scope === 'model') return onModelMessage(data);
  if (data.scope === 'job') return onJobMessage(data);
}

function onModelMessage(data) {
  switch (data.status) {
    case 'progress': {
      const p = data.data;
      if (p?.status === 'progress' && typeof p.progress === 'number') {
        showProgress(`Loading model: ${p.file ?? ''}`, p.progress);
      } else if (p?.status === 'initiate' || p?.status === 'download') {
        showStatus('Loading model…');
      }
      break;
    }
    case 'ready':
      // Don't clobber active transcription status.
      if (!running) showStatus(`Model ready (${data.device}).`);
      hideProgressBar();
      break;
    case 'error':
      setError(data.message);
      break;
  }
}

function onJobMessage(data) {
  const job = jobs.find((j) => j.id === data.jobId);
  if (!job) return;
  switch (data.status) {
    case 'transcribing':
      hideStatus(); // model is loaded; global line no longer needed
      job.status = 'transcribing';
      job.total = data.total ?? 0;
      job.startTime = performance.now();
      updateJobRow(job);
      persistJob(job);
      break;
    case 'update':
      job.processed = data.processed;
      job.total = data.total;
      job.text = data.text;
      updateJobRow(job);
      break;
    case 'checkpoint':
      // Window boundary reached — persist progress so a refresh can resume here.
      job.segments = data.segments ?? [];
      job.processedUpTo = data.processedUpTo ?? 0;
      persistJob(job);
      break;
    case 'complete':
      job.status = 'done';
      job.text = data.text;
      job.segments = data.segments ?? [];
      job.processedUpTo = job.total;
      updateJobRow(job);
      persistJob(job);
      resolveCurrent();
      break;
    case 'canceled':
      job.status = 'canceled';
      updateJobRow(job);
      persistJob(job);
      resolveCurrent();
      break;
    case 'error':
      job.status = 'error';
      job.error = data.message;
      updateJobRow(job);
      persistJob(job);
      resolveCurrent();
      break;
  }
  updateSummary();
}

function resolveCurrent() {
  const resolve = currentResolve;
  currentResolve = null;
  currentJob = null;
  if (resolve) resolve();
}

// --- Persistence -------------------------------------------------------------
// Build the small metadata record stored in IndexedDB (no audio bytes).
function jobMeta(job) {
  return {
    id: job.id,
    order: job.order,
    name: job.file.name,
    type: job.file.type,
    status: job.status,
    text: job.text,
    segments: job.segments,
    processedUpTo: job.processedUpTo,
    settings: job.settings,
  };
}

function persistJob(job) {
  putJobMeta(jobMeta(job)).catch(() => {}); // best-effort; never block the UI
}

function makeJob(file, fields = {}) {
  return {
    id: fields.id ?? crypto.randomUUID(),
    order: fields.order ?? orderSeq++,
    file,
    status: 'pending',
    processed: 0,
    total: 0,
    text: '',
    segments: [],
    processedUpTo: 0,
    settings: null,
    error: '',
    startTime: 0,
    audioUrl: null,
    activeSeg: -1,
    ...fields,
  };
}

// --- File handling -----------------------------------------------------------
function addFiles(fileList) {
  if (!persistRequested) {
    persistRequested = true;
    requestPersistentStorage(); // ask for durable storage (best-effort)
  }
  let added = 0;
  for (const file of fileList) {
    if (!isAudio(file)) continue;
    const job = makeJob(file);
    jobs.push(job);
    createJobRow(job);
    persistJob(job);
    // Store the audio bytes once; needed to resume / re-listen after a reload.
    putAudio(job.id, file).catch(() => {
      // Likely a storage-quota error: keep going (works this session, but this
      // file won't survive a reload).
    });
    added += 1;
  }
  if (added === 0) {
    setError('No audio files found in that drop.');
    return;
  }
  hideError();
  queueSection.hidden = false;
  updateButton();
  updateSummary();
  warmLoad(); // start loading the model now so it's ready by the time we run
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('change', (e) => {
  addFiles(e.target.files);
  fileInput.value = ''; // allow re-selecting the same file
});

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dropzone--dragover');
  }),
);
['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dropzone--dragover');
  }),
);
dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

modelSelect.addEventListener('change', warmLoad);
backendSelect.addEventListener('change', warmLoad);

function warmLoad() {
  if (running) return;
  const key = `${modelSelect.value}|${backendSelect.value}`;
  if (key === lastWarmKey) return;
  lastWarmKey = key;
  getWorker().postMessage({
    type: 'load',
    model: modelSelect.value,
    backend: backendSelect.value,
    modelBase: MODEL_BASE,
  });
}

// --- Audio decoding ----------------------------------------------------------
// Decode any browser-supported audio file into a mono Float32Array at 16 kHz.
async function decodeAudio(file) {
  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const decodeCtx = new AudioCtx();
  let decodedBuffer;
  try {
    decodedBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  } finally {
    decodeCtx.close();
  }

  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(decodedBuffer.duration * WHISPER_SAMPLE_RATE),
    WHISPER_SAMPLE_RATE,
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = decodedBuffer;
  source.connect(offlineCtx.destination);
  source.start();
  const rendered = await offlineCtx.startRendering();
  return rendered.getChannelData(0);
}

function ensureDecoded(job) {
  if (!decoded.has(job.id)) decoded.set(job.id, decodeAudio(job.file));
  return decoded.get(job.id);
}

// --- Queue runner ------------------------------------------------------------
transcribeBtn.addEventListener('click', runQueue);

async function runQueue() {
  if (running) return;
  const firstPending = jobs.find((j) => j.status === 'pending');
  if (!firstPending) return;

  running = true;
  setControlsDisabled(true);
  updateButton();

  let job;
  while ((job = jobs.find((j) => j.status === 'pending'))) {
    try {
      job.status = 'decoding';
      updateJobRow(job);

      const audio = await ensureDecoded(job);

      // Pipeline: start decoding the next file while this one transcribes.
      const next = jobs.find((j) => j.status === 'pending' && j.id !== job.id);
      if (next) ensureDecoded(next).catch(() => {});

      await transcribeOnWorker(job, audio); // resolves on complete/error/canceled
    } catch (err) {
      job.status = 'error';
      job.error = err?.message ?? String(err);
      updateJobRow(job);
    } finally {
      decoded.delete(job.id); // free the (now-transferred) audio
    }
  }

  running = false;
  setControlsDisabled(false);
  updateButton();
  updateSummary();
}

function transcribeOnWorker(job, audio) {
  return new Promise((resolve) => {
    currentJob = job;
    currentResolve = resolve;

    // A resumed job keeps the settings it started with; a fresh job adopts the
    // current UI selection (and remembers it, so a resume stays consistent).
    if (!job.settings) {
      job.settings = {
        model: modelSelect.value,
        language: languageSelect.value,
        backend: backendSelect.value,
      };
    }

    // Resume only the unfinished tail if we have a checkpoint for this job.
    const resumeFrom = job.processedUpTo > 0 ? job.processedUpTo : 0;
    const priorSegments = resumeFrom > 0 ? job.segments.filter((s) => s.start < resumeFrom) : [];

    // Transfer the audio buffer (zero-copy) — `audio` is unusable afterward,
    // which is fine since we drop it.
    getWorker().postMessage(
      {
        type: 'transcribe',
        jobId: job.id,
        audio,
        model: job.settings.model,
        language: job.settings.language,
        backend: job.settings.backend,
        modelBase: MODEL_BASE,
        resumeFrom,
        priorSegments,
      },
      [audio.buffer],
    );
  });
}

// --- Per-job actions ---------------------------------------------------------
function removeJob(job) {
  if (job === currentJob) return abortJob(job); // can't drop the in-flight one
  const i = jobs.indexOf(job);
  if (i !== -1) jobs.splice(i, 1);
  decoded.delete(job.id);
  if (job.audioUrl) URL.revokeObjectURL(job.audioUrl);
  job.els?.root.remove();
  deleteJob(job.id).catch(() => {});
  if (jobs.length === 0) queueSection.hidden = true;
  updateButton();
  updateSummary();
}

function abortJob(job) {
  getWorker().postMessage({ type: 'abort', jobId: job.id });
}

function retryJob(job) {
  job.status = 'pending';
  job.text = '';
  job.segments = [];
  job.error = '';
  job.processed = 0;
  job.processedUpTo = 0; // restart from the beginning
  job.activeSeg = -1;
  updateJobRow(job);
  persistJob(job);
  updateButton();
  updateSummary();
  if (running) return; // active loop will pick it up
}

// --- Bulk actions ------------------------------------------------------------
copyAllBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(allTranscripts());
    flashButton(copyAllBtn, 'Copied!');
  } catch {
    flashButton(copyAllBtn, 'Copy failed');
  }
});

downloadAllBtn.addEventListener('click', () => {
  downloadText('transcripts.txt', allTranscripts());
});

clearDoneBtn.addEventListener('click', () => {
  for (const job of [...jobs]) {
    if (['done', 'error', 'canceled'].includes(job.status)) removeJob(job);
  }
});

function jobText(job, withTimestamps = false) {
  if (withTimestamps && job.segments.length) {
    return job.segments
      .map((s) => `[${formatTime(s.start)}] ${s.text.trim()}`)
      .join('\n');
  }
  return job.text || job.segments.map((s) => s.text).join('').trim();
}

function allTranscripts() {
  const ts = timestampsToggle.checked;
  return jobs
    .filter((j) => j.status === 'done')
    .map((j) => `# ${j.file.name}\n\n${jobText(j, ts)}`)
    .join('\n\n\n');
}

// --- Queue rendering ---------------------------------------------------------
function createJobRow(job) {
  const root = document.createElement('li');
  root.className = 'job';

  const head = document.createElement('div');
  head.className = 'job__head';

  const name = document.createElement('span');
  name.className = 'job__name';
  name.textContent = job.file.name;
  name.title = `${job.file.name} (${formatBytes(job.file.size)})`;

  const status = document.createElement('span');
  status.className = 'job__status';

  const actions = document.createElement('div');
  actions.className = 'job__actions';

  head.append(name, status, actions);

  const bar = document.createElement('div');
  bar.className = 'job__bar';
  bar.hidden = true;
  const fill = document.createElement('div');
  fill.className = 'job__bar-fill';
  bar.appendChild(fill);

  const result = document.createElement('div');
  result.className = 'job__result';
  result.hidden = true;

  // Live streaming text (shown while transcribing, and as a fallback if a
  // finished job produced no timestamped segments).
  const live = document.createElement('textarea');
  live.className = 'job__text';
  live.readOnly = true;
  live.spellcheck = false;

  // Playback of the original file, for re-listening to a segment.
  job.audioUrl = URL.createObjectURL(job.file);
  const audio = document.createElement('audio');
  audio.className = 'job__audio';
  audio.controls = true;
  audio.preload = 'none';
  audio.src = job.audioUrl;
  audio.hidden = true;
  audio.addEventListener('timeupdate', () => highlightActiveSegment(job));

  // Clickable timestamped segments.
  const segs = document.createElement('ol');
  segs.className = 'job__segments';
  segs.hidden = true;

  result.append(live, audio, segs);
  root.append(head, bar, result);
  queueEl.appendChild(root);

  job.els = { root, status, actions, bar, fill, result, live, audio, segs };
  updateJobRow(job);
}

function button(label, className, onClick) {
  const b = document.createElement('button');
  b.className = `btn btn--ghost ${className}`;
  b.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}

function updateJobRow(job) {
  const { els } = job;
  if (!els) return;

  // Status label
  const labels = {
    pending: job.processedUpTo > 0 ? `Pending — resumes from ${formatTime(job.processedUpTo)}` : 'Pending',
    decoding: 'Decoding…',
    transcribing: progressLabel(job),
    done: 'Done',
    error: `Error: ${job.error}`,
    canceled: 'Canceled',
  };
  els.status.textContent = labels[job.status] ?? job.status;
  els.status.dataset.state = job.status;

  // Progress bar (decoding/transcribing only)
  const showBar = job.status === 'decoding' || job.status === 'transcribing';
  els.bar.hidden = !showBar;
  if (job.status === 'transcribing' && job.total) {
    els.fill.style.width = `${Math.round((job.processed / job.total) * 100)}%`;
  } else if (job.status === 'decoding') {
    els.fill.style.width = '0%';
  }

  // Transcript: live streaming text while running; a clickable timestamped
  // segment list (with audio playback) once done.
  const done = job.status === 'done';
  const hasSegments = done && job.segments.length > 0;
  els.result.hidden = !(job.status === 'transcribing' || done || job.text);
  els.live.hidden = hasSegments;
  els.audio.hidden = !hasSegments;
  els.segs.hidden = !hasSegments;

  if (hasSegments) {
    renderSegments(job);
  } else if (els.live.value !== job.text) {
    els.live.value = job.text;
    if (job.status === 'transcribing') els.live.scrollTop = els.live.scrollHeight;
  }

  // Contextual action buttons
  els.actions.replaceChildren(...rowActions(job));
}

// Render the timestamped segment list. Each segment seeks the audio on click.
function renderSegments(job) {
  const rows = job.segments.map((seg, i) => {
    const li = document.createElement('li');
    li.className = 'seg';

    const time = document.createElement('button');
    time.className = 'seg__time';
    time.textContent = formatTime(seg.start);
    time.title = `Play from ${formatTime(seg.start)}`;

    const text = document.createElement('span');
    text.className = 'seg__text';
    text.textContent = seg.text;

    li.append(time, text);
    li.addEventListener('click', () => seekTo(job, seg.start));
    return li;
  });
  job.els.segs.replaceChildren(...rows);
  job.activeSeg = -1;
}

function seekTo(job, start) {
  const audio = job.els.audio;
  audio.currentTime = start;
  audio.play().catch(() => {}); // ignore autoplay rejection
}

// Highlight the segment currently playing (called on audio timeupdate).
function highlightActiveSegment(job) {
  const t = job.els.audio.currentTime;
  const segs = job.segments;
  let active = -1;
  for (let i = 0; i < segs.length; i += 1) {
    if (t >= segs[i].start && (segs[i].end == null || t < segs[i].end)) {
      active = i;
      break;
    }
  }
  if (active === job.activeSeg) return;
  const rows = job.els.segs.children;
  rows[job.activeSeg]?.classList.remove('seg--active');
  rows[active]?.classList.add('seg--active');
  job.activeSeg = active;
}

function rowActions(job) {
  switch (job.status) {
    case 'pending':
      return [button('Remove', 'job__btn', () => removeJob(job))];
    case 'transcribing':
      return [button('Cancel', 'job__btn', () => abortJob(job))];
    case 'done':
      return [
        button('Copy', 'job__btn', (e) => copyJob(job, e.target)),
        button('Download', 'job__btn', () => downloadJob(job)),
        button('✕', 'job__btn job__btn--icon', () => removeJob(job)),
      ];
    case 'error':
    case 'canceled':
      return [
        button('Retry', 'job__btn', () => retryJob(job)),
        button('✕', 'job__btn job__btn--icon', () => removeJob(job)),
      ];
    default:
      return [];
  }
}

function progressLabel(job) {
  if (!job.total) return 'Transcribing…';
  const fraction = Math.min(1, job.processed / job.total);
  const percent = Math.round(fraction * 100);
  let eta = '';
  if (job.startTime && fraction > 0.02 && fraction < 1) {
    const elapsed = (performance.now() - job.startTime) / 1000;
    eta = ` · ~${formatTime((elapsed / fraction) * (1 - fraction))} left`;
  }
  return `Transcribing… ${formatTime(job.processed)} / ${formatTime(job.total)} (${percent}%)${eta}`;
}

function copyJob(job, btn) {
  navigator.clipboard.writeText(jobText(job, timestampsToggle.checked)).then(
    () => flashButton(btn, 'Copied!'),
    () => flashButton(btn, 'Failed'),
  );
}

function downloadJob(job) {
  const base = job.file.name.replace(/\.[^.]+$/, '') || 'transcript';
  downloadText(`${base}.txt`, jobText(job, timestampsToggle.checked));
}

// --- UI helpers --------------------------------------------------------------
function updateButton() {
  const pending = jobs.filter((j) => j.status === 'pending').length;
  if (running) {
    transcribeBtn.textContent = 'Transcribing…';
    transcribeBtn.disabled = true;
  } else if (pending > 0) {
    transcribeBtn.textContent = `Transcribe ${pending} file${pending > 1 ? 's' : ''}`;
    transcribeBtn.disabled = false;
  } else {
    transcribeBtn.textContent = 'Transcribe';
    transcribeBtn.disabled = true;
  }
}

function updateSummary() {
  const total = jobs.length;
  const done = jobs.filter((j) => j.status === 'done').length;
  queueSummary.textContent = total ? `Queue — ${done}/${total} done` : 'Queue';
  const anyDone = jobs.some((j) => j.status === 'done');
  copyAllBtn.hidden = !anyDone;
  downloadAllBtn.hidden = !anyDone;
  timestampsToggleLabel.hidden = !anyDone;
  clearDoneBtn.hidden = !jobs.some((j) => ['done', 'error', 'canceled'].includes(j.status));
}

function setControlsDisabled(disabled) {
  modelSelect.disabled = disabled;
  languageSelect.disabled = disabled;
  backendSelect.disabled = disabled;
}

function showStatus(msg) {
  statusEl.hidden = false;
  statusEl.classList.remove('status--error');
  statusMessage.textContent = msg;
}

function hideStatus() {
  statusEl.hidden = true;
  hideProgressBar();
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
  if (statusEl.classList.contains('status--error')) statusEl.hidden = true;
  statusEl.classList.remove('status--error');
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

function downloadText(filename, content) {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Seconds → "m:ss" (or "h:mm:ss" for long audio).
function formatTime(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  return `${m}:${String(sec).padStart(2, '0')}`;
}

// --- Restore on load ---------------------------------------------------------
// Rebuild the queue from IndexedDB. Finished jobs come back fully usable; a job
// interrupted mid-transcription comes back as "pending" with its checkpoint, so
// running the queue resumes it from the last completed window.
async function restore() {
  let metas;
  try {
    metas = await getAllJobMeta();
  } catch {
    return; // no persisted state (or IndexedDB unavailable)
  }
  if (!metas?.length) return;

  metas.sort((a, b) => a.order - b.order);
  for (const m of metas) {
    const rec = await getAudio(m.id).catch(() => null);
    if (!rec?.blob) {
      // Audio bytes missing (quota failure, or storage cleared) — we can't
      // resume or re-listen, so drop the stale metadata.
      deleteJob(m.id).catch(() => {});
      continue;
    }
    const file = new File([rec.blob], m.name, { type: m.type });
    // An interrupted run was mid-transcribe/decode; reopen it as pending so the
    // checkpoint (processedUpTo + segments) drives a resume.
    const status = m.status === 'transcribing' || m.status === 'decoding' ? 'pending' : m.status;
    const job = makeJob(file, {
      id: m.id,
      order: m.order,
      status,
      text: m.text ?? '',
      segments: m.segments ?? [],
      processedUpTo: m.processedUpTo ?? 0,
      settings: m.settings ?? null,
    });
    jobs.push(job);
    orderSeq = Math.max(orderSeq, m.order + 1);
    createJobRow(job);
  }

  if (jobs.length) {
    // Restore the dropdowns from the most recent job's settings so warm-load
    // (and the visible selection) match what was in use.
    const recent = jobs.filter((j) => j.settings).sort((a, b) => b.order - a.order)[0];
    if (recent) {
      if (MODELS.some((m) => m.id === recent.settings.model)) modelSelect.value = recent.settings.model;
      languageSelect.value = recent.settings.language;
      backendSelect.value = recent.settings.backend;
    }
    queueSection.hidden = false;
    updateButton();
    updateSummary();
    warmLoad();
  }
}

// Warn before leaving while work is in progress (a reload kills the worker).
window.addEventListener('beforeunload', (e) => {
  if (running) {
    e.preventDefault();
    e.returnValue = '';
  }
});

restore();
