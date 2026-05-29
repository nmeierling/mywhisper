// Single source of truth for models and per-backend precision. Imported by
// both the worker (browser) and scripts/download-models.js (Node), so the files
// downloaded to disk always match what the app asks for at runtime.

// Selectable Whisper models. `size` is the approximate on-disk footprint of the
// q8 (WASM) weights; WebGPU variants add more (see below).
export const MODELS = [
  { id: 'Xenova/whisper-tiny', label: 'Tiny — fastest', size: '~40 MB' },
  { id: 'Xenova/whisper-base', label: 'Base — balanced', size: '~80 MB' },
  { id: 'Xenova/whisper-small', label: 'Small — most accurate', size: '~250 MB' },
];

// The Whisper sub-models whose ONNX weights are loaded at runtime.
export const SUBMODELS = ['encoder_model', 'decoder_model_merged'];

// dtype value → ONNX filename suffix (Transformers.js convention).
export const DTYPE_SUFFIX = {
  fp32: '',
  fp16: '_fp16',
  q4: '_q4',
  q8: '_quantized',
};

// Precision per backend:
//   • WASM: q8 everywhere — small and broadly compatible.
//   • WebGPU: fp32 encoder + q4 decoder — the configuration the Transformers.js
//     Whisper WebGPU demo uses; fast with good quality, modest VRAM.
export const DTYPE = {
  wasm: { encoder_model: 'q8', decoder_model_merged: 'q8' },
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
};

// Resolve the ONNX file paths a given backend needs for a model, e.g.
// ['onnx/encoder_model.onnx', 'onnx/decoder_model_merged_q4.onnx'].
export function onnxFilesFor(backend) {
  return SUBMODELS.map(
    (sub) => `onnx/${sub}${DTYPE_SUFFIX[DTYPE[backend][sub]]}.onnx`,
  );
}
