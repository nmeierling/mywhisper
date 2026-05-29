// Web Worker: loads the Whisper model and runs transcription off the main
// thread so the UI stays responsive during inference.
import { env, pipeline } from '@huggingface/transformers';

// Resolve models ONLY from local files — never the network. In dev they are
// served at /models/ from ~/.mywhisper (Vite plugin); in a build they are
// bundled into dist/models/. The Hugging Face hub is never contacted.
// `localModelPath` is set per request from the page's base URL (see below): a
// worker resolves relative URLs against its own script location, not the page,
// so we need an absolute base for the app to work in a deploy subdirectory.
env.allowLocalModels = true;
env.allowRemoteModels = false;

// Pin the precision so the on-disk filenames are deterministic. `q8` maps to
// the "_quantized" ONNX files — keep scripts/download-models.js (ONNX_KEEP) in
// sync with this value.
const DTYPE = 'q8';

// Cache one pipeline instance per model id. Switching models lazily builds a
// new one; previously loaded models stay cached for instant reuse.
class PipelineFactory {
  static instances = new Map();

  static async get(model, progress_callback) {
    if (!this.instances.has(model)) {
      this.instances.set(
        model,
        pipeline('automatic-speech-recognition', model, {
          dtype: DTYPE,
          progress_callback,
        }),
      );
    }
    return this.instances.get(model);
  }
}

self.addEventListener('message', async (event) => {
  const { audio, model, language, modelBase } = event.data;

  try {
    // Absolute URL of the /models/ directory, computed by the main thread from
    // the page location so subdirectory deploys resolve correctly.
    if (modelBase) env.localModelPath = modelBase;

    const transcriber = await PipelineFactory.get(model, (progress) => {
      // Forward model-download progress to the UI.
      self.postMessage({ status: 'progress', data: progress });
    });

    self.postMessage({ status: 'transcribing' });

    const options = {
      // Whisper handles 30s windows; chunk + stride lets us process audio of
      // any length and stitch the pieces back together.
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: false,
    };

    if (language && language !== 'auto') {
      options.language = language;
      options.task = 'transcribe';
    }

    const output = await transcriber(audio, options);

    self.postMessage({ status: 'complete', text: output.text.trim() });
  } catch (error) {
    let message = error?.message ?? String(error);
    // Models load locally only; a missing file usually means it wasn't fetched.
    if (/\b(404|not.*found|could not locate|failed to fetch)\b/i.test(message)) {
      message = `Model "${model}" is not available locally. Run \`npm run download-models ${model}\` and reload.`;
    }
    self.postMessage({ status: 'error', message });
  }
});
