import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { defineConfig } from 'vite';

// Root directory where `npm run download-models` stores model files.
export const MODELS_DIR = path.join(os.homedir(), '.mywhisper', 'models');

const MIME = {
  '.json': 'application/json',
  '.txt': 'text/plain; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.onnx_data': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

// Makes the locally downloaded models available to the app:
//   • dev (`npm run dev`): serves them from ~/.mywhisper/models at /models/
//   • build (`npm run build`): copies them into dist/models/ so the static
//     site is fully self-contained (preview/production serve them statically)
function serveLocalModels() {
  let outModelsDir = null;

  const middleware = (req, res, next) => {
    if (!req.url || !req.url.startsWith('/models/')) return next();

    const rel = decodeURIComponent(req.url.slice('/models/'.length).split('?')[0]);
    const filePath = path.join(MODELS_DIR, rel);

    // Guard against path traversal outside the models directory.
    if (!filePath.startsWith(MODELS_DIR)) {
      res.statusCode = 403;
      return res.end('Forbidden');
    }

    fs.stat(filePath, (err, stat) => {
      if (err || !stat.isFile()) {
        res.statusCode = 404;
        return res.end('Not found');
      }
      res.setHeader('Content-Type', MIME[path.extname(filePath)] || 'application/octet-stream');
      res.setHeader('Content-Length', stat.size);
      res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      fs.createReadStream(filePath).pipe(res);
    });
  };

  return {
    name: 'serve-local-models',
    configResolved(config) {
      outModelsDir = path.resolve(config.root, config.build.outDir, 'models');
    },
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    // Bundle the downloaded models into the build output.
    writeBundle() {
      if (!fs.existsSync(MODELS_DIR) || fs.readdirSync(MODELS_DIR).length === 0) {
        this.warn(
          `No models found in ${MODELS_DIR}. The build will have no models — ` +
            'run `npm run download-models` and rebuild, or the app will not work.',
        );
        return;
      }
      fs.cpSync(MODELS_DIR, outModelsDir, { recursive: true });
      console.log(`\n  copied local models → ${path.relative(process.cwd(), outModelsDir)}`);
    },
  };
}

// Cross-origin isolation (COOP + COEP) enables SharedArrayBuffer, which
// onnxruntime-web needs to run its multi-threaded WASM backend — without it
// model loading fails outright in some browsers. A static host must send these
// same headers (see README); GitHub Pages can't, so use the coi-serviceworker
// shim there.
function crossOriginIsolation() {
  const setHeaders = (_req, res, next) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    next();
  };
  return {
    name: 'cross-origin-isolation',
    configureServer(server) {
      server.middlewares.use(setHeaders);
    },
    configurePreviewServer(server) {
      server.middlewares.use(setHeaders);
    },
  };
}

// `base: './'` keeps asset paths relative so the built site works from any
// subdirectory on a static host (GitHub Pages project sites, etc.).
export default defineConfig({
  base: './',
  plugins: [crossOriginIsolation(), serveLocalModels()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // Transformers.js ships its own wasm/onnx runtime; let Vite pre-bundle it.
    exclude: ['@huggingface/transformers'],
  },
});
