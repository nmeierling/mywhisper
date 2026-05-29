// Minimal IndexedDB wrapper for persisting the queue across reloads.
//
// Two stores so we never rewrite large blobs on frequent metadata updates:
//   • 'jobs'  — small per-job metadata (status, segments, processedUpTo, …),
//                rewritten often (e.g. on every checkpoint).
//   • 'audio' — the original file Blob, written once when a job is added.
const DB_NAME = 'mywhisper';
const DB_VERSION = 1;

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('jobs')) db.createObjectStore('jobs', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('audio')) db.createObjectStore('audio', { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

// Run `fn(store)` in a transaction; resolves with the (optional) request result.
function run(storeName, mode, fn) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(storeName, mode);
        const req = fn(tx.objectStore(storeName));
        tx.oncomplete = () => resolve(req ? req.result : undefined);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      }),
  );
}

export function putJobMeta(meta) {
  return run('jobs', 'readwrite', (s) => s.put(meta));
}

export function getAllJobMeta() {
  return run('jobs', 'readonly', (s) => s.getAll());
}

export function putAudio(id, blob) {
  return run('audio', 'readwrite', (s) => s.put({ id, blob }));
}

export function getAudio(id) {
  return run('audio', 'readonly', (s) => s.get(id));
}

export async function deleteJob(id) {
  await run('jobs', 'readwrite', (s) => s.delete(id));
  await run('audio', 'readwrite', (s) => s.delete(id)).catch(() => {});
}

// Ask the browser to keep our storage durable (not evicted under pressure).
export async function requestPersistentStorage() {
  try {
    return navigator.storage?.persist ? await navigator.storage.persist() : false;
  } catch {
    return false;
  }
}
