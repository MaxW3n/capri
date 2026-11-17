import { createChat as createSceneChat } from './chat.js';

const info = {
  protocol: 'capri-local-v1', runtime: 'browser', size: '27B', layer: 20,
  min_dose: 0, max_dose: 1.2, default_dose: 0.74,
};
const encoder = new TextEncoder();

export function createChat(opts = {}) {
  const nativeFetch = opts.fetch ?? globalThis.fetch;
  const workerUrl = opts.browserWorkerUrl ?? globalThis.CAPRI_CONFIG?.browserWorkerUrl ?? '/capri/browser/inference.worker.js?v=1';
  let worker, pending, sequence = 0, loadStats;
  let browserSelected = false, destroyed = false;
  let status = 'loading', message = 'Starting the browser engine…';

  function terminate() {
    worker?.terminate();
    worker = undefined;
  }

  function fail(error) {
    status = 'error';
    message = error;
    pending?.finish(new Error(error));
    terminate();
  }

  function start() {
    status = 'loading';
    message = 'Starting the browser engine…';
    loadStats = undefined;
    const requestId = ++sequence;
    try {
      const instance = worker = new Worker(workerUrl, { type: 'module' });
      instance.onerror = (event) => {
        if (worker === instance) fail(event.message || 'The browser engine could not start.');
      };
      instance.onmessage = ({ data }) => {
        if (worker !== instance) return;
        if (data.type === 'fatal') fail(data.message);
        else if (data.requestId === requestId) {
          if (data.type === 'progress') message = data.message;
          else if (data.type === 'error') fail(data.error);
          else if (data.type === 'result') {
            const { weightCacheHits, weightCacheMisses, networkWeightBytes, loadMs } = data.value;
            loadStats = { weightCacheHits, weightCacheMisses, networkWeightBytes, loadMs };
            status = 'ready';
            message = 'Capri is ready in this browser.';
          }
        } else if (data.requestId === pending?.id) {
          if (data.type === 'token') pending.emit({ t: data.text });
          else if (data.type === 'error') {
            const error = new Error(data.error);
            error.recoverable = true;
            pending.finish(error);
          }
          else if (data.type === 'result') {
            const { decodeTokensPerSecond, generatedTokens, finish } = data.value;
            pending.emit({ done: true, tps: decodeTokensPerSecond, tokens: generatedTokens, finish });
            pending.finish();
          }
        }
      };
      instance.postMessage({ type: 'load', requestId });
    } catch (error) {
      fail(error.message);
    }
  }

  function generate({ messages, dose = info.default_dose }, signal) {
    const instance = worker, id = ++sequence;
    let cancel = () => {};
    const stream = new ReadableStream({
      start(controller) {
        const finish = (error, close = true) => {
          if (pending?.id !== id) return;
          pending = undefined;
          signal.removeEventListener('abort', abort);
          if (error) controller.error(error);
          else if (close) controller.close();
        };
        cancel = (error, close) => {
          if (pending?.id !== id) return;
          finish(error, close);
          if (worker === instance) instance.postMessage({ type: 'cancel', requestId: id });
        };
        const abort = () => cancel(new DOMException('Aborted', 'AbortError'));
        pending = {
          id, finish,
          emit(event) { controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`)); },
        };
        if (signal.aborted) { abort(); return; }
        signal.addEventListener('abort', abort, { once: true });
        try {
          instance.postMessage({ type: 'generate', requestId: id, messages: messages.slice(-11), dose,
            maxTokens: 96, temperature: 0, enableThinking: false });
        } catch (error) { finish(error); }
      },
      cancel() { cancel(undefined, false); },
    });
    return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
  }

  function fetch(url, options) {
    const { body, signal } = options;
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (!browserSelected) return nativeFetch(url, options);
    if (url.endsWith('/info')) {
      if (!worker && status !== 'error') start();
      return Response.json({ ...info, status, message, ready: status === 'ready', loadStats });
    }
    if (url.endsWith('/chat')) {
      if (!worker || status !== 'ready' || pending) return Response.json({ error: message }, { status: 503 });
      return generate(JSON.parse(body), signal);
    }
    throw new Error(`Unexpected browser engine request: ${url}`);
  }

  const chat = createSceneChat({ pollMs: 500, monitorMs: 1000, ...opts, fetch,
    browserSelected: () => browserSelected });
  const { connect, disconnect, destroy } = chat;
  Object.defineProperty(chat, 'runtime', { get: () => browserSelected ? 'browser' : 'native' });
  chat.connect = (...args) => {
    if (destroyed) return Promise.resolve(false);
    browserSelected = false;
    terminate();
    return connect(...args);
  };
  chat.connectBrowser = () => {
    if (destroyed) return Promise.resolve(false);
    browserSelected = true;
    if (status === 'error') status = 'loading';
    return connect();
  };
  chat.disconnect = () => {
    browserSelected = false;
    terminate();
    return disconnect();
  };
  chat.destroy = () => {
    destroyed = true;
    browserSelected = false;
    terminate();
    destroy();
  };
  return chat;
}
