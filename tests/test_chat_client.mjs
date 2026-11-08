import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setImmediate as tick } from 'node:timers/promises';

const source = await readFile(new URL('../web/3d/chat.js', import.meta.url), 'utf8');
const { createChat } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const info = (status = 'ready') => ({ protocol: 'capri-local-v1', status, ready: status === 'ready',
  message: status === 'loading' ? 'Downloading the model…' : 'Capri is ready.', size: '27B',
  min_dose: 0, max_dose: 1.2, default_dose: 0.74 });
const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const permissionDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
function setPermission(state) {
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
    permissions: { query: async () => ({ state }) },
  } });
}
async function fixture(t, handler, options = {}) {
  const previousFetch = globalThis.fetch;
  setPermission('prompt');
  const requests = [];
  globalThis.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), init });
    if (String(url).endsWith('/demo_lines.json')) return json([{ q: 'demo sample', a: 'RECORDED DEMO' }]);
    return handler(String(url), init);
  };
  const client = createChat({ manualLocal: true, pollMs: 1, monitorMs: 60000, ...options });
  t.after(async () => {
    client.destroy();
    await tick();
    globalThis.fetch = previousFetch;
    if (permissionDescriptor) Object.defineProperty(globalThis, 'navigator', permissionDescriptor);
    else delete globalThis.navigator;
  });
  await client.ready;
  return { client, requests };
}
function sse(text) {
  const bytes = new TextEncoder().encode('data: ' + JSON.stringify({ t: text }) + '\n\n' +
    'data: {"done":true,"tps":25,"tokens":4,"finish":"stop"}\n\n');
  return new Response(new ReadableStream({ start(controller) {
    for (let start = 0; start < bytes.length; start += 3) controller.enqueue(bytes.slice(start, start + 3));
    controller.close();
  } }), { headers: { 'Content-Type': 'text/event-stream' } });
}

test('hosted page starts in demo without requesting loopback access', async (t) => {
  const { client, requests } = await fixture(t, () => { throw new Error('Unexpected loopback request'); });
  await tick();
  assert.equal(client.mode, 'demo');
  assert.equal(client.connection.status, 'idle');
  assert.equal(requests.length, 1);
  assert.ok(requests.every(({ url }) => !url.startsWith('http://127.0.0.1')));
});

test('explicit connection resets demo history and streams real replies over loopback', async (t) => {
  let chatBody;
  const tokens = [];
  const { client, requests } = await fixture(t, (url, init) => {
    if (url.endsWith('/info')) return json(info());
    if (url.endsWith('/chat')) { chatBody = JSON.parse(init.body); return sse('Café, choom.'); }
    throw new Error('Unexpected URL ' + url);
  }, { onToken: (token) => tokens.push(token) });
  client.history.push({ role: 'user', content: 'Old demo question' });
  assert.equal(await client.connect(), true);
  assert.equal(client.mode, 'live');
  assert.deepEqual(client.history, []);
  const result = await client.send('A new real question', 0.74);
  assert.equal(result.reply, 'Café, choom.');
  assert.equal(tokens.join(''), result.reply);
  assert.equal(result.error, null);
  assert.deepEqual(chatBody.messages, [{ role: 'user', content: 'A new real question' }]);
  assert.equal(chatBody.dose, 0.74);
  for (const { url, init } of requests.filter((r) => r.url.startsWith('http://'))) {
    assert.equal(new URL(url).origin, 'http://127.0.0.1:8777');
    assert.equal(init.targetAddressSpace, 'loopback');
    assert.equal(init.mode, 'cors');
    assert.equal(init.credentials, 'omit');
  }
  client.disconnect();
  assert.equal(client.mode, 'demo');
  assert.equal(client.connection.status, 'idle');
  assert.deepEqual(client.history, []);
});

test('loading is displayed and polled without claiming a live connection', async (t) => {
  let attempts = 0;
  const changes = [];
  const { client } = await fixture(t, () => json(info(++attempts < 3 ? 'loading' : 'ready')),
    { onConnection: (state) => changes.push(state.status) });
  assert.equal(await client.connect(), true);
  assert.ok(changes.includes('loading'));
  assert.equal(changes.at(-1), 'ready');
  assert.equal(client.mode, 'live');
});

test('denied browser permission is explained without attempting a local request', async (t) => {
  const { client, requests } = await fixture(t, () => { throw new Error('Must not request a blocked permission'); });
  setPermission('denied');
  assert.equal(await client.connect(), false);
  assert.equal(client.connection.status, 'denied');
  assert.match(client.connection.message, /Site settings/);
  assert.equal(client.mode, 'disconnected');
  assert.equal(requests.length, 1);
});

test('a missing server or wrong protocol stays disconnected until an explicit demo choice', async (t) => {
  let validProtocol = false;
  const { client } = await fixture(t, () => {
    if (!validProtocol) throw new TypeError('Failed to fetch');
    return json({ ...info(), protocol: 'unrelated-service' });
  });
  assert.equal(await client.connect(), false);
  assert.equal(client.mode, 'disconnected');
  assert.match(client.connection.message, /Terminal/);
  validProtocol = true;
  assert.equal(await client.connect(), false);
  assert.match(client.connection.message, /different service/);
  client.disconnect();
  assert.equal(client.mode, 'demo');
});

test('a failed live request never substitutes a recorded demo answer', async (t) => {
  const chunks = [];
  const { client, requests } = await fixture(t, (url) => {
    if (url.endsWith('/info')) return json(info());
    throw new TypeError('Connection closed');
  }, { onToken: (chunk) => chunks.push(chunk) });
  await client.connect();
  const result = await client.send('demo sample');
  assert.equal(result.reply, '');
  assert.match(result.error, /Connection lost/);
  assert.equal(client.mode, 'disconnected');
  assert.equal(client.connection.status, 'lost');
  assert.deepEqual(chunks, []);
  assert.deepEqual(client.history, []);
  assert.equal(requests.filter(({ url }) => url.endsWith('/chat')).length, 1);
});

test('disconnect cancels a pending connection without a stale callback changing demo mode', async (t) => {
  let started = false;
  const { client } = await fixture(t, (_url, init) => new Promise((_resolve, reject) => {
    started = true;
    init.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  }));
  const pending = client.connect();
  while (!started) await tick();
  client.disconnect();
  assert.equal(await pending, false);
  assert.equal(client.mode, 'demo');
  assert.equal(client.connection.status, 'idle');
});

test('older Chromium enum falls back to local without extra network attempts', async (t) => {
  const originalRequest = globalThis.Request;
  globalThis.Request = class extends originalRequest {
    constructor(url, options) {
      if (options?.targetAddressSpace === 'loopback') throw new TypeError('Unsupported address-space enum');
      super(url, options);
    }
  };
  t.after(() => { globalThis.Request = originalRequest; });
  const { client, requests } = await fixture(t, () => json(info()));
  assert.equal(await client.connect(), true);
  const localRequests = requests.filter(({ url }) => url.startsWith('http://127.0.0.1'));
  assert.equal(localRequests.length, 1);
  assert.equal(localRequests[0].init.targetAddressSpace, 'local');
});
