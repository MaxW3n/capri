// chat.js: the /chat SSE client, the conversation history and the demo-mode fallback.
//
// createChat({ onToken, onStart, onDone, onError, onMode }) ->
//   { send(text, dose) -> Promise<{ reply, aborted, error, restore? }>, stop(), history, mode, info, ready }
//
// At start it asks GET /info (1.5 s timeout). If the model server answers, mode = 'live' and replies come from
// POST /chat (SSE "data: {json}\n\n": {t} text chunks, then {done, tps, finish, tokens} or {error}). Otherwise
// mode = 'demo' and replies come from demo_lines.json (real Capri answers from the 27B test set), picked by word
// overlap with the question and streamed as fake tokens. A network failure in live mode falls back to demo mode
// and answers the same message from the demo lines, so the page never just dies.
//
// Callbacks: onStart(), onToken(textChunk), onDone({ reply, tps, finish, tokens, mode }), onError(message),
// onMode(mode, info). Extras (additive): `ready` (Promise resolved once the mode is known), `lastTps`, `reset()`.

const STOP = new Set(('a an the and or but if to of in on at for with by from is are was were be been am do does did i you he she it we ' +
  'they me my your our its this that these those how what why when where who which can could would should will just so ' +
  'any some there here have has had get got not no yes about into out up than then them his her him im ive dont u ur').split(' '));
const words = (s) => (String(s).toLowerCase().match(/[a-z0-9']+/g) || [])
  .map((w) => w.replace(/'s$|'/g, '')).filter((w) => w.length > 1 && !STOP.has(w))
  .map((w) => (w.length > 4 ? w.replace(/(ing|ed|es|s)$/, '') : w));

// demo-only lines for bare greetings / "who are you" (the test set has none)
const DEMO_GREET = ["Yeah, hi. What do you need?", "Hey. Make it quick, choom.", "Hi. You gonna say something or just stare?"];
const DEMO_WHO = ["Capri. Edgerunner. That's all you get.", "Who's asking? ...Capri. Happy?"];
// demo-only: when nothing in the recorded lines fits, deflect in character instead of answering something unrelated
const DEMO_DEFLECT = ["Not my department. Ask me something weirder.", "Huh. No clue. Ask the real me, she's smarter. Start server.py.",
  "That's a lot of words. Try me on something I care about.", "Pass. Next question.", "Dunno. Ask me about goldfish or the moon or something."];

export function createChat(opts = {}) {
  const config = globalThis.CAPRI_CONFIG || {};
  const manualLocal = opts.manualLocal ?? config.manualLocal ?? false;
  const base = String(opts.apiBase ?? config.apiBase ?? (manualLocal ? 'http://127.0.0.1:8777' : '')).replace(/\/$/, '');
  const crossOrigin = !!base;
  const pollMs = opts.pollMs ?? 1500, monitorMs = opts.monitorMs ?? 6000;
  const cb = (name, ...args) => { try { opts[name]?.(...args); } catch (e) { console.error(e); } };
  const history = [];
  let lines = [], ctl = null, epoch = 0, seq = 0;
  let lastPick = -1, lastDeflect = -1;
  let connectionEpoch = 0, connectionAbort = null, monitorTimer = null, destroyed = false;
  const api = {
    history, mode: 'demo', info: null, lastTps: 0, ready: null,
    connection: { status: 'idle', message: 'Demo · recorded replies. Local access starts only when you connect.' },
    send, connect, disconnect,
    stop() { ctl?.abort(); },
    reset() { epoch++; ctl?.abort(); ctl = null; history.length = 0; api.lastTps = 0; },
    destroy() { destroyed = true; ++connectionEpoch; connectionAbort?.abort(); clearTimeout(monitorTimer); api.reset(); },
    get busy() { return !!ctl; },
  };
  const linesP = fetch('/web/3d/demo_lines.json').then((r) => r.json()).then((d) => { lines = Array.isArray(d) ? d : []; }).catch(() => {});

  function connection(status, message, info = null) {
    api.connection = { status, message, info };
    cb('onConnection', api.connection);
    if (typeof globalThis.CustomEvent === 'function' && globalThis.dispatchEvent) {
      globalThis.dispatchEvent(new CustomEvent('capri:connection', { detail: api.connection }));
    }
  }
  function setMode(mode) {
    if (api.mode === mode) return;
    api.mode = mode;
    cb('onMode', mode, api.info);
  }
  async function permission() {
    if (!crossOrigin || !globalThis.navigator?.permissions?.query) return 'prompt';
    for (const name of ['loopback-network', 'local-network-access', 'local-network']) {
      try { return (await navigator.permissions.query({ name })).state; } catch { /* unsupported permission name */ }
    }
    return 'prompt';
  }
  function wait(ms, signal) {
    return new Promise((resolve, reject) => {
      if (signal.aborted) return reject(new DOMException('Aborted', 'AbortError'));
      const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, ms);
      const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
      signal.addEventListener('abort', abort, { once: true });
    });
  }
  let addressOptions = null;
  function requestOptions() {
    if (!crossOrigin) return {};
    if (addressOptions) return addressOptions;
    // Current Chromium calls this space "loopback". Older implementations used
    // "local" for both spaces; detect the enum without making a network request.
    for (const space of ['loopback', 'local']) {
      try {
        new Request(base + '/info', { targetAddressSpace: space });
        return addressOptions = { mode: 'cors', targetAddressSpace: space };
      } catch { /* older browser enum */ }
    }
    return addressOptions = { mode: 'cors' };
  }
  async function infoRequest(signal) {
    const ac = new AbortController(), abort = () => ac.abort();
    if (signal.aborted) ac.abort();
    signal.addEventListener('abort', abort, { once: true });
    // A permission prompt can stay open while the visitor reads it.
    const timer = setTimeout(abort, 30000);
    try {
      const response = await fetch(base + '/info', { signal: ac.signal, cache: 'no-store', credentials: 'omit',
        ...requestOptions() });
      if (!response.ok) throw new Error('Local Capri returned HTTP ' + response.status + '. Check Terminal and try again.');
      const info = await response.json();
      if (typeof info.max_dose !== 'number' || (manualLocal && info.protocol !== 'capri-local-v1')) {
        throw new Error('A different service is running on port 8777. Start the current Capri server, then try again.');
      }
      return info;
    } finally { clearTimeout(timer); signal.removeEventListener('abort', abort); }
  }
  async function failure(error, id, automatic = false, lost = false) {
    if (destroyed || id !== connectionEpoch) return false;
    const denied = await permission() === 'denied';
    if (destroyed || id !== connectionEpoch) return false;
    api.reset();
    setMode(automatic ? 'demo' : 'disconnected');
    const status = denied ? 'denied' : lost ? 'lost' : 'error';
    const message = denied
      ? 'Browser access is blocked. Allow local or loopback network access for this site in browser Site settings, then try again.'
      : error instanceof TypeError || error?.name === 'AbortError'
        ? 'Cannot reach local Capri. Keep Terminal open, wait for the server, then try again. If your browser blocks the link, open the local chat below.'
        : error?.message || 'Local Capri could not connect. Check Terminal and try again.';
    connection(status, message);
    return false;
  }
  function scheduleMonitor(id) {
    clearTimeout(monitorTimer);
    monitorTimer = setTimeout(async () => {
      if (destroyed || id !== connectionEpoch || api.mode !== 'live') return;
      try {
        const info = await infoRequest(connectionAbort.signal);
        if (id !== connectionEpoch || destroyed) return;
        if (info.status === 'error') throw new Error(info.message || 'The local model stopped. Check Terminal and try again.');
        if (info.status === 'loading' || info.ready === false) throw new Error('The local model is restarting. Reconnect when Terminal says it is ready.');
        api.info = info;
        scheduleMonitor(id);
      } catch (error) { await failure(error, id, false, true); }
    }, monitorMs);
  }
  async function connect({ automatic = false } = {}) {
    const id = ++connectionEpoch;
    connectionAbort?.abort();
    clearTimeout(monitorTimer);
    connectionAbort = new AbortController();
    const signal = connectionAbort.signal;
    api.reset();
    setMode('connecting');
    connection('connecting', 'Connecting to Capri on this Mac. Choose Allow if your browser asks.');
    try {
      if (await permission() === 'denied') {
        const error = new Error('Browser access is blocked.');
        return await failure(error, id, automatic);
      }
      while (!destroyed && id === connectionEpoch) {
        const info = await infoRequest(signal);
        if (id !== connectionEpoch || destroyed) return false;
        api.info = info;
        if (info.status === 'error') throw new Error(info.message || 'Capri could not load. Check Terminal for details.');
        if (info.status === 'loading' || info.ready === false) {
          setMode('loading');
          connection('loading', info.message || 'Capri is loading. Leave Terminal open; this page will connect when she is ready.', info);
          await wait(pollMs, signal);
          continue;
        }
        setMode('live');
        connection('ready', 'Connected · replies are generated by Capri on this Mac.', info);
        scheduleMonitor(id);
        return true;
      }
      return false;
    } catch (error) { return await failure(error, id, automatic); }
  }
  function disconnect() {
    ++connectionEpoch;
    connectionAbort?.abort();
    clearTimeout(monitorTimer);
    api.reset();
    api.info = null;
    setMode('demo');
    connection('idle', 'Demo · recorded replies. Your local model is disconnected from this page.');
    return 'demo';
  }
  // Hosted pages never probe loopback on load. Even permission queries below
  // happen only inside an explicit Connect click. Same-origin local UI can start automatically.
  api.ready = manualLocal
    ? Promise.resolve().then(() => { cb('onMode', 'demo', null); return 'demo'; })
    : Promise.resolve().then(() => connect({ automatic: true })).then(() => api.mode);

  const whenAborted = (signal) => new Promise((resolve) => {
    if (signal.aborted) return resolve();
    signal.addEventListener('abort', resolve, { once: true });
  });
  const STALE = () => ({ reply: '', aborted: true, error: null, restore: null, stale: true });
  async function send(text, dose) {
    text = String(text || '').trim();
    if (!text) return { reply: '', aborted: false };
    if (!['demo', 'live'].includes(api.mode)) {
      const message = api.mode === 'loading' ? 'Capri is still loading. Leave Terminal open and wait.'
        : 'Connect local Capri or choose Use demo before sending another message.';
      cb('onError', message);
      return { reply: '', aborted: false, error: message, restore: text };
    }
    ctl?.abort();
    const my = ctl = new AbortController(), myEpoch = epoch, mySeq = ++seq;
    const stale = () => myEpoch !== epoch || mySeq !== seq;
    await Promise.race([api.ready, whenAborted(my.signal)]);
    if (stale()) return STALE();
    const userMsg = { role: 'user', content: text };
    history.push(userMsg);
    cb('onStart');
    let reply = '', error = null, aborted = false, meta = {};
    const onT = (chunk) => { reply += chunk; if (!stale()) cb('onToken', chunk); };
    try {
      meta = api.mode === 'live'
        ? await live(history.slice(), dose, my.signal, onT)
        : await demo(text, my.signal, onT);
    } catch (caught) {
      if (caught.name === 'AbortError' || my.signal.aborted) aborted = true;
      else {
        error = caught instanceof TypeError ? 'Connection lost. Reconnect local Capri, or choose Use demo.' : caught.message || String(caught);
        // A failed live request never gets replaced by a recorded answer.
        if (api.mode === 'live') {
          clearTimeout(monitorTimer);
          setMode('disconnected');
          connection('lost', error);
        }
      }
    } finally { if (ctl === my) ctl = null; }
    if (stale()) return STALE();
    const undo = () => { const i = history.lastIndexOf(userMsg); if (i >= 0) history.splice(i, 1); };
    if (reply && !error) history.push({ role: 'assistant', content: reply });
    else undo();
    if (meta?.tps) api.lastTps = meta.tps;
    if (error) { history.length = 0; cb('onError', error); }
    else cb('onDone', { reply, aborted, mode: api.mode, ...meta });
    return { reply, aborted, error, restore: reply && !error ? null : text };
  }
  async function live(messages, dose, signal, onToken) {
    const body = { messages };
    if (typeof dose === 'number' && isFinite(dose)) body.dose = dose;
    const response = await fetch(base + '/chat', { method: 'POST', credentials: 'omit',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
      ...requestOptions() });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Local Capri returned HTTP ' + response.status);
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffered = '', meta = {}, sawDone = false;
    function handle(part) {
      if (!part.startsWith('data: ')) return;
      const event = JSON.parse(part.slice(6));
      if (event.t) onToken(event.t);
      if (event.error) throw new Error(event.error);
      if (event.done) { sawDone = true; meta = { tps: event.tps, finish: event.finish, tokens: event.tokens }; }
    }
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      const parts = buffered.split('\n\n');
      buffered = parts.pop();
      for (const part of parts) handle(part.trim());
    }
    buffered += decoder.decode();
    if (buffered.trim()) handle(buffered.trim());
    if (!sawDone && !signal.aborted) throw new TypeError('stream ended early');
    return meta;
  }

  function pick(q) {
    const w = words(q), ws = new Set(w);
    const bare = q.toLowerCase().replace(/[^a-z ]/g, '').trim();
    if (/^(hi+|hey+|hello+|yo+|sup|hiya|howdy|heya|good (morning|evening|afternoon))( capri| there)?$/.test(bare)) return DEMO_GREET[(Math.random() * DEMO_GREET.length) | 0];
    if (/\b(who are you|your name|who r u|what are you)\b/.test(bare)) return DEMO_WHO[(Math.random() * DEMO_WHO.length) | 0];
    if (!lines.length) return "...Link's dead. Start server.py and try me again.";
    let best = -1, bestS = 0, bestN = 0;
    lines.forEach((l, i) => {
      const lw = new Set(words(l.q));
      let s = 0, n = 0; for (const x of ws) if (lw.has(x)) { s += x.length > 5 ? 1.5 : 1; n++; }
      s /= Math.sqrt(lw.size + 2);
      if (s > bestS) { bestS = s; best = i; bestN = n; }
    });
    // one shared word out of a long question ("explain") is a coincidence, not a match
    if (best < 0 || bestS < 0.3 || (bestN < 2 && ws.size >= 2)) {
      let k; do k = (Math.random() * DEMO_DEFLECT.length) | 0; while (k === lastDeflect);
      lastDeflect = k; return DEMO_DEFLECT[k];
    }
    lastPick = best;
    return lines[best].a;
  }

  async function demo(q, signal, onT) {
    await linesP;
    const a = pick(q);
    const sleep = (ms) => new Promise((res, rej) => {
      if (signal.aborted) return rej(new DOMException('aborted', 'AbortError'));
      const t = setTimeout(res, ms);
      signal.addEventListener('abort', () => { clearTimeout(t); rej(new DOMException('aborted', 'AbortError')); }, { once: true });
    });
    await sleep(450 + Math.random() * 650); // "thinking"
    let i = 0, n = 0;
    while (i < a.length) {
      const k = 1 + ((Math.random() * 4) | 0);
      onT(a.slice(i, i + k)); i += k; n++;
      await sleep(20 + Math.random() * 40);
    }
    return { tps: 0, finish: 'stop', tokens: n, demo: true };
  }

  return api;
}
