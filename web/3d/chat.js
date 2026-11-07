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
  const cb = (name, ...a) => { try { opts[name] && opts[name](...a); } catch (e) { console.error(e); } };
  const history = [];
  let lines = [];
  let ctl = null;          // AbortController of the running request (live or demo)
  let lastPick = -1, lastDeflect = -1;
  let epoch = 0;           // bumped by reset(): a request from an older conversation must not touch history

  const api = {
    history,
    mode: 'demo',
    info: null,
    lastTps: 0,
    ready: null,
    send,
    stop() { if (ctl) ctl.abort(); },
    // new conversation: abort whatever is running and empty the history (a late reply can't leak into it)
    reset() { epoch++; if (ctl) ctl.abort(); ctl = null; history.length = 0; },
    get busy() { return !!ctl; },
  };

  const linesP = fetch('/web/3d/demo_lines.json').then((r) => r.json()).then((d) => { lines = Array.isArray(d) ? d : []; }).catch(() => {});

  // server.py is single-threaded, so while the page is still pulling three.js / the model off it a probe can
  // queue for a while: allow a generous timeout, and retry once at start before settling on demo mode.
  let lastProbe = 0, probeMs = 20000;
  // a 404 means "a plain static server, no model server here": look again only now and then (every 60 s, or when
  // the tab comes back), so demo mode doesn't fill the console with red 404s or delay each send. A network error
  // (nothing listening) keeps the normal 20 s cadence.
  async function probe(ms = 1500, signal) {
    lastProbe = Date.now();
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), ms);
    const onAbort = () => ac.abort();
    if (signal) signal.addEventListener('abort', onAbort, { once: true });
    try {
      const r = await fetch('/info', { signal: ac.signal, cache: 'no-store' });
      if (r.status === 404) probeMs = 60000;
      if (!r.ok) throw new Error(r.statusText);
      const info = await r.json();
      if (typeof info.max_dose !== 'number') { probeMs = 60000; throw new Error('not the model server'); }
      api.info = info; probeMs = 20000; return true;
    } catch { return false; }
    finally { clearTimeout(timer); if (signal) signal.removeEventListener('abort', onAbort); }
  }
  api.ready = probe(5000).then((ok) => ok || (probeMs > 20000 ? false : probe(5000))).then((ok) => { api.mode = ok ? 'live' : 'demo'; cb('onMode', api.mode, api.info); return api.mode; });
  // in demo mode, look for the server again now and then (it may have been started after the page)
  const reprobe = async () => {
    if (api.mode !== 'demo' || ctl || document.hidden || Date.now() - lastProbe < probeMs - 500) return;
    if (await probe()) setMode('live');
  };
  setInterval(reprobe, 10000);
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', () => { if (!document.hidden && Date.now() - lastProbe > 8000) { lastProbe = 0; reprobe(); } });

  function setMode(m) { if (api.mode !== m) { api.mode = m; cb('onMode', m, api.info); } }

  // resolves when the signal aborts (so a send waiting on the first probe can be superseded at once)
  const whenAborted = (signal) => new Promise((res) => { if (signal.aborted) res(); else signal.addEventListener('abort', res, { once: true }); });
  let seq = 0; // bumped by every send: only the newest send may fire callbacks or touch the history

  async function send(text, dose) {
    text = String(text || '').trim();
    if (!text) return { reply: '', aborted: false };
    if (ctl) ctl.abort();
    const my = (ctl = new AbortController()), myEpoch = epoch, mySeq = ++seq;
    // superseded by a newer send (or by reset()): resolve quietly, no callbacks, no history changes
    const stale = () => myEpoch !== epoch || mySeq !== seq;
    const STALE = () => ({ reply: '', aborted: true, error: null, restore: null, stale: true });
    await Promise.race([api.ready, whenAborted(my.signal)]);
    if (stale()) return STALE();
    // demo mode, but the server may have come up since the last look (only when the last probe didn't find a
    // plain static server: see probeMs)
    if (api.mode === 'demo' && !my.signal.aborted && Date.now() - lastProbe > Math.max(4000, probeMs - 500) && await probe(1500, my.signal)) setMode('live');
    if (stale()) return STALE();
    if (my.signal.aborted) { if (ctl === my) ctl = null; cb('onDone', { reply: '', aborted: true, mode: api.mode }); return { reply: '', aborted: true, error: null, restore: text }; }
    const userMsg = { role: 'user', content: text };
    history.push(userMsg);
    cb('onStart');
    let reply = '', error = null, aborted = false, meta = {};
    const onT = (t) => { reply += t; if (!stale()) cb('onToken', t); };
    try {
      if (api.mode === 'live') {
        try {
          meta = await live(history.slice(), dose, my.signal, onT);
        } catch (e) {
          // the server went away (TypeError = network) before anything arrived: answer from the demo lines
          if (e instanceof TypeError && !reply && !my.signal.aborted) {
            setMode('demo');
            meta = await demo(text, my.signal, onT);
          } else throw e;
        }
      } else {
        meta = await demo(text, my.signal, onT);
      }
    } catch (e) {
      if (e.name === 'AbortError' || my.signal.aborted) aborted = true;
      else if (e instanceof TypeError) setMode('demo');
      if (!aborted) error = e instanceof TypeError ? (e.message === 'stream ended early' ? 'The model server stopped mid-reply (did server.py crash?). Restart it with: .venv/bin/python server.py' : "The model server isn't answering. Start it with: .venv/bin/python server.py") : (e.message || String(e));
    } finally {
      if (ctl === my) ctl = null;
    }
    // undo this turn (by identity: a newer send may already have pushed its own message after ours)
    const undo = () => { const i = history.lastIndexOf(userMsg); if (i >= 0) history.splice(i, 1); };
    if (stale()) { if (myEpoch === epoch) undo(); return STALE(); } // reset() or a newer send ran meanwhile
    // keep a turn only if it came back whole; a reply cut off by a server error (or nothing at all) is undone so the
    // next request doesn't carry a dangling user message or a half sentence the user never saw
    if (reply && !error) history.push({ role: 'assistant', content: reply });
    else undo();
    if (meta && meta.tps) api.lastTps = meta.tps;
    if (error) cb('onError', error);
    else cb('onDone', { reply, aborted, mode: api.mode, ...meta });
    return { reply, aborted, error, restore: reply && !error ? null : text };
  }

  async function live(msgs, dose, signal, onT) {
    const body = { messages: msgs };
    if (typeof dose === 'number' && isFinite(dose)) body.dose = dose;
    const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `server said ${res.status} ${res.statusText}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = '', meta = {}, sawDone = false;
    const handle = (p) => {
      if (!p.startsWith('data: ')) return;
      let ev; try { ev = JSON.parse(p.slice(6)); } catch { return; }
      if (ev.t) onT(ev.t);
      if (ev.error) throw new Error(ev.error);
      if (ev.done) { sawDone = true; meta = { tps: ev.tps, finish: ev.finish, tokens: ev.tokens }; }
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const p of parts) handle(p.trim());
    }
    buf += dec.decode();
    if (buf.trim()) handle(buf.trim());
    // server.py speaks HTTP/1.0 without Content-Length, so a process that dies mid-reply looks like a clean end of
    // stream: without the {done} event the reply is incomplete (TypeError = treated like a network failure)
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
