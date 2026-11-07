// voice.js: Capri's "talk blips" — tiny, quiet, slightly bit-crushed bleeps, one per revealed character.
//
// createVoice() -> { resume(), blip(char), zap('out'|'in'), setEnabled(bool), enabled }
//
// Each blip: a square + triangle pair through a band-pass and a stepped wave-shaper (the "crushed" cyber
// timbre), with a ~60 ms envelope. Pitch is deterministic per character (vowels sit on the low, open notes
// of a minor pentatonic, consonants higher and shorter), so the same word always "sounds" the same. Spaces
// and punctuation are silent. Blips are rate-limited (and every other one is skipped when text runs fast)
// so 30 chars/s stays a soft chatter instead of a buzz. Nothing is created until enabled + resumed.

const VOWELS = 'aeiouy';
const SCALE = [0, 3, 5, 7, 10, 12, 15];          // minor pentatonic-ish, semitones
const BASE = 392;                                  // G4: she's small and fast
const MIN_GAP = 0.052;                             // s between blips

function hashCh(c) { let h = c.charCodeAt(0) * 2654435761; h ^= h >>> 13; return (h >>> 0) % 997; }

export function createVoice() {
  let ctx = null, master = null, crush = null, enabled = false, lastAt = 0, n = 0, zapNoise = null;

  function init() {
    if (ctx) return ctx;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    try { ctx = new AC(); } catch { return null; }
    master = ctx.createGain(); master.gain.value = 0.055;
    // gentle low-pass at the end keeps the square's top end from biting
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 4200; lp.Q.value = 0.4;
    // 'bit crush': quantise the waveform to a few levels
    crush = ctx.createWaveShaper();
    const N = 1024, steps = 7, curve = new Float32Array(N);
    for (let i = 0; i < N; i++) { const x = (i / (N - 1)) * 2 - 1; curve[i] = Math.round(x * steps) / steps; }
    crush.curve = curve; crush.oversample = 'none';
    crush.connect(lp); lp.connect(master); master.connect(ctx.destination);
    return ctx;
  }

  const api = {
    get enabled() { return enabled; },
    setEnabled(on) { enabled = !!on; if (enabled) api.resume(); },
    resume() {
      if (!enabled) return;
      const c = init();
      if (c && c.state === 'suspended') c.resume().catch(() => {});
    },
    blip(ch) {
      if (!enabled || !ctx || ctx.state !== 'running' || !ch) return;
      const c = String(ch).toLowerCase();
      if (!/[a-z0-9]/.test(c)) return;              // spaces, punctuation, emoji: silent
      const now = ctx.currentTime;
      if (now - lastAt < MIN_GAP) return;
      lastAt = now; n++;
      const h = hashCh(c), vowel = VOWELS.includes(c), digit = /[0-9]/.test(c);
      const deg = vowel ? SCALE[h % 3] : SCALE[2 + (h % 5)];
      const f = BASE * Math.pow(2, (deg + (digit ? 12 : 0)) / 12) * (1 + ((n % 3) - 1) * 0.012); // a hair of wobble
      const dur = vowel ? 0.075 : 0.045;

      const env = ctx.createGain();
      env.gain.setValueAtTime(0.0001, now);
      env.gain.exponentialRampToValueAtTime(vowel ? 1 : 0.7, now + 0.004);
      env.gain.exponentialRampToValueAtTime(0.0001, now + dur);
      const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = f * 2.2; bp.Q.value = 1.1;

      const o1 = ctx.createOscillator(); o1.type = 'square'; o1.frequency.setValueAtTime(f, now);
      o1.frequency.exponentialRampToValueAtTime(f * (vowel ? 0.94 : 1.05), now + dur);
      const o2 = ctx.createOscillator(); o2.type = 'triangle'; o2.frequency.value = f * 1.5;
      const g2 = ctx.createGain(); g2.gain.value = 0.35;
      o1.connect(bp); o2.connect(g2); g2.connect(bp); bp.connect(env); env.connect(crush);
      o1.start(now); o2.start(now); o1.stop(now + dur + 0.02); o2.stop(now + dur + 0.02);
      o1.onended = () => { try { env.disconnect(); bp.disconnect(); g2.disconnect(); } catch { /* already gone */ } };
    },
    // zap('out' | 'in'): the teleport. A crushed square chirp (falling as she breaks up, rising as she lands) over a
    // crackle of band-passed noise, with a short hard "crack" timed to her flash (out: ~0.25 s in, in: ~0.03 s in).
    zap(kind) {
      if (!enabled || !ctx || ctx.state !== 'running') return;
      const now = ctx.currentTime, out = kind !== 'in';
      const dur = out ? 0.34 : 0.4, crackAt = now + (out ? 0.25 : 0.03);
      if (!zapNoise) {
        zapNoise = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.5), ctx.sampleRate);
        const d = zapNoise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      }
      const out1 = ctx.createGain(); out1.gain.value = 1.6; out1.connect(crush);
      const parts = [out1];
      // chirp
      const o = ctx.createOscillator(); o.type = 'square';
      const f0 = out ? 1500 : 260, f1 = out ? 180 : 1300;
      o.frequency.setValueAtTime(f0, now); o.frequency.exponentialRampToValueAtTime(f1, now + dur * 0.85);
      const og = ctx.createGain();
      og.gain.setValueAtTime(0.0001, now); og.gain.exponentialRampToValueAtTime(out ? 0.35 : 0.45, now + 0.02);
      og.gain.setValueAtTime(out ? 0.35 : 0.45, now + dur * 0.6); og.gain.exponentialRampToValueAtTime(0.0001, now + dur);
      // stutter: chop the chirp into glitchy gates
      for (let k = 0; k < 6; k++) { const tk = now + 0.03 + k * dur / 7; og.gain.setValueAtTime(0.06, tk); og.gain.setValueAtTime(out ? 0.35 : 0.45, tk + 0.012); }
      o.connect(og); og.connect(out1); parts.push(o, og);
      // crackle (band-passed noise, gated)
      const n = ctx.createBufferSource(); n.buffer = zapNoise;
      const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.setValueAtTime(out ? 2400 : 900, now); bp.frequency.exponentialRampToValueAtTime(out ? 700 : 3000, now + dur); bp.Q.value = 2.5;
      const ng = ctx.createGain(); ng.gain.setValueAtTime(0.0001, now); ng.gain.exponentialRampToValueAtTime(0.5, now + 0.03); ng.gain.exponentialRampToValueAtTime(0.0001, now + dur);
      n.connect(bp); bp.connect(ng); ng.connect(out1); parts.push(n, bp, ng);
      // the crack on the flash
      const c = ctx.createBufferSource(); c.buffer = zapNoise;
      const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 1200;
      const cg = ctx.createGain(); cg.gain.setValueAtTime(0.0001, now); cg.gain.setValueAtTime(0.0001, crackAt - 0.004);
      cg.gain.exponentialRampToValueAtTime(1.1, crackAt); cg.gain.exponentialRampToValueAtTime(0.0001, crackAt + 0.09);
      c.connect(hp); hp.connect(cg); cg.connect(out1); parts.push(c, hp, cg);
      o.start(now); n.start(now); c.start(now, 0.1);
      o.stop(now + dur + 0.05); n.stop(now + dur + 0.05); c.stop(crackAt + 0.12);
      o.onended = () => { for (const p of parts) { try { p.disconnect(); } catch { /* gone */ } } };
    },
  };
  return api;
}
