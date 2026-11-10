// main.js: renderer, camera, post-processing, the speech clock and the HTML overlay. Wires everything.
//
// Render order per frame:
//   bloom: half-res render of the scene with Capri darkened -> UnrealBloomPass (see renderBloom)
//   composer: RenderPass(scene: city + Capri) -> "cyber" ShaderPass (adds bloom, glitch/CA/grain, tone maps, to screen)
//   then bubbleScene on top (autoClear off, depth cleared) so the reply text is crisp and never bloomed.
//
// The speech clock (see README) turns bursty model tokens into speaking-paced characters; each revealed char
// goes to bubble.append, capri.speak and voice.blip together.

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { PALETTE } from './palette.js';
import { createChat } from './chat.js';
import { createVoice } from './voice.js';

const $ = (id) => document.getElementById(id);
const clamp = THREE.MathUtils.clamp;
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));
const store = {
  get(k, d) { try { const v = localStorage.getItem('capri.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('capri.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const reducedMQ = matchMedia('(prefers-reduced-motion: reduce)');
let reduced = reducedMQ.matches;
reducedMQ.addEventListener?.('change', (e) => { reduced = e.matches; });
const coarse = matchMedia('(pointer: coarse)').matches;

// ------------------------------------------------------------------------------------------------
// renderer / scenes / camera
// ------------------------------------------------------------------------------------------------
const canvas = $('scene');
let renderer;
try {
  // no backbuffer MSAA: the scene is drawn into the composer's multisampled target (below), and only the post quad
  // and the (SDF-anti-aliased) bubble reach the default framebuffer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
} catch (e) {
  $('boot-t').textContent = 'WebGL is not available in this browser. Try /classic.';
  throw e;
}
// Pixel ratio: capped at 1.5 (the post chain + bloom pre-render are fill-rate bound on a Retina MacBook), and
// stepped down further by the adaptive check in frameTick if frames keep running long.
const DPR_CAP = 1.5;
let dprScale = 1;
const pixelRatio = () => Math.max(1, Math.min(devicePixelRatio || 1, DPR_CAP) * dprScale);
renderer.setPixelRatio(pixelRatio());
renderer.setSize(innerWidth, innerHeight, false);
// Neutral (not ACES): ACES bleaches her acid green #19f267 to mint; Neutral keeps hue and saturation.
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(PALETTE.void);
const bubbleScene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.05, 1500);

const quality = coarse || (navigator.hardwareConcurrency || 8) <= 4 ? 'low' : 'high';

// ------------------------------------------------------------------------------------------------
// post: bloom + the "cyber" pass
// ------------------------------------------------------------------------------------------------
const CyberShader = {
  uniforms: {
    tDiffuse: { value: null },
    tBloom: { value: null },
    uTime: { value: 0 },
    uRes: { value: new THREE.Vector2(1, 1) },
    uDpr: { value: 1 },
    uGlitch: { value: 0 },     // 0..1 burst
    uAmt: { value: 0.8 },      // overall strength (dose-scaled)
    uTalk: { value: 0 },
    uFlash: { value: 0 },      // teleport: a brief pink-white lift of the whole frame (kept low: no strobe)
    uSplit: { value: 0 },      // teleport: extra horizontal RGB split, peaks with her flashes
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    precision highp float;
    uniform sampler2D tDiffuse, tBloom; uniform float uTime, uGlitch, uAmt, uTalk, uDpr, uFlash, uSplit; uniform vec2 uRes;
    varying vec2 vUv;
    float h11(float p){ p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
    float h21(vec2 p){ vec3 q = fract(vec3(p.xyx) * .1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
    void main(){
      vec2 uv = vUv;
      float g = uGlitch;
      // glitch: a few horizontal bands slip sideways, a block or two flashes
      float tq = floor(uTime * 24.0);
      float band = floor(uv.y * mix(14.0, 38.0, h11(tq)));
      float r = h21(vec2(band, tq));
      float slip = step(1.0 - g * 0.35, r) * (h11(band + tq * 1.7) - 0.5) * 0.09 * g;
      uv.x += slip;
      // chromatic aberration: radial, stronger toward the edges and during glitches
      vec2 d = uv - 0.5;
      // (kept subtle: a lens fringe at the corners, not a split display; the strong split is for glitch bursts only)
      float ca = (0.00025 + 0.0006 * dot(d, d) * 4.0) * uAmt + g * 0.009 + abs(slip) * 0.4;
      vec2 caH = vec2(g * 0.009 + abs(slip) * 0.4 + uSplit * 0.014, 0.0) * 0.5;
      vec3 col;
      col.r = texture2D(tDiffuse, uv + d * ca * 2.0 + caH).r;
      col.g = texture2D(tDiffuse, uv).g;
      col.b = texture2D(tDiffuse, uv - d * ca * 2.0 - caH).b;
      col += texture2D(tBloom, uv).rgb; // selective bloom (see renderBloom), added here to save a full-res pass
      // teleport flash: a pink-white lift, strongest in the middle of the frame
      col += (vec3(1.0, 0.18, 0.42) * 0.55 + vec3(0.45) * uFlash) * uFlash * (1.0 - 0.5 * dot(d, d) * 2.0);
      // glitch tint flashes in her colours
      float blk = step(0.985 - g * 0.05, h21(floor(uv * vec2(18.0, 30.0)) + tq)) * g;
      col = mix(col, col.gbr * vec3(1.2, 0.6, 1.4) + vec3(0.25, 0.0, 0.08) * g, blk * 0.7);
      // scanlines (in CSS px so they read the same on retina)
      float y = gl_FragCoord.y / uDpr;
      float sl = 0.5 + 0.5 * sin(y * 3.14159 * 0.66 + uTime * 2.0);
      col *= 1.0 - (0.055 + 0.04 * g) * uAmt * sl;
      // slow rolling bar
      col *= 1.0 - 0.035 * uAmt * smoothstep(0.0, 0.08, abs(fract(uv.y * 0.6 - uTime * 0.05) - 0.5) - 0.42);
      // vignette
      float v = smoothstep(1.05, 0.25, length(d * vec2(1.05, 1.25)));
      col *= mix(0.5, 1.0, v);
      // film grain (in linear HDR, before tone mapping: keep it tiny)
      float n = h21(gl_FragCoord.xy + fract(uTime * 7.13) * 413.0) - 0.5;
      col += n * (0.022 + 0.02 * g) * uAmt * (0.35 + col);
      gl_FragColor = vec4(max(col, 0.0), 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`,
};

// Selective bloom: Capri's unlit texture (acid green ~0.65 luma) would bloom like a neon sign and smear her
// cel look, so the bloom gets its own half-res render of the scene with her colours darkened (her glitch
// tints and talk glow still get through), and the result is added onto the normal full-res render.
const bloomRT = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType });
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.8, 0.6, 0.24);
// the scene render target is multisampled (4x), so the city and her cel outline are anti-aliased (setSize keeps samples)
const composerRT = new THREE.WebGLRenderTarget(Math.max(2, innerWidth * renderer.getPixelRatio()), Math.max(2, innerHeight * renderer.getPixelRatio()), { type: THREE.HalfFloatType, samples: 4 });
const composer = new EffectComposer(renderer, composerRT);
composer.setPixelRatio(renderer.getPixelRatio());
composer.setSize(innerWidth, innerHeight);
composer.addPass(new RenderPass(scene, camera));
// one pass after the scene: bloom add + glitch/CA/scanlines/grain + tone mapping + sRGB (was three passes)
const cyber = new ShaderPass(CyberShader);
cyber.uniforms.tBloom.value = bloom.renderTargetsHorizontal[0].texture; // set after the clone (RT textures can't be cloned)
composer.addPass(cyber);

// ------------------------------------------------------------------------------------------------
// framing
// ------------------------------------------------------------------------------------------------
const frame = { pos: new THREE.Vector3(0.4, 1.35, 3.6), target: new THREE.Vector3(0, 1.25, 0), fov: 38 };
// the per-aspect shot fitted to her on perches[0] (frame = that, adjusted for the perch she's on now: see perchShot)
const baseFrame = { pos: new THREE.Vector3(), target: new THREE.Vector3(), fov: 38 };
// perches: city.perches (README); filled once Capri has loaded
let perches = [], perchIdx = 0;
// postures per perch (capri.js POSTURES): [name, weight]. She lands in one when she jumps there (set while she is
// invisible); home keeps her modelled seated pose most of the time.
const PERCH_POSTURES = {
  parapet: [['seated', 3], ['float', 1]],
  crates: [['crouch', 1]],
  'stair-hut': [['lie', 1]],
  'water-tank': [['float', 1]], // (too narrow for the recline)
};
function rollPosture(p) {
  const opts = PERCH_POSTURES[p.id] || [['seated', 1]];
  let r = Math.random() * opts.reduce((a, o) => a + o[1], 0);
  for (const [n, w] of opts) if ((r -= w) <= 0) return (p.posture = n);
  return (p.posture = opts[0][0]);
}

function computeFrame() {
  const a = innerWidth / innerHeight;
  if (a >= 1.25) {          // desktop / landscape: her a little left of centre so the bubble has the right side
    frame.fov = 38;
    frame.pos.set(0.4, 1.35, 3.6);
    frame.target.set(0.3, 1.27, 0);
    if (a > 2) { frame.pos.z = 3.3; }
  } else if (a >= 0.8) {    // squareish tablets
    frame.fov = 42;
    frame.pos.set(0.35, 1.45, 4.3);
    frame.target.set(0.05, 1.42, 0);
  } else {                  // portrait phones: pull back + raise the target so she sits low, bubble above her head
    frame.fov = 46;
    const k = clamp((0.8 - a) / 0.34, 0, 1); // 0 at a=0.8, 1 at a~0.46 (iPhone)
    frame.pos.set(0.25, 1.5 + 0.15 * k, 4.6 + 1.3 * k);
    frame.target.set(0.0, 1.62 + 0.28 * k, 0);
  }
  fitToBand(a);
  baseFrame.pos.copy(frame.pos); baseFrame.target.copy(frame.target); baseFrame.fov = frame.fov;
  perchShot(perches[perchIdx], a, frame);
  try { city?.setAspect?.(a); city?.setFraming?.({ pos: frame.pos, target: frame.target, fov: frame.fov, aspect: a }); } catch { /* optional */ }
  camera.fov = frame.fov;
  camera.aspect = a;
  camera.updateProjectionMatrix();
}
// Once Capri is loaded, slide the camera along its framing direction (and re-aim it vertically) so she fits
// between the header and the prompt bar: hair spikes under the header, the dangling foot just above the bar.
let fitBox = null; // { top, bot } world-space heights of her rest pose
function fitToBand(a) {
  if (!fitBox) return;
  const H = innerHeight;
  const hdr = document.querySelector('header')?.getBoundingClientRect().bottom || 56;
  // the prompt bar's top, ignoring the on-screen keyboard (she shouldn't jump when it opens)
  const bar = (document.getElementById('form')?.getBoundingClientRect().top ?? H - 90) + kbPx;
  const portrait = a < 0.8;
  // portrait phones keep room above her head for the bubble: the band starts lower
  const topPx = portrait ? hdr + H * 0.26 : hdr - 6, botPx = bar - (portrait ? 6 : 18); // (empirical: her feet are nearer the camera than her hair)
  if (botPx - topPx < 120) return;
  const tan = Math.tan(THREE.MathUtils.degToRad(frame.fov) / 2);
  const yTop = 1 - (2 * topPx) / H, yBot = 1 - (2 * botPx) / H;
  const dir = tmpFit.copy(frame.pos).sub(frame.target);
  const d0 = dir.length(); dir.divideScalar(d0);
  // (landscape: ~7% wider than a tight fit, so the master shot holds the foreground corners and the fore perch reads
  // clearly bigger than the parapet: she jumps between them inside this one shot)
  const d = clamp((fitBox.top - fitBox.bot) / ((yTop - yBot) * tan), d0 * 0.9, d0 * 1.35) * (portrait ? 1 : a >= 1.25 ? 1.07 : 1.03);
  const yc = fitBox.top - yTop * d * tan; // height on her plane that lands at the screen centre
  frame.target.y = yc;
  frame.pos.copy(frame.target).addScaledVector(dir, d);
}
const tmpFit = new THREE.Vector3();
let kbPx = 0;

// ---- perches: where she sits, and a camera that keeps her whole on each one ----
// root transform for a perch: root = seat - seatPoint*scale (rotated by the perch yaw)
function perchRoot(p, outPos) {
  const s = p.scale || 1;
  const sp = (capri ? capri.seatPoint : new THREE.Vector3(-0.05, 0.675, 0.13)).clone().multiplyScalar(s).applyAxisAngle(Y_UP, yawOf(p));
  return outPos.copy(p.seat).sub(sp);
}
const Y_UP = new THREE.Vector3(0, 1, 0);
// her root yaw on perch p: the perch's own, unless the posture faces along the ledge (POSTURES[..].ledgeYaw: yaw
// relative to the ledge's outward normal, e.g. 0 = body square to the edge, facing out)
function yawOf(p) {
  const d = capri?.POSTURES?.[p.posture || 'seated'];
  return d && d.ledgeYaw !== undefined ? (p.ledgeYaw ?? 0) + d.ledgeYaw : p.rotationY;
}
const _pm = new THREE.Matrix4(), _pq = new THREE.Quaternion(), _ps = new THREE.Vector3(), _pp = new THREE.Vector3();
// her rest-pose silhouette sitting on perch p, as world points (capri.getRestPoints: ~220 body vertices; falls back
// to the 8 box corners). p.cropY (optional, model-space height): only the part of her above it has to fit (a close
// shot may cut her legs off at the frame's bottom edge, outside the prompt bar)
const restPts = {}; // per posture
function perchCorners(p, out = []) {
  const pose = p.posture || 'seated';
  if (!restPts[pose]) {
    const lb = capri?.getLocalBounds?.(); if (!lb) return null;
    restPts[pose] = capri.getPosturePoints?.(pose) || capri.getRestPoints?.() || Array.from({ length: 8 }, (_, i) => new THREE.Vector3(i & 1 ? lb.max.x : lb.min.x, i & 2 ? lb.max.y : lb.min.y, i & 4 ? lb.max.z : lb.min.z));
  }
  const pts = restPts[pose];
  _pm.compose(perchRoot(p, _pp), _pq.setFromAxisAngle(Y_UP, yawOf(p)), _ps.setScalar(p.scale || 1));
  const y0 = p.cropY ?? -1e9; let n = 0;
  for (const q of pts) { if (q.y < y0) continue; (out[n] || (out[n] = new THREE.Vector3())).copy(q).applyMatrix4(_pm); n++; }
  out.length = n;
  return out;
}
const fitCam = new THREE.PerspectiveCamera();
const _corn = [], _corn0 = [], _c0 = new THREE.Vector3(), _c1 = new THREE.Vector3(), _fv = new THREE.Vector3();
// The safe screen rect (CSS px) she must stay inside: under the header, clear of the prompt bar (portrait phones
// keep the top ~quarter for the bubble); landscape also keeps one side free for the bubble. Her legs may run off the
// bottom edge beside the dock (a close 'fore' perch), never behind the prompt bar / chips.
function safeRect(a, away) { // away: a perch other than home (portrait: a back perch may come a little higher)
  const H = innerHeight, W = innerWidth;
  const hdr = document.querySelector('header')?.getBoundingClientRect().bottom || 56;
  const f = document.getElementById('form')?.getBoundingClientRect();
  const bar = (f?.top ?? H - 90) + kbPx;
  const ch = document.getElementById('chips'), cr = ch && !ch.hidden ? ch.getBoundingClientRect() : null;
  const portrait = a < 0.8, pad = 12;
  return {
    W, H, top: portrait ? hdr + H * (away ? 0.215 : 0.26) : hdr - 4, bot: bar - (portrait ? 4 : 14), left: W * 0.015, right: W * 0.985,
    side: a >= 1.25 ? Math.min(520, W * 0.4) : 0,
    dock: f ? { x0: f.left - pad, x1: f.right + pad } : { x0: 0, x1: W },
    // (only the chips' buttons count, not the empty part of their row)
    chips: cr && cr.height > 0 ? chipRect(ch, pad) : null,
  };
}
function chipRect(ch, pad) {
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (const b of ch.children) { const r = b.getBoundingClientRect(); if (!r.width) continue; x0 = Math.min(x0, r.left); x1 = Math.max(x1, r.right); y0 = Math.min(y0, r.top); y1 = Math.max(y1, r.bottom); }
  return x1 > x0 ? { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad } : null;
}
// her on-screen extent for a camera: x0..x1 / y0 (top) over the points that are on screen, y1 = lowest point, and
// hit = some point lands behind the prompt bar (or below it, inside the dock's width) or on the chips
function screenBox(pts, pos, target, fov, a, R) {
  fitCam.fov = fov; fitCam.aspect = a; fitCam.near = 0.05; fitCam.far = 100; fitCam.updateProjectionMatrix();
  fitCam.position.copy(pos); fitCam.lookAt(target); fitCam.updateMatrixWorld();
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, hit = false;
  const C = R.chips;
  for (const c of pts) {
    _fv.copy(c).project(fitCam);
    if (_fv.z > 1) { hit = true; continue; } // behind the camera
    const x = (_fv.x + 1) / 2 * R.W, y = (1 - _fv.y) / 2 * R.H;
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    if (y <= R.H) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
    if (y > R.bot && x > R.dock.x0 && x < R.dock.x1) hit = true;
    else if (C && x > C.x0 && x < C.x1 && y > C.y0 && y < C.y1) hit = true;
  }
  return { x0, x1, y0, y1, hit };
}
const fits = (b, R) => !b.hit && b.y0 >= R.top && b.x0 >= R.left && b.x1 <= R.right && (R.side === 0 || b.x0 - R.left >= R.side || R.right - b.x1 >= R.side);
// out = the shot for perch p. Jumps should happen *inside one shot* (the owner wants to see her flash across the
// rooftop, the after-image left at the old spot), so the master shot (the per-aspect base shot) is used whenever she
// fits there; otherwise the city's suggested view for the perch, else the smallest change that fits her: pan part of
// the way toward her (s = 1 moves the camera rigidly with her) and dolly back along the shot's own direction.
// Cheap (a few thousand projections), run on resize / at a jump.
const _homeSeated = {};
function perchShot(p, a, out) {
  out.pos.copy(baseFrame.pos); out.target.copy(baseFrame.target); out.fov = baseFrame.fov;
  // home in her modelled seated pose is the master shot itself; any other pose there (float) is fitted like an away spot
  const home = p === perches[0], seatedHome = Object.assign(_homeSeated, perches[0], { posture: 'seated' });
  if (!p || !perches.length || (home && (p.posture || 'seated') === 'seated') || !perchCorners(p, _corn) || !perchCorners(seatedHome, _corn0)) return out;
  const R = safeRect(a, !home);
  if (fits(screenBox(_corn, out.pos, out.target, out.fov, a, R), R)) return out; // the master shot holds both spots
  // the city's suggested camera for this perch (landscape: view, portrait: viewPortrait), else the base shot
  const v = a < 0.8 ? p.viewPortrait : a >= 1.25 ? p.view : null;
  const base = v ? { pos: v.pos, target: v.target, fov: v.fov || out.fov } : baseFrame;
  out.pos.copy(base.pos); out.target.copy(base.target); out.fov = base.fov;
  if (fits(screenBox(_corn, out.pos, out.target, out.fov, a, R), R)) return out;
  const avg = (arr, o) => { o.set(0, 0, 0); for (const c of arr) o.add(c); return o.multiplyScalar(1 / Math.max(1, arr.length)); };
  avg(_corn, _c1);
  if (v) _c0.copy(base.target); else avg(_corn0, _c0);
  const delta = _c1.sub(_c0), dir = tmpFit.copy(base.pos).sub(base.target); const d0 = dir.length(); dir.divideScalar(d0);
  let best = null, bestCost = 1e9;
  const T = new THREE.Vector3(), P = new THREE.Vector3();
  for (let si = 0; si <= 20; si++) {
    const s = si / 20;
    for (let mi = 0; mi <= 40; mi++) {
      const m = 0.8 + mi * 0.04, cost = s * 1.0 + Math.abs(m - 1) * 1.4;
      if (cost >= bestCost) continue;
      T.copy(base.target).addScaledVector(delta, s); P.copy(T).addScaledVector(dir, d0 * m);
      if (fits(screenBox(_corn, P, T, out.fov, a, R), R)) { bestCost = cost; best = { s, m }; }
    }
  }
  if (!best) best = { s: 1, m: 1 };
  out.target.copy(base.target).addScaledVector(delta, best.s);
  out.pos.copy(out.target).addScaledVector(dir, d0 * best.m);
  return out;
}
// is perch i inside the current master shot? (main.js keeps the camera still for those jumps)
function inMaster(i, a = innerWidth / innerHeight) {
  const p = perches[i]; if (!p || !perchCorners(p, _corn)) return false;
  if (i === 0) return true;
  const R = safeRect(a, true); return fits(screenBox(_corn, baseFrame.pos, baseFrame.target, baseFrame.fov, a, R), R);
}
computeFrame();
camera.position.copy(frame.pos);
camera.lookAt(frame.target);

// ------------------------------------------------------------------------------------------------
// modules (fail soft: each piece can be missing or throw and the page still runs)
// ------------------------------------------------------------------------------------------------
let city = null, capri = null, bubble = null;
const head = new THREE.Vector3(-0.15, 1.65, 0.2);
const tmpV = new THREE.Vector3();

async function tryImport(path) {
  try { return await import(path); } catch (e) { console.warn('[capri] could not load', path, e); return null; }
}

// boot progress: the bar fills as each stage finishes (city, Capri, shader warm-up), then the overlay fades
const bootBar = document.querySelector('#boot .bar i');
function bootStep(p, text) { if (bootBar) bootBar.style.width = Math.round(p * 100) + '%'; if (text) $('boot-t').textContent = text; }
// let the browser paint the boot screen (whole, every tile) before a stage that blocks the page for seconds; a
// background tab never fires rAF, so a timeout backs it up
const letPaint = () => new Promise((r) => { let done = false; const go = () => { if (!done) { done = true; r(); } };
  requestAnimationFrame(() => setTimeout(go, 0)); setTimeout(go, 120); });
bootStep(0.08);
await letPaint();
const bubbleMod = await tryImport('/web/3d/bubble.js');
try {
  if (bubbleMod) bubble = bubbleMod.createBubble(bubbleScene, { maxWidth: 440, renderer, scene });
} catch (e) { console.error('[capri] bubble failed', e); }
if (!bubble) bubble = { group: new THREE.Group(), update() {}, begin() {}, append() {}, end() {}, setError() {}, setAnchor() {}, hide() {} };

bootStep(0.15);
await letPaint();
const cityMod = await tryImport('/web/3d/city.js');
try {
  if (cityMod) city = cityMod.createCity(scene, renderer, { quality });
} catch (e) { console.error('[capri] city failed', e); }
if (city && city.recommendedFar) { camera.far = Math.max(camera.far, city.recommendedFar); camera.updateProjectionMatrix(); }
const seat = city ? city.seat.clone() : new THREE.Vector3(0, 0.8, 0);
const seatRotY = city ? city.seatRotationY || 0 : 0;

bootStep(0.5, 'JACKING IN… waking Capri');
await letPaint();
const capriMod = await tryImport('/web/3d/capri.js');
try {
  if (capriMod) {
    // turn her so the camera sits between her body's +Z and where her (turned) face looks (README: cameraYaw + 0.6)
    const camYaw = Math.atan2(frame.pos.x - seat.x, frame.pos.z - seat.z);
    const rotY = seatRotY + camYaw + 0.6;
    capri = await capriMod.createCapri(scene, { url: '/capri_project_capricorn/scene.gltf', rotationY: rotY, scale: 1, camera });
    perches = buildPerches(rotY);
    perches.forEach(rollPosture); perches[0].posture = 'seated'; // first look: her modelled pose
    placeCapri(perches[0]);
    capri.setCamera?.(camera);
    const bb = capri.getBounds();
    fitBox = { top: bb.max.y, bot: bb.min.y };
    computeFrame();
  }
} catch (e) { console.error('[capri] capri failed', e); capri = null; }
if (!capri) {
  toast('Capri’s model failed to load. She can still talk.', 'ERR');
}

// contact shadows on the wet coping under her hips and hands (one draw call; the hand blobs follow her hands
// and fade as they lift off the ledge)
let contact = null, backCard = null;
if (capri) {
  const B = capri._debug?.bones;
  const blobs = [
    { model: new THREE.Vector3(-0.04, 0, -0.06), sx: 0.78, sz: 0.5, a: 0.62, bone: null },
    { model: null, sx: 0.26, sz: 0.2, a: 0.55, bone: B?.handR },
    { model: null, sx: 0.24, sz: 0.18, a: 0.55, bone: B?.handL },
  ];
  const n = blobs.length, pos = new Float32Array(n * 12), uv = new Float32Array(n * 8), al = new Float32Array(n * 4), idx = [];
  for (let i = 0; i < n; i++) { uv.set([0, 0, 1, 0, 1, 1, 0, 1], i * 8); const o = i * 4; idx.push(o, o + 2, o + 1, o, o + 3, o + 2); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('aA', new THREE.BufferAttribute(al, 1));
  g.setIndex(idx);
  const m = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    vertexShader: 'attribute float aA; varying vec2 vUv; varying float vA; void main(){ vUv = uv; vA = aA; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'varying vec2 vUv; varying float vA; void main(){ float r = length(vUv * 2.0 - 1.0); float a = vA * (1.0 - smoothstep(0.15, 1.0, r)); a *= a > 0.0 ? 1.0 : 0.0; gl_FragColor = vec4(0.0, 0.0, 0.0, a); }',
  });
  const mesh = new THREE.Mesh(g, m); mesh.frustumCulled = false; mesh.renderOrder = 2;
  scene.add(mesh);
  // re-placed on every perch: her yaw / hips, the ledge top and its outward normal (blobs stay on the top surface)
  const ax = new THREE.Vector3(), az = new THREE.Vector3(), hipW = new THREE.Vector3(), ledgeN = new THREE.Vector3(), seatP = new THREE.Vector3();
  const c = new THREE.Vector3();
  let topY = seat.y + 0.004, sc = 1;
  contact = {
    place(p) {
      const yaw = capri.root.rotation.y; sc = p.scale || 1;
      ax.set(Math.cos(yaw), 0, -Math.sin(yaw)); az.set(Math.sin(yaw), 0, Math.cos(yaw));
      capri.root.updateMatrixWorld(true);
      hipW.copy(capri.root.localToWorld(blobs[0].model.clone()));
      seatP.copy(p.seat); topY = p.seat.y + 0.004;
      const ly = p.ledgeYaw ?? 0; ledgeN.set(Math.sin(ly), 0, Math.cos(ly));
    },
    update(vis = 1) {
      mesh.visible = vis > 0.01;
      for (let i = 0; i < n; i++) {
        const b = blobs[i]; let a = b.a * vis;
        if (b.bone) {
          b.bone.getWorldPosition(c);
          const h = c.y - seatP.y; // wrist height above the ledge top
          a *= 1 - THREE.MathUtils.smoothstep(h, 0.06 * sc, 0.3 * sc);
          c.addScaledVector(ax, Math.sign((c.x - hipW.x) * ax.x + (c.z - hipW.z) * ax.z) * 0.06 * sc); // under the palm, past the wrist
        } else c.copy(hipW);
        // stay on the top surface: no further out than just behind the front edge
        const over = (c.x - seatP.x) * ledgeN.x + (c.z - seatP.z) * ledgeN.z + 0.02; if (over > 0) c.addScaledVector(ledgeN, -over);
        c.y = topY;
        for (let k = 0; k < 4; k++) {
          const sx = (k === 1 || k === 2 ? 0.5 : -0.5) * b.sx * sc, sz = (k >= 2 ? -0.5 : 0.5) * b.sz * sc;
          _cv.set(c.x + ax.x * sx + az.x * sz, c.y, c.z + ax.z * sx + az.z * sz);
          const o2 = (_cv.x - seatP.x) * ledgeN.x + (_cv.z - seatP.z) * ledgeN.z; if (o2 > 0) _cv.addScaledVector(ledgeN, -o2);
          pos[(i * 4 + k) * 3] = _cv.x; pos[(i * 4 + k) * 3 + 1] = _cv.y; pos[(i * 4 + k) * 3 + 2] = _cv.z;
          al[i * 4 + k] = a;
        }
      }
      g.attributes.position.needsUpdate = true; g.attributes.aA.needsUpdate = true;
    },
  };
}
const _cv = new THREE.Vector3();

// a soft backlight behind her torso and hair (one additive card, depth-tested so the coping and her body hide it
// where they're in front): her black suit, hair underside and outline separate from the dark city behind
if (capri) {
  const bm = new THREE.ShaderMaterial({
    uniforms: { uPink: { value: new THREE.Color(PALETTE.pink) }, uBlue: { value: new THREE.Color(0x2f7dff) }, uA: { value: 0.16 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `uniform vec3 uPink, uBlue; uniform float uA; varying vec2 vUv;
      void main(){ vec2 c = vUv * 2.0 - 1.0; float r = length(c * vec2(1.0, 0.85));
        float a = exp(-r * r * 2.6) * (1.0 - smoothstep(0.55, 1.0, r));
        vec3 col = mix(uBlue, uPink, smoothstep(-0.7, 0.6, c.x * 0.6 - c.y * 0.8));
        gl_FragColor = vec4(col * a * uA, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
  });
  const card = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 2.3), bm);
  card.renderOrder = -1; card.frustumCulled = false;
  scene.add(card);
  backCard = {
    place(p) {
      const s = p.scale || 1, bb = capri.getBounds(), c = bb.getCenter(new THREE.Vector3());
      const back = new THREE.Vector3().subVectors(c, frame.pos).setY(0).normalize();
      card.position.set(c.x, (bb.min.y + bb.max.y) / 2 + 0.25 * s, c.z).addScaledVector(back, 0.75 * s);
      card.scale.setScalar(s);
      card.lookAt(card.position.clone().sub(back));
    },
    // vis: how much of her is there; flash: the teleport flash briefly lights the air behind her
    update(vis, flash) { bm.uniforms.uA.value = 0.16 * vis + 0.5 * flash; bm.uniforms.uPink.value.setHex(PALETTE.pink); card.visible = bm.uniforms.uA.value > 0.002; },
  };
}

if (capri) { contact?.place(perches[0]); backCard?.place(perches[0]); }

// perches: city.perches (README contract); without a city only the default seat (no teleports)
function buildPerches(rotY0) {
  const src = Array.isArray(city?.perches) && city.perches.length ? city.perches : null;
  if (src) {
    return src.map((p, i) => ({
      id: p.id || 'perch' + i, seat: p.seat.clone(), rotationY: p.rotationY ?? rotY0, scale: p.scale || 1,
      depth: p.depth || 'mid', portrait: p.portrait !== false,
      ledgeYaw: p.ledgeYaw ?? ((p.rotationY ?? rotY0) - rotY0 + seatRotY), ledgeW: p.ledgeW || 1.5,
      view: p.view || null, viewPortrait: p.viewPortrait || null, // the city's suggested cameras (optional)
      cropY: p.cropY ?? null, // optional: only her part above this model-space height has to fit
    }));
  }
  // guard only (no city): just the default seat, so she never teleports into thin air
  return [{ id: 'parapet', seat: seat.clone(), rotationY: rotY0, scale: 1, depth: 'mid', portrait: true, ledgeYaw: seatRotY, view: null, viewPortrait: null }];
}
function placeCapri(p) {
  if (!capri || !p) return;
  capri.setPosture?.(p.posture || 'seated');
  capri.root.rotation.y = yawOf(p);
  capri.root.scale.setScalar(p.scale || 1);
  perchRoot(p, capri.root.position);
  capri.root.updateMatrixWorld(true);
  document.body.classList.toggle('perch-fore', p.depth === 'fore');
}

// ------------------------------------------------------------------------------------------------
// teleport: now and then she jumps to another perch (capri.teleportOut -> move -> capri.teleportIn), never
// mid-sentence: while idle every ~25-60 s, and sometimes (~25%) right after you send, before her reply starts
// (the reply is held until she has landed). The camera re-frames over ~1.1 s if the new perch needs it, the bubble
// glitches out across the jump and comes back at her new head, and the post pass flashes / splits with her.
// ------------------------------------------------------------------------------------------------
const TP = {
  busy: false, pendingAt: 0, nextIdleAt: 40 + Math.random() * 25, lastAt: -100, prevIdx: -1,
  bubbleHide: 0, bubbleWant: 0, bubbleBackAt: 0, lastRect: null,
  cam: null, // { fromPos, fromTgt, t0, dur }
  fxK: 1,    // screen flash / glitch scale for the jump (fxScale)
  seq: 0,    // bumped per jump: a jump that never finished (capri died mid-jump) can't clobber a later one
};
const portraitNow = () => innerWidth / innerHeight < 0.8;
function perchOk(i) { const p = perches[i]; return !!p && (p.portrait || !portraitNow()); }
function pickPerch() {
  const cand = [], a = innerWidth / innerHeight;
  for (let i = 0; i < perches.length; i++) {
    if (i === perchIdx || !perchOk(i)) continue;
    // never the spot she just came from twice in a row (when there's a choice); home (perches[0]) a bit more often;
    // spots inside the master shot far more often than ones that need a camera cut (the jump should read as her
    // flashing across the rooftop, not as a scene edit)
    const w = (i === TP.prevIdx && perches.length > 2 ? 0.25 : 1) * (i === 0 ? 1.6 : 1) * (inMaster(i, a) ? 1 : 0.2);
    cand.push([i, w]);
  }
  if (!cand.length) return -1;
  let r = Math.random() * cand.reduce((a, c) => a + c[1], 0);
  for (const [i, w] of cand) { if ((r -= w) <= 0) return i; }
  return cand[cand.length - 1][0];
}
const surfaceOf = (p) => ({ point: p.seat, yaw: p.ledgeYaw ?? 0, w: p.ledgeW || 1.5 });
// how much of the frame's height she fills on perch p (current shot) -> the screen-wide flash / glitch scale
function fxScale(p) {
  try {
    if (!p || !perchCorners(Object.assign(_noCrop, p, { cropY: null }), _corn)) return 1;
    const a = innerWidth / innerHeight, R = safeRect(a), b = screenBox(_corn, frame.pos, frame.target, frame.fov, a, R);
    const hf = clamp((Math.min(b.y1, R.H) - Math.max(0, b.y0)) / R.H, 0.05, 1);
    return clamp(0.42 / hf, 0.45, 1);
  } catch { return 1; }
}
const zap = (k) => { try { voice.zap?.(k); } catch { /* sound must never break a jump */ } };
// re-frame for the perch she's on now; returns true when that's a hard camera cut (the target isn't in the master shot)
function reframeForJump() {
  const fromPos = camBase.pos.clone(), fromTgt = camBase.target.clone();
  computeFrame();
  const dMove = fromPos.distanceTo(frame.pos) + fromTgt.distanceTo(frame.target);
  const ease = !reduced && dMove > 1e-3 && dMove < 0.35;
  TP.cam = ease ? { fromPos, fromTgt, t0: clock, dur: 0.45 } : null;
  return dMove >= 0.35;
}
// why: 'idle' (her old line is dropped: the bubble stays down until she next speaks), 'send' (her reply is held and
// the bubble comes back with her last arrival tick), or undefined (tests)
async function teleportTo(idx, why) {
  if (!capri?.teleportOut || TP.busy || perches.length < 2) return false;
  if (idx === undefined || idx === null) idx = pickPerch();
  idx = +idx;
  if (!(idx >= 0 && idx < perches.length) || idx === perchIdx) return false;
  TP.busy = true; TP.lastAt = clock; TP.visAt = 0; TP.inAt = 0; TP.fxK = fxScale(perches[perchIdx]);
  const my = ++TP.seq;
  const fromIdx = perchIdx, from = perches[fromIdx], to = perches[idx];
  let moved = false;
  try {
    TP.bubbleWant = 1;
    zap('out');
    await capri.teleportOut({ surface: surfaceOf(from), reduced, fx: TP.fxK });
    if (why === 'idle' && !state.net && !state.speaking && !state.queue.length) { try { bubble.hide(); } catch { /* optional */ } }
    await capri.wait(reduced ? 0.15 : 0.11);
    TP.prevIdx = fromIdx; perchIdx = idx;
    rollPosture(from); // next time she lands there she may pick another pose
    placeCapri(to); moved = true;
    // the camera holds still when the new spot is inside the master shot (she flashes across it, the after-image at
    // the old spot); otherwise a hard cut while she's gone (the show's quick cuts; with reduced motion too), and the
    // new spot is held empty for a beat before she snaps in
    const cut = reframeForJump();
    if (cut) capri.clearAfterImage?.(); // the after-image would hang where the new shot no longer matches
    contact?.place(to); backCard?.place(to);
    TP.fxK = fxScale(to);
    if (cut && !reduced) await capri.wait(0.1);
    zap('in');
    TP.inAt = clock;
    await capri.teleportIn({ surface: surfaceOf(to), reduced, fx: TP.fxK });
  } catch (e) {
    console.error('[capri] teleport failed', e);
    await recoverJump(moved ? idx : fromIdx, fromIdx);
  }
  if (my !== TP.seq) return false; // the watchdog already gave up on this jump
  TP.busy = false;
  TP.bubbleBackAt = clock; TP.bubbleWant = 0;
  TP.nextIdleAt = clock + 25 + Math.random() * 35;
  return true;
}
// never leave her invisible: something threw (or hung) between teleportOut and teleportIn -> put her on a valid perch
// (the target if she got there, else where she came from), re-frame, and bring her back in
async function recoverJump(i, fallback) {
  try {
    const st = capri?.teleportState; if (!capri || !st || (st.phase !== 'gone' && st.visible > 0.99)) return;
    try { perchIdx = i; placeCapri(perches[i]); } catch { perchIdx = fallback; placeCapri(perches[fallback]); }
    try { reframeForJump(); contact?.place(perches[perchIdx]); backCard?.place(perches[perchIdx]); } catch (e) { console.error(e); }
    if (st.phase === 'gone' || st.phase === null) await capri.teleportIn({ surface: surfaceOf(perches[perchIdx]), reduced }).catch(() => {});
  } catch (e) { console.error('[capri] teleport recovery failed', e); }
}
function teleportTick(dt) {
  // a jump that can't progress (capri errored out and was dropped, or it hangs) must never hold her reply forever,
  // nor leave her invisible
  if (TP.busy && (!capri || clock - TP.lastAt > 3)) { TP.seq++; TP.busy = false; TP.bubbleWant = 0; TP.bubbleBackAt = clock; TP.cam = null; recoverJump(perchIdx, 0); }
  if (!capri?.teleportOut) { TP.pendingAt = 0; return; }
  // the bubble comes back with her last arrival glitch tick (~0.12 s after she's whole again), not when the jump settles
  const ts = capri.teleportState;
  if (TP.busy && TP.inAt && ts && ts.visible >= 0.99) { if (!TP.visAt) TP.visAt = clock; else if (clock - TP.visAt > 0.12) TP.bubbleWant = 0; }
  if (perches.length < 2 || !booted) return;
  // a send-triggered jump, scheduled by submit() (only if her reply hasn't started)
  if (TP.pendingAt && clock >= TP.pendingAt) { TP.pendingAt = 0; if (!TP.busy && !state.firstChar) teleportTo(undefined, 'send'); }
  // idle jumps
  if (clock >= TP.nextIdleAt) {
    const idle = !TP.busy && !state.speaking && !state.net && !state.queue.length && state.capriState === 'idle'
      && clock - typingAt > 6 && !drawerOpen && !document.hidden && clock - TP.lastAt > 20;
    if (idle) teleportTo(undefined, 'idle'); else TP.nextIdleAt = clock + 3 + Math.random() * 4;
  }
  // the current perch isn't allowed any more (a phone turned to portrait): jump home, but not mid-sentence
  if (!TP.busy && !perchOk(perchIdx) && perchOk(0) && !state.speaking && !state.net && !state.queue.length) teleportTo(0, 'idle');
}
// after bubble.update: hide (glitch) the bubble across the jump; bring it back once it has re-anchored at her new head
const _bubU = () => bubble.group?.children?.[0]?.material?.uniforms;
let bubbleSpill = null;
scene.traverse((o) => { if (!bubbleSpill && o.isMesh && o.renderOrder === 9 && o.material?.uniforms?.uA && o.material.blending === THREE.AdditiveBlending) bubbleSpill = o; });
function bubbleTeleportFx(dt) {
  const r = bubble.getScreenRect?.();
  let settled = true;
  if (r && TP.lastRect) settled = Math.abs(r.x - TP.lastRect.x) + Math.abs(r.y - TP.lastRect.y) < 60 * dt; // < ~60 px/s
  if (r) { TP.lastRect = TP.lastRect || { x: 0, y: 0 }; TP.lastRect.x = r.x; TP.lastRect.y = r.y; } else TP.lastRect = null;
  let want = TP.bubbleWant;
  if (!want && TP.bubbleBackAt && clock - TP.bubbleBackAt < 0.6 && !settled) want = 1; // still springing over to her
  TP.bubbleHide = want ? Math.min(1, TP.bubbleHide + dt / 0.12) : Math.max(0, TP.bubbleHide - dt / 0.14);
  if (TP.bubbleHide <= 0) return;
  const u = _bubU();
  // (fade-in: the panel is opaque again within ~0.14 s; the glitch only rides the first half of it)
  if (u) { u.uAlpha.value *= 1 - TP.bubbleHide; u.uGlitch.value = Math.max(u.uGlitch.value, Math.min(1, Math.max(0, TP.bubbleHide * 2 - 0.6))); }
  if (bubbleSpill) bubbleSpill.material.uniforms.uA.value *= 1 - TP.bubbleHide;
}
const smoother = (x) => x * x * x * (x * (x * 6 - 15) + 10);
const camBase = { pos: new THREE.Vector3(), target: new THREE.Vector3() };

// ------------------------------------------------------------------------------------------------
// overlay: status, settings, sound, transcript, toast
// ------------------------------------------------------------------------------------------------
const els = {
  form: $('form'), input: $('input'), send: $('send'), lbl: $('send').querySelector('.lbl'), chips: $('chips'),
  status: $('status'), dose: $('dose'), doseOut: $('dose-out'), log: $('log'), sr: $('sr'), drawer: $('drawer'),
  scrim: $('scrim'), settings: $('settings'), btnSet: $('btn-set'), btnLog: $('btn-log'), btnSound: $('btn-sound'),
  swSound: $('sw-sound'), dock: $('dock'),
};

let toastTimer = 0;
function toast(msg, kind = 'ERR', ms = 4200) {
  const t = $('toast');
  $('toast-k').textContent = kind;
  $('toast-t').textContent = msg;
  t.classList.toggle('ok', kind !== 'ERR');
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), ms);
}

let info = null, mode = 'connecting', lastTps = 0;
function renderStatus() {
  const s = els.status, txt = s.querySelector('.txt');
  s.classList.toggle('live', mode === 'live');
  if (mode === 'connecting' || mode === 'loading' || mode === 'disconnected') {
    const label = mode === 'loading' ? 'LOCAL MODEL LOADING' : mode === 'disconnected' ? 'LOCAL LINK OFFLINE' : 'CONNECTING LOCAL CAPRI';
    txt.textContent = label;
    s.title = mode === 'disconnected' ? 'Run locally to reconnect, or choose Use demo.' : 'Leave the Capri Terminal window open.';
  } else if (mode === 'live') {
    const size = info && info.size ? String(info.size).toUpperCase().replace(/[^A-Z0-9.]/g, '').slice(0, 12) : '27B';
    const tps = lastTps ? ` · <b>${(+lastTps).toFixed(0)}</b> tok/s` : '';
    txt.innerHTML = `<span class="long">LINK: LOCAL · ${size}${tps}</span><span class="short">LIVE · ${size}</span>`;
    s.title = 'Talking to the steered model on this machine';
  } else {
    txt.innerHTML = '<span class="long">DEMO · RECORDED REPLIES</span><span class="short">DEMO</span>';
    s.title = 'Demo: replies are recordings. Run locally to generate new replies on your Mac.';
  }
  $('set-mode').textContent = mode === 'live' ? 'LIVE' : mode === 'demo' ? 'DEMO' : mode === 'loading' ? 'LOADING' : 'OFFLINE';
  // the dose only steers the live model: in demo mode the slider is dimmed and says so
  els.settings.classList.toggle('demo', mode !== 'live');
  els.dose.setAttribute('aria-disabled', String(mode !== 'live'));
  const dh = $('demo-hint'); if (dh) dh.hidden = mode !== 'demo';
}

// dose
function setDoseRange(min, max, def) {
  const d = els.dose;
  d.min = min; d.max = max;
  d.step = max - min > 3 ? 0.1 : 0.05;
  $('tick-min').textContent = (+min).toFixed(1).replace(/\.0$/, '');
  $('tick-max').textContent = (+max).toFixed(1).replace(/\.0$/, '');
  const saved = store.get('dose', null);
  d.value = saved !== null && saved >= min && saved <= max ? saved : def;
  onDose();
}
function doseVal() { return parseFloat(els.dose.value); }
function onDose() {
  const d = doseVal(), min = +els.dose.min, max = +els.dose.max;
  els.doseOut.textContent = d.toFixed(2);
  els.dose.style.setProperty('--p', `${((d - min) / (max - min || 1)) * 100}%`);
  $('side-dose').textContent = 'DOSE ' + d.toFixed(2);
}
els.dose.addEventListener('input', () => { onDose(); store.set('dose', doseVal()); });
setDoseRange(0, 1.2, 1.0);

// sound
const voice = createVoice();
function setSound(on, user) {
  voice.setEnabled(on);
  els.btnSound.setAttribute('aria-pressed', String(on));
  els.swSound.setAttribute('aria-checked', String(on));
  if (user) { store.set('sound', on); voice.resume(); }
}
setSound(!!store.get('sound', false), false);
els.btnSound.addEventListener('click', () => setSound(!voice.enabled, true));
els.swSound.addEventListener('click', () => setSound(!voice.enabled, true));
// audio can only start from a gesture
const gesture = () => voice.resume();
addEventListener('pointerdown', gesture, { passive: true });
addEventListener('keydown', gesture);

// settings popover
function openSettings(on) {
  els.settings.hidden = !on;
  els.btnSet.setAttribute('aria-expanded', String(on));
  if (on) els.dose.focus({ preventScroll: true });
  layout(); // the bubble keeps clear of the open panel
}
// keyboard users: tabbing out of the panel closes it (focus moving to the settings button itself doesn't)
els.settings.addEventListener('focusout', (e) => {
  const to = e.relatedTarget;
  if (!els.settings.hidden && to && !els.settings.contains(to) && !els.btnSet.contains(to)) openSettings(false);
});
els.btnSet.addEventListener('click', () => openSettings(els.settings.hidden));
document.addEventListener('pointerdown', (e) => {
  if (!els.settings.hidden && !els.settings.contains(e.target) && !els.btnSet.contains(e.target)) openSettings(false);
});

// transcript drawer
let drawerOpen = false;
function openDrawer(on) {
  drawerOpen = on;
  els.drawer.classList.toggle('open', on);
  els.drawer.setAttribute('aria-hidden', String(!on));
  els.drawer.inert = !on;
  els.scrim.classList.toggle('on', on);
  els.btnLog.setAttribute('aria-expanded', String(on));
  if (on) { $('btn-close').focus(); els.log.scrollTop = els.log.scrollHeight; } else els.btnLog.focus({ preventScroll: true });
}
els.drawer.inert = true;
// the drawer is modal (it has a scrim): Tab / Shift+Tab cycle inside it
els.drawer.addEventListener('keydown', (e) => {
  if (e.key !== 'Tab' || !drawerOpen) return;
  const f = [...els.drawer.querySelectorAll('button, a[href], [tabindex]:not([tabindex="-1"])')].filter((x) => !x.disabled && x.offsetParent !== null);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
});
document.addEventListener('focusin', (e) => { if (drawerOpen && !els.drawer.contains(e.target)) $('btn-close').focus(); });
els.btnLog.addEventListener('click', () => openDrawer(!drawerOpen));
$('btn-close').addEventListener('click', () => openDrawer(false));
els.scrim.addEventListener('click', () => openDrawer(false));
addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (drawerOpen) openDrawer(false);
  else if (!els.settings.hidden) { openSettings(false); els.btnSet.focus(); }
  else if (busy()) stop();
});

function logAdd(who, text, cls = '') {
  els.log.querySelector('.empty')?.remove();
  const li = document.createElement('li');
  li.className = who + (cls ? ' ' + cls : '');
  const w = document.createElement('span'); w.className = 'who';
  // (the dose only applies live; demo replies were recorded at 1.00)
  w.textContent = who === 'you' ? 'you' : who === 'sys' ? 'sys' : `capri${cls !== 'local' && mode === 'live' ? ' · ' + doseVal().toFixed(2) : cls !== 'local' ? ' · demo' : ''}`;
  const t = document.createElement('span'); t.className = 'txt'; t.textContent = text;
  li.append(w, t); els.log.append(li);
  els.log.scrollTop = els.log.scrollHeight;
  return t;
}
const cleanMd = (s) => s.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/^#{1,4} /gm, '');

function resetConversation(greet = true) {
  convId++;            // any submit() still awaiting the old conversation bails out when it resolves
  stop();
  chat.reset ? chat.reset() : (chat.history.length = 0);
  state.cut = null; state.net = false;
  hideEcho(true);
  els.log.innerHTML = '<li class="empty">Nothing yet. Say something.</li>';
  showChips(true);
  if (drawerOpen) openDrawer(false);
  if (greet) sayLocal(pickOpener(true));
  else {
    state.queue = ''; state.spoken = ''; state.speaking = false; state.local = false;
    bubble.clear(); bubble.hide(); els.sr.textContent = '';
  }
  syncSend();
}
$('btn-clear').addEventListener('click', () => resetConversation());

// ------------------------------------------------------------------------------------------------
// layout: keyboard (visualViewport), bubble insets
// ------------------------------------------------------------------------------------------------
function layout() {
  const vv = window.visualViewport;
  let kb = 0;
  if (vv) kb = Math.max(0, innerHeight - vv.height - vv.offsetTop);
  if (kb < 60) kb = 0;
  kbPx = kb;
  document.documentElement.style.setProperty('--kb', kb + 'px');
  document.body.classList.toggle('kb', kb > 0);
  const dockTop = els.dock.getBoundingClientRect().top;
  const dockH = innerHeight - kb - dockTop;
  document.documentElement.style.setProperty('--dock-h', dockH + 'px');
  const hdr = document.querySelector('header').getBoundingClientRect().bottom;
  const narrow = innerWidth < 640;
  let right = narrow ? 12 : 24, top = Math.round(hdr + 10);
  if (els.settings && !els.settings.hidden) {
    const r = els.settings.getBoundingClientRect();
    if (narrow) top = Math.max(top, Math.round(r.bottom + 10)); else right = Math.max(right, Math.round(innerWidth - r.left + 12));
  }
  bubble.setInsets?.({ top, bottom: Math.round(innerHeight - dockTop + 12), left: narrow ? 12 : 24, right });
}
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setPixelRatio(pixelRatio());
  renderer.setSize(w, h, false);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(w, h);
  const bw = Math.max(2, Math.round(w * renderer.getPixelRatio() * 0.5)), bh = Math.max(2, Math.round(h * renderer.getPixelRatio() * 0.5));
  bloomRT.setSize(bw, bh);
  bloom.setSize(bw, bh);
  cyber.uniforms.uRes.value.set(w, h);
  cyber.uniforms.uDpr.value = renderer.getPixelRatio();
  computeFrame();
  layout();
}
addEventListener('resize', resize);
visualViewport?.addEventListener('resize', layout);
visualViewport?.addEventListener('scroll', layout);
resize();

// ------------------------------------------------------------------------------------------------
// input: auto-grow textarea, Enter to send (IME-safe), chips
// ------------------------------------------------------------------------------------------------
let typingAt = -10;
function grow() {
  const ta = els.input;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 132) + 'px';
  layout();
}
// where the caret sits along the prompt bar (0..1): her eyes follow it while she reads
const caretFrac = () => {
  const ta = els.input, w = ta.clientWidth || 300;
  const before = ta.value.slice(0, ta.selectionStart ?? ta.value.length).split('\n').pop();
  return clamp((before.length * 8.4) % w / w, 0, 1);
};
els.input.addEventListener('input', () => { typingAt = clock; grow(); syncSend(); try { capri?.typing?.(caretFrac()); } catch { /* optional */ } });
els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    if (!busy()) els.form.requestSubmit();
    else if (els.input.value.trim()) nudgeStop();
  }
});
// Enter while she's still replying: nothing is sent (the reply would be cut); the STOP button pulses and the
// placeholder-style hint says how to cut her off
let nudgeT = 0;
function nudgeStop() {
  els.send.classList.remove('nudge'); void els.send.offsetWidth; els.send.classList.add('nudge');
  const h = $('busy-hint'); if (h) { h.hidden = false; els.dock.classList.add('hinting'); requestAnimationFrame(() => h.classList.add('on')); }
  clearTimeout(nudgeT);
  nudgeT = setTimeout(() => { els.send.classList.remove('nudge'); if (h) { h.classList.remove('on'); els.dock.classList.remove('hinting'); setTimeout(() => { if (!h.classList.contains('on')) h.hidden = true; }, 300); } }, 2200);
}
els.form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (busy()) { stop(); return; }
  submit(els.input.value);
});
els.input.addEventListener('focus', () => setTimeout(layout, 250));
els.input.addEventListener('blur', () => setTimeout(layout, 250));

const OPENERS = ['Who are you, anyway?', 'Tell me a joke about vegetables.', 'How far away is the moon?',
  'Would you rather live without cheese or without chocolate?', 'Is golf a real sport?', 'I just got a goldfish. Name him.',
  'Why do we have leap years?', 'my new shoes squeak on every step. fix??', 'Name my goldfish.', 'Why leap years?'];
function showChips(on) {
  if (on) {
    els.chips.innerHTML = '';
    // phones: two short chips (one row) so the stack doesn't bury her
    const narrow = innerWidth < 560;
    const pool = OPENERS.slice(1).filter((q) => !narrow || q.length <= 22).sort(() => Math.random() - 0.5);
    const picks = [OPENERS[0], ...pool.slice(0, narrow ? 1 : 2)];
    for (const q of picks) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip';
      const s = document.createElement('span'); s.textContent = q; b.append(s);
      b.addEventListener('click', () => { if (!busy()) submit(q); });
      els.chips.append(b);
    }
  }
  els.chips.hidden = !on;
  layout();
}
showChips(true);

// ------------------------------------------------------------------------------------------------
// conversation + the speech clock
// ------------------------------------------------------------------------------------------------
const state = {
  queue: '',          // received, not yet spoken
  spoken: '',         // revealed so far (this reply)
  wait: 0,            // seconds until the next char
  net: false,         // a request is running
  firstChar: false,   // has this reply started speaking
  speaking: false,    // the bubble holds a reply in progress
  lastCharAt: -10,
  local: false,       // current speech is a local line (greeting), not a model reply
  logEl: null,
  glitch: 0,
  capriState: 'idle',
  cut: null,         // set by STOP while a request is running: what she'd said when cut off
};

const chat = createChat({
  onMode(m, inf) {
    const was = mode; mode = m; if (inf) info = inf;
    if (was !== m) { lastTps = 0; resetConversation(false); }
    if (inf && m === 'live') setDoseRange(inf.min_dose ?? 0, inf.max_dose ?? 1.2, inf.default_dose ?? 1);
    renderStatus();
    // toasts only for real changes after the first mode was decided (not on the initial probe)
    if (m === 'disconnected') toast('Local connection lost. Run locally to retry, or choose Use demo.', 'LINK', 8000);
    else if (m === 'live') toast('Local Capri connected. Starting a new conversation.', 'LINK');
    else if (m === 'demo' && was !== 'connecting') toast('Demo mode · recorded replies. Starting a new conversation.', 'DEMO');
  },
  onStart() {},
  onToken(t) {
    if (state.local) return;
    state.queue += t;
  },
  onDone(r) {
    state.net = false;
    if (r && r.tps) { lastTps = r.tps; renderStatus(); }
    if (!r.reply && !r.aborted) { bubble.end(); }
    syncSend();
  },
  onError(msg) {
    state.net = false;
    state.queue = '';
    // in the bubble: an in-character line (plus whatever she'd already said); the technical message goes to the
    // toast and the transcript as a SYS line, not as her speech
    bubble.setError(ERROR_LINES[(Math.random() * ERROR_LINES.length) | 0]);
    toast(msg, 'ERR', 7000);
    capri?.react('glitch');
    state.glitch = Math.max(state.glitch, 0.8);
    if (state.logEl && !state.spoken) { state.logEl.closest('li')?.remove(); state.logEl = null; }
    logAdd('sys', msg, 'err');
    hideEcho(); // the prompt comes back into the input box
    syncSend();
  },
});

const ERROR_LINES = ['Signal dropped. Try again.', 'Tch. Lost you there. Again?', 'Static. Say that again.'];
function busy() { return state.net || (!state.local && (state.queue.length > 0 || state.speaking)); }

function syncSend() {
  const b = busy();
  els.send.classList.toggle('stop', b);
  els.lbl.textContent = b ? 'STOP' : 'SEND';
  els.send.setAttribute('aria-label', b ? 'Stop her reply' : 'Send');
  const canSend = mode === 'demo' || mode === 'live';
  els.input.disabled = !canSend;
  els.input.placeholder = canSend ? 'Say something to Capri…' : mode === 'loading' ? 'Capri is loading on your Mac…' : 'Reconnect or choose Use demo to chat';
  els.send.disabled = !b && (!els.input.value.trim() || !canSend);
  els.form.classList.toggle('busy', b);
  if (!b) { const h = $('busy-hint'); if (h && !h.hidden) { h.classList.remove('on'); h.hidden = true; els.dock.classList.remove('hinting'); } }
}

async function submit(text) {
  text = String(text || '').trim();
  if (!text || busy()) return;
  voice.resume();
  if (state.local) { state.queue = ''; finishSpeech(true); } // cut the greeting short
  els.input.value = ''; grow();
  showChips(false);
  logAdd('you', text);
  showEcho(text);
  try { capri?.react('nod'); } catch (e) { console.error(e); } // "got it", before she starts thinking
  lastReactAt = clock - 1.5; cue.reset();
  const my = convId, mySub = ++submitSeq;
  // new reply
  state.queue = ''; state.spoken = ''; state.wait = 0.25; state.firstChar = false;
  state.speaking = true; state.local = false; state.net = true; state.logEl = null;
  bubble.begin();
  syncSend();
  // now and then she jumps to another perch first, then answers from there (her reply is held until she lands)
  if (capri?.teleportOut && perches.length > 1 && !TP.busy && clock - TP.lastAt > 12 && Math.random() < 0.25) TP.pendingAt = clock + 0.5;
  state.cut = null;
  const res = await chat.send(text, doseVal());
  if (my !== convId || mySub !== submitSeq || (res && res.stale)) return res; // the chat was cleared / a newer send took over
  if (state.cut != null) {
    if (res && res.reply) {
      trimHistory(state.cut);
      if (!state.cut.trim() && !els.input.value) { els.input.value = text; grow(); } // cut before a word: give the text back
    }
    state.cut = null;
  }
  if (res && res.restore) {
    // nothing came back (aborted or failed before any text): give the text back
    if (!els.input.value) { els.input.value = res.restore; grow(); }
    if (!res.error) finishSpeech(true); else { state.speaking = false; state.local = false; }
  }
  syncSend();
  return res;
}

// STOP cuts her off mid-sentence: the model history should hold what she actually said, not the unspoken rest
// (in demo mode the whole reply has usually arrived already; live, the aborted request may have run ahead).
function trimHistory(said) {
  const h = chat.history, last = h[h.length - 1];
  if (!last || last.role !== 'assistant') return;
  said = said.trim();
  if (said) last.content = said;
  else h.splice(h.length - 2, 2); // nothing was spoken: drop the whole turn
}
function stop() {
  const wasNet = state.net, cutting = !state.local && (state.queue.length > 0 || state.speaking);
  chat.stop();
  state.net = false;
  TP.pendingAt = 0; // a send-triggered jump that hasn't started yet is dropped with the reply
  if (cutting) {
    const said = state.spoken;
    if (wasNet) state.cut = said; // the request is still settling: submit() trims once chat.send resolves
    else trimHistory(said);
  }
  if (state.queue.length || state.speaking) {
    state.queue = '';
    if (state.spoken && !/[.!?…]$/.test(state.spoken.trim())) { bubble.append('—'); state.spoken += '—'; }
    finishSpeech();
  }
  syncSend();
}

function finishSpeech(silent) {
  const was = state.speaking;
  state.speaking = false;
  state.queue = '';
  if (!was) return;
  bubble.end();
  TP.nextIdleAt = Math.max(TP.nextIdleAt, clock + 8); // no jump right after she finishes a line
  if (!silent && state.spoken && !state.local) { try { capri?.endUtterance?.(); } catch (e) { console.error(e); } }
  echoFadeAt = clock + 3.5;
  if (!silent && state.spoken) {
    const txt = cleanMd(state.spoken);
    if (state.logEl) state.logEl.textContent = txt; else if (!state.local) logAdd('capri', txt);
    els.sr.textContent = 'Capri: ' + txt;
  }
  state.local = false;
}

// a local (not-in-history) line, e.g. the greeting
function sayLocal(text) {
  state.queue = text; state.spoken = ''; state.wait = 0.5; state.firstChar = false;
  state.speaking = true; state.local = true; state.net = false; state.logEl = null;
  cue.reset();
  bubble.begin();
  logAdd('capri', cleanMd(text), 'local'); // shown in the transcript, but not part of the model history
}
function pickOpener(again) {
  const first = ['You lost? ...Fine. Talk.', 'Oh. Company. Talk.'];
  const more = ['New channel. Go on, then.', 'Clean slate. Don’t waste it.', 'Hey. Make it quick.'];
  const visits = store.get('visits', 0);
  if (!again) store.set('visits', visits + 1);
  if (again) return more[(Math.random() * more.length) | 0];
  return visits === 0 ? first[0] : [...first, ...more][(Math.random() * (first.length + more.length)) | 0];
}

// ---- emotional cues: scan the revealed text and fire capri.react() at the matching character
// (one react every ~2.5 s at most so she doesn't twitch)
let lastReactAt = -10;
const CUES = {
  laugh: new Set(['ha', 'hah', 'haha', 'hahaha', 'heh', 'hehe', 'lol', 'lmao', 'pfft']),
  shrug: new Set(['dunno', 'whatever', 'idk', 'maybe', 'eh', 'meh', 'shrug', 'guess', 'probably', 'kinda']),
  nod: new Set(['yeah', 'yes', 'sure', 'right', 'fine', 'yep', 'yup', 'okay', 'ok', 'obviously', 'totally', 'definitely']),
  surprised: new Set(['what', 'wait', 'whoa', 'woah', 'huh', 'seriously', 'really']),
};
const cue = { word: '', words: 0, sentenceWords: 0, prev: '', reset() { this.word = ''; this.words = 0; this.sentenceWords = 0; this.prev = ''; } };
function fireReact(kind) {
  if (!capri || clock - lastReactAt < 2.5) return false;
  lastReactAt = clock;
  try { capri.react(kind); } catch (e) { console.error(e); }
  return true;
}
function cueTick(ch) {
  if (/[\p{L}']/u.test(ch)) { cue.word += ch.toLowerCase(); return; }
  const w = cue.word.replace(/'/g, ''); cue.word = '';
  if (w) {
    cue.words++; cue.sentenceWords++;
    if (CUES.laugh.has(w)) fireReact('laugh');
    else if (CUES.surprised.has(w) && (ch === '?' || ch === '!' || (w === 'wait' && cue.sentenceWords === 1))) fireReact('surprised');
    else if (CUES.nod.has(w) && cue.words <= 2) fireReact('nod');
    else if (CUES.shrug.has(w) || (w === 'know' && cue.prev === 'dont')) fireReact('shrug');
    cue.prev = w;
  }
  if (ch === '!') {
    // "?!" = surprise; a short dry punchline ending in "!" = a laugh, sometimes
    if (state.spoken.endsWith('?!')) fireReact('surprised');
    else if (cue.sentenceWords >= 2 && cue.sentenceWords <= 9 && Math.random() < 0.45) fireReact('laugh');
  }
  if ('.!?…\n'.includes(ch)) cue.sentenceWords = 0;
}

// ---- the user's last line, echoed just above the prompt bar while she answers
let echoFadeAt = Infinity;
function showEcho(text) {
  const e = $('echo'); if (!e) return;
  $('echo-t').textContent = text.length > 140 ? text.slice(0, 137) + '…' : text;
  e.hidden = false; requestAnimationFrame(() => e.classList.add('on'));
  echoFadeAt = Infinity;
  layout();
}
function hideEcho(now) {
  const e = $('echo'); if (!e) return;
  e.classList.remove('on'); echoFadeAt = Infinity;
  if (now) { e.hidden = true; layout(); } else setTimeout(() => { if (!e.classList.contains('on')) { e.hidden = true; layout(); } }, 450);
}
let convId = 0, submitSeq = 0;

const BASE_CPS = 30;
function pauseFor(ch, next) {
  if (ch === '\n') return 0.22;
  if (ch === '.' && next === '.') return 0.09;       // "..." trails
  if ('.!?…'.includes(ch)) return next && next !== ' ' && next !== '\n' && next !== '"' ? 0 : 0.34;
  if (ch === ',' || ch === ';' || ch === ':' || ch === '—') return 0.14;
  return 0;
}
function speechTick(dt) {
  if (!state.speaking) return;
  if (!state.queue.length) {
    // all spoken: done if the network is done too
    if (!state.net) { finishSpeech(); syncSend(); }
    return;
  }
  const backlog = state.queue.length;
  const cps = BASE_CPS * (backlog > 120 ? 1 + (backlog - 120) / 50 : 1);
  const pauseK = backlog > 120 ? 0.35 : 1;
  state.wait -= dt;
  let guard = 0;
  while (state.wait <= 0 && state.queue.length && guard++ < 64) {
    // whole code points, so an emoji never arrives (or gets cut by STOP) as half a surrogate pair
    const ch = String.fromCodePoint(state.queue.codePointAt(0));
    state.queue = state.queue.slice(ch.length);
    if (!state.firstChar) {
      state.firstChar = true;
      state.glitch = Math.max(state.glitch, 0.7);
      if (!state.local && state.logEl === null) state.logEl = logAdd('capri', '');
    }
    bubble.append(ch);
    try { capri?.speak(ch, state.queue.length); } catch (e) { console.error(e); }
    voice.blip(ch);
    state.spoken += ch;
    cueTick(ch);
    state.lastCharAt = clock;
    state.wait += 1 / cps + pauseFor(ch, state.queue[0]) * pauseK;
  }
  if (state.wait < -0.25) state.wait = -0.25;
  if (state.logEl && (frameN & 7) === 0) state.logEl.textContent = cleanMd(state.spoken);
}

// ------------------------------------------------------------------------------------------------
// render loop
// ------------------------------------------------------------------------------------------------
let clock = 0, frameN = 0, frameMs = 0, last = performance.now(), booted = false, rafId = 0;
const pointer = new THREE.Vector2(), pointerS = new THREE.Vector2();
addEventListener('pointermove', (e) => {
  if (e.pointerType === 'touch') return;
  pointer.set((e.clientX / innerWidth) * 2 - 1, (e.clientY / innerHeight) * 2 - 1);
}, { passive: true });
// the reply bubble scrolls back with the wheel / a drag over it / PageUp-PageDown
const overBubble = (x, y) => { const r = bubble.getScreenRect?.(); return !!r && x >= r.x - 8 && x <= r.x + r.w + 8 && y >= r.y - 8 && y <= r.y + r.h + 8; };
canvas.addEventListener('wheel', (e) => {
  if (!bubble.scrollBy || !overBubble(e.clientX, e.clientY)) return;
  const dy = e.deltaY * (e.deltaMode === 1 ? 18 : e.deltaMode === 2 ? (bubble.pageHeight || 200) : 1);
  if (bubble.scrollBy(dy)) e.preventDefault();
}, { passive: false });
addEventListener('keydown', (e) => {
  if ((e.key !== 'PageUp' && e.key !== 'PageDown') || !bubble.scrollable || drawerOpen) return;
  if (bubble.scrollBy((e.key === 'PageUp' ? -1 : 1) * (bubble.pageHeight || 200) * 0.8)) e.preventDefault();
});
// touch: drag over the bubble scrolls it; anywhere else on the scene gives a little parallax, springs back on release
let touchStart = null;
canvas.addEventListener('touchstart', (e) => {
  const t = e.touches[0];
  touchStart = { x: t.clientX, y: t.clientY, lastY: t.clientY, scroll: !!bubble.scrollable && overBubble(t.clientX, t.clientY) };
}, { passive: true });
canvas.addEventListener('touchmove', (e) => {
  if (!touchStart) return; const t = e.touches[0];
  if (touchStart.scroll) { bubble.scrollBy(touchStart.lastY - t.clientY); touchStart.lastY = t.clientY; return; }
  pointer.set(clamp((t.clientX - touchStart.x) / 140, -1, 1), clamp((t.clientY - touchStart.y) / 140, -1, 1));
}, { passive: true });
canvas.addEventListener('touchend', () => { touchStart = null; pointer.set(0, 0); }, { passive: true });

const camPos = new THREE.Vector3(), camTgt = new THREE.Vector3();
let talkI = 0;

function updateCapriState() {
  let s = 'idle';
  // once she has started, she stays 'talking' while the reply is still streaming in, even if the text runs dry for
  // a moment (a slow model): capri.js then holds a pause (mouth shut, eyes on the bubble) instead of switching to the
  // 'thinking' pose mid-sentence
  const talking = state.speaking && state.firstChar && (state.queue.length > 0 || state.net || clock - state.lastCharAt < 0.6);
  if (talking) s = 'talking';
  else if (state.net || (state.speaking && !state.firstChar && !state.local)) s = 'thinking';
  else if (clock - typingAt < 1.5 && document.activeElement === els.input) s = 'listening';
  if (s !== state.capriState) {
    state.capriState = s;
    try { capri?.setState(s); capri?.lookAt(s === 'listening' ? promptWorld() : null); } catch (e) { console.error(e); }
  }
  return s;
}

// a world point ~2.5 m in front of the camera, behind the prompt bar's centre on screen (so she reads what you type)
const _pw = new THREE.Vector3();
function promptWorld() {
  const r = els.form.getBoundingClientRect();
  const nx = ((r.left + r.right) / 2 / innerWidth) * 2 - 1, ny = 1 - ((r.top + r.bottom) / 2 / innerHeight) * 2;
  _pw.set(nx, ny, 0.5).unproject(camera).sub(camera.position).normalize();
  return _pw.multiplyScalar(1.2).add(camera.position).clone(); // (near the camera: a gentle downward look, face still visible)
}
const maxFps = +(new URLSearchParams(location.search).get('fps')) || 0; // debug: cap the frame rate (?fps=5)
// The page shares the GPU with the model. Uncapped, a 120 Hz screen draws 120 frames a second and halves her
// generation speed, so the page draws at most 60 frames a second, and 30 while a reply is generating.
const fpsCap = () => maxFps || (state.net ? 30 : 60);
let lastDraw = 0;
// adaptive resolution: if frames keep arriving slowly (GPU/fill-rate bound), step the pixel ratio down
// ...and back up after a sustained fast stretch (so one slow patch, e.g. shader compiles or Low Power Mode, doesn't
// blur a Retina screen for the rest of the session). The first ~4 s after boot / a context restore are ignored.
let slowT = 0, perfT = -4, fastT = 0;
function adaptRes(rawDt, cap) {
  if (maxFps || cap < 60 || document.hidden || rawDt > 0.25) return; // ignore debug caps, the 30 fps reply cap, tab switches, hitches
  perfT += rawDt;
  if (perfT < 0) return;
  slowT = rawDt > 0.0215 ? slowT + rawDt : Math.max(0, slowT - rawDt * 0.5); // slower than ~46 fps
  fastT = rawDt < 0.0185 ? fastT + rawDt : Math.max(0, fastT - rawDt * 2);    // holding the 60 fps cap
  if (perfT > 3 && slowT > 2 && dprScale > 0.67 && pixelRatio() > 1) {
    dprScale = Math.max(0.67, dprScale - 0.17); slowT = 0; perfT = 0; fastT = 0;
    resize();
  } else if (fastT > 20 && dprScale < 1) {  // capped frames can't show headroom, so wait longer before stepping up
    dprScale = Math.min(1, dprScale + 0.17); slowT = 0; perfT = 0; fastT = 0;
    resize();
  }
}
canvas.addEventListener('webglcontextrestored', () => { perfT = -4; slowT = 0; fastT = 0; });
function frameTick(now) {
  rafId = requestAnimationFrame(frameTick);
  const cap = fpsCap();
  if (now - lastDraw < 1000 / cap - 2) return;  // 2 ms slack, so a 60 Hz screen still draws every frame at cap 60
  lastDraw = now;
  const rawDt = Math.max(0, (now - last) / 1000);
  const dt = Math.min(0.1, rawDt);
  last = now; clock += dt; frameN++;
  const t = clock;
  adaptRes(rawDt, cap);
  if (clock > echoFadeAt) hideEcho();

  // her reply waits while she jumps (and until the bubble is back at her new head), only while the jump can progress
  const jumpHold = capri && (TP.busy || TP.pendingAt || (TP.bubbleHide > 0.2 && clock - TP.bubbleBackAt < 1));
  if (!jumpHold) speechTick(dt);
  const cs = updateCapriState();
  teleportTick(dt);
  talkI = damp(talkI, cs === 'talking' ? 1 : cs === 'thinking' ? 0.3 : 0, 3, dt);
  city?.setIntensity(cs === 'talking' ? 0.85 + 0.15 * Math.sin(t * 3) : cs === 'thinking' ? 0.3 : 0);

  // camera: parallax + slow idle drift (+ a tiny push-in while she talks)
  const motion = reduced ? 0.25 : 1;
  pointerS.x = damp(pointerS.x, pointer.x, 2.2, dt);
  pointerS.y = damp(pointerS.y, pointer.y, 2.2, dt);
  // the shot: frame (per aspect + perch), eased over ~1 s after a jump to a perch that needs a different shot
  camBase.pos.copy(frame.pos); camBase.target.copy(frame.target);
  if (TP.cam) {
    const k = clamp((clock - TP.cam.t0) / TP.cam.dur, 0, 1), e = smoother(k);
    camBase.pos.lerpVectors(TP.cam.fromPos, frame.pos, e); camBase.target.lerpVectors(TP.cam.fromTgt, frame.target, e);
    if (k >= 1) TP.cam = null;
  }
  camPos.copy(camBase.pos);
  camPos.x += (pointerS.x * 0.16 + Math.sin(t * 0.11) * 0.07) * motion;
  camPos.y += (-pointerS.y * 0.07 + Math.sin(t * 0.17 + 1.3) * 0.03) * motion;
  camPos.z += (Math.sin(t * 0.07) * 0.08 - talkI * 0.12) * motion;
  camTgt.copy(camBase.target);
  camTgt.x += pointerS.x * 0.04 * motion;
  camera.position.copy(camPos);
  camera.lookAt(camTgt);
  camera.rotateZ(Math.sin(t * 0.09) * 0.006 * motion);

  try { city?.update(dt, t); } catch (e) { console.error(e); city = null; }
  if (capri) {
    try {
      capri.update(dt, t); capri.getHeadWorldPosition(head);
      const ts = capri.teleportState;
      contact?.update(ts ? ts.visible : 1); backCard?.update(ts ? ts.visible : 1, ts && !reduced ? ts.flash * TP.fxK : 0);
    } catch (e) { console.error(e); capri = null; }
  }
  if ((avoidT -= dt) <= 0) { avoidT = 0.2; updateBubbleAvoid(); }
  bubble.setAnchor(head);
  bubble.update(dt, t, camera);
  bubbleTeleportFx(dt);

  // post uniforms
  const dmax = +els.dose.max || 1.2;
  const dn = clamp(doseVal() / Math.max(0.01, (info && info.default_dose) || 1), 0, 1.6);
  state.glitch = Math.max(0, state.glitch - dt * 2.2);
  // dose > default: rare spontaneous micro-glitches (she "comes apart")
  if (!reduced && doseVal() > (((info && info.default_dose) || 1) * 1.05) && Math.random() < dt * 0.25 * (doseVal() / dmax)) state.glitch = Math.max(state.glitch, 0.35);
  const U = cyber.uniforms;
  // teleport: the frame flashes / splits with her two flashes (reduced motion: none)
  const tps = capri?.teleportState;
  // (scaled down when she is big on screen, TP.fxK: the burst is about her, never a frame-wide blast)
  if (tps && !reduced) { const k = TP.fxK; state.glitch = Math.max(state.glitch, tps.glitch * 0.6 * k); U.uFlash.value = tps.flash * 0.1 * k; U.uSplit.value = tps.flash * 0.8 * k; }
  else { U.uFlash.value = 0; U.uSplit.value = 0; }
  U.uTime.value = t % 1000;
  U.uGlitch.value = reduced ? state.glitch * 0.2 : state.glitch;
  U.uAmt.value = 0.45 + 0.55 * Math.min(1, dn);
  U.uTalk.value = talkI;
  bloom.strength = 0.8 + talkI * 0.12;

  renderer.autoClear = true;
  renderBloom(dt);
  composer.render(dt);
  renderer.autoClear = false;
  renderer.clearDepth();
  renderer.render(bubbleScene, camera);
  frameMs = performance.now() - now;
}

// the bubble keeps clear of her hair: the right edge of her rest silhouette around head height (live camera)
let avoidT = 0;
const _avoidPts = [], _hv = new THREE.Vector3(), _noCrop = {};
function updateBubbleAvoid() {
  if (!capri || !bubble.setAvoid) return;
  const p = perches[perchIdx]; if (!p || !perchCorners(Object.assign(_noCrop, p, { cropY: null }), _avoidPts)) return;
  _hv.copy(head).project(camera); const hy = (1 - _hv.y) / 2 * innerHeight;
  let x1 = -1e9;
  for (const c of _avoidPts) { _fv.copy(c).project(camera); const y = (1 - _fv.y) / 2 * innerHeight; if (y > hy - 190 && y < hy + 70) x1 = Math.max(x1, (_fv.x + 1) / 2 * innerWidth); }
  try { bubble.setAvoid(x1 > -1e8 ? x1 : null); } catch { /* optional */ }
}
const capriMats = [];
if (capri) capri.root.traverse((o) => { if (o.isMesh && o.material && o.material.color && !capriMats.some((m) => m.mat === o.material)) capriMats.push({ mat: o.material, col: o.material.color.clone() }); });
const BLOOM_DARK = 0.4;
const bloomSkip = capri?.teleportFx?.bloomSkip || [], _bsVis = [];
function renderBloom(dt) {
  for (const m of capriMats) m.mat.color.copy(m.col).multiplyScalar(BLOOM_DARK);
  for (let i = 0; i < bloomSkip.length; i++) { _bsVis[i] = bloomSkip[i].visible; bloomSkip[i].visible = false; }
  renderer.setRenderTarget(bloomRT);
  renderer.clear();
  renderer.render(scene, camera);
  for (let i = 0; i < bloomSkip.length; i++) bloomSkip[i].visible = _bsVis[i];
  for (const m of capriMats) m.mat.color.copy(m.col);
  bloom.render(renderer, null, bloomRT, dt, false);
  renderer.setRenderTarget(null);
}

// browsers already pause rAF in hidden tabs; on return just restart the clock so dt doesn't jump
// (one loop only: the id is kept so nothing can ever queue a second chain)
document.addEventListener('visibilitychange', () => { if (!document.hidden) { last = performance.now(); slowT = 0; perfT = 0; fastT = 0; } });

bootStep(0.88);
// compile the teleport FX programs once at boot, behind the boot overlay (otherwise the first jump hitches on
// shader compiles right inside its 0.3 s break-up)
try {
  const warm = capri?.teleportFx?.warm || [];
  if (warm.length) {
    const vis = warm.map((o) => o.visible);
    warm.forEach((o) => { o.visible = true; });
    // (into the composer's target: programs are keyed on the target's colour space / tone mapping; the bloom
    // pre-render uses the same variant, and its skip list keeps the burst / ghosts out of it anyway)
    renderer.setRenderTarget(composerRT);
    renderer.compile(scene, camera);
    renderer.setRenderTarget(null);
    warm.forEach((o, i) => { o.visible = vis[i]; });
  }
} catch (e) { console.warn('[capri] warm-up failed', e); }

rafId = requestAnimationFrame((n) => { last = n; frameTick(n); });

// ------------------------------------------------------------------------------------------------
// boot
// ------------------------------------------------------------------------------------------------
bootStep(1);
// lift the boot screen once the scene has rendered a few frames (the first ones compile shaders and stall), so its
// fade runs smoothly instead of hanging near opaque over a half-drawn scene
const bootFrames = frameN;
const bootWhenDrawn = (tries = 0) => (frameN - bootFrames >= 4 || tries > 40 ? bootReady() : setTimeout(() => bootWhenDrawn(tries + 1), 50));
setTimeout(() => bootWhenDrawn(), 200);
function bootReady() {
  document.body.classList.add('booted');
  $('boot').classList.add('gone');
  booted = true;
  try { capri?.react('greet'); } catch (e) { console.error(e); }
  state.glitch = 1;
  setTimeout(() => { if (!state.speaking && !state.net && !chat.history.length) sayLocal(pickOpener(false)); }, 650);
  syncSend();
}

// for tests / the console
window.__capri = {
  send: (text) => submit(text),
  stop,
  newChat: () => resetConversation(false),
  capri, bubble, city, chat, voice, renderer, camera, composer,
  state, frame, computeFrame,
  get mode() { return mode; },
  get frameMs() { return frameMs; },
  get frames() { return frameN; },
  glitch(x = 1) { state.glitch = x; capri?.react('glitch'); },
  // teleport(perchIndex?) -> Promise<boolean>: jump now (random allowed perch when omitted)
  teleport: (i, why) => teleportTo(i, why),
  get perches() { return perches; },
  get perchIdx() { return perchIdx; },
  TP,
  // shotInfo(i): perch i's suggested view vs the safe rect (her fit box in CSS px, and whether it fits there as given)
  shotInfo(i, master) {
    const p = perches[i], a = innerWidth / innerHeight; if (!p || !perchCorners(p, _corn)) return null;
    const v = (master ? null : (a < 0.8 ? p.viewPortrait : a >= 1.25 ? p.view : null)) || baseFrame, R = safeRect(a, i > 0);
    const bx = screenBox(_corn, v.pos, v.target, v.fov || baseFrame.fov, a, R);
    return { box: [bx.x0, bx.y0, bx.x1, bx.y1].map(Math.round), hit: bx.hit, safe: R, fits: fits(bx, R), inMaster: inMaster(i, a) };
  },
};

window.dispatchEvent(new CustomEvent('capri:ready', { detail: { chat } }));
