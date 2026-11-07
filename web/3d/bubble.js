// bubble.js: the holographic speech bubble Capri's replies stream into.
//
// One draw call. The bubble is a single quad whose fragment shader draws the whole thing as a signed
// distance field in *CSS-pixel* units: a chamfered (clipped-corner) panel unioned with a wedge tail that
// points at the anchor (her face), a glowing pink rim, acid-green corner ticks, scanlines, a slow
// holographic shimmer, the reveal/glitch effects, the "..." thinking blocks and the cursor block.
// The text is drawn into a CanvasTexture (devicePixelRatio-aware, no mipmaps) that only covers the panel.
//
// Layout happens in screen space so the text is always a fixed, readable CSS size (~17px desktop,
// ~16px phones) no matter how main.js frames the shot: every frame we pick where the panel should sit
// on screen next to the anchor's projection (right of her head, else left, else above), spring toward it,
// then place the group in 3D at roughly her depth and scale it so 1 local unit == 1 CSS px. The group
// billboards toward the camera with lag, a velocity tilt and a slight yaw toward her, plus a float bob,
// so it reads as a physical object hanging in the scene rather than a HUD sticker.
//
// Contract (README.md): createBubble(bubbleScene, { maxWidth }) ->
//   { group, update(dt, t, camera), begin(), append(text), end(), setError(msg), setAnchor(v3), hide() }
// Optional extras: opts.renderer (viewport size + anisotropy), opts.insets {top,bottom,left,right} (CSS px kept
// clear for the header / prompt bar), opts.label, opts.channel, opts.fontSize, opts.maxLines, opts.injectFonts;
// methods setInsets(obj), setViewport(w, h), setSide('auto'|'right'|'left'|'top'), clear(), and getters
// state ('hidden'|'thinking'|'talking'|'done'|'error') and text.

import * as THREE from 'three';
import { PALETTE } from './palette.js';

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));

const LABEL_FONT = '"Share Tech Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace'; // the small CAPRI header
const BODY_FONT = '"Exo 2", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const FONTS_HREF = 'https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&family=Share+Tech+Mono&display=swap';

const ERR = '#ff2f45';

// ------------------------------------------------------------------------------------------------
// shader
// ------------------------------------------------------------------------------------------------
const VERT = /* glsl */`
uniform vec4 uBox;           // centre.xy, size.zw of the quad, local px
varying vec2 vP;
void main() {
  vP = uBox.xy + position.xy * uBox.zw;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(vP, 0.0, 1.0);
}`;

const FRAG = /* glsl */`
precision highp float;
uniform vec2 uHalf;          // panel half size (px)
uniform vec2 uChamfer;       // big (TL/BR), small (TR/BL) chamfers
uniform vec2 uT0, uT1, uT2;  // tail triangle (px)
uniform float uTailOn;
uniform sampler2D uText;
uniform vec2 uTexSize;       // canvas size in css px (panel top-left = canvas origin)
uniform float uTime;
uniform float uRevX, uRevY, uAlpha;
uniform float uGlitch, uError, uEnergy;
uniform vec4 uCursor;        // x, y (centre, local px), w, h
uniform float uCursorA;
uniform vec3 uDots;          // x, y of first block, alpha
uniform float uDotSize;
uniform vec3 uPink, uGreen, uBlue, uRed, uFill;
varying vec2 vP;

float hash(float n) { return fract(sin(n) * 43758.5453123); }
float sdSeg(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
float sdTri(vec2 p, vec2 p0, vec2 p1, vec2 p2) {
  vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
  vec2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
  vec2 pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  vec2 pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  vec2 pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  float s = sign(e0.x * e2.y - e0.y * e2.x);
  vec2 d = min(min(vec2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                   vec2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                   vec2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}
float sdPanel(vec2 p, vec2 b) {
  vec2 q = abs(p) - b;
  float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
  float c = (p.x * p.y < 0.0) ? uChamfer.x : uChamfer.y;
  float dc = (abs(p.x) + abs(p.y) - (b.x + b.y - c)) * 0.70710678;
  return max(d, dc);
}
vec4 over(vec4 dst, vec3 c, float a) {
  float oa = a + dst.a * (1.0 - a);
  return vec4((c * a + dst.rgb * dst.a * (1.0 - a)) / max(oa, 1e-5), oa);
}
float aa(float d) { float w = max(fwidth(d), 0.5); return clamp(0.5 - d / w, 0.0, 1.0); }

void main() {
  vec2 p = vP;
  // glitch: horizontal slices jump sideways
  float g = uGlitch + uError * 0.12 * step(0.86, hash(floor(uTime * 9.0)));
  float row = floor(p.y / 6.0);
  float rnd = hash(row * 1.37 + floor(uTime * 40.0));
  float shift = (g > 0.001 && rnd < g * 0.6) ? (hash(row + floor(uTime * 40.0) * 3.1) - 0.5) * 38.0 * g : 0.0;
  p.x += shift;

  // reveal clip: a line grows horizontally, then opens vertically
  vec2 ext = uHalf + vec2(90.0);
  float clipX = uRevX * ext.x, clipY = max(uRevY * ext.y, 1.5);
  if ((uRevX < 0.999 || uRevY < 0.999) && (abs(p.x) > clipX || abs(p.y) > clipY)) discard; // (fully open: no clip, the tail can reach past it)
  float revEdge = (1.0 - smoothstep(0.0, 3.0, clipY - abs(p.y))) * step(uRevY, 0.98);

  float dPanel = sdPanel(p, uHalf);
  float dTail = uTailOn > 0.5 ? sdTri(p, uT0, uT1, uT2) : 1e4;
  float d = min(dPanel, dTail);

  vec3 edge = mix(uPink, uRed, uError);
  vec4 col = vec4(0.0);

  // soft dark halo first, so a bright sign behind the bubble doesn't fight the text
  col = over(col, uFill * 0.6, exp(-max(dPanel, 0.0) / 16.0) * 0.55 * step(0.0, d));
  col = over(col, uFill * 0.6, exp(-max(dTail, 0.0) / 3.0) * 0.4 * step(0.0, d));
  // outer glow (we're drawn after bloom, so the glow is ours); tight around the tail
  float glow = exp(-max(dPanel, 0.0) / 7.0) * 0.5 + exp(-max(dPanel, 0.0) / 22.0) * 0.18;
  glow = max(glow, exp(-max(dTail, 0.0) / 3.5) * 0.45);
  glow *= (0.85 + 0.15 * sin(uTime * 2.3)) * (0.75 + 0.35 * uEnergy);
  col = over(col, edge, glow * step(0.0, d) * 0.9);

  float inside = aa(d);
  if (inside > 0.0) {
    // fill: translucent void, a touch lighter at the top, faint pink inner glow at the rim
    vec3 fill = uFill + vec3(0.025, 0.012, 0.03) * smoothstep(-uHalf.y, uHalf.y, p.y);
    fill += edge * exp(d / 9.0) * 0.16;
    fill += uBlue * 0.05 * smoothstep(uHalf.y * 0.2, -uHalf.y, p.y);
    // scanlines
    float scan = 0.5 + 0.5 * sin(p.y * 2.094 + uTime * 0.0);
    fill *= 0.9 + 0.1 * scan;
    // holographic shimmer: a slow diagonal sweep tinted pink -> blue
    float sweep = fract((p.x * 0.55 + p.y) / (2.4 * (uHalf.x + uHalf.y)) - uTime * 0.075);
    float band = smoothstep(0.0, 0.06, sweep) * (1.0 - smoothstep(0.06, 0.2, sweep));
    fill += mix(uPink, uBlue, sweep * 5.0) * band * 0.07;
    // near-opaque fill (the scene behind mustn't show through the text); the tail is solid
    float inTail = step(dTail, 0.0) * step(0.0, dPanel);
    float fa = mix(0.93 + 0.04 * band, 0.97, inTail);
    col = over(col, fill, fa * inside);
    // tail: an acid-green inner stroke so it separates from her pink hair
    if (uTailOn > 0.5) {
      float ts = aa(abs(dTail + 2.6) - 0.7) * step(0.5, dPanel);
      col = over(col, mix(uGreen, uRed, uError), ts * 0.9 * inside);
    }

    // text (rgb split while glitching)
    vec2 tuv = vec2((p.x + uHalf.x) / uTexSize.x, 1.0 - (uHalf.y - p.y) / uTexSize.y);
    if (tuv.x > 0.0 && tuv.x < 1.0 && tuv.y > 0.0 && tuv.y < 1.0 && dPanel < 0.0) {
      vec4 tx = texture2D(uText, tuv);
      float split = (g + uError * 0.15) * 3.0 / uTexSize.x;
      if (split > 0.0) {
        float r = texture2D(uText, tuv + vec2(split, 0.0)).a;
        float b = texture2D(uText, tuv - vec2(split, 0.0)).a;
        tx.rgb = vec3(mix(tx.r, 1.0, r * 0.6) * max(r, tx.a), tx.g * tx.a, mix(tx.b, 1.0, b * 0.6) * max(b, tx.a)) / max(max(r, b), max(tx.a, 1e-4));
        tx.a = max(tx.a, max(r, b) * 0.8);
      }
      col = over(col, tx.rgb, tx.a * (0.94 + 0.06 * scan));
    }

    // cursor block
    if (uCursorA > 0.0) {
      vec2 cq = abs(p - uCursor.xy) - uCursor.zw * 0.5;
      float cd = max(cq.x, cq.y);
      vec3 cc = mix(uGreen, uRed, uError);
      col = over(col, cc, aa(cd) * uCursorA * inside);
      col = over(col, cc, exp(-max(cd, 0.0) / 4.0) * 0.35 * uCursorA * step(0.0, cd) * inside);
    }
    // thinking blocks
    if (uDots.z > 0.0) {
      for (int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 c = vec2(uDots.x + fi * uDotSize * 1.9, uDots.y);
        vec2 q = abs(p - c) - vec2(uDotSize * 0.5);
        float dd = max(q.x, q.y);
        float ph = fract(uTime * 1.4 - fi * 0.18);
        float pulse = 0.35 + 0.65 * pow(1.0 - ph, 3.0);
        col = over(col, uGreen, aa(dd) * pulse * uDots.z);
        col = over(col, uGreen, exp(-max(dd, 0.0) / 5.0) * 0.4 * pulse * uDots.z * step(0.0, dd));
      }
    }
  }

  // rim: bright core + a short acid-green accent on the top edge after the clipped corner
  float rimW = 1.6;
  float rim = aa(abs(d) - rimW * 0.5);
  float accent = step(uHalf.y - 3.0, p.y) * step(-uHalf.x + uChamfer.x + 4.0, p.x) * step(p.x, -uHalf.x + uChamfer.x + 58.0);
  accent += step(p.y, -uHalf.y + 3.0) * step(uHalf.x - uChamfer.x - 30.0, p.x) * step(p.x, uHalf.x - uChamfer.x - 4.0);
  vec3 rimC = mix(edge * 1.15 + 0.12, mix(uGreen, uRed, uError), clamp(accent, 0.0, 1.0));
  col = over(col, rimC, rim);

  // corner ticks (outside the rim): L-brackets on the square-ish corners, a dash along each big chamfer
  float o = 5.0, L = 11.0;
  vec2 tr = uHalf + vec2(o), bl = -uHalf - vec2(o);
  float t = min(min(sdSeg(p, tr, tr - vec2(L, 0.0)), sdSeg(p, tr, tr - vec2(0.0, L))),
                min(sdSeg(p, bl, bl + vec2(L, 0.0)), sdSeg(p, bl, bl + vec2(0.0, L))));
  float c = uChamfer.x;
  vec2 n = vec2(-0.70710678, 0.70710678) * 6.0;
  vec2 a0 = vec2(-uHalf.x, uHalf.y - c) + n, a1 = vec2(-uHalf.x + c, uHalf.y) + n;
  t = min(t, sdSeg(p, mix(a0, a1, 0.22), mix(a0, a1, 0.78)));
  a0 = vec2(uHalf.x, -uHalf.y + c) - n; a1 = vec2(uHalf.x - c, -uHalf.y) - n;
  t = min(t, sdSeg(p, mix(a0, a1, 0.22), mix(a0, a1, 0.78)));
  vec3 tickC = mix(uGreen, uRed, uError);
  col = over(col, tickC, aa(t - 0.8) * 0.95);
  col = over(col, tickC, exp(-t / 4.0) * 0.25);

  // tail tip: a tiny target pip
  if (uTailOn > 0.5) {
    float pip = length(p - uT2);
    col = over(col, mix(uGreen, uRed, uError), aa(pip - 2.2) * 0.95);
  }

  // rain sliding past in front of the panel (faint over the text, a bit stronger over the rim / glow)
  {
    vec2 rq = vec2(p.x + p.y * 0.11, p.y);
    float lane = floor(rq.x / 9.0);
    float hr = hash(lane * 7.31);
    float fall = fract((rq.y + uTime * (520.0 + 260.0 * hr)) / (300.0 + 240.0 * hash(lane * 3.7)) + hr);
    float streak = step(0.72, hash(lane * 1.93)) * smoothstep(0.0, 0.05, fall) * (1.0 - smoothstep(0.05, 0.16, fall));
    float xw = 1.0 - smoothstep(0.25, 0.9, abs(fract(rq.x / 9.0) - 0.5) * 9.0);
    float onRim = 1.0 - smoothstep(0.0, 12.0, abs(dPanel));
    col = over(col, vec3(0.75, 0.82, 1.0), streak * xw * (0.07 + 0.22 * onRim) * step(dPanel, 30.0));
  }
  // reveal scan edge
  col = over(col, vec3(1.0, 0.75, 0.85), revEdge * 0.9 * step(abs(p.x), clipX));
  // glitch flash
  col.rgb += edge * g * 0.35 * col.a;

  float a = col.a * uAlpha;
  if (a < 0.002) discard;
  gl_FragColor = vec4(col.rgb, a);
}`;

// ------------------------------------------------------------------------------------------------
// markdown-ish tokenizer: **bold** toggles, \n breaks. Returns paragraphs of words, each word a list of pieces.
// ------------------------------------------------------------------------------------------------
function tokenize(raw, streaming) {
  let s = raw.replace(/\r/g, '');
  if (streaming) s = s.replace(/(^|[^*])\*$/, '$1'); // a lone trailing '*' may be the start of '**'
  const paras = [[]];
  let bold = false, word = null;
  const flush = () => { if (word) { paras[paras.length - 1].push(word); word = null; } };
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '*' && s[i + 1] === '*') { bold = !bold; i++; continue; }
    if (ch === '\n') { flush(); paras.push([]); continue; }
    if (ch === ' ' || ch === '\t') { flush(); continue; }
    if (!word) word = [];
    const last = word[word.length - 1];
    if (last && last.bold === bold) last.text += ch; else word.push({ text: ch, bold });
  }
  flush();
  return paras;
}

// ------------------------------------------------------------------------------------------------
export function createBubble(bubbleScene, opts = {}) {
  const group = new THREE.Group();
  group.name = 'bubble';
  group.visible = false;
  bubbleScene.add(group);

  const renderer = opts.renderer || null;
  const insets = Object.assign({ top: 64, bottom: 124, left: 16, right: 16 }, opts.insets || {});
  const maxWidthOpt = opts.maxWidth || 440;
  const labelText = opts.label || 'CAPRI';
  const channelText = opts.channel || ''; // a small dim tag at the header's right (none by default)
  let forcedSide = opts.side || 'auto';
  let viewW = 0, viewH = 0, viewForced = false;

  if (opts.injectFonts !== false && typeof document !== 'undefined') injectFonts();

  // ---------- canvas / texture ----------
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace; // we output raw sRGB values ourselves
  if (renderer) tex.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());

  const col3 = (hex) => new THREE.Color().setHex(hex, THREE.NoColorSpace);
  const U = {
    uBox: { value: new THREE.Vector4(0, 0, 1, 1) },
    uHalf: { value: new THREE.Vector2(200, 60) },
    uChamfer: { value: new THREE.Vector2(16, 4) },
    uT0: { value: new THREE.Vector2() }, uT1: { value: new THREE.Vector2() }, uT2: { value: new THREE.Vector2() },
    uTailOn: { value: 0 },
    uText: { value: tex },
    uTexSize: { value: new THREE.Vector2(1, 1) },
    uTime: { value: 0 },
    uRevX: { value: 0 }, uRevY: { value: 0 }, uAlpha: { value: 1 },
    uGlitch: { value: 0 }, uError: { value: 0 }, uEnergy: { value: 0 },
    uCursor: { value: new THREE.Vector4() }, uCursorA: { value: 0 },
    uDots: { value: new THREE.Vector3() }, uDotSize: { value: 7 },
    uPink: { value: col3(PALETTE.pink) }, uGreen: { value: col3(PALETTE.green) }, uBlue: { value: col3(PALETTE.blue) },
    uRed: { value: col3(0xff2f45) }, uFill: { value: col3(PALETTE.void) },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms: U, vertexShader: VERT, fragmentShader: FRAG,
    transparent: true, depthTest: false, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 10;
  group.add(mesh);

  // ---------- optional light spill into the main scene (opts.scene): a soft additive pink glow on whatever sits
  // behind the panel (rain, the parapet, walls), so the bubble feels like a light source hanging in the scene.
  // One extra draw call, depth-tested, rendered before bloom.
  let spill = null;
  if (opts.scene) {
    const sm = new THREE.ShaderMaterial({
      uniforms: { uA: { value: 0 }, uPink: { value: new THREE.Color(PALETTE.pink) }, uBlue: { value: new THREE.Color(PALETTE.blue) }, uTime: U.uTime },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: `uniform float uA; uniform vec3 uPink, uBlue; uniform float uTime; varying vec2 vUv;
        void main(){ vec2 c = vUv * 2.0 - 1.0; float r = length(c * vec2(1.0, 1.25));
          float a = exp(-r * r * 3.2) * (0.85 + 0.15 * sin(uTime * 2.3));
          gl_FragColor = vec4(mix(uPink, uBlue, smoothstep(-0.6, 0.9, -c.y)) * a * uA, 1.0); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    });
    spill = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), sm);
    spill.frustumCulled = false; spill.renderOrder = 9; spill.visible = false;
    opts.scene.add(spill);
  }

  // ---------- text state ----------
  let raw = '', errMsg = '';
  let state = 'hidden';
  let dirty = true, fontsDirty = true;
  let lines = [];               // { pieces:[{text,bold,x,err}], y, w } (y = top of line, px from text-area top)
  let contentH = 0, maxLineW = 0;
  let Wc = 300, Wv = 0, drawnW = 0;   // current (animated) panel width, shrink-wraps short replies
  let scroll = 0, scrollTarget = 0, scrollRaw = 0, follow = true; // follow = auto-scroll to the newest line (off while the user scrolls back)
  let lastDraw = -1;
  let lastAppend = -1, endAt = -1;
  let cursorPx = { x: 0, y: 0 }; // canvas coords of cursor centre (before scroll)
  const widthCache = new Map();

  // metrics (recomputed by configure())
  const M = { W: 400, font: 17, lineH: 25, padX: 18, padTop: 12, padBot: 16, headH: 20, maxLines: 7, paraGap: 8, dpr: 1, textH: 175 };
  let texH = 0, texW = 0;

  function configure(vw, vh) {
    const phone = vw < 560;
    // the reply should hold attention on big screens too: ~17.5px at laptop sizes, up to 22px at 1920+
    M.font = opts.fontSize || (phone ? 16 : clamp(Math.round(vw * 0.0116 * 2) / 2, 17.5, 22));
    M.lineH = Math.round(M.font * 1.45);
    M.padX = phone ? 15 : 19;
    M.padTop = phone ? 10 : 12;
    M.padBot = phone ? 14 : 17;
    M.headH = Math.round(M.font * (phone ? 1.12 : 1.1));
    M.paraGap = Math.round(M.lineH * 0.35);
    M.W = Math.round(clamp(Math.min(maxWidthOpt * (M.font / 17.5), vw - insets.left - insets.right - 8), 220, 2000));
    const availH = vh - insets.top - insets.bottom - 40;
    const fitLines = Math.floor((availH * (phone ? 0.5 : 0.62) - M.padTop - M.padBot - M.headH) / M.lineH);
    M.maxLines = clamp(opts.maxLines || fitLines, 3, opts.maxLines || (phone ? 7 : 8));
    M.textH = M.maxLines * M.lineH;
    // texture scale: a little supersampling, capped at 2x (each redraw is a full texImage2D upload)
    M.dpr = clamp((typeof devicePixelRatio === 'number' ? devicePixelRatio : 1) * (phone ? 1 : 1.25), 1.25, 2);
    const cw = Math.ceil(M.W * M.dpr), ch = Math.ceil((M.padTop + M.headH + M.textH + M.padBot + 2) * M.dpr);
    if (cw !== canvas.width || ch !== canvas.height) {
      canvas.width = cw; canvas.height = ch;
      tex.dispose(); // reallocate at the new size
    }
    texW = M.W; texH = ch / M.dpr;
    U.uTexSize.value.set(texW, texH);
    U.uDotSize.value = Math.round(M.font * 0.5);
    U.uChamfer.value.set(phone ? 14 : 18, 4);
    widthCache.clear();
    dirty = true;
  }

  const fontFor = (bold) => `${bold ? 700 : 400} ${M.font}px ${BODY_FONT}`;
  function measure(text, bold) {
    const k = (bold ? 'b' : 'n') + text;
    let w = widthCache.get(k);
    if (w === undefined) { ctx.font = fontFor(bold); w = ctx.measureText(text).width; widthCache.set(k, w); }
    return w;
  }

  function layout() {
    const maxW = M.W - M.padX * 2;
    const out = [];
    let y = 0;
    const addParas = (paras, err) => {
      paras.forEach((words, pi) => {
        if (pi > 0 || out.length) { if (pi > 0 && !words.length) { y += M.paraGap; return; } }
        let line = { pieces: [], y, w: 0 };
        const push = () => { out.push(line); y += M.lineH; line = { pieces: [], y, w: 0 }; };
        for (const word of words) {
          let ww = 0; for (const pc of word) ww += measure(pc.text, pc.bold);
          const sp = line.pieces.length ? measure(' ', false) : 0;
          if (line.pieces.length && line.w + sp + ww > maxW) push();
          if (ww > maxW) { // hard-split a monster word by characters
            for (const pc of word) for (const ch of pc.text) {
              const cw = measure(ch, pc.bold);
              if (line.w + cw > maxW && line.pieces.length) push();
              line.pieces.push({ text: ch, bold: pc.bold, x: line.w, err }); line.w += cw;
            }
            continue;
          }
          let x = line.w + (line.pieces.length ? measure(' ', false) : 0);
          for (const pc of word) { line.pieces.push({ text: pc.text, bold: pc.bold, x, err }); x += measure(pc.text, pc.bold); }
          line.w = x;
        }
        push();
      });
    };
    if (raw.trim()) addParas(tokenize(raw.replace(/^\s+/, ''), state === 'talking' || state === 'thinking'), false);
    if (errMsg) { if (out.length) y += M.paraGap; addParas([[...errMsg.split(/\s+/).filter(Boolean).map((w) => [{ text: w, bold: false }])]], true); }
    lines = out;
    const last = out[out.length - 1];
    contentH = last ? last.y + M.lineH : 0;
    maxLineW = 0; for (const ln of out) maxLineW = Math.max(maxLineW, ln.w);
    cursorPx.x = last ? last.w + M.font * 0.32 : 0;
    cursorPx.y = last ? last.y : 0;
    // scroll so the top visible line starts on a line boundary (no half-cut line under the header)
    const need = contentH - M.textH;
    if (follow) {
      scrollTarget = 0;
      if (need > 0) { scrollTarget = need; for (const ln of out) if (ln.y >= need - 0.5) { scrollTarget = ln.y; break; } }
    } else scrollTarget = clamp(scrollTarget, 0, Math.max(0, need));
  }

  function drawLabel(x, y, text, color, size, spacing, weight = 600) {
    ctx.font = `400 ${size}px ${LABEL_FONT}`; // Share Tech Mono has one weight: no faux bold
    ctx.fillStyle = color;
    for (const ch of text) { ctx.fillText(ch, x, y); x += ctx.measureText(ch).width + spacing; }
    return x;
  }

  function draw() {
    const d = M.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.textBaseline = 'alphabetic';
    const err = state === 'error';
    // header: just her name tag (it's speech, not a HUD window); a dim status only when something's wrong
    const hy = M.padTop + M.headH * 0.66;
    const lsz = Math.round(M.font * 0.66);
    let x = drawLabel(M.padX, hy, labelText, err ? ERR : PALETTE.css.green, lsz, 2.4, 700);
    if (err) drawLabel(x + 8, hy, '// SIGNAL LOST', ERR, lsz - 1, 1.4, 500);
    else if (channelText) drawLabel(x + 10, hy, channelText, 'rgba(254,3,83,0.7)', lsz - 1, 1.4, 500);
    // a short green lead line under the tag (fixed length: no redraw needed while the width animates)
    ctx.fillStyle = err ? ERR : PALETTE.css.green;
    ctx.fillRect(M.padX, M.padTop + M.headH - 1, Math.max(22, x - M.padX - 2.4), 1.5);
    const DW = M.W;

    // body text, clipped to the text area
    const top = M.padTop + M.headH + 4;
    ctx.save();
    ctx.beginPath(); ctx.rect(0, top - 2, M.W, M.textH + 2); ctx.clip(); // (the label rule sits at top - 5)
    const baseOff = Math.round(M.lineH * 0.5 + M.font * 0.36);
    for (const ln of lines) {
      const yy = top + ln.y - scroll;
      if (yy + M.lineH < top - 2 || yy > top + M.textH) continue;
      for (const pc of ln.pieces) {
        ctx.font = fontFor(pc.bold);
        ctx.fillStyle = pc.err ? '#ff6b7a' : pc.bold ? '#ffffff' : '#ebe7ec';
        if (pc.bold && !pc.err) { ctx.shadowColor = 'rgba(254,3,83,0.85)'; ctx.shadowBlur = 6; }
        ctx.fillText(pc.text, M.padX + pc.x, yy + baseOff);
        ctx.shadowBlur = 0;
        if (pc.bold && !pc.err) { // pink underline nib under bold runs
          ctx.fillStyle = 'rgba(254,3,83,0.7)';
          ctx.fillRect(M.padX + pc.x, yy + baseOff + 3, measure(pc.text, true), 1.2);
        }
      }
    }
    ctx.restore();
    // soft edges wherever text is clipped: a fade at the top while there are lines above, at the bottom while there
    // are lines below (constant, not only while moving)
    const maxS = Math.max(0, contentH - M.textH);
    const fade = (y0, h, down) => {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      const gr = ctx.createLinearGradient(0, y0, 0, y0 + h);
      gr.addColorStop(down ? 1 : 0, 'rgba(0,0,0,1)'); gr.addColorStop(down ? 0 : 1, 'rgba(0,0,0,0)');
      ctx.fillStyle = gr; ctx.fillRect(0, y0, M.W, h);
      ctx.restore();
    };
    const fhT = Math.min(M.lineH * 0.42, scroll * 0.5), fhB = Math.min(M.lineH * 0.6, (maxS - scroll) * 0.6);
    if (fhT > 0.5) fade(top - 2, fhT + 2, false);
    if (fhB > 0.5) fade(top + M.textH - fhB, fhB, true);
    // scroll affordances: little chevrons at the right of the text area when lines are hidden above / below
    const chev = (cx, cy, up) => { ctx.beginPath(); ctx.moveTo(cx - 5, cy + (up ? 3 : -3)); ctx.lineTo(cx, cy + (up ? -2 : 2)); ctx.lineTo(cx + 5, cy + (up ? 3 : -3)); ctx.stroke(); };
    ctx.strokeStyle = 'rgba(25,242,103,0.9)'; ctx.lineWidth = 1.6;
    const rx = Math.round(Wc) - M.padX * 0.55;
    if (scrollTarget > 0.5 && scroll > 0.5) chev(rx, top + 6, true);
    if (scrollTarget < maxS - 0.5) chev(rx, top + M.textH - 6, false);
    drawnW = Wc;
    tex.needsUpdate = true;
  }

  // ---------- animation state ----------
  let anchor = new THREE.Vector3(0, 2.0, 0), anchorSet = false;
  let side = 'right';
  let H = 90, Hv = 0;                 // current panel height (px), spring
  let sx = null, sy = null, svx = 0, svy = 0; // panel centre on screen (px, y down), spring
  let rev = 0, revTarget = 0, revT = 0; // reveal 0..1
  let glitch = 0, nextGlitch = 4;
  let dotsA = 0, cursorA = 0, energy = 0, errorA = 0;
  let time = 0;
  let tailSide = 'left';
  const tilt = new THREE.Quaternion(), qTarget = new THREE.Quaternion(), e = new THREE.Euler();
  const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3(), fwd = new THREE.Vector3(), camPos = new THREE.Vector3();
  const inv = new THREE.Matrix4();
  let qInit = false, camQ0Init = false;
  const camQ0 = new THREE.Quaternion();

  function viewport() {
    if (viewForced) return;
    if (renderer && renderer.domElement && renderer.domElement.clientWidth) { viewW = renderer.domElement.clientWidth; viewH = renderer.domElement.clientHeight; }
    else { viewW = window.innerWidth; viewH = window.innerHeight; }
  }
  let cfgW = 0, cfgH = 0;

  function bodyHeightTarget() {
    const textH = clamp(contentH, M.lineH, M.textH);
    return Math.round(M.padTop + M.headH + 4 + textH + M.padBot);
  }

  // where should the panel centre be on screen?
  // setAvoid(x | null): the right edge (CSS px) of her silhouette around head height; a right-side panel keeps ~20 px
  // clear of it (her big hair reaches further out than the head radius up close)
  let avoidX1 = -1e9;
  function place(A, rPx, W, Wn, Ht) {
    const vw = viewW, vh = viewH, L = insets.left, R = vw - insets.right, T = insets.top, B = vh - insets.bottom;
    const tail = 30;
    const cy = (top) => clamp(top + Ht / 2, T + Ht / 2, Math.max(T + Ht / 2, B - Ht / 2));
    const opt = {
      right: () => { const left = Math.max(A.x + rPx * 0.95 + tail, avoidX1 + 20); const fits = left + W <= R || A.x + rPx * 0.45 + tail + W <= R; return { fits, x: Math.min(left, R - W) + Wn / 2, y: cy(A.y - rPx * 0.35 - Ht * 0.25) }; },
      left: () => { const right = A.x - rPx * 0.95 - tail; const fits = right - W >= L || A.x - rPx * 0.45 - tail - W >= L; return { fits, x: Math.max(right, L + W) - Wn / 2, y: cy(A.y - rPx * 0.35 - Ht * 0.25) }; },
      top: () => { const bottom = A.y - rPx * 1.9 - tail; const fits = bottom - Ht >= T; return { fits, x: clamp(A.x + Wn * 0.12, L + Wn / 2, Math.max(L + Wn / 2, R - Wn / 2)), y: Math.max(bottom - Ht / 2, T + Ht / 2) }; },
    };
    if (forcedSide !== 'auto' && opt[forcedSide]) { side = forcedSide; return opt[side](); }
    const cur = opt[side] ? opt[side]() : null;
    if (cur && cur.fits) return cur;
    for (const s of ['right', 'left', 'top']) { const r = opt[s](); if (r.fits) { side = s; return r; } }
    side = 'top'; return opt.top();
  }

  function update(dt, t, camera) {
    dt = Math.min(dt || 0.016, 0.1);
    time = t !== undefined ? t : time + dt;
    U.uTime.value = time;
    if (!camera) return;
    viewport();
    if (viewW !== cfgW || viewH !== cfgH || fontsDirty) { cfgW = viewW; cfgH = viewH; fontsDirty = false; configure(viewW, viewH); }

    // reveal / hide
    if (revTarget > rev) { revT += dt; rev = Math.min(1, revT / 0.42); }
    else if (revTarget < rev) { revT -= dt * 1.6; rev = Math.max(0, revT / 0.42); if (rev <= 0) { group.visible = false; if (state === 'hidden') { raw = ''; errMsg = ''; dirty = true; } } }
    if (!group.visible) { if (spill) spill.visible = false; return; }
    const revX = clamp(rev / 0.45, 0, 1), revY = clamp((rev - 0.38) / 0.62, 0, 1);
    U.uRevX.value = 1 - Math.pow(1 - revX, 3);
    U.uRevY.value = 1 - Math.pow(1 - revY, 3);
    U.uAlpha.value = clamp(rev * 3, 0, 1);

    // ambient micro-glitches
    glitch = Math.max(0, glitch - dt * 3.2);
    nextGlitch -= dt;
    if (nextGlitch <= 0) { nextGlitch = 5 + Math.random() * 9; if (state === 'talking' || state === 'done') glitch = Math.max(glitch, 0.18 + Math.random() * 0.14); }
    U.uGlitch.value = glitch;
    errorA = damp(errorA, state === 'error' ? 1 : 0, 10, dt);
    U.uError.value = errorA;
    energy = damp(energy, state === 'talking' && time - lastAppend < 0.5 ? 1 : 0, 6, dt);
    U.uEnergy.value = energy;

    // text: re-layout + redraw (a full texture upload) at most ~16x a second while she streams, so a burst of
    // characters costs one upload instead of one each; anything else (state change, scroll) redraws at once
    const throttle = state === 'talking' && time - lastDraw < 0.06;
    const relayout = dirty && !throttle;
    if (relayout) layout();
    const prevScroll = scroll;
    scroll = damp(scroll, scrollTarget, 9, dt);
    if (Math.abs(scroll - scrollTarget) < 0.3) scroll = scrollTarget;
    if (relayout || Math.abs(scroll - prevScroll) > 0.01) { draw(); dirty = false; lastDraw = time; }

    // panel height spring
    const Ht = bodyHeightTarget();
    const steps = Math.ceil(dt / (1 / 120)), h = dt / steps;
    const k = 260, c = 2 * Math.sqrt(k) * 0.9;
    for (let i = 0; i < steps; i++) { Hv += (k * (Ht - H) - c * Hv) * h; H += Hv * h; }
    if (Math.abs(Ht - H) < 0.05 && Math.abs(Hv) < 0.05) { H = Ht; Hv = 0; }
    // width: shrink-wrap to the longest line (min = what the header needs), spring toward it
    const minW = Math.min(M.W, Math.round(M.font * (state === 'error' ? 22 : 17)));
    const Wt = clamp(Math.ceil(maxLineW + M.padX * 2 + M.font * 0.9), minW, M.W);
    if (rev < 0.05) { Wc = Wt; Wv = 0; }
    for (let i = 0; i < steps; i++) { Wv += (k * (Wt - Wc) - c * Wv) * h; Wc += Wv * h; }
    if (Math.abs(Wt - Wc) < 0.05 && Math.abs(Wv) < 0.05) { Wc = Wt; Wv = 0; }
    Wc = Math.min(Wc, M.W);
    if (Math.abs(Wc - drawnW) > 0.5 && contentH > M.textH) draw(); // only the scroll chevrons depend on the width
    const W = Wc;
    U.uHalf.value.set(W / 2, H / 2);

    // ---- screen placement ----
    camera.updateMatrixWorld();
    camPos.setFromMatrixPosition(camera.matrixWorld);
    camera.getWorldDirection(fwd);
    const depthA = Math.max(0.2, tmp.copy(anchor).sub(camPos).dot(fwd));
    const pxPerWorld = (D) => camera.isPerspectiveCamera ? viewH / (2 * D * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) : viewH * camera.zoom / (camera.top - camera.bottom);
    tmp.copy(anchor).project(camera);
    const A = { x: (tmp.x + 1) / 2 * viewW, y: (1 - tmp.y) / 2 * viewH };
    const rPx = 0.42 * pxPerWorld(depthA) * (camera.isPerspectiveCamera ? 1 : 1);
    const target = place(A, rPx, M.W, W, Ht);
    if (sx === null || rev < 0.05) { sx = target.x; sy = target.y; svx = svy = 0; }
    const ks = 70, cs = 2 * Math.sqrt(ks) * 0.85;
    for (let i = 0; i < steps; i++) {
      svx += (ks * (target.x - sx) - cs * svx) * h; svy += (ks * (target.y - sy) - cs * svy) * h;
      sx += svx * h; sy += svy * h;
    }
    const bob = Math.sin(time * 1.15) * 3 + Math.sin(time * 0.53 + 1.3) * 1.5;
    const swayX = Math.sin(time * 0.7) * 1.5;

    // ---- to world ----
    const D = Math.max(camera.near * 4, depthA - 0.25);
    const ppw = pxPerWorld(D);
    const ndcX = (sx + swayX) / viewW * 2 - 1, ndcY = 1 - (sy + bob) / viewH * 2;
    tmp.set(ndcX, ndcY, 0.5).unproject(camera).sub(camPos).normalize();
    const along = D / Math.max(1e-3, tmp.dot(fwd));
    group.position.copy(camPos).addScaledVector(tmp, along);
    const pop = 0.9 + 0.1 * (1 - Math.pow(1 - clamp(rev / 0.7, 0, 1), 3)) + 0.025 * Math.sin(clamp(rev, 0, 1) * Math.PI);
    group.scale.setScalar(pop / ppw);

    // billboard with lag + tilt: face the camera, yaw toward her (~10deg, a real perspective keystone), lean with
    // velocity, and only half-follow the camera's own turning, so pointer parallax visibly swings the panel
    const yawToHer = side === 'right' ? 0.19 : side === 'left' ? -0.19 : 0;
    const pitchToHer = side === 'top' ? -0.12 : 0.05;
    e.set(pitchToHer + clamp(svy * 0.0011, -0.18, 0.18) + Math.sin(time * 0.9) * 0.02,
          yawToHer + clamp(-svx * 0.0011, -0.22, 0.22) + Math.sin(time * 0.6 + 0.7) * 0.035,
          clamp(-svx * 0.0003, -0.06, 0.06) + Math.sin(time * 0.8) * 0.008, 'XYZ');
    tilt.setFromEuler(e);
    // the direction from the panel to the camera (not the camera's view axis): moving the camera sideways turns the
    // panel against the screen a little, like a real object would
    if (!camQ0Init || rev < 0.05) { camQ0.copy(camera.quaternion); camQ0Init = true; }
    camQ0.slerp(camera.quaternion, 1 - Math.exp(-0.6 * dt));
    qTarget.copy(camQ0).slerp(camera.quaternion, 0.35).multiply(tilt);
    if (!qInit || rev < 0.05) { group.quaternion.copy(qTarget); qInit = true; }
    else group.quaternion.slerp(qTarget, 1 - Math.exp(-5 * dt));
    group.updateMatrixWorld(true);
    if (spill) {
      // ~0.9 m behind the panel along the view ray, 1.7x its size
      spill.visible = true;
      spill.position.copy(group.position).addScaledVector(fwd, 0.9);
      spill.quaternion.copy(camera.quaternion);
      const k = (D + 0.9) / D * pop / ppw;
      spill.scale.set(W * 1.9 * k, H * 2.3 * k, 1);
      spill.material.uniforms.uA.value = 0.07 * clamp(rev, 0, 1) * (0.7 + 0.5 * energy) * (1 - errorA * 0.5);
    }

    // ---- tail: intersect camera->anchor with the panel plane, in local px ----
    inv.copy(group.matrixWorld).invert();
    const C = tmp.copy(camPos).applyMatrix4(inv), Aw = tmp2.copy(anchor).applyMatrix4(inv);
    let tx, ty;
    const dz = Aw.z - C.z;
    if (Math.abs(dz) > 1e-6) { const s = -C.z / dz; tx = C.x + (Aw.x - C.x) * s; ty = C.y + (Aw.y - C.y) * s; } else { tx = Aw.x; ty = Aw.y; }
    // no tail (Max, Nov 6): just the box and its text; buildTail is kept but unused
    U.uTailOn.value = 0; void tx; void ty;

    // cursor / dots / box
    const top = H / 2 - M.padTop - M.headH - 4;
    const cyL = top - (cursorPx.y - scroll) - M.lineH / 2;
    const blinkOn = (time - lastAppend < 0.45) ? 1 : (Math.sin(time * 9.4) > -0.2 ? 1 : 0);
    let cTarget = 0;
    if (state === 'talking') cTarget = blinkOn;
    else if (state === 'done') cTarget = time - endAt < 1.6 ? blinkOn : 0;
    // (no cursor in the error state: she isn't typing anything)
    cursorA = cTarget > cursorA ? cTarget : damp(cursorA, cTarget, 18, dt);
    const inArea = cyL < top + 1 && cyL > -H / 2 + M.padBot - 2;
    U.uCursorA.value = inArea && raw && state !== 'error' ? cursorA : 0;
    U.uCursor.value.set(-W / 2 + M.padX + cursorPx.x + M.font * 0.28, cyL, Math.round(M.font * 0.56), Math.round(M.font * 1.0));
    dotsA = damp(dotsA, state === 'thinking' ? 1 : 0, 10, dt);
    U.uDots.value.set(-W / 2 + M.padX + U.uDotSize.value * 0.5, top - M.lineH / 2, dotsA);

    // quad bounds: panel + tail + glow margin
    const m = 46;
    let x0 = -W / 2, x1 = W / 2, y0 = -H / 2, y1 = H / 2;
    if (U.uTailOn.value) { for (const v of [U.uT0.value, U.uT1.value, U.uT2.value]) { x0 = Math.min(x0, v.x); x1 = Math.max(x1, v.x); y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y); } }
    x0 -= m; x1 += m; y0 -= m; y1 += m;
    U.uBox.value.set((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0);
    rect.x = sx - W / 2; rect.y = sy - H / 2; rect.w = W; rect.h = H; rect.visible = rev > 0.5;
  }
  const rect = { x: 0, y: 0, w: 0, h: 0, visible: false }; // panel on screen, CSS px (for wheel / touch scrolling)

  function buildTail(tx, ty, hw, hh, rPx) {
    // which edge faces the anchor (with hysteresis so it doesn't flicker on the diagonal)
    const ox = Math.abs(tx) - hw, oy = Math.abs(ty) - hh;
    const horiz = tailSide === 'left' || tailSide === 'right';
    const bias = 18;
    const wantH = horiz ? ox + bias > oy : ox > oy + bias;
    const ns = wantH ? (tx < 0 ? 'left' : 'right') : (ty < 0 ? 'bottom' : 'top');
    tailSide = ns;
    const inside = ox < 0 && oy < 0;
    if (inside) { U.uTailOn.value = 0; return; }
    U.uTailOn.value = 1;
    const c = U.uChamfer.value.x, baseW = Math.max(12, Math.min(50, 2 * (wantH ? hh : hw) - c * 2 - 10));
    let bx, by, dirx, diry;
    if (wantH) {
      const sgn = tx < 0 ? -1 : 1;
      bx = sgn * (hw - 3);
      by = clamp(ty, -hh + c + baseW * 0.5 + 4, hh - c - baseW * 0.5 - 4);
      if (hh - c - baseW * 0.5 - 4 < -hh + c + baseW * 0.5 + 4) by = 0;
      dirx = sgn; diry = 0;
    } else {
      const sgn = ty < 0 ? -1 : 1;
      by = sgn * (hh - 3);
      bx = clamp(tx, -hw + c + baseW * 0.5 + 6, hw - c - baseW * 0.5 - 6);
      dirx = 0; diry = sgn;
    }
    // tip points at the anchor but stops short of her face
    let vx = tx - bx, vy = ty - by;
    const dist = Math.hypot(vx, vy) || 1;
    vx /= dist; vy /= dist;
    // keep the tip within ~60deg of the edge normal so the wedge stays a wedge
    const dn = vx * dirx + vy * diry;
    if (dn < 0.5) { const px = vx - dn * dirx, py = vy - dn * diry; const pl = Math.hypot(px, py) || 1; vx = dirx * 0.5 + px / pl * 0.866; vy = diry * 0.5 + py / pl * 0.866; }
    // reaches toward her mouth (the anchor) but stops ~24 px short of it, so the tip never sits on her eyes or face
    // plate; a tail out of the bottom edge (bubble above her head, phones) stops at the top of her hair instead of
    // spearing down through it
    const stop = Math.max(22, rPx * 0.17);
    const len = wantH ? clamp(dist - stop, 16, 160) : clamp(Math.min(dist - stop, Math.max(26, dist - rPx * 1.25)), 16, 70);
    const half = baseW * 0.5;
    // asymmetric wedge: the edge nearer the panel's "leading" corner is longer, for an angular feel
    const ex = -diry, ey = dirx;
    U.uT0.value.set(bx + ex * half, by + ey * half);
    U.uT1.value.set(bx - ex * half * 0.6, by - ey * half * 0.6);
    U.uT2.value.set(bx + vx * len + 3 * dirx, by + vy * len + 3 * diry);
  }

  // ---------- API ----------
  function show() {
    if (!group.visible || revTarget === 0) { group.visible = true; if (rev <= 0) { revT = 0; rev = 0; sx = null; qInit = false; } }
    revTarget = 1;
  }
  function begin() {
    raw = ''; errMsg = ''; scroll = 0; scrollTarget = 0; scrollRaw = 0; contentH = 0; follow = true;
    state = 'thinking'; dirty = true;
    if (group.visible && rev > 0.5) { revT = 0.42 * 0.55; rev = 0.55; } // re-pop from a partly collapsed state
    show();
    glitch = 1;
  }
  function append(text) {
    if (!text) return;
    if (state === 'hidden' || state === 'error') begin();
    if (state === 'thinking' || state === 'done') { state = 'talking'; glitch = Math.max(glitch, 0.35); }
    raw += text; dirty = true; lastAppend = time;
  }
  function end() { if (state === 'talking' || state === 'thinking') { state = raw ? 'done' : 'done'; endAt = time; dirty = true; } }
  function setError(message) {
    errMsg = String(message || 'connection lost'); state = 'error'; dirty = true; glitch = 1; lastAppend = time;
    show();
  }
  function setAnchor(v) { if (v) { anchor.copy(v); anchorSet = true; } }
  function hide() { if (state !== 'hidden') { state = 'hidden'; revTarget = 0; revT = rev * 0.42; glitch = Math.max(glitch, 0.6); dirty = true; } }
  function clear() { raw = ''; errMsg = ''; dirty = true; }

  // fonts: re-measure / redraw when web fonts arrive
  const refont = () => { fontsDirty = true; };
  if (typeof document !== 'undefined' && document.fonts) {
    document.fonts.ready.then(refont);
    document.fonts.addEventListener && document.fonts.addEventListener('loadingdone', refont);
    const want = [`400 17px ${BODY_FONT}`, `700 17px ${BODY_FONT}`, `400 13px ${LABEL_FONT}`];
    const loadAll = () => Promise.all(want.map((f) => document.fonts.load(f).catch(() => null))).then(refont);
    loadAll();
    const link = document.querySelector('link[data-bubble-fonts]');
    if (link) link.addEventListener('load', loadAll);
  }

  return {
    group, update, begin, append, end, setError, setAnchor, hide,
    clear,
    setInsets(o) { const before = JSON.stringify(insets); Object.assign(insets, o || {}); if (JSON.stringify(insets) !== before) cfgW = 0; },
    setViewport(w, h) { if (w && h) { viewW = w; viewH = h; viewForced = true; } else viewForced = false; cfgW = 0; },
    setSide(s) { forcedSide = s || 'auto'; },
    // scroll the reply by dy CSS px (wheel / drag / PageUp); returns true if it moved. Scrolling to the bottom
    // resumes auto-follow.
    scrollBy(dy) {
      const max = Math.max(0, contentH - M.textH);
      if (max <= 0 || state === 'hidden') return false;
      const before = scrollTarget;
      // small trackpad deltas accumulate; the target always lands on a whole line (no half-cut line at the top)
      if (Math.abs(scrollRaw - scrollTarget) > M.lineH) scrollRaw = scrollTarget;
      scrollRaw = clamp(scrollRaw + dy, 0, max);
      let best = max;
      if (scrollRaw < max - M.lineH * 0.5) { best = 0; for (const ln of lines) { if (Math.abs(ln.y - scrollRaw) < Math.abs(best - scrollRaw)) best = ln.y; } }
      scrollTarget = clamp(best, 0, max);
      follow = scrollTarget >= max - 1;
      return scrollTarget !== before;
    },
    get scrollable() { return contentH > M.textH && state !== 'hidden'; },
    getScreenRect() { return rect.visible ? { x: rect.x, y: rect.y, w: rect.w, h: rect.h } : null; },
    get pageHeight() { return M.textH; },
    get state() { return state; },
    get text() { return raw; },
    get side() { return side; },
    setAvoid(x) { avoidX1 = Number.isFinite(x) ? x : -1e9; },
  };
}

function injectFonts() {
  if (document.querySelector('link[data-bubble-fonts]')) return;
  const have = [...document.querySelectorAll('link[rel=stylesheet]')].some((l) => /Exo\+2/.test(l.href) && /Share\+Tech\+Mono/.test(l.href));
  if (have) return;
  const l = document.createElement('link');
  l.rel = 'stylesheet'; l.href = FONTS_HREF; l.dataset.bubbleFonts = '1';
  l.onerror = () => l.remove(); // offline: system fallbacks
  document.head.appendChild(l);
}
