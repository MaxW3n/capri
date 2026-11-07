// city.js: the cyberpunk environment Capri sits in.
//
// Night, rain, a neon megacity. Capri sits on the concrete parapet of a rooftop high above the streets;
// the city sprawls behind and below her. Everything is procedural (no textures loaded): building windows
// are computed in a shader, signs and billboards are drawn once into two small canvas atlases, rain /
// traffic (traffic.js) / train / searchlights are animated entirely on the GPU from a time uniform. The whole thing is
// about 18 draw calls + traffic.js's 5.
//
// createCity(scene, renderer, { quality: 'high' | 'low' }) ->
//   { update(dt, t), seat, seatRotationY, setIntensity(x), dispose(), group, recommendedFar }
//
// Coordinates: metres, Y up. Capri faces +Z (toward the camera); the city is toward -Z.
// Emissive levels are tuned for UnrealBloomPass with threshold ~0.2: walls stay ~0.01, windows ~0.2-0.4,
// only neon cores (signs, trims, billboard lines, vehicle heads) go above 1.

import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { PALETTE } from "./palette.js";
import { createTraffic } from "./traffic.js";

const SEAT = new THREE.Vector3(0, 0.8, 0); // top-front edge of the parapet, where her hips go
const FLOOR = -0.62;                        // the lower roof terrace in front of the parapet (the camera side)
const Y_AXIS = new THREE.Vector3(0, 1, 0);
// perches (see PERCHES below): seat = top-front edge, ledgeYaw = the edge's outward normal (atan2(nx, nz))
const P_FORE = { seat: new THREE.Vector3(-0.6, 0.45, 1.15), ledgeYaw: 0.3 };
const RIGHT_FG = { c: new THREE.Vector3(1.5, 0, 2.35), yaw: -0.3 }; // the bottom-right foreground crates (not a perch)
const P_HUT = { seat: new THREE.Vector3(-1.12, 1.45, -2.42), ledgeYaw: 0 };
const P_TANK = { seat: new THREE.Vector3(0.6, 1.75, -4.6), ledgeYaw: 0 };
// city.perches (README): rotationY = Capri's final root yaw there (= ledgeYaw + camera yaw offset + 0.6, like main.js does
// for the parapet); view = an optional suggested desktop camera for that perch (null = main.js's default shot)
const PERCHES = [
  { id: "parapet", seat: SEAT.clone(), rotationY: 0.71, ledgeYaw: 0, ledgeW: 1.5, depth: "mid", portrait: true, view: null },
  { id: "crates", seat: P_FORE.seat.clone(), rotationY: 0.9, ledgeYaw: P_FORE.ledgeYaw, ledgeW: 0.55, depth: "fore", portrait: true,
    // in the master shot (no view of its own on desktop): the stack sits on the terrace ~1 m in front of the parapet,
    // left of her legs, so she reads ~1.3x bigger and lower in the frame. cropY (model space): below her hips she may
    // run off the frame's bottom edge / behind the near crate edge (main.js still keeps her clear of the prompt bar)
    cropY: 0.6, view: null,
    viewPortrait: { pos: new THREE.Vector3(0.1, 1.45, 5.0), target: new THREE.Vector3(-0.55, 1.75, 1.12), fov: 46 } },
  { id: "stair-hut", seat: P_HUT.seat.clone(), rotationY: 0.83, ledgeYaw: 0, ledgeW: 1.06, depth: "back", portrait: true,
    view: { pos: new THREE.Vector3(0.7, 1.75, 4.6), target: new THREE.Vector3(-0.5, 1.95, -2.42), fov: 38 }, // clear of the pole / blade sign
    viewPortrait: { pos: new THREE.Vector3(-1.0, 1.75, 5.3), target: new THREE.Vector3(-1.05, 2.2, -2.42), fov: 46 } },
  // (portrait: a closer shot from just in front of the parapet, so she isn't a 170 px figure on a phone)
  { id: "water-tank", seat: P_TANK.seat.clone(), rotationY: 0.57, ledgeYaw: 0, ledgeW: 0.8, depth: "back", portrait: true, view: null,
    viewPortrait: { pos: new THREE.Vector3(0.45, 1.75, 1.7), target: new THREE.Vector3(0.62, 2.85, -4.6), fov: 46 } },
];
const FENCE = { x0: -1.0, z0: 1.35, x1: -0.56, z1: 2.65, h: 1.85 };
const NEAR_CAB = { x: -0.3, z: 2.45, yaw: 0.25, top: 0.6 }; // utility cabinet near the lens, in front of the fore perch
const POLE = { x: -0.98, z: 3.05 };      // the utility pole, just off the left edge of the master shot (its cables cross the top-left)
const RECOMMENDED_FAR = 1500;              // the far skyline sits ~450 m out; the sky dome never clips

// ------------------------------------------------------------------------------------------------
// small helpers
// ------------------------------------------------------------------------------------------------

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const lin = (hex, k = 1) => new THREE.Color(hex).multiplyScalar(k); // sRGB hex -> linear Color

// neon colours (linear). Capri's palette first; cyan / yellow / violet are sparing accents.
const NEON = {
  pink: lin(PALETTE.pink),
  green: lin(PALETTE.green),
  blue: lin(0x2f7dff),     // a brighter take on her eye blue so it reads as neon
  eye: lin(PALETTE.blue),
  cyan: lin(0x3ee8ff),
  yellow: lin(0xffc93d),
  violet: lin(0xa64dff),
  white: lin(PALETTE.white),
};

// ------------------------------------------------------------------------------------------------
// shared GLSL: hashes, noise, sky colour, fog. Every custom material includes this.
// ------------------------------------------------------------------------------------------------

const COMMON = /* glsl */ `
uniform float uTime;
uniform float uI;            // smoothed "loudness" 0..1
uniform vec3 uSkyHorizon;    // violet glow at the horizon, sides
uniform vec3 uSkyHorizonC;   // deep teal-blue at the horizon, centre (behind her head)
uniform vec3 uSkyZenith;
uniform vec3 uSkyBelow;      // smog lit by the streets
uniform float uFogDensity;

float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), f.x), mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm3(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 3; i++){ s += a * vnoise(p); p = p * 2.07 + 13.7; a *= 0.5; } return s; }

vec3 skyCol(vec3 d){
  float side = clamp(abs(d.x) * 1.5, 0.0, 1.0);
  vec3 hor = mix(uSkyHorizonC, uSkyHorizon, side);
  vec3 c = mix(hor, uSkyZenith, pow(smoothstep(-0.02, 0.55, d.y), 0.6));
  c = mix(c, uSkyBelow * mix(0.75, 1.0, side), smoothstep(0.0, -0.22, d.y));
  return c;
}
float fogAmt(float dist){ float x = uFogDensity * dist; return 1.0 - exp(-x * x); }
// opaque things fade into the sky colour in their direction (slightly darker, so far towers read as silhouettes)
vec3 fogMix(vec3 col, vec3 wp){ vec3 d = wp - cameraPosition; float L = length(d); return mix(col, skyCol(d / L) * 0.86, fogAmt(L)); }
// additive things just fade out
float fogKeep(vec3 wp){ return 1.0 - fogAmt(length(wp - cameraPosition)); }
`;
const COMMON_FS = /* glsl */ `
// anti-aliased band |d| < w
float bandAA(float d, float w){ float px = max(fwidth(d), 1e-4); return clamp((w - abs(d)) / px + 0.5, 0.0, 1.0) * min(1.0, 2.0 * w / px); }
`;

const OUT = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

// ------------------------------------------------------------------------------------------------
// canvas atlases
// ------------------------------------------------------------------------------------------------

const JP_FONT = '"Hiragino Kaku Gothic ProN","Hiragino Sans","Yu Gothic","Noto Sans JP","Noto Sans CJK JP","IPAGothic","WenQuanYi Zen Hei",sans-serif';
const LAT_FONT = '"Arial Narrow","Helvetica Neue","Roboto Condensed",Arial,sans-serif';

function cssOf(c) { const s = c.clone().convertLinearToSRGB(); return `rgb(${(s.r * 255) | 0},${(s.g * 255) | 0},${(s.b * 255) | 0})`; }
function cssMixWhite(c, k) { const s = c.clone().convertLinearToSRGB().lerp(new THREE.Color(1, 1, 1), k); return `rgb(${(s.r * 255) | 0},${(s.g * 255) | 0},${(s.b * 255) | 0})`; }

// invented shop names. vertical: stacked glyphs (+ small latin); horizontal: latin + katakana subline.
const V_SIGNS = [
  { t: "山羊座", s: "BAR", c: "blue" },        // 0: the blade sign beside Capri (blue: hot pink is kept for her)
  { t: "ラーメン", s: "", c: "yellow" },
  { t: "電脳医", s: "24H", c: "cyan" },
  { t: "ホテル", s: "KAGE", c: "violet" },
  { t: "カラオケ", s: "", c: "pink" },
  { t: "酒場", s: "SAKE", c: "green" },
  { t: "義体", s: "MODS", c: "blue" },
  { t: "夜市", s: "", c: "green" },
];
const H_SIGNS = [
  { t: "NEKO DATA", s: "ネコ・データ", c: "green" },
  { t: "BYTE NOODLE", s: "バイト麺", c: "pink" },
  { t: "GOAT 24H", s: "ヤギ", c: "green" },      // 2: the small lightbox on the parapet
  { t: "KUMO HOTEL", s: "クモ・ホテル", c: "blue" },
  { t: "SYNTH SUSHI", s: "シンセ寿司", c: "cyan" },
  { t: "ZERO PARLOR", s: "ゼロ", c: "violet" },
  { t: "OPEN", s: "営業中", c: "pink" },
  { t: "CHROME+BONE", s: "修理・改造", c: "green" },
];

function vRect(i) { return [i * 128, 0, 128, 512]; }
function hRect(j) { return [(j % 2) * 512, 512 + (j >> 1) * 128, 512, 128]; }
function rectUV([x, y, w, h], S = 1024) { return [x / S, 1 - (y + h) / S, (x + w) / S, 1 - y / S]; } // u0 v0 u1 v1

function makeSignAtlas() {
  const S = 1024, cv = document.createElement("canvas"); cv.width = cv.height = S;
  const g = cv.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, S, S);
  const frame = (x, y, w, h, col) => {
    g.save();
    g.fillStyle = "rgb(10,6,16)"; g.fillRect(x + 4, y + 4, w - 8, h - 8);
    g.strokeStyle = col; g.lineWidth = 5; g.shadowColor = col; g.shadowBlur = 14;
    g.strokeRect(x + 10, y + 10, w - 20, h - 20);
    g.restore();
  };
  const glowText = (txt, x, y, font, col, core) => {
    g.save(); g.font = font; g.textAlign = "center"; g.textBaseline = "middle";
    g.shadowColor = col; g.shadowBlur = 22; g.fillStyle = col; g.fillText(txt, x, y);
    g.shadowBlur = 6; g.fillStyle = core; g.fillText(txt, x, y);
    g.restore();
  };
  V_SIGNS.forEach((sg, i) => {
    const [x, y, w, h] = vRect(i), c = NEON[sg.c], col = cssOf(c), core = cssMixWhite(c, 0.55);
    frame(x, y, w, h, col);
    const chars = [...sg.t], n = chars.length, room = sg.s ? 400 : 460;
    const step = Math.min(110, room / n), fs = Math.min(92, step * 0.86);
    const y0 = y + 26 + (room - step * n) / 2 + step / 2;
    chars.forEach((ch, k) => glowText(ch, x + w / 2, y0 + k * step, `bold ${fs}px ${JP_FONT}`, col, core));
    if (sg.s) glowText(sg.s, x + w / 2, y + h - 52, `bold 34px ${LAT_FONT}`, col, core);
  });
  H_SIGNS.forEach((sg, j) => {
    const [x, y, w, h] = hRect(j), c = NEON[sg.c], col = cssOf(c), core = cssMixWhite(c, 0.55);
    frame(x, y, w, h, col);
    g.save(); g.translate(x + w / 2, y + 52); g.scale(0.82, 1);
    glowText(sg.t, 0, 0, `bold 64px ${LAT_FONT}`, col, core); g.restore();
    glowText(sg.s, x + w / 2, y + 100, `bold 26px ${JP_FONT}`, col, core);
  });
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  return tex;
}

// goat-horn emblem in white lines: the "Project Capricorn" mark (original drawing)
function drawGoatLogo(g, cx, cy, s) {
  g.save(); g.translate(cx, cy); g.scale(s, s);
  g.strokeStyle = "#fff"; g.fillStyle = "#fff"; g.lineCap = "round"; g.lineJoin = "round";
  g.shadowColor = "#fff"; g.shadowBlur = 10;
  // outer ring + ticks
  g.lineWidth = 7; g.beginPath(); g.arc(0, 0, 210, 0, Math.PI * 2); g.stroke();
  g.lineWidth = 3; g.beginPath(); g.arc(0, 0, 190, 0, Math.PI * 2); g.stroke();
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2, r0 = i % 4 ? 172 : 160;
    g.beginPath(); g.moveTo(Math.cos(a) * r0, Math.sin(a) * r0); g.lineTo(Math.cos(a) * 184, Math.sin(a) * 184); g.stroke();
  }
  // horns: two swept, curling strokes
  g.lineWidth = 22;
  for (const m of [-1, 1]) {
    g.beginPath();
    g.moveTo(m * 26, -20);
    g.bezierCurveTo(m * 30, -120, m * 120, -150, m * 140, -70);
    g.bezierCurveTo(m * 152, -20, m * 105, 5, m * 88, -40);
    g.stroke();
  }
  // face: a sharp shield
  g.lineWidth = 9;
  g.beginPath(); g.moveTo(-52, -18); g.lineTo(52, -18); g.lineTo(20, 110); g.lineTo(0, 140); g.lineTo(-20, 110); g.closePath(); g.stroke();
  // eyes
  g.beginPath(); g.moveTo(-34, 22); g.lineTo(-10, 30); g.lineTo(-30, 38); g.closePath(); g.fill();
  g.beginPath(); g.moveTo(34, 22); g.lineTo(10, 30); g.lineTo(30, 38); g.closePath(); g.fill();
  g.restore();
}

function makeBoardAtlas() {
  const S = 1024, cv = document.createElement("canvas"); cv.width = cv.height = S;
  const g = cv.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, S, S);
  // top half: landscape billboard (1024x512): emblem left, copy right
  drawGoatLogo(g, 250, 256, 1.0);
  g.save(); g.fillStyle = "#fff"; g.shadowColor = "#fff"; g.shadowBlur = 8; g.textBaseline = "alphabetic";
  g.font = `bold 46px ${LAT_FONT}`; g.fillText("PROJECT", 500, 150);
  g.save(); g.translate(496, 262); g.font = `900 124px ${LAT_FONT}`; const cw = g.measureText("CAPRICORN").width; g.scale(Math.min(0.85, 500 / cw), 1); g.fillText("CAPRICORN", 0, 0); g.restore();
  g.font = `bold 40px ${JP_FONT}`; g.fillText("プロジェクト・カプリコーン", 500, 330);
  g.fillRect(500, 360, 470, 5);
  g.font = `bold 36px ${LAT_FONT}`; g.fillText("ONE DOSE.  A NEW YOU.", 500, 420);
  g.font = `24px ${LAT_FONT}`; g.fillText("STEERED MINDS // EST. 2077-Q4 // 山羊座", 500, 462);
  g.restore();
  // bottom-left (512x512): emblem alone, for the floating hologram
  drawGoatLogo(g, 256, 768, 1.1);
  // bottom-right: vertical katakana banner
  g.save(); g.fillStyle = "#fff"; g.shadowColor = "#fff"; g.shadowBlur = 8; g.font = `bold 64px ${JP_FONT}`; g.textAlign = "center";
  [..."カプリコーン"].forEach((ch, k) => g.fillText(ch, 640, 600 + k * 70));
  g.font = `bold 34px ${LAT_FONT}`; [..."CAPRI"].forEach((ch, k) => g.fillText(ch, 760, 600 + k * 70));
  g.restore();
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return tex;
}

// ------------------------------------------------------------------------------------------------
// createCity
// ------------------------------------------------------------------------------------------------

export function createCity(scene, renderer, opts = {}) {
  const hi = (opts.quality || "high") !== "low";
  const R = mulberry32(0xc0ffee);
  const rr = (a, b) => a + (b - a) * R();
  const group = new THREE.Group(); group.name = "city";
  scene.add(group);
  const trash = []; // things to dispose

  const U = {
    uTime: { value: 0 },
    uI: { value: 0 },
    uSkyHorizon: { value: lin(0x1c1244) },
    uSkyHorizonC: { value: lin(0x3a0c3c) },
    uSkyZenith: { value: lin(0x030207) },
    uSkyBelow: { value: lin(0x3a1438) },
    uFogDensity: { value: 0.0052 },
  };
  const prevFog = scene.fog, prevBg = scene.background;
  scene.fog = new THREE.FogExp2(0x1a0f2c, 0.012); // for non-city materials (Capri, props): only matters up close
  scene.background = new THREE.Color(PALETTE.void);

  const shader = (o) => {
    const m = new THREE.ShaderMaterial({
      uniforms: { ...U, ...(o.uniforms || {}) },
      vertexShader: COMMON + o.vs, fragmentShader: COMMON + COMMON_FS + o.fs,
      transparent: !!o.additive, depthWrite: !o.additive,
      blending: o.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      side: o.side ?? THREE.FrontSide,
    });
    trash.push(m);
    return m;
  };
  const add = (obj, order = 0) => { obj.renderOrder = order; group.add(obj); if (obj.geometry) trash.push(obj.geometry); return obj; };

  // ------------------------------------------------------------------ sky dome
  {
    const geo = new THREE.SphereGeometry(1, 32, 16);
    const mat = shader({
      side: THREE.BackSide,
      vs: `varying vec3 vDir; void main(){ vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position * 1000.0 + cameraPosition, 1.0); gl_Position = p.xyww; }`,
      fs: `varying vec3 vDir; void main(){
        vec3 d = normalize(vDir);
        vec3 c = skyCol(d);
        // two smog / cloud layers drifting at different speeds, lit hot pink from below by the city (strongest near
        // the horizon, fading to violet higher up), with darker gaps between them so the sky has depth
        vec2 q = vec2(atan(d.x, -d.z) * 3.0, d.y * 9.0);
        float cl = fbm3(q * vec2(2.6, 2.4) + vec2(uTime * 0.03, 0.0));
        float cl2 = fbm3(q * vec2(1.4, 3.6) + vec2(-uTime * 0.015, 3.1));
        float band = smoothstep(0.01, 0.08, d.y) * (1.0 - smoothstep(0.14, 0.42, d.y));
        float band2 = smoothstep(0.1, 0.2, d.y) * (1.0 - smoothstep(0.28, 0.6, d.y));
        // pink only right above the skyline; violet, then her eye blue higher up (so her pink hair, which sits a few
        // degrees above the horizon in frame, keeps its contrast against the sky)
        vec3 lit = mix(vec3(1.0, 0.06, 0.32), vec3(0.3, 0.07, 0.55), smoothstep(0.015, 0.09, d.y));
        lit = mix(lit, vec3(0.08, 0.2, 0.62), smoothstep(0.12, 0.3, d.y));
        float centre = 1.0 - smoothstep(0.0, 0.7, abs(d.x - 0.12)); // brightest in the open canyon right of her head
        float k1 = smoothstep(0.38, 0.78, cl), k2 = smoothstep(0.45, 0.85, cl2);
        c *= 1.0 - 0.55 * (1.0 - k1) * band;                  // dark gaps between the lit smog banks
        c += lit * k1 * k1 * band * (0.12 + 0.3 * centre);
        c += lit * k2 * k2 * band2 * (0.05 + 0.14 * centre);
        // horizon glow of the far city, a thin bright band right at the skyline
        c += vec3(0.9, 0.1, 0.4) * 0.05 * exp(-abs(d.y - 0.012) * 40.0) * (0.6 + 0.4 * centre);
        gl_FragColor = vec4(c, 1.0);
        ${OUT}
      }`,
    });
    mat.depthWrite = false;
    const sky = add(new THREE.Mesh(geo, mat), -10);
    sky.frustumCulled = false;
    let warned = false;
    sky.onBeforeRender = (_r, _s, cam) => {
      if (!warned && cam.isPerspectiveCamera && cam.far < 600) { warned = true; console.warn(`[city] camera.far=${cam.far}: set it to >= ${RECOMMENDED_FAR} (city.recommendedFar) or the skyline is clipped`); }
    };
  }

  // ------------------------------------------------------------------ buildings (one InstancedMesh)
  // per instance: aStyle (hue 0..1, lit fraction, window cell scale, seed), aTrim (rgb neon, code = type + width/2)
  const B = []; // {x,z,w,d,top,bottom,rot,hue,lit,cell,trim:[r,g,b],ttype,tw}
  const addB = (o) => { B.push(Object.assign({ bottom: -170, rot: 0, hue: R(), lit: 0.3, cell: 1, trim: null, ttype: 0, tw: 0.4 }, o)); return B[B.length - 1]; };
  const trimCols = [NEON.pink, NEON.green, NEON.blue, NEON.cyan, NEON.violet];
  const pickHue = (x) => {
    // windows: mostly warm or cool white; the centre (behind her) leans blue/teal
    const r = R();
    if (Math.abs(x) < 25 && r < 0.6) return 0.3 + R() * 0.25;
    return r < 0.45 ? R() * 0.15 : r < 0.85 ? 0.25 + R() * 0.3 : 0.6 + R() * 0.4;
  };

  // hero buildings, placed by hand for the composition
  // the billboard tower sits at the right edge, the billboard high up on it: the band right of her head (where the
  // reply bubble floats) stays dark and open toward the far skyline
  const heroB = addB({ x: 31, z: -74, w: 16, d: 14, top: 56, hue: 0.32, lit: 0.3, trim: NEON.blue, ttype: 2, tw: 0.07 }); // carries the big billboard
  const BOARD = { x: heroB.x - 6, y: 15.6, z: heroB.z + heroB.d / 2 + 1.2, w: 16, h: 8, yaw: -0.3 };
  // a low block to her right (its roof sits well above the parapet line, so its trim can't read as a seam in the
  // ledge neon); carries a readable shop sign below the bubble
  const heroR = addB({ x: 12, z: -30, w: 11, d: 9, top: 1.2, hue: 0.47, lit: 0.17, cell: 0.72, trim: NEON.blue, ttype: 1, tw: 0.12 });
  const boardBase = addB({ x: -17, z: -52, w: 16, d: 12, top: -9, hue: 0.12, lit: 0.36 });
  const holoBase = addB({ x: -50, z: -104, w: 16, d: 16, top: 6, hue: 0.4, lit: 0.3, trim: NEON.cyan, ttype: 1, tw: 0.3 });
  addB({ x: -46, z: -320, w: 40, d: 38, top: 165, hue: 0.45, lit: 0.22, cell: 2.2, trim: NEON.blue, ttype: 3, tw: 0.35 }); // far megatower, just left of her head (the canyon right of it stays open)
  addB({ x: 70, z: -240, w: 34, d: 34, top: 120, hue: 0.62, lit: 0.25, cell: 2.0, trim: NEON.violet, ttype: 2, tw: 0.4 });
  addB({ x: -95, z: -210, w: 30, d: 30, top: 95, hue: 0.9, lit: 0.25, cell: 2.0, trim: NEON.pink, ttype: 3, tw: 0.3 });

  // keep sightlines from the camera to the two hero billboards clear: anything in front of one is capped below it
  const SIGHT = [
    { x: -50, z: -104, y0: 8.0, half: 0.09 },   // emblem hologram (left)
    { x: BOARD.x, z: BOARD.z, y0: BOARD.y - BOARD.h / 2, half: 0.13 },   // big billboard (right)
  ];
  // a sightline over the left third of the frame (between the emblem hologram and her) and right of her head:
  // mid-distance roofs there stay under the horizon so the hazy far skyline, the horizon glow and a searchlight
  // read behind her
  const VISTA = [{ a0: -0.42, a1: -0.2, rise: 0.012 }, { a0: -0.03, a1: 0.31, rise: 0.028 }];
  // the open canyon also holds for the far field: towers out there stay low inside this wedge, so the sky, the lit
  // smog and the horizon glow show behind her head and the bubble
  const FAR_VISTA = { a0: -0.06, a1: 0.33, rise: 0.04 };
  const clampTop = (x, z, w, top) => {
    for (const t of SIGHT) {
      if (z <= t.z + 4) continue;
      const db = Math.hypot(x - 0.4, 3.6 - z), dt = Math.hypot(t.x - 0.4, 3.6 - t.z);
      const da = Math.abs(Math.atan2(x - 0.4, 3.6 - z) - Math.atan2(t.x - 0.4, 3.6 - t.z));
      if (da < t.half + Math.atan((w * 0.75) / db)) top = Math.min(top, 1.35 + (t.y0 - 1.35) * (db / dt) - 0.5);
    }
    const db = Math.hypot(x - 0.4, 3.6 - z), a = Math.atan2(x - 0.4, 3.6 - z), half = Math.atan((w * 0.6) / db);
    if (z > -150) {
      for (const v of VISTA) if (a + half > v.a0 && a - half < v.a1) top = Math.min(top, 1.35 + db * v.rise);
    } else if (a + half > FAR_VISTA.a0 && a - half < FAR_VISTA.a1) {
      // a soft edge: full cap in the middle of the wedge, rising toward its edges
      const m = Math.min(a + half - FAR_VISTA.a0, FAR_VISTA.a1 - (a - half)) / 0.08;
      top = Math.min(top, 1.35 + db * (FAR_VISTA.rise + 0.25 * Math.max(0, 1 - m) ** 2));
    }
    return top;
  };

  const nNear = hi ? 26 : 14, nMid = hi ? 46 : 22, nFar = hi ? 60 : 30;
  for (let i = 0, tries = 0; i < nNear && tries < 600; tries++) {
    const z = rr(-16, -48), x = rr(-1.2, 1.2) * (Math.abs(z) + 10);
    if (Math.abs(x) < Math.abs(z) * 0.32 + 5) continue; // keep a canyon open behind her
    if (Math.abs(x - heroB.x) < 9 && Math.abs(z - heroB.z) < 10) continue;
    if (Math.abs(x - heroR.x) < 11 && Math.abs(z - heroR.z) < 10) continue;
    const w = rr(6, 14), top = clampTop(x, z, w, x > 20 && R() < 0.25 ? rr(0, 14) : rr(-26, -4));
    addB({ x, z, w, d: rr(6, 14), top, hue: pickHue(x), lit: rr(0.18, 0.42), cell: rr(0.9, 1.2),
      trim: R() < 0.35 ? trimCols[(R() * trimCols.length) | 0] : null, ttype: R() < 0.6 ? 1 : 3, tw: rr(0.15, 0.3) });
    i++;
  }
  for (let i = 0, tries = 0; i < nMid && tries < 800; tries++) {
    const z = rr(-55, -140), x = rr(-1.15, 1.15) * Math.abs(z);
    const tall = Math.abs(x) > Math.abs(z) * 0.3 && R() < 0.3;
    const w = rr(10, 24), top = clampTop(x, z, w, tall ? rr(12, 42) : rr(-32, 4));
    if (Math.abs(x) < Math.abs(z) * 0.12 + 4 && top > -8) continue;
    if (Math.abs(x - heroB.x) < 14 && Math.abs(z - heroB.z) < 14) continue;
    addB({ x, z, w, d: rr(10, 24), top, hue: pickHue(x), lit: rr(0.15, 0.4), cell: rr(1.1, 1.5),
      trim: R() < 0.3 ? trimCols[(R() * trimCols.length) | 0] : null, ttype: [1, 2, 3][(R() * 3) | 0], tw: rr(0.2, 0.35) });
    i++;
  }
  for (let i = 0; i < nFar; i++) {
    const z = rr(-150, -460), x = rr(-1.1, 1.1) * Math.abs(z);
    const tall = R() < 0.25;
    const w = rr(18, 45);
    addB({ x, z, w, d: rr(18, 45), top: clampTop(x, z, w, tall ? rr(40, 110) : rr(-15, 30)), rot: rr(-0.3, 0.3), hue: pickHue(x),
      lit: rr(0.12, 0.32), cell: rr(1.6, 2.4), trim: R() < 0.3 ? trimCols[(R() * trimCols.length) | 0] : null, ttype: [1, 3][(R() * 2) | 0], tw: rr(0.3, 0.6) });
  }
  // small rooftop sheds / water tanks on near and mid roofs
  const roofCount = B.length;
  for (let i = 0; i < roofCount; i++) {
    const b = B[i];
    if (b.z < -140 || R() < 0.45) continue;
    const k = 1 + ((R() * 2) | 0);
    for (let j = 0; j < k; j++) {
      const w = rr(2, 5), d = rr(2, 5), sx = b.x + rr(-0.3, 0.3) * (b.w - w), sz = b.z + rr(-0.3, 0.3) * (b.d - d);
      const st = clampTop(sx, sz, w, b.top + rr(1.5, 4));
      if (st < b.top + 0.8) continue;
      addB({ x: sx, z: sz, w, d, bottom: b.top, top: st,
        hue: b.hue, lit: R() < 0.5 ? 0.0 : 0.5, cell: 0.8, trim: R() < 0.25 ? trimCols[(R() * trimCols.length) | 0] : null, ttype: 1, tw: 0.12 });
    }
  }
  // elevated rail: track + pillars
  const TRACK = { z: -46, y: -3.4 };
  addB({ x: 0, z: TRACK.z, w: 900, d: 4, bottom: TRACK.y - 1.0, top: TRACK.y, lit: 0, trim: NEON.pink, ttype: 1, tw: 0.08 });
  for (let x = -260; x <= 260; x += 26) addB({ x, z: TRACK.z, w: 1.6, d: 1.6, top: TRACK.y - 1.0, lit: 0 });

  // sign placements (filled while placing, built below)
  const signs = []; // {c:center Vector3, r:right Vector3, u:up Vector3, hw, hh, uv:[u0,v0,u1,v1], near, bright, kind}
  const addSign = (o) => signs.push(Object.assign({ near: 0, bright: 1.9, kind: 0 }, o));
  const Z = new THREE.Vector3(0, 0, 1), Xp = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0);

  // signs on near / mid buildings: on the front face, or on the side face that looks toward the canyon
  B.forEach((b, i) => {
    if (b.w > 100 || b.w < 5 || b.z < -150 || b.top < -18) return;
    if (i < 8 && b !== heroB && b !== heroR) return;
    const nSigns = b === heroB ? 4 : b === heroR ? 0 : R() < 0.75 ? 1 + ((R() * 2) | 0) : 0;
    for (let s = 0; s < nSigns; s++) {
      const vertical = b === heroB ? true : R() < 0.6;
      const side = (b === heroB || (R() < 0.5 && Math.abs(b.x) > 8)) ? -Math.sign(b.x) : 0; // 0 = front face
      const sc = b.z > -60 ? 1 : 1.6;
      const hw = (vertical ? rr(0.7, 1.1) : rr(2.6, 4.2)) * sc, hh = vertical ? hw * 4 : hw / 4;
      const yTop = Math.min(b.top - 1, 30), yBot = Math.max(-22, b.top - 26);
      if (yTop - yBot < hh * 2 + 1) continue;
      const cy = b === heroB ? (s < 2 ? 34 : 4) - (s % 2) * 9 : rr(yBot + hh, yTop - hh);
      let c, r;
      if (side) {
        const fx = b.x + side * (b.w / 2 + 0.25), span = b.d / 2 - hw - 0.5;
        if (span < 0) continue;
        c = new THREE.Vector3(fx, cy, b.z + (b === heroB ? (s % 2 ? -1 : 1) * Math.min(span, 2.5) : rr(-span, span)));
        r = new THREE.Vector3(0, 0, -side);
      } else {
        const span = b.w / 2 - hw - 0.5;
        if (span < 0) continue;
        c = new THREE.Vector3(b.x + rr(-span, span), cy, b.z + b.d / 2 + 0.25);
        r = Xp.clone();
      }
      const idx = vertical ? 1 + ((R() * 7) | 0) : [0, 1, 3, 4, 5, 6, 7][(R() * 7) | 0];
      addSign({ c, r, u: Y, hw, hh, uv: rectUV(vertical ? vRect(idx) : hRect(idx)), near: THREE.MathUtils.clamp(1 - Math.abs(c.z) / 40, 0, 1) * 0.6, bright: rr(0.9, 1.5) });
      // dark backing box (no windows)
      addB({ x: c.x - (side ? side * 0.12 : 0), z: c.z - (side ? 0 : 0.12), w: side ? 0.25 : hw * 2 + 0.3, d: side ? hw * 2 + 0.3 : 0.25,
        bottom: cy - hh - 0.15, top: cy + hh + 0.15, lit: 0, hue: 0 });
    }
  });
  // the big horizontal sign on heroR's roof
  // dark backing panel behind the big billboard so the tower's windows don't show through the hologram
  addB({ x: BOARD.x + 0.35 * Math.sin(BOARD.yaw), z: BOARD.z - 0.35 * Math.cos(BOARD.yaw), w: BOARD.w + 0.6, d: 0.3, bottom: BOARD.y - BOARD.h / 2 - 0.4, top: BOARD.y + BOARD.h / 2 + 0.4, rot: BOARD.yaw, lit: 0, hue: 0 });
  // readable shop signs either side of her, below the bubble band (on heroR's face, and a low block at the left)
  addSign({ c: new THREE.Vector3(13.6, -1.4, heroR.z + heroR.d / 2 + 0.25), r: Xp, u: Y, hw: 2.3, hh: 0.575, uv: rectUV(hRect(4)), near: 0.25, bright: 1.25 });
  // a rooftop sign on poles out in the left vista (sits on the horizon, left of her)
  {
    // (x: clear of the vent stack on the roof in front of it)
    const sx = -18.8, sy = 2.5, sz = -40, yaw = 0.38, r = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    addB({ x: sx - 0.18 * Math.sin(yaw), z: sz - 0.18 * Math.cos(yaw), w: 5.3, d: 0.25, bottom: sy - 0.78, top: sy + 0.78, rot: yaw, lit: 0, hue: 0 });
    for (const k of [-1.6, 1.6]) addB({ x: sx + r.x * k - 0.2 * Math.sin(yaw), z: sz + r.z * k - 0.2 * Math.cos(yaw), w: 0.22, d: 0.22, bottom: -24, top: sy - 0.78, lit: 0, hue: 0 });
    addSign({ c: new THREE.Vector3(sx, sy, sz), r, u: Y, hw: 2.5, hh: 0.625, uv: rectUV(hRect(0)), near: 0.2, bright: 1.2 });
  }

  {
    const n = B.length;
    const geo = new THREE.BoxGeometry(1, 1, 1); geo.translate(0, 0.5, 0);
    const style = new Float32Array(n * 4), trim = new Float32Array(n * 4);
    const mat = shader({
      vs: `
        attribute vec4 aStyle; attribute vec4 aTrim;
        varying vec3 vLocal; varying vec3 vN; varying vec3 vW; varying vec3 vS; varying vec4 vStyle; varying vec4 vTrim;
        void main(){
          vec3 s = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
          vS = s; vLocal = position * s; vN = normal; vStyle = aStyle; vTrim = aTrim;
          vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
          vW = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fs: `
        varying vec3 vLocal; varying vec3 vN; varying vec3 vW; varying vec3 vS; varying vec4 vStyle; varying vec4 vTrim;
        vec3 winCol(float h, float v){
          vec3 warm = vec3(1.0, 0.55, 0.25), cool = vec3(0.62, 0.78, 1.0), cyan = vec3(0.2, 0.7, 1.0), green = vec3(0.25, 1.0, 0.5), pink = vec3(1.0, 0.18, 0.42);
          vec3 c = h < 0.2 ? warm : h < 0.4 ? cool : h < 0.6 ? cyan : h < 0.8 ? mix(cool, green, 0.6) : mix(warm, pink, 0.7);
          return mix(c, v < 0.5 ? warm : cool, step(0.72, fract(v * 7.31)) * 0.6);
        }
        void main(){
          vec3 n = normalize(vN);
          float isSide = 1.0 - step(0.5, abs(n.y));
          float onX = step(0.5, abs(n.x));
          vec2 uv = mix(vec2(vLocal.x * sign(n.z), vLocal.y), vec2(-vLocal.z * sign(n.x), vLocal.y), onX);
          float halfW = mix(vS.x, vS.z, onX) * 0.5;
          float h = vS.y;
          float seed = vStyle.w;
          // wall: near-black, a touch of coloured street glow from below
          float upl = clamp(exp(-(vW.y + 28.0) / 16.0), 0.0, 1.0);
          vec3 glow = mix(vec3(0.5, 0.05, 0.25), vec3(0.03, 0.25, 0.35), smoothstep(-60.0, 60.0, vW.x));
          vec3 col = vec3(0.010, 0.009, 0.016) + glow * upl * 0.045;
          col += vec3(0.010, 0.008, 0.016) * (1.0 - smoothstep(-25.0, -60.0, vW.z)) * smoothstep(-30.0, 2.0, vW.y); // near walls: faint albedo
          // windows
          vec2 cs = vec2(1.1, 1.8) * vStyle.z;
          vec2 cell = vec2(uv.x + halfW, uv.y) / cs;
          vec2 id = floor(cell), f = fract(cell);
          float faceId = dot(n, vec3(1.0, 2.0, 3.0));
          float rnd = hash12(id + seed * 37.0 + faceId * 11.0);
          float floorDark = step(0.8, hash12(vec2(id.y * 1.3, seed * 13.0 + faceId)));
          float lit = step(rnd, vStyle.y) * (1.0 - floorDark);
          vec2 fw = fwidth(cell);
          float detail = 1.0 - smoothstep(0.15, 0.55, max(fw.x, fw.y));
          float variant = floor(fract(seed * 0.377) * 3.0);   // 0 punched windows, 1 ribbon windows, 2 tall slits
          vec2 e = variant < 0.5 ? vec2(0.16, 0.26) : variant < 1.5 ? vec2(0.03, 0.32) : vec2(0.32, 0.12);
          if (variant > 0.5 && variant < 1.5) { rnd = hash12(vec2(floor(id.x / 5.0), id.y) + seed * 37.0 + faceId * 11.0); lit = step(rnd, vStyle.y) * (1.0 - floorDark); }
          vec2 a = smoothstep(e - fw, e + fw, f), b = 1.0 - smoothstep(1.0 - e - fw, 1.0 - e + fw, f);
          float rect = a.x * a.y * b.x * b.y;
          float hv = hash12(id * 1.7 + seed + faceId);
          // a few windows flicker (TVs, broken tubes)
          float fl = mix(1.0, 0.55 + 0.45 * step(0.5, hash11(floor(uTime * 9.0) + rnd * 91.0)), step(0.985, hv));
          // interiors: brighter toward the sill, some with blinds, a few tinted her colours
          float blinds = mix(1.0, 0.55 + 0.45 * step(0.5, fract(f.y * 7.0)), step(0.6, hash12(id + 7.7)) * detail);
          float grad = mix(0.65, 1.15, 1.0 - f.y);
          vec3 wc = winCol(vStyle.x, hv);
          float tint = hash12(id + 5.3);
          wc = tint > 0.93 ? vec3(1.0, 0.1, 0.35) : tint > 0.86 ? vec3(0.15, 1.0, 0.45) : tint > 0.8 ? vec3(0.15, 0.45, 1.0) : wc;
          vec3 nearW = wc * mix(0.3, 1.0, hash12(id + 3.1)) * lit * rect * fl * blinds * grad;
          vec3 farW = winCol(vStyle.x, 0.5) * vStyle.y * 0.8 * 0.26;
          vec3 win = mix(farW, nearW, detail) * 0.26;
          win *= (1.0 - step(halfW - 0.45 * vStyle.z, abs(uv.x))) * (1.0 - step(h - 1.2, uv.y)) * isSide;
          col += win;
          // neon trims
          float tt = floor(vTrim.a + 0.001), tw = fract(vTrim.a + 0.001) * 2.0;
          float m = 0.0;
          m += step(0.5, tt) * step(tt, 1.5) * bandAA(uv.y - (h - tw), tw) * isSide;
          m += step(1.5, tt) * step(tt, 2.5) * bandAA(abs(uv.x) - (halfW - tw), tw) * isSide;
          m += step(2.5, tt) * bandAA(fract(uv.y / 14.0 + 0.5) - 0.5, tw / 14.0) * isSide;
          col += vTrim.rgb * m * 1.15;
          // roofs: dark, faint reflection of the sky
          col = mix(col, vec3(0.012, 0.010, 0.02), 1.0 - isSide);
          gl_FragColor = vec4(fogMix(col, vW), 1.0);
          ${OUT}
        }`,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, n);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3();
    B.forEach((b, i) => {
      const hgt = Math.max(0.1, b.top - b.bottom);
      q.setFromAxisAngle(Y, b.rot); p.set(b.x, b.bottom, b.z); s.set(b.w, hgt, b.d);
      mesh.setMatrixAt(i, m4.compose(p, q, s));
      style.set([b.hue, b.lit, b.cell, R() * 10], i * 4);
      const t = b.trim;
      trim.set(t ? [t.r, t.g, t.b, b.ttype + Math.min(0.49, b.tw / 2)] : [0, 0, 0, 0], i * 4);
    });
    geo.setAttribute("aStyle", new THREE.InstancedBufferAttribute(style, 4));
    geo.setAttribute("aTrim", new THREE.InstancedBufferAttribute(trim, 4));
    mesh.computeBoundingSphere();
    add(mesh, 0);
  }

  // ------------------------------------------------------------------ foreground rooftop
  // concrete: MeshStandardMaterial with a tiny procedural grime/wetness tweak, lit by 3 neon point lights
  // and reflecting a small PMREM environment made of neon panels (that's the wet sheen).
  let envRT = null;
  {
    const pm = new THREE.PMREMGenerator(renderer);
    const es = new THREE.Scene(); es.background = lin(0x05030a);
    const panel = (col, k, pos, size) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshBasicMaterial({ color: col.clone().multiplyScalar(k) }));
      m.position.set(...pos); es.add(m);
    };
    panel(NEON.pink, 6, [-4, 1, 1], [0.6, 4, 1.5]);
    panel(NEON.blue, 3, [4, 3, -4], [3, 2, 0.5]);
    panel(NEON.cyan, 2, [0, 6, -9], [10, 2, 0.5]);
    panel(NEON.green, 3, [3, 0.6, 1], [0.6, 0.4, 1]);
    panel(lin(0x2a1240), 1.5, [0, 12, 0], [40, 0.5, 40]);
    panel(lin(0x0b2a44), 2, [0, 2, -14], [40, 6, 0.5]);
    envRT = pm.fromScene(es, 0.03);
    es.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
    pm.dispose();
  }
  // wet tops (concrete: the parapet coping, the hut roof): sharp-edged puddles that mirror the nearest neon as vertical
  // streaks rippled by rain rings (analytic: a light's reflected direction vs the view's reflected ray)
  const WET_L = [
    [new THREE.Vector3(1.38, 0.99, -0.215), 0.05, lin(PALETTE.green, 1.5)],       // the GOAT 24H sign standing on the coping
    [new THREE.Vector3(25, 15.6, -66), 0.05, lin(0x9a6cff, 0.9)],                  // the big billboard
    [new THREE.Vector3(2.8, 3.0, -1.2), 0.12, lin(0x2f7dff, 0.7)],                 // the blue glow beyond the corner
    [new THREE.Vector3(-7, 4.5, -9), 0.08, lin(PALETTE.pink, 0.8)],                // the hologram ram on the left
  ];
  const fgPatch = (mat, wetTop) => {
    mat.customProgramCacheKey = () => 'city-fg-' + (wetTop ? 'wet' : 'dry'); // (same onBeforeCompile source: without a key both shared one program)
    mat.onBeforeCompile = (sh) => {
      if (wetTop) {
        sh.uniforms.uTime = U.uTime;
        sh.uniforms.uWL = { value: WET_L.map((l) => new THREE.Vector4(l[0].x, l[0].y, l[0].z, l[1])) };
        sh.uniforms.uWC = { value: WET_L.map((l) => l[2]) };
      }
      sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 vWP; varying vec3 vWN;")
        .replace("#include <worldpos_vertex>", "#include <worldpos_vertex>\nvWP = (modelMatrix * vec4(transformed, 1.0)).xyz; vWN = normalize(mat3(modelMatrix) * objectNormal);");
      sh.fragmentShader = sh.fragmentShader.replace("#include <common>", `#include <common>
        varying vec3 vWP; varying vec3 vWN;
        float fh(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
        float fn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f); return mix(mix(fh(i), fh(i+vec2(1,0)), f.x), mix(fh(i+vec2(0,1)), fh(i+vec2(1,1)), f.x), f.y); }`)
        .replace("#include <color_fragment>", `#include <color_fragment>
          vec2 gp = abs(vWN.y) > 0.5 ? vWP.xz : (abs(vWN.x) > 0.5 ? vWP.zy : vWP.xy);
          float grime = fn(gp * 3.0) * 0.6 + fn(gp * 11.0) * 0.4;
          float streak = fn(vec2(gp.x * 9.0, gp.y * 0.6));
          float vert = step(abs(vWN.y), 0.5);
          float seam = 1.0 - 0.6 * vert * (1.0 - smoothstep(0.004, 0.012, abs(fract(gp.x / 1.6) - 0.5) * 1.6));
          diffuseColor.rgb *= (0.65 + 0.5 * grime - 0.25 * streak * vert) * seam;`)
        .replace("#include <roughnessmap_fragment>", `#include <roughnessmap_fragment>
          float puddle = smoothstep(0.45, 0.62, fn(gp * 1.7 + 4.0));
          roughnessFactor = mix(roughnessFactor, ${wetTop ? "0.06" : "0.2"}, step(0.5, vWN.y) * puddle);
          roughnessFactor *= 0.85 + 0.3 * fn(gp * 7.0);
          roughnessFactor = mix(roughnessFactor, roughnessFactor * 0.45, vert * smoothstep(0.55, 0.8, streak)); // rain running down`);
      if (wetTop) {
        sh.fragmentShader = sh.fragmentShader.replace("#include <common>", `#include <common>
          uniform float uTime; uniform vec4 uWL[${WET_L.length}]; uniform vec3 uWC[${WET_L.length}];`)
          .replace("#include <opaque_fragment>", `#include <opaque_fragment>
          if (vWN.y > 0.5) {
            vec2 wp = vWP.xz;
            float pm = smoothstep(0.5, 0.53, fn(wp * 1.7 + 4.0));      // hard-edged puddles on a rougher, matte top
            // rain rings
            vec2 cell = floor(wp * 4.0), f = fract(wp * 4.0) - 0.5;
            float hh = fh(cell), tt = fract(uTime * 0.85 + hh);
            vec2 d = f - (vec2(fh(cell + 7.1), fh(cell + 3.3)) - 0.5) * 0.5; float L = length(d);
            float ring = sin((L - tt * 0.45) * 70.0) * (1.0 - tt) * (1.0 - smoothstep(0.0, 0.05, abs(L - tt * 0.45)));
            vec3 Nw = normalize(vec3(d.x / max(L, 1e-3) * ring * 0.12, 1.0, d.y / max(L, 1e-3) * ring * 0.12));
            vec3 Vw = normalize(vWP - cameraPosition), Rw = reflect(Vw, Nw);
            vec2 ra = vec2(atan(Rw.x, -Rw.z), Rw.y);
            vec3 refl = vec3(0.0);
            for (int i = 0; i < ${WET_L.length}; i++) {
              vec3 dl = normalize(uWL[i].xyz - vWP);
              vec2 la = vec2(atan(dl.x, -dl.z), dl.y);
              float dx = (ra.x - la.x) / uWL[i].w, dy = (ra.y - la.y) / (uWL[i].w * 7.0);
              refl += uWC[i] * exp(-dx * dx - dy * dy);
            }
            float fres = 0.08 + 0.92 * pow(1.0 - max(-Vw.y, 0.0), 5.0);
            gl_FragColor.rgb = mix(gl_FragColor.rgb * 1.08, gl_FragColor.rgb * 0.55 + refl * fres * 1.3, pm);
          }`);
      }
    };
  };
  const concrete = new THREE.MeshStandardMaterial({ color: 0x2c2933, roughness: 0.42, metalness: 0.0, envMap: envRT.texture, envMapIntensity: 1.0 });
  fgPatch(concrete, true);
  const metal = new THREE.MeshStandardMaterial({ color: 0x16151c, roughness: 0.35, metalness: 0.6, envMap: envRT.texture, envMapIntensity: 1.2 });
  fgPatch(metal, false);
  const neonMat = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: true, fog: false });
  trash.push(concrete, metal, neonMat, envRT);

  const box = (x0, x1, y0, y1, z0, z1) => { const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0); g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2); return g; };
  const cyl = (r, h, x, y, z, seg = 12) => { const g = new THREE.CylinderGeometry(r, r, h, seg); g.translate(x, y + h / 2, z); return g; };
  const tube = (pts, r, seg = 40) => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), seg, r, 5, false);
  const sag = (a, b, s, n = 8) => { const out = []; for (let i = 0; i <= n; i++) { const t = i / n; const p = a.clone().lerp(b, t); p.y -= Math.sin(Math.PI * t) * s; out.push(p); } return out; };
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  const BLADE_DX = 0.3; // the blade sign sits this far in from x -2 so more of its glyphs are in frame
  const DECK = -0.25, BX0 = -2.6, BX1 = 1.9; // the right end is the building's corner: past it you look down into the city
  const conc = [
    box(BX0, BX1, -40, DECK, -7.5, -0.04),            // the building (facade face at z ~ 0)
    box(BX0, BX1, DECK, 0.7, -0.46, -0.04),           // front parapet wall
    box(BX0 - 0.05, BX1 + 0.05, 0.7, SEAT.y, -0.55, SEAT.z), // coping: top surface = seat.y, front edge = seat.z
    box(BX0, BX0 + 0.42, DECK, 0.8, -7.5, -0.4),      // side parapets
    box(BX1 - 0.42, BX1, DECK, 0.8, -7.5, -0.4),
    box(BX0 - 0.1, BX1 + 0.1, -0.62, -0.42, -0.04, 0.16), // cornice below
    box(BX0, BX1, -0.42, -0.36, -0.04, 0.1),
  ];
  // stains / chipped blocks on the coping so the top edge isn't a perfect line
  for (let i = 0; i < 6; i++) { const x = rr(BX0 + 0.3, BX1 - 0.5), w = rr(0.2, 0.6); conc.push(box(x, x + w, 0.79, 0.805, -0.5, -0.3)); }
  const mets = [
    // left: a boxy AC unit beside the stairwell hut (the hut, a perch, is built in the depth section below) and a vent
    // stack with an elbow coming out of the hut's roof
    box(-2.1, -1.25, DECK, 1.0, -5.0, -4.0),
    cyl(0.18, 0.9, -1.85, 1.45, -3.3), tube([V(-1.85, 2.2, -3.3), V(-1.85, 2.45, -3.2), V(-1.7, 2.55, -2.95)], 0.18, 8),
    cyl(0.32, 0.12, -1.67, 1.0, -4.5, 16),
    // right: antenna mast with cross bars, water tank on legs
    cyl(0.05, 5.6, 1.6, DECK, -6.5, 6), box(1.25, 1.95, 3.2, 3.24, -6.52, -6.48), box(1.35, 1.85, 4.0, 4.04, -6.52, -6.48), box(1.45, 1.75, 4.6, 4.63, -6.52, -6.48),
    cyl(0.12, 1.6, 0.4, DECK, -7.0, 8),
    // the blade sign's housing (left end of the parapet) + bracket
    box(-2.03 + BLADE_DX, -1.87 + BLADE_DX, -1.6, 2.15, 0.02, 0.76), box(-2.15 + BLADE_DX, -1.75 + BLADE_DX, 0.7, 0.82, -0.3, 0.1),
    // small lightbox on the parapet (right)
    box(1.1, 1.66, SEAT.y + 0.06, 1.12, -0.36, -0.22), box(1.2, 1.24, SEAT.y, SEAT.y + 0.06, -0.32, -0.26), box(1.52, 1.56, SEAT.y, SEAT.y + 0.06, -0.32, -0.26),
    // cables: from the mast off to the right, a long sag over the back, conduits along the parapet
    tube(sag(V(1.6, 4.9, -6.5), V(14, 9, -14), 1.6, 10), 0.018, 30),
    tube(sag(V(1.6, 4.5, -6.5), V(-10, 6.5, -18), 2.2, 12), 0.016, 40),
    tube(sag(V(-2.0, 1.9, -3.2), V(-9, 5.6, -10), 0.45, 8), 0.02, 20), // (passes above the NEKO DATA sign, not across it)
    tube([V(BX0, 0.5, 0.01), V(-1.8, 0.48, 0.03), V(-0.6, 0.52, 0.03), V(0.8, 0.47, 0.03), V(BX1, 0.5, 0.01)], 0.022, 30),
    tube([V(1.1, 0.82, -0.3), V(1.2, 0.6, 0.03), V(1.3, 0.1, 0.05), V(1.25, -0.35, 0.1), V(1.3, -1.5, 0.18), V(1.2, -3.5, 0.05)], 0.02, 30),
  ];

  // ------------------------------------------------------------------ depth: lower terrace, perches, foreground
  // In front of her parapet (+Z, the camera side) there is a lower roof terrace (floor at FLOOR, its old cornice is now
  // a curb at the wall's foot). On it, and on the roof behind her, sit the props she can teleport onto (city.perches);
  // close to the camera sit dark, rim-lit foreground props that frame the shot from the corners.
  // A perch box is built in its ledge frame: local +Z = the ledge's outward normal (yaw `ledgeYaw`), the top at
  // seat.y, the front face through seat. Her body yaw is ledgeYaw + ~0.6 (the README's cameraYaw + 0.6 rule), so her
  // right hand reaches out past the front-left corner and her left hand rests on the top, like on the parapet.
  const depthNeon = []; // [geometry, colour, k] added to the neon tube mesh below
  const sil = [];       // geometries for the foreground silhouette material
  const ledgeBox = (seat, yaw, x0, x1, y0, y1, z0, z1) => { const g = box(x0, x1, y0, y1, z0, z1); g.rotateY(yaw); g.translate(seat.x, 0, seat.z); return g; };
  const ledgeV = (seat, yaw, x, y, z) => V(x, y, z).applyAxisAngle(Y_AXIS, yaw).add(V(seat.x, 0, seat.z));

  // terrace walls: the facade between the floor and the old cornice; the curb's pink tube moves onto its top edge
  conc.push(box(BX0 - 0.1, BX1 + 0.1, FLOOR - 0.3, -0.62, -0.04, 0.16));

  // --- perch 2 (back, left): the stairwell hut on the roof behind her; she sits on its roof slab over the door
  {
    const s = P_HUT.seat; // ledge yaw 0 (axis aligned with the building)
    conc.push(
      box(s.x - 1.0, s.x + 1.0, DECK, s.y - 0.08, s.z - 1.2, s.z - 0.1),      // walls
      box(s.x - 1.06, s.x + 1.06, s.y - 0.08, s.y, s.z - 1.3, s.z),          // roof slab: top = seat.y, front edge = seat.z
    );
    mets.push(
      box(s.x + 0.15, s.x + 0.8, DECK, s.y - 0.5, s.z - 0.12, s.z - 0.07),     // door
      box(s.x + 0.1, s.x + 0.85, s.y - 0.5, s.y - 0.46, s.z - 0.13, s.z - 0.06),
      box(s.x - 0.75, s.x - 0.2, s.y - 0.75, s.y - 0.45, s.z - 0.12, s.z - 0.08), // sign plate
    );
    depthNeon.push([box(s.x + 0.2, s.x + 0.75, s.y - 0.11, s.y - 0.09, s.z - 0.09, s.z - 0.06), NEON.green, 1.3]); // lamp under the slab
    depthNeon.push([box(s.x - 0.72, s.x - 0.23, s.y - 0.72, s.y - 0.48, s.z - 0.075, s.z - 0.07), NEON.cyan, 0.16]);
    depthNeon.push([box(s.x - 0.72, s.x - 0.23, s.y - 0.665, s.y - 0.655, s.z - 0.068, s.z - 0.066), NEON.white, 0.6]);
  }

  // --- perch 3 (back, centre-right): a sectional steel water tank on a stand behind her
  {
    const s = P_TANK.seat, x0 = s.x - 0.8, x1 = s.x + 0.8, z1 = s.z - 0.02, z0 = z1 - 1.1, yb = s.y - 0.84;
    for (const lx of [x0 + 0.06, x1 - 0.06]) for (const lz of [z0 + 0.06, z1 - 0.06]) mets.push(box(lx - 0.04, lx + 0.04, DECK, yb, lz - 0.04, lz + 0.04));
    mets.push(
      box(x0, x1, yb - 0.06, yb, z0, z1),                                       // platform
      box(x0 + 0.02, x1 - 0.02, yb, s.y - 0.04, z0 + 0.02, z1 - 0.02),          // tank
      box(x0 - 0.01, x1 + 0.01, s.y - 0.04, s.y, z0 - 0.01, s.z),               // lid rim: top = seat.y, front edge = seat.z
      box(x0 + 0.1, x1 - 0.1, DECK + 0.25, DECK + 0.29, z1 - 0.07, z1 - 0.03),  // cross brace
      cyl(0.06, yb - DECK, x0 + 0.35, DECK, z0 + 0.5, 8),                       // outlet pipe
    );
    for (const y of [yb + 0.25, yb + 0.5]) mets.push(box(x0, x1, y, y + 0.025, z0, z1 + 0.012)); // panel seams
    for (const x of [x0 + 0.53, x0 + 1.07]) mets.push(box(x - 0.012, x + 0.012, yb, s.y - 0.04, z1 - 0.02, z1 + 0.012));
    // ladder up the right side
    mets.push(box(x1 + 0.02, x1 + 0.05, DECK, s.y + 0.3, z1 - 0.45, z1 - 0.42), box(x1 + 0.02, x1 + 0.05, DECK, s.y + 0.3, z1 - 0.15, z1 - 0.12));
    for (let y = DECK + 0.25; y < s.y + 0.2; y += 0.28) mets.push(box(x1 + 0.025, x1 + 0.045, y, y + 0.02, z1 - 0.43, z1 - 0.13));
    depthNeon.push([box(x0 + 0.08, x0 + 0.11, yb + 0.08, s.y - 0.12, z1 + 0.013, z1 + 0.02), NEON.cyan, 1.1]); // level gauge
    depthNeon.push([box(x0 + 0.12, x1 - 0.12, s.y - 0.075, s.y - 0.06, z1 - 0.02, z1 - 0.008), NEON.pink, 1.0]); // strip under the lid, behind her legs
    depthNeon.push([box(x0 - 0.01, x1 + 0.01, s.y - 0.006, s.y, s.z, s.z + 0.006), NEON.cyan, 0.45]);              // lit lid edge, so the tank reads
    for (const lx of [x0 + 0.06, x1 - 0.06]) depthNeon.push([box(lx + 0.04, lx + 0.046, DECK, yb, z1 - 0.02, z1 - 0.014), NEON.blue, 0.35]); // rim on the front legs
  }

  // --- foreground (near the camera, dark + rim-lit): a crate stack bottom-right, a chain-link fence panel and a
  // utility pole bottom-left / left edge, cables off the pole. They frame the default shot from the corners.
  {
    // a crate: k = surface pattern for the silhouette shader (1 horizontal planks, 2 vertical slats, 3 plain)
    const crate = (cx, cz, yaw, w, d, y0, h, k = 1) => {
      const c = V(cx, 0, cz), e = 0.03;
      const out = [ledgeBox(c, yaw, -w / 2, w / 2, y0, y0 + h, -d / 2, d / 2)];
      // edge frames (slightly proud) so the silhouette reads as a crate, not a box
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) out.push(ledgeBox(c, yaw, sx * w / 2 - e, sx * w / 2 + e, y0, y0 + h, sz * d / 2 - e, sz * d / 2 + e));
      for (const yy of [y0, y0 + h - 0.045]) { out.push(ledgeBox(c, yaw, -w / 2 - e, w / 2 + e, yy, yy + 0.045, d / 2 - e, d / 2 + e)); out.push(ledgeBox(c, yaw, -w / 2 - e, -w / 2 + e, yy, yy + 0.045, -d / 2, d / 2)); }
      out.forEach((g) => { g.userData.k = k; }); sil.push(...out);
    };
    // a box rotated about its own z (in the ledge frame) then yawed: chevrons / strokes of a stencil
    const ledgeBoxR = (c, yaw, cx, cy, cz, w, h, d, rz) => { const g = new THREE.BoxGeometry(w, h, d); g.rotateZ(rz); g.translate(cx, cy, cz); g.rotateY(yaw); g.translate(c.x, 0, c.z); return g; };
    // perch 1 (fore): a crate stack on the terrace in front of the parapet, left of her legs; the top crate's front
    // edge is the seat. In the master shot she sits here ~1.3x bigger than on the parapet. Broken up so it reads as
    // stacked cargo, not one cabinet: a slatted pallet, two crates of different sizes / offsets / yaws (gaps between
    // them), a small box, a strap, a stencil and a hazard tag on the camera-facing faces
    const ps = P_FORE.seat, py = P_FORE.ledgeYaw, b = ledgeV(ps, py, -0.11, 0, -0.36); // (top crate: local x -0.64..0.42, clear of the parapet perch's shins in the master shot)
    {
      const pc = V(b.x + 0.03, 0, b.z - 0.02), pyw = py + 0.05;
      for (const x of [-0.52, 0, 0.5]) sil.push(ledgeBox(pc, pyw, x - 0.06, x + 0.06, FLOOR, FLOOR + 0.1, -0.42, 0.42));         // runners
      for (const z of [-0.36, -0.12, 0.12, 0.36]) sil.push(ledgeBox(pc, pyw, -0.62, 0.6, FLOOR + 0.1, FLOOR + 0.14, z - 0.065, z + 0.065)); // deck boards (gaps)
    }
    crate(b.x - 0.02, b.z - 0.02, py + 0.07, 1.16, 0.8, FLOOR + 0.14, 0.52, 1);
    crate(b.x, b.z, py, 1.06, 0.72, FLOOR + 0.68, ps.y - FLOOR - 0.68, 2);
    {
      // small box on top, back left (clear of her hands / hips)
      const sb = ledgeV(V(b.x, 0, b.z), py, -0.36, 0, -0.18);
      const g = ledgeBox(V(sb.x, 0, sb.z), py - 0.32, -0.17, 0.17, ps.y, ps.y + 0.26, -0.14, 0.14); g.userData.k = 1; sil.push(g);
      // a strap over the top crate (left end) with a buckle
      sil.push(ledgeBox(V(b.x, 0, b.z), py, -0.5, -0.45, ps.y, ps.y + 0.012, -0.37, 0.37));
      sil.push(ledgeBox(V(b.x, 0, b.z), py, -0.5, -0.45, FLOOR + 0.68, ps.y + 0.012, 0.355, 0.37));
      sil.push(ledgeBox(V(b.x, 0, b.z), py, -0.52, -0.43, ps.y - 0.26, ps.y - 0.19, 0.365, 0.385));
    }
    // her legs hang in front of the top crate: a green LED strip right under its front lip + a soft wash on the face
    // (below, with the other washes), so her black shins read against it
    depthNeon.push([ledgeBox(V(b.x, 0, b.z), py, -0.46, 0.46, ps.y - 0.07, ps.y - 0.055, 0.362, 0.37), NEON.green, 1.3]);
    // a pink hazard tag and an original stencil (three chevrons + a bar) on the bottom crate's camera-facing face
    {
      const bc = V(b.x + 0.05, 0, b.z - 0.02), byaw = py + 0.07, fz = 0.405;
      depthNeon.push([ledgeBox(bc, byaw, 0.3, 0.56, FLOOR + 0.52, FLOOR + 0.55, fz, fz + 0.008), NEON.pink, 0.9]);
      for (let i = 0; i < 3; i++) {
        const x = -0.5 + i * 0.12;
        depthNeon.push([ledgeBoxR(bc, byaw, x, FLOOR + 0.45, fz + 0.004, 0.1, 0.02, 0.006, 0.8), NEON.cyan, 0.3]);
        depthNeon.push([ledgeBoxR(bc, byaw, x, FLOOR + 0.38, fz + 0.004, 0.1, 0.02, 0.006, -0.8), NEON.cyan, 0.3]);
      }
      depthNeon.push([ledgeBox(bc, byaw, -0.22, 0.08, FLOOR + 0.26, FLOOR + 0.28, fz, fz + 0.006), NEON.cyan, 0.24]);
    }
    // the right-hand foreground (the default shot's bottom-right corner, close to the lens): a lower, staggered pair of
    // crates with a gap, a cable reel and a lit corner tag, so its silhouette breaks up instead of reading as a cabinet
    {
      const c = RIGHT_FG.c, y = RIGHT_FG.yaw;
      crate(c.x, c.z, y, 1.3, 0.9, FLOOR, 0.76, 1);
      const c2 = ledgeV(V(c.x, 0, c.z), y, 0.28, 0, -0.12);
      crate(c2.x, c2.z, y - 0.16, 0.86, 0.7, FLOOR + 0.76, 0.6, 2);
      { const c4 = ledgeV(V(c2.x, 0, c2.z), y - 0.16, 0.12, 0, -0.08); const g = ledgeBox(V(c4.x, 0, c4.z), y + 0.3, -0.2, 0.2, FLOOR + 1.36, FLOOR + 1.62, -0.16, 0.16); g.userData.k = 1; sil.push(g); }
      const c3 = ledgeV(V(c.x, 0, c.z), y, -0.98, 0, 0.12);
      crate(c3.x, c3.z, y + 0.22, 0.52, 0.5, FLOOR, 0.4, 3);
      const rc = ledgeV(V(c.x, 0, c.z), y, -0.55, 0, 0.75);
      for (const [r, x0, x1] of [[0.3, -0.18, -0.14], [0.3, 0.14, 0.18], [0.18, -0.14, 0.14]]) {
        const cg = new THREE.CylinderGeometry(r, r, x1 - x0, 14); cg.rotateZ(Math.PI / 2); cg.translate((x0 + x1) / 2, FLOOR + 0.3, 0); cg.rotateY(y + 0.6); cg.translate(rc.x, 0, rc.z);
        sil.push(cg);
      }
      const t0 = ledgeV(V(rc.x, 0, rc.z), y + 0.6, 0.0, 0, 0.18);
      sil.push(tube([V(t0.x, FLOOR + 0.36, t0.z), V(t0.x - 0.2, FLOOR + 0.1, t0.z + 0.2), V(t0.x - 0.05, FLOOR + 0.02, t0.z + 0.5)], 0.016, 10));
      // corner tag + a thin LED edge on the upper crate (they catch the eye, so the shapes read as boxes)
      depthNeon.push([ledgeBox(V(c2.x, 0, c2.z), y - 0.16, -0.43, -0.4, FLOOR + 0.8, FLOOR + 1.3, 0.35, 0.36), NEON.cyan, 0.9]);
      depthNeon.push([ledgeBox(V(c.x, 0, c.z), y, -0.6, -0.2, FLOOR + 0.6, FLOOR + 0.63, 0.45, 0.458), NEON.pink, 0.8]);
    }
    // a utility cabinet right in front of the fore perch, close to the lens (bottom-left corner of the master shot,
    // mostly behind the prompt bar's left end): her shins hang behind it there, so she never runs under the UI, and it is
    // the nearest layer in that corner. A thin lit edge on its lid + a status LED so the near-black block reads as a box.
    {
      const c = V(NEAR_CAB.x, 0, NEAR_CAB.z), y = NEAR_CAB.yaw, top = NEAR_CAB.top, cab = [];
      cab.push(ledgeBox(c, y, -0.22, 0.22, FLOOR + 0.12, top, -0.22, 0.22));                 // body
      cab.push(ledgeBox(c, y, -0.25, 0.25, top, top + 0.04, -0.25, 0.25));                  // lid (overhang)
      for (const sx of [-0.2, 0.2]) cab.push(ledgeBox(c, y, sx - 0.03, sx + 0.03, FLOOR, FLOOR + 0.12, -0.2, 0.2)); // feet
      cab.push(ledgeBox(c, y, -0.012, 0.012, FLOOR + 0.2, top - 0.06, 0.22, 0.23));          // door seam
      cab.push(cyl(0.03, top - FLOOR - 0.1, c.x - 0.18, FLOOR, c.z - 0.3, 8));                // conduit behind
      cab.forEach((g) => { g.userData.k = 3; }); sil.push(...cab);
      depthNeon.push([ledgeBox(c, y, -0.25, 0.25, top + 0.04, top + 0.048, 0.252, 0.262), NEON.cyan, 0.7]);
      depthNeon.push([ledgeBox(c, y, 0.1, 0.16, top - 0.14, top - 0.12, 0.22, 0.226), NEON.green, 1.6]);
      depthNeon.push([ledgeBox(c, y, -0.18, 0.05, top - 0.3, top - 0.28, 0.22, 0.226), NEON.pink, 0.5]);
    }
    // utility pole at the left edge + a junction box, cables
    const PX = POLE.x, PZ = POLE.z;
    sil.push(cyl(0.065, 3.6, PX, FLOOR, PZ, 10));
    sil.push(box(PX - 0.02, PX + 0.16, 0.75, 1.05, PZ - 0.09, PZ + 0.07));
    sil.push(box(PX - 0.4, PX + 0.1, 2.55, 2.62, PZ - 0.04, PZ + 0.04));
    depthNeon.push([box(PX + 0.16, PX + 0.17, 0.8, 1.0, PZ - 0.07, PZ + 0.05), NEON.green, 1.2]); // status edge on the box
    sil.push(tube(sag(V(PX - 0.35, 2.58, PZ), V(-4.5, 2.2, 1.0), 0.6, 10), 0.016, 24));
    // two more cables off the pole top, sagging across the top-left of the master shot toward the stairwell hut and
    // the parapet's far end: thin near-lens lines that sit in front of everything (k 4: dim rim)
    {
      const cab = [tube(sag(V(PX + 0.05, 2.62, PZ), V(-1.6, 2.35, -2.2), 0.55, 14), 0.011, 30), tube(sag(V(PX + 0.05, 2.5, PZ), V(-2.6, 1.15, 0.0), 0.3, 12), 0.01, 24)];
      cab.forEach((g) => { g.userData.k = 4; }); sil.push(...cab);
    }
    // fence posts + rails (the mesh itself is a shader panel)
    const fx = (t) => FENCE.x0 + (FENCE.x1 - FENCE.x0) * t, fz = (t) => FENCE.z0 + (FENCE.z1 - FENCE.z0) * t;
    sil.push(cyl(0.03, FENCE.h, fx(0), FLOOR, fz(0), 8));
    sil.push(tube([V(fx(0), FLOOR + FENCE.h, fz(0)), V(fx(1), FLOOR + FENCE.h, fz(1))], 0.022, 4));
    sil.push(tube([V(fx(0), FLOOR + 0.1, fz(0)), V(fx(1), FLOOR + 0.1, fz(1))], 0.015, 4));
  }
  const fgConc = add(new THREE.Mesh(mergeGeometries(conc.map((g) => g.toNonIndexed())), concrete), 1);
  const fgMet = add(new THREE.Mesh(mergeGeometries(mets.map((g) => (g.index ? g.toNonIndexed() : g))), metal), 1);
  conc.concat(mets).forEach((g) => g.dispose());
  void fgConc; void fgMet;

  // foreground silhouettes: near-black, slightly lifted toward the haze (they read as out of focus) with a neon rim
  // (violet-blue on the left, cyan-green on the right) so they pop as the closest layer
  {
    const m = shader({
      vs: `attribute float aK; varying float vK; varying vec3 vW; varying vec3 vN; void main(){ vK = aK; vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `varying vec3 vW; varying vec3 vN; varying float vK;
        void main(){
          vec3 N = normalize(vN), Vd = normalize(cameraPosition - vW);
          // crates: plank grooves on the sides that catch a thread of rim light
          // (k: 1 horizontal planks, 2 vertical slats, 3 plain; the crate faces keep only a thin rim, edges catch it)
          float gy = abs(fract(vW.y / 0.155 + 0.5) - 0.5) * 0.155;
          float sx = vW.x * 0.96 + vW.z * 0.276, gx = abs(fract(sx / 0.19 + 0.5) - 0.5) * 0.19;
          float isC = step(0.5, vK);
          float groove = (vK < 1.5 ? 1.0 - smoothstep(0.007, 0.014, gy) : vK < 2.5 ? 1.0 - smoothstep(0.007, 0.014, gx) : 0.0) * isC * step(abs(N.y), 0.5);
          // per-plank value variation (k 1 / 2): the boards read as boards from the master shot, not one flat face
          float pid = vK < 1.5 ? floor(vW.y / 0.155 + 0.5) : floor(sx / 0.19 + 0.5);
          float plank = vK < 2.5 ? 0.7 + 0.6 * fract(sin(pid * 12.9898 + vK * 4.1) * 43758.5453) : 1.0;
          float rimK = mix(1.0, 0.55, isC);
          float fr = pow(1.0 - abs(dot(N, Vd)), 4.0);
          vec3 rc = mix(vec3(0.22, 0.25, 1.0), vec3(0.1, 0.95, 0.7), smoothstep(-0.4, 1.4, vW.x));
          float grime = fbm3(vec2(vW.x + vW.z, vW.y) * 6.0);
          vec3 c = vec3(0.014, 0.011, 0.02) * (0.7 + 0.6 * grime) * mix(1.0, plank, isC);
          // a little key from the city (violet-pink from the right / front, cold blue from the left) on the crates, so
          // their faces separate and the planks catch it
          c += isC * plank * (vec3(0.5, 0.2, 0.55) * 0.07 * max(dot(N, normalize(vec3(0.55, 0.45, 0.7))), 0.0)
                            + vec3(0.15, 0.3, 1.0) * 0.045 * max(dot(N, normalize(vec3(-0.8, 0.3, 0.45))), 0.0));
          c += rc * fr * 0.22 * rimK;
          c += vec3(0.05, 0.03, 0.1) * 0.25 * max(N.y, 0.0);                  // tops catch the sky glow
          c += rc * 0.05 * rimK * smoothstep(0.6, 1.0, abs(N.x)) * (0.5 + 0.5 * grime); // the side facing the neon
          c = mix(c, rc * 0.09 + c, groove);
          c = mix(c, vec3(0.03, 0.02, 0.05), 0.15);                          // a touch of haze: reads as out of focus
          gl_FragColor = vec4(c, 1.0);
          ${OUT}
        }`,
    });
    const g = mergeGeometries(sil.map((x) => { const q = x.index ? x.toNonIndexed() : x; if (q.attributes.uv) q.deleteAttribute("uv"); q.setAttribute("aK", new THREE.BufferAttribute(new Float32Array(q.attributes.position.count).fill(x.userData.k || 0), 1)); return q; }));
    sil.forEach((x) => x.dispose());
    add(new THREE.Mesh(g, m), 1);
  }

  // chain-link fence panel (procedural diamond mesh, alpha blended)
  {
    const L = Math.hypot(FENCE.x1 - FENCE.x0, FENCE.z1 - FENCE.z0);
    const g = new THREE.PlaneGeometry(L, FENCE.h);
    g.rotateY(-Math.atan2(FENCE.z1 - FENCE.z0, FENCE.x1 - FENCE.x0));
    g.translate((FENCE.x0 + FENCE.x1) / 2, FLOOR + FENCE.h / 2, (FENCE.z0 + FENCE.z1) / 2);
    const mat = shader({
      side: THREE.DoubleSide, uniforms: { uSize: { value: new THREE.Vector2(L, FENCE.h) } },
      vs: `varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `uniform vec2 uSize; varying vec2 vUv; varying vec3 vW;
        void main(){
          vec2 q = vUv * uSize / 0.075;
          vec2 r = vec2(q.x + q.y, q.x - q.y) * 0.7071;
          vec2 f = abs(fract(r) - 0.5);
          float d = 0.5 - max(f.x, f.y);           // distance to the nearest wire
          float px = max(fwidth(d), 1e-4);
          float wire = clamp((0.055 - d) / px + 0.5, 0.0, 1.0);
          // a torn corner and a sagging top: holes in the mesh
          float hole = step(0.82, vUv.x) * step(vUv.y, 0.35 + 0.4 * (vUv.x - 0.82));
          wire *= 1.0 - hole;
          // dark wire, glinting where it catches the neon behind (blue on the left, green toward the pole)
          vec3 glint = mix(vec3(0.15, 0.2, 0.9), vec3(0.1, 0.8, 0.5), vUv.x) * (0.12 + 0.25 * pow(1.0 - abs(vUv.y - 0.6), 4.0));
          vec3 c = vec3(0.02, 0.018, 0.03) + glint;
          gl_FragColor = vec4(c, wire * 0.92);
          ${OUT}
        }`,
    });
    mat.transparent = true; mat.depthWrite = false;
    add(new THREE.Mesh(g, mat), 6);
  }

  // the wet terrace floor: dark concrete with puddles that mirror the sky and stretch the nearby neon into streaks,
  // rain rings rippling them (no real reflections: a few analytic streaks + the sky colour in the reflected direction)
  {
    const g = new THREE.PlaneGeometry(1, 1); g.rotateX(-Math.PI / 2);
    g.scale(15, 1, 10.5); g.translate(-0.5, FLOOR, 5.2);
    // streak sources: xyz = the light, w = width (rad); colours below
    const Ls = [
      [V(-1.815 + BLADE_DX, 0.3, 0.39), 0.035, lin(0x2f7dff, 1.3)],     // blade sign
      [V(-0.6, 0.39, 1.17), 0.012, lin(PALETTE.green, 1.4)],               // fore crate LED strip
      [V(-0.13, 0.9, 2.95), 0.01, lin(PALETTE.green, 1.0)],        // junction box
      [V(P_HUT.seat.x + 0.47, 1.35, -2.5), 0.02, lin(PALETTE.green, 0.6)],       // stairwell hut lamp
      [V(2.8, 3.0, -1.2), 0.08, lin(0x2f7dff, 0.25)],                     // the blue glow beyond the corner
    ];
    const uL = Ls.map((l) => new THREE.Vector4(l[0].x, l[0].y, l[0].z, l[1])), uLC = Ls.map((l) => l[2]);
    const mat = shader({
      uniforms: { uL: { value: uL }, uLC: { value: uLC }, uFloor: { value: FLOOR }, uPinkC: { value: NEON.pink } },
      vs: `varying vec3 vW; void main(){ vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `uniform vec4 uL[5]; uniform vec3 uLC[5]; uniform float uFloor; uniform vec3 uPinkC; varying vec3 vW;
        vec2 ripple(vec2 p, float sc, float sp){
          vec2 cell = floor(p * sc), f = fract(p * sc) - 0.5;
          float h = hash12(cell), t = fract(uTime * sp + h);
          vec2 o = (vec2(hash12(cell + 7.1), hash12(cell + 3.3)) - 0.5) * 0.5;
          vec2 d = f - o; float L = length(d), r = t * 0.5;
          float w = sin((L - r) * 60.0) * (1.0 - t) * (1.0 - smoothstep(0.0, 0.06, abs(L - r)));
          return d / max(L, 1e-3) * w;
        }
        void main(){
          vec2 p = vW.xz;
          float n = fbm3(p * 1.3);
          float puddle = smoothstep(0.4, 0.56, fbm3(p * 0.6 + vec2(3.1, 7.7)));
          vec2 rp = ripple(p, 3.0, 0.9) + ripple(p + 0.37, 4.7, 1.3) * 0.7;
          vec3 N = normalize(vec3(rp.x * 0.09 + (n - 0.5) * 0.05 * (1.0 - puddle), 1.0, rp.y * 0.09));
          vec3 V = normalize(vW - cameraPosition);
          vec3 R = reflect(V, N);
          vec3 refl = skyCol(R) * 0.7;
          // neon streaks: compare the reflected ray with the direction to each light (stretched along elevation)
          vec2 ra = vec2(atan(R.x, -R.z), R.y);
          for (int i = 0; i < 5; i++){
            vec3 dl = normalize(uL[i].xyz - vW);
            vec2 la = vec2(atan(dl.x, -dl.z), dl.y);
            float dx = (ra.x - la.x) / uL[i].w, dy = (ra.y - la.y) / (uL[i].w * 9.0);
            refl += uLC[i] * exp(-dx * dx - dy * dy) * 1.2;
          }
          // the pink curb tube: a horizontal line at the wall's foot, mirrored as a band that wobbles with the ripples
          float pz = vW.z - 0.17;
          refl += uPinkC * 0.55 * exp(-pow(max(pz, 0.0) * 1.6 + abs(N.x) * 6.0, 2.0)) * smoothstep(${(BX0 - 0.4).toFixed(2)}, ${BX0.toFixed(2)}, vW.x) * (1.0 - smoothstep(${BX1.toFixed(2)}, ${(BX1 + 0.4).toFixed(2)}, vW.x));
          float fres = 0.04 + 0.96 * pow(1.0 - max(dot(-V, N), 0.0), 5.0);
          vec3 base = vec3(0.022, 0.019, 0.03) * (0.6 + 0.8 * n);
          // soft coloured light pools from the props (diffuse)
          base += lin3(0.1, 0.25, 1.0) * 0.05 * exp(-length(p - vec2(-1.5, 0.6)) * 1.6);
          base += vec3(0.1, 0.9, 0.4) * 0.03 * exp(-length(p - vec2(1.05, 2.7)) * 3.0);
          base += uPinkC * 0.03 * exp(-max(pz, 0.0) * 4.0);
          vec3 col = base * (1.0 - 0.6 * puddle) + refl * mix(0.1, 0.9, puddle) * mix(0.35, 1.0, fres);
          col = fogMix(col, vW);
          gl_FragColor = vec4(col, 1.0);
          ${OUT}
        }`.replace("lin3(0.1, 0.25, 1.0)", "vec3(0.1, 0.25, 1.0)"),
    });
    add(new THREE.Mesh(g, mat), 0);
  }

  // rain right in front of the lens: a few big, soft, out-of-focus streaks in view space (they follow any camera)
  {
    const n = hi ? 70 : 30;
    const geo = new THREE.PlaneGeometry(1, 1);
    const d = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) d.set([R(), R(), R(), R()], i * 4);
    geo.setAttribute("aDrop", new THREE.InstancedBufferAttribute(d, 4));
    const mat = shader({
      additive: true, side: THREE.DoubleSide,
      vs: `attribute vec4 aDrop; varying vec2 vUv; varying vec3 vC; varying float vA;
        void main(){
          float depth = mix(0.35, 1.3, aDrop.z);
          float th = 1.0 / projectionMatrix[1][1], tw = 1.0 / projectionMatrix[0][0];
          float sp = mix(1.2, 1.8, aDrop.w);
          float ph = fract(aDrop.y - uTime * sp / 2.6);
          vec3 c = vec3((aDrop.x * 2.0 - 1.0) * tw * 1.15, (1.0 - ph * 2.0) * th * 1.3, -1.0) * depth;
          c.x -= ph * 0.12 * depth;
          vec2 axis = normalize(vec2(-0.09, -1.0)), side = vec2(-axis.y, axis.x);
          float len = 0.07 * depth, w = 0.0075 * depth;
          c.xy += axis * position.y * len + side * position.x * w;
          vUv = position.xy * 2.0;
          float sx = aDrop.x;
          vC = mix(vec3(0.35, 0.4, 1.0), vec3(0.3, 0.95, 0.85), smoothstep(0.3, 0.9, sx)) * 0.7 + vec3(0.25);
          vA = (0.05 + 0.05 * aDrop.w) * smoothstep(0.0, 0.08, ph) * (1.0 - smoothstep(0.85, 1.0, ph));
          gl_Position = projectionMatrix * vec4(c, 1.0);
        }`,
      fs: `varying vec2 vUv; varying vec3 vC; varying float vA;
        void main(){
          float a = (1.0 - vUv.x * vUv.x) * (1.0 - vUv.y * vUv.y);
          gl_FragColor = vec4(vC * a * a * vA, 1.0);
          ${OUT}
        }`,
    });
    const m = add(new THREE.InstancedMesh(geo, mat, n), 9);
    m.frustumCulled = false;
  }

  // neon tubes (unlit, vertex coloured, values > 1 so they bloom)
  {
    const parts = [];
    const tubeC = (g, c, k) => { const n = g.attributes.position.count, col = new Float32Array(n * 3); for (let i = 0; i < n; i++) col.set([c.r * k, c.g * k, c.b * k], i * 3); g.setAttribute("color", new THREE.BufferAttribute(col, 3)); parts.push(g.index ? g.toNonIndexed() : g); };
    tubeC(box(BX0 - 0.05, BX1 + 0.05, 0.676, 0.7, -0.02, 0.0), NEON.green, 0.45);       // under the coping lip
    tubeC(box(BX0 - 0.1, BX1 + 0.1, -0.45, -0.425, 0.16, 0.175), NEON.pink, 0.9);      // along the top of the curb (the old cornice)
    // blue tube outline around the blade sign
    tubeC(box(-1.85 + BLADE_DX, -1.835 + BLADE_DX, -1.55, 2.1, 0.78, 0.8), NEON.blue, 1.0);
    tubeC(box(-1.85 + BLADE_DX, -1.835 + BLADE_DX, 2.08, 2.1, 0.05, 0.8), NEON.blue, 1.0);
    tubeC(box(-1.85 + BLADE_DX, -1.835 + BLADE_DX, -1.57, -1.55, 0.05, 0.8), NEON.blue, 1.0);
    // green / cyan status strips on the AC unit
    tubeC(box(-2.05, -1.3, 0.82, 0.85, -3.98, -3.96), NEON.green, 1.4);
    tubeC(box(-2.05, -1.3, 0.45, 0.47, -3.98, -3.96), NEON.cyan, 1.0);
    // the depth layer's neon (perch props, foreground rims)
    for (const [g, c, k] of depthNeon) tubeC(g, c, k);
    const g = mergeGeometries(parts.map((p) => { const q = p.clone(); q.deleteAttribute("uv"); q.deleteAttribute("normal"); return q; }));
    parts.forEach((p) => p.dispose());
    add(new THREE.Mesh(g, neonMat), 2);
  }

  // the two foreground signs go in the sign mesh (they're "near" = 1, so they react to her talking)
  addSign({ c: V(-1.815 + BLADE_DX, 0.3, 0.39), r: V(0, 0, -1), u: Y, hw: 0.34, hh: 1.6, uv: rectUV(vRect(0)), near: 1, bright: 0.85, kind: 0 });
  addSign({ c: V(1.38, 0.99, -0.215), r: Xp, u: Y, hw: 0.27, hh: 0.0675, uv: rectUV(hRect(2)), near: 1, bright: 0.7, kind: 1 });

  // ------------------------------------------------------------------ signs (one merged additive mesh)
  {
    const n = signs.length, pos = new Float32Array(n * 12), uv = new Float32Array(n * 8), at = new Float32Array(n * 16), idx = [];
    const tmp = new THREE.Vector3();
    signs.forEach((s, i) => {
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
      corners.forEach(([a, b], k) => {
        tmp.copy(s.c).addScaledVector(s.r, a * s.hw).addScaledVector(s.u, b * s.hh);
        pos.set([tmp.x, tmp.y, tmp.z], (i * 4 + k) * 3);
        uv.set([a < 0 ? s.uv[0] : s.uv[2], b < 0 ? s.uv[1] : s.uv[3]], (i * 4 + k) * 2);
        at.set([R() * 100, s.near, s.bright, s.kind], (i * 4 + k) * 4);
      });
      const o = i * 4; idx.push(o, o + 1, o + 2, o, o + 2, o + 3);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    g.setAttribute("aSign", new THREE.BufferAttribute(at, 4));
    g.setIndex(idx);
    const tex = makeSignAtlas(); trash.push(tex);
    const mat = shader({
      additive: true, side: THREE.DoubleSide, uniforms: { uMap: { value: tex } },
      vs: `attribute vec4 aSign; varying vec2 vUv; varying vec4 vS; varying vec3 vW;
        void main(){ vUv = uv; vS = aSign; vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `uniform sampler2D uMap; varying vec2 vUv; varying vec4 vS; varying vec3 vW;
        void main(){
          vec3 c = texture2D(uMap, vUv).rgb;
          float seed = vS.x, near = vS.y, k = uI * near;
          // hum, random tube drop-outs (more when she talks), a talking pulse for the near signs
          float hum = 0.94 + 0.06 * sin(uTime * 110.0 + seed);
          float r = hash11(floor(uTime * (7.0 + 10.0 * k)) + seed * 13.0);
          float drop = step(r, 0.025 + 0.18 * k + 0.22 * step(0.5, vS.w));
          float pulse = 1.0 + k * (0.35 + 0.35 * sin(uTime * 13.0 + seed));
          float b = vS.z * hum * mix(1.0, 0.18, drop) * pulse;
          gl_FragColor = vec4(c * b * fogKeep(vW), 1.0);
          ${OUT}
        }`,
    });
    add(new THREE.Mesh(g, mat), 3);
  }

  // ------------------------------------------------------------------ holographic billboards (one merged mesh)
  const boards = [];
  const boardPortrait = { value: 0 };
  {
    const tex = makeBoardAtlas(); trash.push(tex);
    const list = [
      // landscape board on the upper front of the tall tower at the left, angled toward her
      { c: V(BOARD.x, BOARD.y, BOARD.z), w: BOARD.w, h: BOARD.h, yaw: BOARD.yaw, uv: [0, 0.5, 1, 1], id: 0 },
      // floating emblem hologram over the tower on the right
      { c: V(holoBase.x, holoBase.top + 11, holoBase.z), w: 17, h: 17, yaw: 0.44, uv: [0, 0, 0.5, 0.5], id: 1 },
      // a tall narrow kana banner: a pink accent in the left part of the frame
      { c: V(-25, -1, -50), w: 3.2, h: 13, yaw: 0.45, uv: [0.56, 0.0, 0.8, 0.44], id: 2 },
      // portrait phones only (setAspect): a second emblem hologram up beside her hair, right of her head, since the
      // narrow frame crops out all the hero signs (collapsed to nothing in landscape)
      { c: V(9.0, 9.2, -55), w: 8.5, h: 8.5, yaw: -0.12, uv: [0, 0, 0.5, 0.5], id: 3 },
    ];
    const pos = [], uv = [], at = [], idx = [];
    list.forEach((b, i) => {
      const r = V(Math.cos(b.yaw), 0, -Math.sin(b.yaw));
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, c]) => {
        const p = b.c.clone().addScaledVector(r, a * b.w / 2).addScaledVector(Y, c * b.h / 2);
        pos.push(p.x, p.y, p.z); uv.push((a + 1) / 2, (c + 1) / 2);
        at.push(b.id, b.uv[0], b.uv[1], b.uv[2], b.uv[3]);
      });
      idx.push(i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3);
      boards.push(b);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    const atArr = new Float32Array(at), ia = new THREE.InterleavedBuffer(atArr, 5);
    g.setAttribute("aId", new THREE.InterleavedBufferAttribute(ia, 1, 0));
    g.setAttribute("aRect", new THREE.InterleavedBufferAttribute(ia, 4, 1));
    g.setIndex(idx);
    const mat = shader({
      additive: true, side: THREE.DoubleSide,
      uniforms: { uMap: { value: tex }, uPink: { value: NEON.pink }, uBlue: { value: NEON.blue }, uGreen: { value: NEON.green }, uCyan: { value: NEON.cyan }, uPortrait: boardPortrait },
      vs: `attribute float aId; attribute vec4 aRect; uniform float uPortrait; varying vec2 vUv; varying float vId; varying vec4 vRect; varying vec3 vW;
        void main(){ vUv = uv; vRect = aRect;
          // id 3 = the portrait-only emblem: drawn as the emblem (id 1); collapsed to a point unless portrait
          vec3 pos = position; vId = aId;
          if (aId > 2.5) { vId = 1.0; pos = mix(vec3(9.0, 9.2, -55.0), pos, uPortrait); }
          vec4 wp = modelMatrix * vec4(pos, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `uniform sampler2D uMap; uniform vec3 uPink, uBlue, uGreen, uCyan;
        varying vec2 vUv; varying float vId; varying vec4 vRect; varying vec3 vW;
        void main(){
          vec2 uv = vUv;
          float t = uTime;
          // glitch: occasional horizontal slice shifts (more often when she talks)
          float gTick = floor(t * 7.0) + vId * 17.0;
          float glitch = step(0.9 - 0.05 * uI, hash11(gTick)); // ~10%, ~15% while she talks (1 tick = 1/7 s)
          float slice = floor(uv.y * 18.0);
          uv.x += glitch * (hash12(vec2(slice, gTick)) - 0.5) * 0.12 * step(0.5, hash12(vec2(slice * 3.1, gTick)));
          // the emblem hologram slowly turns (fake it by squashing u)
          float turn = cos(t * 0.35);
          if (vId > 0.5 && vId < 1.5) uv.x = (uv.x - 0.5) / (0.6 + 0.4 * turn) + 0.5;
          float inside = step(0.0, uv.x) * step(uv.x, 1.0);
          vec2 tuv = mix(vRect.xy, vRect.zw, clamp(uv, 0.0, 1.0));
          float split = 0.0008 + glitch * 0.01;
          float cr = texture2D(uMap, tuv + vec2(split, 0.0)).r, cg = texture2D(uMap, tuv).g, cb = texture2D(uMap, tuv - vec2(split, 0.0)).b;
          vec3 content = vec3(cr, cg, cb) * inside;
          // colour: calm = blue/green/cyan; talking = her pink takes over
          vec3 calm = vId < 0.5 ? mix(uBlue, uCyan, 0.35 + 0.35 * sin(t * 0.3)) : vId < 1.5 ? mix(uGreen, uCyan, 0.5) : uPink;
          vec3 tint = mix(calm, uPink, uI * (vId < 1.5 ? 1.0 : 0.0));
          if (vId < 0.5) tint *= 1.0 - 0.45 * uI; // the big billboard eases off while she talks
          // backdrop: soft diagonal gradient bands drifting
          float bands = 0.5 + 0.5 * sin((vUv.x * 1.3 + vUv.y) * 7.0 - t * 0.8);
          vec3 bg = mix(tint, mix(uBlue, uPink, 0.3 + 0.7 * uI), vUv.y) * (0.035 + 0.035 * bands);
          if (vId > 0.5 && vId < 1.5) bg *= 0.0; // the floating emblem has no panel
          float scan = 0.72 + 0.28 * sin(vUv.y * 380.0 - t * 14.0);
          float roll = 0.85 + 0.15 * smoothstep(0.0, 0.04, abs(fract(vUv.y - t * 0.11) - 0.5));
          float flick = 0.92 + 0.08 * hash11(floor(t * 24.0) + vId);
          // frame
          vec2 fe = min(vUv, 1.0 - vUv);
          float frame = (vId > 0.5 && vId < 1.5) ? 0.0 : (1.0 - smoothstep(0.0, 0.012, min(fe.x, fe.y * 2.0)));
          vec3 col = (bg + content * tint * 1.9 + frame * tint * 1.4) * scan * roll * flick;
          // edge falloff so it reads as a projection
          col *= smoothstep(0.0, 0.04, fe.x) * smoothstep(0.0, 0.04, fe.y) * 0.6 + 0.4;
          gl_FragColor = vec4(col * fogKeep(vW), 1.0);
          ${OUT}
        }`,
    });
    add(new THREE.Mesh(g, mat), 4);
  }

  // ------------------------------------------------------------------ haze / smog / rain sheets (one merged additive mesh)
  {
    // kind 0 smog, 1 rain sheet, 2 street glow (bright at the bottom)
    const cards = [
      { c: V(0, -4, -95), w: 260, h: 44, col: lin(0x0e4a6a), a: 0.45, k: 0 },
      { c: V(0, 20, -230), w: 700, h: 120, col: lin(0x2a1250), a: 0.28, k: 0 },
      { c: V(-26, 0, -54), w: 30, h: 26, col: NEON.pink, a: 0.05, k: 0 },
      { c: V(13, 6, -52), w: 34, h: 24, col: NEON.blue, a: 0.06, k: 0 },
      { c: V(-48, 16, -98), w: 50, h: 40, col: NEON.cyan, a: 0.05, k: 0 },
      { c: V(-60, -22, -75), w: 160, h: 34, col: lin(0xff3a7a), a: 0.16, k: 2 },
      { c: V(55, -20, -60), w: 140, h: 30, col: lin(0x19f2c7), a: 0.1, k: 2 },
      { c: V(0, -26, -130), w: 300, h: 40, col: lin(0x8a2ad0), a: 0.14, k: 2 },
      { c: V(0, -5, -20), w: 70, h: 18, col: lin(0x3a2a70), a: 0.25, k: 0 },
      { c: V(0, 4, -12), w: 40, h: 22, col: lin(0x7080c0), a: 0.07, k: 1 },
      { c: V(0, 10, -36), w: 90, h: 50, col: lin(0x7080c0), a: 0.06, k: 1 },
      // soft wash on the parapet face behind her legs (so her black shin reads as a silhouette), and the
      // horizon glow band the left vista looks out on
      { c: V(-0.35, 0.3, -0.03), w: 1.6, h: 0.9, col: NEON.pink, a: 0.2, k: 3 },
      { c: V(0.45, 0.3, -0.025), w: 1.5, h: 0.9, col: lin(0x2f7dff), a: 0.28, k: 3 },
      { c: V(-120, 6, -330), w: 380, h: 70, col: lin(0x6a2a90), a: 0.35, k: 2 },
      // washes on the back perches' faces, behind her legs (stairwell hut door, water tank)
      // (kept small and under the lip: from the parapet shot they must not read as glowing panels)
      { c: V(P_HUT.seat.x + 0.1, P_HUT.seat.y - 0.36, P_HUT.seat.z - 0.05), w: 1.6, h: 0.72, col: lin(0x2f7dff), a: 0.3, k: 3 },
      { c: V(P_TANK.seat.x, P_TANK.seat.y - 0.3, P_TANK.seat.z - 0.005), w: 1.05, h: 0.5, col: lin(0x7a4dff), a: 0.17, k: 3 },
      // the fore crate's front face, under the seat (yawed with the crate)
      { c: ledgeV(P_FORE.seat, P_FORE.ledgeYaw, -0.06, P_FORE.seat.y - 0.33, 0.012), w: 1.1, h: 0.6, col: lin(0xd0306a), a: 0.34, k: 3, yaw: P_FORE.ledgeYaw },
    ].filter((c, i) => hi || i !== 3);
    const pos = [], uv = [], at = [], col = [], idx = [];
    cards.forEach((cd, i) => {
      const cy = Math.cos(cd.yaw || 0), sy = Math.sin(cd.yaw || 0);
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, b]) => {
        pos.push(cd.c.x + a * cd.w / 2 * cy, cd.c.y + b * cd.h / 2, cd.c.z - a * cd.w / 2 * sy); uv.push((a + 1) / 2, (b + 1) / 2);
        at.push(R() * 50, cd.a, cd.k, cd.w / cd.h); col.push(cd.col.r, cd.col.g, cd.col.b);
      });
      idx.push(i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setAttribute("aH", new THREE.Float32BufferAttribute(at, 4));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    const mat = shader({
      additive: true, side: THREE.DoubleSide,
      vs: `attribute vec4 aH; attribute vec3 color; varying vec2 vUv; varying vec4 vH; varying vec3 vC; varying vec3 vW;
        void main(){ vUv = uv; vH = aH; vC = color; vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `varying vec2 vUv; varying vec4 vH; varying vec3 vC; varying vec3 vW;
        void main(){
          vec2 c = vUv * 2.0 - 1.0;
          float edge = (1.0 - smoothstep(0.4, 1.0, abs(c.x))) * (1.0 - smoothstep(0.2, 1.0, abs(c.y)));
          vec2 p = vec2(vUv.x * vH.w, vUv.y) * 2.5 + vH.x;
          float a;
          if (vH.z < 0.5) {
            a = fbm3(p + vec2(uTime * 0.03, uTime * 0.008)) ;
            a = smoothstep(0.25, 0.85, a) * edge;
          } else if (vH.z < 1.5) {
            // rain sheet: thin vertical streaks scrolling down
            vec2 q = vec2(vUv.x * vH.w * 60.0, vUv.y * 3.0 + uTime * 2.2);
            float col = floor(q.x);
            float s = fract(q.y + hash11(col) * 10.0);
            a = step(0.82, hash11(col * 1.7)) * smoothstep(0.0, 0.2, s) * (1.0 - smoothstep(0.2, 0.35, s)) * edge;
            a *= 0.6 + 0.4 * fbm3(p * 0.5);
          } else if (vH.z > 2.5) {
            // wall wash: brightest just under the coping lip, soft at the sides
            a = (0.25 + 0.75 * smoothstep(0.0, 1.0, vUv.y)) * (1.0 - smoothstep(0.3, 1.0, abs(c.x))) * smoothstep(0.0, 0.3, vUv.y);
          } else {
            a = (1.0 - smoothstep(0.0, 1.0, vUv.y)) * (1.0 - smoothstep(0.3, 1.0, abs(c.x))) * (0.55 + 0.45 * fbm3(p * 0.7 + uTime * 0.02));
          }
          gl_FragColor = vec4(vC * a * vH.y * fogKeep(vW * 0.6 + cameraPosition * 0.4), 1.0);
          ${OUT}
        }`,
    });
    add(new THREE.Mesh(g, mat), 5);
  }

  // ------------------------------------------------------------------ beams: searchlights + hologram projector (axial billboards)
  {
    const beams = [
      { a: V(-120, -30, -300), dir: V(0.15, 1, 0.1), len: 420, w0: 2, w1: 22, col: lin(0x6a8cff), alpha: 0.16, sweep: 0.25, speed: 0.11 },
      { a: V(60, -30, -330), dir: V(-0.25, 1, 0.05), len: 420, w0: 2, w1: 20, col: lin(0x3ee8c0), alpha: 0.12, sweep: 0.22, speed: 0.09 },
      { a: V(140, -30, -340), dir: V(-0.2, 1, 0.05), len: 420, w0: 2, w1: 26, col: lin(0xff4d8a), alpha: 0.08, sweep: 0.3, speed: 0.08 },
      { a: V(-20, -30, -420), dir: V(0.05, 1, 0), len: 460, w0: 3, w1: 30, col: lin(0x3ee8c0), alpha: 0.05, sweep: 0.2, speed: 0.06 },
      // projector cone from the tower roof up to the emblem hologram
      { a: V(holoBase.x, holoBase.top + 0.3, holoBase.z), dir: V(0, 1, 0), len: 12, w0: 0.6, w1: 17, col: NEON.cyan, alpha: 0.12, sweep: 0, speed: 0 },
      // light spilling down from the left billboard into the rain
      { a: V(BOARD.x, BOARD.y - BOARD.h / 2 - 0.2, BOARD.z + 0.4), dir: V(-0.1, -1, 0.3), len: 30, w0: 13, w1: 30, col: lin(0x5070ff), alpha: 0.06, sweep: 0, speed: 0, pinkable: 1 },
    ];
    const pos = [], A = [], Bv = [], W = [], C = [], P = [], idx = [];
    beams.forEach((b, i) => {
      const end = b.a.clone().addScaledVector(b.dir.clone().normalize(), b.len);
      [[-1, 0], [1, 0], [1, 1], [-1, 1]].forEach(([s, t]) => {
        pos.push(s, t, 0); A.push(b.a.x, b.a.y, b.a.z); Bv.push(end.x, end.y, end.z); W.push(b.w0, b.w1);
        C.push(b.col.r, b.col.g, b.col.b); P.push(R() * 10, b.sweep, b.speed, b.alpha + (b.pinkable ? 10 : 0));
      });
      idx.push(i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("aA", new THREE.Float32BufferAttribute(A, 3));
    g.setAttribute("aB", new THREE.Float32BufferAttribute(Bv, 3));
    g.setAttribute("aW", new THREE.Float32BufferAttribute(W, 2));
    g.setAttribute("aC", new THREE.Float32BufferAttribute(C, 3));
    g.setAttribute("aP", new THREE.Float32BufferAttribute(P, 4));
    g.setIndex(idx);
    const mat = shader({
      additive: true, side: THREE.DoubleSide, uniforms: { uPink: { value: NEON.pink } },
      vs: `attribute vec3 aA, aB, aC; attribute vec2 aW; attribute vec4 aP;
        varying vec2 vUv; varying vec3 vC; varying float vAlpha; varying vec3 vW;
        uniform vec3 uPink;
        void main(){
          vec3 axis = aB - aA; float L = length(axis); axis /= L;
          // sweep: rotate the axis around z (and a bit around x)
          float ang = aP.y * sin(uTime * aP.z + aP.x);
          float cs = cos(ang), sn = sin(ang);
          axis = vec3(axis.x * cs - axis.y * sn, axis.x * sn + axis.y * cs, axis.z + 0.3 * aP.y * sin(uTime * aP.z * 0.7 + aP.x * 2.0));
          axis = normalize(axis);
          vec3 p = aA + axis * (position.y * L);
          vec3 toCam = normalize(cameraPosition - p);
          vec3 side = normalize(cross(axis, toCam));
          float w = mix(aW.x, aW.y, position.y) * 0.5;
          p += side * position.x * w;
          vUv = vec2(position.x, position.y);
          float pink = step(5.0, aP.w);
          vAlpha = aP.w - pink * 10.0;
          vC = mix(aC, uPink, pink * uI);
          vW = p;
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fs: `varying vec2 vUv; varying vec3 vC; varying float vAlpha; varying vec3 vW;
        void main(){
          float across = 1.0 - vUv.x * vUv.x;
          float along = (1.0 - vUv.y) * smoothstep(0.0, 0.05, vUv.y);
          float dust = 0.75 + 0.25 * vnoise(vec2(vUv.x * 3.0, vUv.y * 20.0 - uTime * 0.5));
          float a = across * across * along * dust * vAlpha;
          gl_FragColor = vec4(vC * a * fogKeep(vW * 0.5 + cameraPosition * 0.5), 1.0);
          ${OUT}
        }`,
    });
    const m = add(new THREE.Mesh(g, mat), 6); m.frustumCulled = false;
  }

  // ------------------------------------------------------------------ flying traffic: actual hover cars (traffic.js)
  const traffic = createTraffic(group, { quality: hi ? "high" : "low", sky: { uSkyHorizon: U.uSkyHorizon, uSkyHorizonC: U.uSkyHorizonC, uSkyZenith: U.uSkyZenith, uSkyBelow: U.uSkyBelow, uFogDensity: U.uFogDensity } });

  // ------------------------------------------------------------------ elevated train
  let train;
  {
    const cars = 6, carL = 13, gap = 0.8, L = cars * carL + (cars - 1) * gap;
    const geo = new THREE.BoxGeometry(L, 2.7, 3.0); geo.translate(0, 1.35, 0);
    const mat = shader({
      uniforms: { uL: { value: L } },
      vs: `varying vec3 vL; varying vec3 vN; varying vec3 vW;
        void main(){ vL = position; vN = normal; vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
      fs: `uniform float uL; varying vec3 vL; varying vec3 vN; varying vec3 vW;
        void main(){
          float x = vL.x + uL * 0.5;                 // 0..L along the train
          float carF = x / 13.8, car = floor(carF), cf = fract(carF) * 13.8;
          float inGap = step(13.0, cf);
          float side = step(0.5, abs(vN.z));
          vec3 col = vec3(0.012, 0.011, 0.018);
          // window band
          float wy = step(1.05, vL.y) * step(vL.y, 2.0);
          float wx = step(0.25, fract(cf / 1.6)) * step(0.6, cf) * step(cf, 12.4);
          vec3 wc = mix(vec3(1.0, 0.85, 0.6), vec3(0.45, 0.9, 1.0), step(0.6, hash11(car + 3.0)));
          col += wc * 0.42 * wy * wx * side * (1.0 - inGap);
          // pink roof line + door lights
          col += vec3(1.0, 0.02, 0.15) * 2.0 * step(2.55, vL.y) * step(vL.y, 2.62) * side * (1.0 - inGap);
          col += vec3(0.1, 1.0, 0.45) * 1.2 * step(0.35, vL.y) * step(vL.y, 0.42) * side * (1.0 - inGap);
          // headlight at the front end
          float front = step(uL - 0.05, x) * step(0.5, vN.x);
          col += vec3(1.0, 0.95, 0.85) * 4.0 * front * step(0.6, vL.y) * step(vL.y, 1.2);
          gl_FragColor = vec4(fogMix(col, vW), 1.0);
          ${OUT}
        }`,
    });
    train = add(new THREE.Mesh(geo, mat), 1);
    train.position.set(-400, TRACK.y, TRACK.z);
    train.frustumCulled = false;
  }

  // ------------------------------------------------------------------ points: aviation blinkers, distant street specks
  {
    const P = [], A = [];
    B.forEach((b) => {
      if (b.top > 14 && b.w < 100 && R() < 0.8) {
        P.push(b.x + b.w * 0.35, b.top + 0.6, b.z + b.d * 0.35); A.push(0, b.z < -150 ? 3.5 : 1.2, rr(0.6, 1.2), R() * 6);
        if (b.top > 60) { P.push(b.x - b.w * 0.35, b.top + 0.6, b.z - b.d * 0.35); A.push(0, 3.5, 1, R() * 6); }
      }
    });
    P.push(1.6, 5.4, -6.5); A.push(0, 0.14, 0.8, 0);            // the antenna beside Capri
    const nSpecks = hi ? 500 : 200;
    for (let i = 0; i < nSpecks; i++) {
      const z = rr(-30, -240), x = rr(-1.1, 1.1) * Math.abs(z);
      P.push(x, rr(-45, -14), z); A.push(1 + ((R() * 4) | 0), rr(0.4, 1.4), 0, R() * 6);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(P, 3));
    g.setAttribute("aP", new THREE.Float32BufferAttribute(A, 4));
    const mat = shader({
      additive: true, uniforms: { uPx: { value: 800 } },
      vs: `attribute vec4 aP; uniform float uPx; varying vec3 vC; varying float vB; varying vec3 vW;
        void main(){
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp(aP.y * uPx / -mv.z, 1.5, 40.0);
          float k = aP.x;
          vC = k < 0.5 ? vec3(1.0, 0.08, 0.12) : k < 1.5 ? vec3(1.0, 0.6, 0.3) : k < 2.5 ? vec3(0.3, 0.8, 1.0) : k < 3.5 ? vec3(1.0, 0.2, 0.5) : vec3(0.4, 1.0, 0.6);
          vB = k < 0.5 ? smoothstep(0.75, 1.0, sin(uTime * 3.14159 * aP.z + aP.w)) * 3.0 : 0.6 + 0.4 * sin(uTime * 0.7 + aP.w * 10.0);
          vW = (modelMatrix * vec4(position, 1.0)).xyz;
        }`,
      fs: `varying vec3 vC; varying float vB; varying vec3 vW;
        void main(){
          vec2 c = gl_PointCoord * 2.0 - 1.0; float r = dot(c, c);
          float a = exp(-r * 4.0) * (1.0 - smoothstep(0.7, 1.0, r));
          gl_FragColor = vec4(vC * a * vB * fogKeep(vW), 1.0);
          ${OUT}
        }`,
    });
    const pts = add(new THREE.Points(g, mat), 7);
    pts.frustumCulled = false;
    const sz = new THREE.Vector2();
    pts.onBeforeRender = (r, _s, cam) => {
      r.getDrawingBufferSize(sz);
      mat.uniforms.uPx.value = sz.y * 0.5 * (cam.projectionMatrix.elements[5] || 1);
    };
  }

  // ------------------------------------------------------------------ rain (instanced streaks, GPU animated)
  const RAIN_MIN = V(-7, -3, -12), RAIN_SIZE = V(14, 11, 15.2);
  {
    const n = hi ? 6500 : 2200;
    const geo = new THREE.PlaneGeometry(1, 1);
    const d = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) d.set([R(), R(), R(), R()], i * 4);
    geo.setAttribute("aDrop", new THREE.InstancedBufferAttribute(d, 4));
    const mat = shader({
      additive: true, side: THREE.DoubleSide,
      uniforms: { uMin: { value: RAIN_MIN }, uSize: { value: RAIN_SIZE }, uPink: { value: NEON.pink }, uCyan: { value: NEON.cyan } },
      vs: `attribute vec4 aDrop; uniform vec3 uMin, uSize; uniform vec3 uPink, uCyan;
        varying vec2 vUv; varying vec3 vC; varying float vA;
        void main(){
          float speed = mix(8.0, 11.0, aDrop.w);
          vec2 wind = vec2(-1.1, 0.35);
          float ph = fract(aDrop.y - uTime * speed / uSize.y);
          float fallT = (1.0 - ph) * uSize.y / speed;
          vec3 c = uMin + vec3(aDrop.x, ph, aDrop.z) * uSize;
          c.x += wind.x * fallT; c.z += wind.y * fallT;
          c.x = uMin.x + mod(c.x - uMin.x, uSize.x);
          vec3 axis = normalize(vec3(wind.x, -speed, wind.y));
          float len = 0.2 + 0.1 * aDrop.w;
          vec3 toCam = cameraPosition - c; float dist = length(toCam);
          vec3 side = normalize(cross(axis, toCam / dist));
          float w = 0.003 + 0.0009 * dist;
          vec3 p = c + axis * position.y * len + side * position.x * w;
          vUv = position.xy * 2.0;
          // tint from nearby neon: pink on the left (blade sign), cyan-blue on the right
          vec3 base = vec3(0.45, 0.55, 0.8) * 0.6;
          float lp = exp(-length(c - vec3(-1.8, 0.4, 0.5)) * 0.55);
          float lc = smoothstep(0.0, 6.0, c.x) * 0.6;
          vC = base + vec3(0.18, 0.49, 1.0) * lp * 1.1 + uCyan * lc * 0.5;
          vA = (0.06 + 0.06 * aDrop.x) * smoothstep(0.4, 1.4, dist) * (1.0 - smoothstep(9.0, 16.0, dist));
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fs: `varying vec2 vUv; varying vec3 vC; varying float vA;
        void main(){
          float a = (1.0 - vUv.x * vUv.x) * smoothstep(-1.0, 0.3, vUv.y) * (1.0 - smoothstep(0.6, 1.0, vUv.y));
          gl_FragColor = vec4(vC * a * vA, 1.0);
          ${OUT}
        }`,
    });
    const m = add(new THREE.InstancedMesh(geo, mat, n), 8);
    m.frustumCulled = false;
  }

  // ------------------------------------------------------------------ splashes on the wet coping and the cornice
  {
    const n = hi ? 70 : 30;
    const geo = new THREE.PlaneGeometry(1, 1); geo.rotateX(-Math.PI / 2);
    const d = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) d.set([R(), R() * 10, rr(1.4, 2.6), i < n * 0.8 ? 0 : 1], i * 4);
    geo.setAttribute("aSp", new THREE.InstancedBufferAttribute(d, 4));
    const mat = shader({
      additive: true,
      vs: `attribute vec4 aSp; varying vec2 vUv; varying float vLife; varying vec3 vW;
        void main(){
          float k = uTime * aSp.z + aSp.y;
          float cyc = floor(k); vLife = fract(k);
          float onCornice = aSp.w;
          float x = mix(-2.5, 2.7, hash11(cyc * 1.31 + aSp.x * 71.0));
          float z = mix(mix(-0.5, -0.02, hash11(cyc * 2.17 + aSp.x * 13.0)), mix(-0.02, 0.14, hash11(cyc * 2.17 + aSp.x * 13.0)), onCornice);
          float y = mix(0.803, -0.418, onCornice);
          float r = 0.015 + 0.05 * vLife;
          vec3 p = vec3(x, y, z) + vec3(position.x, 0.0, position.z) * r * 2.0;
          vUv = position.xz * 2.0; vW = p;
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fs: `varying vec2 vUv; varying float vLife; varying vec3 vW;
        void main(){
          float d = length(vUv);
          float ring = smoothstep(0.65, 0.85, d) * (1.0 - smoothstep(0.85, 1.0, d));
          float a = ring * (1.0 - vLife) * (1.0 - vLife) * 0.45;
          vec3 c = mix(vec3(0.55, 0.65, 1.0), vec3(0.3, 0.55, 1.0), smoothstep(-0.5, -2.0, vW.x));
          gl_FragColor = vec4(c * a, 1.0);
          ${OUT}
        }`,
    });
    const m = add(new THREE.InstancedMesh(geo, mat, n), 8);
    m.frustumCulled = false;
  }

  // ------------------------------------------------------------------ lights for the foreground props (Capri is unlit)
  const hemi = new THREE.HemisphereLight(0x3a2a6a, 0x120612, 0.9);
  // (the blade sign's light: blue now, and further off the parapet face so it washes instead of making a hotspot)
  const lPink = new THREE.PointLight(0x2f7dff, 3, 6, 2); lPink.position.set(-1.25, 0.7, 1.15);
  const lBlue = new THREE.PointLight(0x2f7dff, 22, 14, 2); lBlue.position.set(2.8, 3.0, -1.2);
  const lGreen = new THREE.PointLight(PALETTE.green, 0.9, 3, 2); lGreen.position.set(1.38, 1.15, 0.35);
  const lCyan = new THREE.PointLight(0x3ee8ff, 10, 16, 2); lCyan.position.set(-1.5, 2.5, -4.5);
  group.add(hemi, lPink, lBlue, lGreen, lCyan);

  // ------------------------------------------------------------------ API
  let target = 0, I = 0;
  const TRAIN_PERIOD = 23, TRAIN_SPEED = 28; // slower: in frame for ~4 s of every pass
  const hsh = (x) => { const s = Math.sin(x * 127.1) * 43758.5453; return s - Math.floor(s); };

  function update(dt, t) {
    dt = Math.min(dt || 0, 0.1);
    I += (target - I) * (1 - Math.exp(-dt * 3.5));
    U.uTime.value = t % 3600; // keep shader time small (float precision); one seamless-enough wrap per hour
    U.uI.value = I;
    traffic.update(dt, t);
    // train: one pass every ~23 s, alternating direction
    const pass = Math.floor(t / TRAIN_PERIOD), ph = t - pass * TRAIN_PERIOD;
    const dir = pass % 2 ? -1 : 1;
    train.position.x = dir * (-320 + ph * TRAIN_SPEED);
    train.scale.x = dir;
    // light flicker (roughly tracking the sign shader's drop-outs) and the talking pulse
    const drop = hsh(Math.floor(t * (7 + 10 * I)) + 1.7) < 0.025 + 0.18 * I;
    lPink.intensity = (drop ? 0.6 : 3) * (1 + I * (0.4 + 0.4 * Math.sin(t * 13)));
    const gdrop = hsh(Math.floor(t * (7 + 10 * I)) + 9.1) < 0.25 + 0.18 * I;
    lGreen.intensity = gdrop ? 0.1 : 0.9;
    lBlue.intensity = 22 * (0.9 + 0.1 * Math.sin(t * 0.7));
  }

  function setIntensity(x) { target = THREE.MathUtils.clamp(+x || 0, 0, 1); traffic.setIntensity(target); }
  // optional: the viewport aspect (main.js calls it when framing). Portrait (< 0.8) shows an extra hero hologram.
  function setAspect(a) { boardPortrait.value = a < 0.8 ? 1 : 0; }

  function dispose() {
    traffic.dispose();
    scene.remove(group);
    for (const o of trash) o.dispose && o.dispose();
    trash.length = 0;
    scene.fog = prevFog; scene.background = prevBg;
  }

  const cloneView = (v) => v && { pos: v.pos.clone(), target: v.target.clone(), fov: v.fov };
  const perches = PERCHES.map((p) => ({ ...p, seat: p.seat.clone(), view: cloneView(p.view), viewPortrait: cloneView(p.viewPortrait) }));
  // setFraming({ pos, target, fov, aspect }): forwarded to traffic.js (its close lanes are planned in screen terms)
  const setFraming = (o) => traffic.setFraming?.(o);
  return { update, seat: SEAT.clone(), seatRotationY: 0, perches, floorY: FLOOR, setIntensity, setAspect, setFraming, dispose, group, recommendedFar: RECOMMENDED_FAR };
}
