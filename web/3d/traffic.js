// traffic.js: flying traffic for the city behind Capri.
//
// createTraffic(parent, { quality: 'high' | 'low' }) -> { update(dt, t), setIntensity(x), dispose(), group, stats }
//
// Three original low-poly hover-vehicle designs, each a handful of extruded side profiles + boxes merged into one
// geometry: a sleek fastback coupe ("kestrel"), a boxy cab / van ("mule") and a hover bike ("wasp"). One
// InstancedMesh per design (3 draw calls), one additive InstancedMesh for every light effect the near cars throw
// (headlight flares + beams in the rain, tail-light flares, underglow, thruster plumes, short light trails: 1 draw
// call), and one for the far field (> ~150 m) where a car is only a head / tail light pair (1 draw call). 5 in all.
//
// Everything moves on the GPU: each car's lane (height, depth, yaw, speed, phase, a gentle sideways wiggle) is an
// instance attribute and the vertex shader computes where it is from a time uniform, banks it into the wiggle,
// bobs it and points the nose along the direction of travel. update() only sets the time.
//
// Shading is a cheap fake, no lights: dark glossy paint = a little ambient + a fresnel-weighted reflection of a fixed
// "neon city" environment (hot pink city glow below the horizon, a bright horizon streak, cold blue sky above, pink /
// cyan rim from fixed directions, neon-sign bands that slide over the body as it moves through the city), city
// underlight on the belly, the car's own underglow colour on its underside, and emissive strips for the lights.
// Fog and the output conventions are copied from city.js so far cars fade exactly like the buildings.
//
// Coordinates: metres, Y up, the city is toward -Z, Capri sits at ~(0, 0.8, 0), the camera is at ~(0.4, 1.35, 3.6).
// Car model space: +X = nose, +Y = up, +Z = the car's right.

import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { PALETTE } from "./palette.js";

// ------------------------------------------------------------------------------------------------
// helpers
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

// ------------------------------------------------------------------------------------------------
// shared GLSL. COMMON (sky colour + fog) is a copy of city.js's so far cars fade into the same haze.
// ------------------------------------------------------------------------------------------------

const COMMON = /* glsl */ `
uniform float uTime;
uniform float uI;
uniform vec3 uSkyHorizon;
uniform vec3 uSkyHorizonC;
uniform vec3 uSkyZenith;
uniform vec3 uSkyBelow;
uniform float uFogDensity;
uniform vec4 uFr;    // framing: (camera y, camera z, tan(vfov/2), view slope) of the current shot (setFraming)
uniform vec4 uBand;  // screen-fraction offsets per lane class 1..4 (near upper, near lower, fly-by, hero)
uniform vec4 uWin;   // fly-by screen window: (ndc x from, ndc x to, fly-bys / hero on 0|1, 0)

float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }

vec3 skyCol(vec3 d){
  float side = clamp(abs(d.x) * 1.5, 0.0, 1.0);
  vec3 hor = mix(uSkyHorizonC, uSkyHorizon, side);
  vec3 c = mix(hor, uSkyZenith, pow(smoothstep(-0.02, 0.55, d.y), 0.6));
  c = mix(c, uSkyBelow * mix(0.75, 1.0, side), smoothstep(0.0, -0.22, d.y));
  return c;
}
float fogAmt(float dist){ float x = uFogDensity * dist; return 1.0 - exp(-x * x); }
vec3 fogMix(vec3 col, vec3 wp){ vec3 d = wp - cameraPosition; float L = length(d); return mix(col, skyCol(d / L) * 0.86, fogAmt(L)); }
float fogKeep(vec3 wp){ return 1.0 - fogAmt(length(wp - cameraPosition)); }
`;

const OUT = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

// Where a car is. Instance attributes:
//   aLane  = (height y, depth z, speed m/s signed: + = travels toward +X, phase 0..1)
//   aLane2 = (lane yaw rad, loop span m, wiggle amplitude m, seed 0..1)
//   aLane3 = (lane-change half-amplitude m, altitude-step half-amplitude m, surge m, 0)
// Out: O = origin (between the hover pods, at "ground" level of the car), F = nose direction, U = up, Rt = right.
// Besides a gentle weave, a car may change lanes (smooth steps sideways, banking ~15-20 deg through them), step its
// altitude (nose pitching with it) and surge (+-1 m/s), so the traffic reads as piloted rather than on rails.
const CAR_FRAME = /* glsl */ `
attribute vec4 aLane;
attribute vec4 aLane2;
attribute vec4 aLane3;
// a smooth periodic step: v = 2*S(sin(phi)) - 1 with S a smoothstep over sin in -c..c; d1 / d2 = derivatives wrt x
void stepWave(float x, float k, float ph, out float v, out float d1, out float d2){
  const float c = 0.8;
  float u = sin(x * k + ph), u1 = k * cos(x * k + ph), u2 = -k * k * u;
  float t = clamp((u + c) / (2.0 * c), 0.0, 1.0), inside = step(abs(u), c);
  float S = t * t * (3.0 - 2.0 * t), S1 = 6.0 * t * (1.0 - t) / (2.0 * c), S2 = inside * (6.0 - 12.0 * t) / (4.0 * c * c);
  v = 2.0 * S - 1.0; d1 = 2.0 * S1 * u1; d2 = 2.0 * (S2 * u1 * u1 + S1 * u2);
}
void carFrame(out vec3 O, out vec3 F, out vec3 U, out vec3 Rt){
  float s = abs(aLane.z), dir = sign(aLane.z);
  float span = aLane2.y, sd = aLane2.w * 6.2832;
  float surge = aLane3.z * sin(uTime * (0.21 + 0.13 * aLane2.w) + sd * 1.7);
  float x = (mod(aLane.w * span + uTime * s + surge, span) - span * 0.5) * dir;   // position along the lane axis
  float yaw = aLane2.x, A = aLane2.z;
  vec3 D = vec3(cos(yaw), 0.0, sin(yaw)), N = vec3(-sin(yaw), 0.0, cos(yaw));
  float k = 6.2832 / 70.0;                                                  // one sideways weave every 70 m
  float w = A * sin(x * k + sd), w1 = A * k * cos(x * k + sd), w2 = -A * k * k * sin(x * k + sd);
  float lc, lc1, lc2; stepWave(x, 6.2832 / 120.0, sd * 1.3 + 1.0, lc, lc1, lc2);   // lane changes
  float hs, hs1, hs2; stepWave(x, 6.2832 / 150.0, sd * 0.7 + 2.0, hs, hs1, hs2);   // altitude steps
  w += aLane3.x * lc; w1 += aLane3.x * lc1; w2 += aLane3.x * lc2;
  float h = aLane3.y * hs, h1 = aLane3.y * hs1;
  float bt = uTime * (0.55 + 0.3 * aLane2.w) + sd;
  float bob = 0.18 * sin(bt) + 0.07 * sin(bt * 2.3 + 1.0);
  float bobV = (0.18 * cos(bt) + 0.16 * cos(bt * 2.3 + 1.0)) * (0.55 + 0.3 * aLane2.w);
  // lane height: class 0 = absolute (aLane.x is y); 1..4 = planned in screen terms for the current shot (aLane.x is
  // the screen fraction, 0 = top, where the car's TOP edge lands; aLane3.w = class + top height / 10)
  float cls = floor(aLane3.w + 0.001), ly = aLane.x, dive = 0.0;
  if (cls > 0.5) {
    float top = fract(aLane3.w) * 10.0;
    float f = aLane.x + (cls < 1.5 ? uBand.x : cls < 2.5 ? uBand.y : cls < 3.5 ? uBand.z : uBand.w);
    ly = uFr.x + (uFr.y - aLane.y) * ((1.0 - 2.0 * f) * uFr.z + uFr.w) - top;
    if (cls > 2.5 && uWin.z < 0.5) ly -= 500.0; // fly-bys / hero off (portrait phones)
    if (cls > 2.5 && cls < 3.5) {
      // fly-bys only cross the open part of the frame: outside the window they dive in from / climb out to the sky
      vec4 cp = projectionMatrix * viewMatrix * vec4(D * x + vec3(0.0, ly, aLane.y), 1.0);
      float sx = cp.x / max(cp.w, 0.01);
      dive = 1.0 - smoothstep(uWin.x - 0.35, uWin.x, sx) * (1.0 - smoothstep(uWin.y, uWin.y + 0.35, sx));
      dive = dive * dive * (3.0 - 2.0 * dive);
    }
  }
  O = vec3(0.0, ly + h + dive * 9.0, aLane.y) + D * x + N * w + vec3(0.0, bob, 0.0);
  F = normalize((D + N * w1 + vec3(0.0, h1, 0.0)) * dir);
  F = normalize(F + vec3(0.0, bobV / max(s, 4.0), 0.0) - vec3(0.0, 0.015, 0.0)); // nose follows the bob, slightly nose-down
  vec3 up = vec3(0.0, 1.0, 0.0);
  Rt = normalize(cross(F, up));
  // bank into the curve: lateral acceleration toward the right side / g
  float lat = dot(N * w2, Rt) * s * s / 9.8;
  U = normalize(up + Rt * clamp(lat, -0.6, 0.6) * 0.65);
  // the hero banks toward the camera (its roof and canopy show, not just its belly); fly-bys pitch through their dive
  if (cls > 3.5) U = normalize(U + vec3(0.0, 0.0, 0.32));
  Rt = normalize(cross(F, U));
  U = cross(Rt, F);
}
`;

// ------------------------------------------------------------------------------------------------
// vehicle geometry
// ------------------------------------------------------------------------------------------------

// part ids (vertex attribute aPart)
const HULL = 0, HEAD = 1, TAIL = 2, ACCENT = 3, THRUST = 4, TRIM = 5, GLASS = 6, CABIN = 7;

function tag(g, part) {
  if (g.index) g = g.toNonIndexed();
  g.deleteAttribute("uv");
  g.computeVertexNormals(); // non-indexed: flat facets (the low-poly look Capri has too)
  const n = g.attributes.position.count;
  g.setAttribute("aPart", new THREE.Float32BufferAttribute(new Float32Array(n).fill(part), 1));
  return g;
}
// a side profile (x, y points, any winding) extruded across the car, centred on z = 0. The bevel tapers the sides
// toward the profile (tumblehome / chamfered edges).
function profile(pts, width, part, bt = 0.08, bs = 0.05) {
  const shape = new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y)));
  const depth = Math.max(0.01, width - 2 * bt);
  const g = new THREE.ExtrudeGeometry(shape, { depth, steps: 1, curveSegments: 1, bevelEnabled: bt > 0, bevelThickness: bt, bevelSize: bs, bevelSegments: 1 });
  g.translate(0, 0, -depth / 2);
  return tag(g, part);
}
// axis-aligned box from min / max corners
function box(x0, x1, y0, y1, z0, z1, part) {
  const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
  g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  return tag(g, part);
}
// a box mirrored to both sides (z0..z1 and -z1..-z0)
function pair(x0, x1, y0, y1, z0, z1, part) { return [box(x0, x1, y0, y1, z0, z1, part), box(x0, x1, y0, y1, -z1, -z0, part)]; }
// a thin slab lying on the profile segment p0 -> p1 (pushed `out` metres along the segment's normal), width across z
function slab(p0, p1, width, thick, out, part) {
  const dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.hypot(dx, dy);
  let nx = -dy / L, ny = dx / L;
  const g = new THREE.BoxGeometry(L, thick, width);
  g.rotateZ(Math.atan2(dy, dx));
  // normal pointing away from the car's centre (roughly up / forward)
  if (ny < 0 || (Math.abs(ny) < 0.2 && nx * (p0[0] + p1[0]) < 0)) { nx = -nx; ny = -ny; }
  g.translate((p0[0] + p1[0]) / 2 + nx * out, (p0[1] + p1[1]) / 2 + ny * out, 0);
  return tag(g, part);
}
function rotZ(g, a, cx, cy) { g.translate(-cx, -cy, 0); g.rotateZ(a); g.translate(cx, cy, 0); g.computeVertexNormals(); return g; }

// glow anchors per design, for the light-effects mesh: dims = (half length, headlight |z|, headlight y, tail y),
// dims2 = (half width, tail |z|, hover-pod bottom y, beam scale)
const DESIGNS = {};

// "Kestrel": a low fastback coupe on four hover pods, slanted headlight slits, a full-width tail bar, side neon line
function kestrel() {
  const parts = [];
  parts.push(profile([[-2.3, 0.3], [-1.7, 0.2], [1.75, 0.2], [2.3, 0.33], [2.44, 0.48], [2.05, 0.66], [0.75, 0.79], [-1.95, 0.86], [-2.38, 0.74]], 1.9, HULL, 0.1, 0.05));
  parts.push(profile([[0.92, 0.74], [0.08, 1.17], [-0.8, 1.21], [-1.85, 0.82], [-1.0, 0.74]], 1.5, GLASS, 0.2, 0.04));
  // a thin bright line along the canopy's base (the lit window edge that reads as glass in anime style)
  for (const g of pair(-1.75, 0.85, 0.828, 0.847, 0.76, 0.8, CABIN)) parts.push(rotZ(g, -0.025, -0.45, 0.8375));
  // a dark spine down the roof so the canopy reads as glass with a body-colour roof between
  parts.push(slab([0.25, 1.16], [-0.9, 1.21], 0.5, 0.03, 0.035, HULL));
  // hover pods (where wheels would be) with glowing emitters underneath
  for (const x of [-1.5, 1.45]) {
    parts.push(...pair(x - 0.48, x + 0.48, 0.1, 0.4, 0.78, 1.12, TRIM));
    parts.push(...pair(x - 0.38, x + 0.38, 0.07, 0.11, 0.82, 1.08, THRUST));
    parts.push(...pair(x - 0.4, x + 0.4, 0.36, 0.39, 1.12, 1.135, ACCENT));
  }
  // headlights: two slanted slits on the nose
  for (const s of [-1, 1]) parts.push(rotZ(box(2.28, 2.4, 0.5, 0.55, s > 0 ? 0.2 : -0.86, s > 0 ? 0.86 : -0.2, HEAD), -0.5, 2.34, 0.52));
  // a full-width tail bar + a small spoiler
  parts.push(box(-2.44, -2.36, 0.64, 0.7, -0.84, 0.84, TAIL));
  parts.push(box(-2.5, -2.12, 0.92, 0.96, -0.82, 0.82, TRIM));
  parts.push(...pair(-2.3, -2.22, 0.82, 0.93, 0.6, 0.66, TRIM));
  // side neon line
  parts.push(...pair(-2.05, 2.05, 0.35, 0.38, 0.95, 0.965, ACCENT));
  DESIGNS.kestrel = { dims: [2.45, 0.55, 0.53, 0.67], dims2: [1.0, 0.75, 0.08, 1.0] };
  return mergeGeometries(parts);
}

// "Mule": a cab / van with a car's side silhouette: a low nose wedge, a raked windscreen, a stepped roof (raised cargo
// box at the back), a dark glass band with a lit cabin, hover pods hanging below the body line, twin rear thrusters
function mule() {
  const parts = [];
  const P = [[-2.55, 0.34], [-2.42, 0.24], [1.95, 0.24], [2.6, 0.4], [2.66, 0.74], [2.25, 0.9], [1.25, 1.5], [-0.75, 1.56], [-0.95, 1.8], [-2.48, 1.84], [-2.62, 1.66]];
  parts.push(profile(P, 2.1, HULL, 0.1, 0.05));
  // windscreen on the raked line, slightly proud, and the side glass band (a parallelogram following the rake)
  parts.push(slab([2.22, 0.94], [1.28, 1.48], 1.86, 0.03, 0.035, GLASS));
  parts.push(profile([[-0.7, 1.0], [1.55, 1.0], [1.12, 1.42], [-0.7, 1.46]], 2.13, GLASS, 0, 0));
  parts.push(...pair(-0.6, 1.2, 1.4, 1.43, 1.066, 1.072, CABIN));             // cabin light line along the top of the glass
  parts.push(profile([[-0.7, 0.98], [1.57, 0.98], [1.6, 0.95], [-0.7, 0.95]], 2.135, TRIM, 0, 0)); // window sill
  parts.push(...pair(-2.35, 1.95, 0.5, 0.53, 1.05, 1.07, ACCENT));            // side stripe (low on the body)
  parts.push(box(-2.2, -1.2, 1.84, 1.96, -0.42, 0.42, ACCENT));                // roof sign on the cargo box
  parts.push(box(2.62, 2.68, 0.58, 0.66, -0.9, 0.9, HEAD));                    // headlight bar under the nose
  parts.push(...pair(-2.69, -2.61, 0.62, 1.5, 0.78, 0.96, TAIL));              // tail light columns
  for (const sgn of [-1, 1]) {                                                 // twin rear thrusters
    parts.push(box(-2.45, -1.25, 0.06, 0.4, sgn * 0.42 - 0.28, sgn * 0.42 + 0.28, TRIM));
    parts.push(box(-2.48, -2.44, 0.11, 0.35, sgn * 0.42 - 0.22, sgn * 0.42 + 0.22, THRUST));
  }
  for (const x of [1.45, -1.55]) {                                             // pods hang below the body line
    parts.push(...pair(x - 0.5, x + 0.5, -0.12, 0.3, 0.92, 1.24, TRIM));
    parts.push(...pair(x - 0.4, x + 0.4, -0.15, -0.11, 0.96, 1.2, THRUST));
  }
  DESIGNS.mule = { dims: [2.65, 0.75, 0.62, 1.05], dims2: [1.15, 0.87, -0.12, 1.1] };
  return mergeGeometries(parts);
}

// "Wasp": a hover bike with a hunched rider, ring thrusters fore and aft
function wasp() {
  const parts = [];
  parts.push(profile([[-1.3, 0.38], [0.95, 0.32], [1.45, 0.48], [0.95, 0.7], [0.25, 0.76], [-0.45, 0.92], [-1.35, 0.66]], 0.56, HULL, 0.08, 0.04));
  parts.push(rotZ(box(-0.6, 0.12, 0.8, 1.16, -0.21, 0.21, HULL), -0.35, -0.25, 0.9)); // rider's back
  parts.push(rotZ(box(-0.05, 0.75, 0.7, 0.84, -0.27, 0.27, TRIM), -0.25, 0.3, 0.77)); // arms / fairing
  parts.push(box(0.05, 0.36, 1.04, 1.3, -0.13, 0.13, HULL));                            // helmet
  parts.push(box(0.355, 0.375, 1.13, 1.19, -0.1, 0.1, ACCENT));                          // visor slit
  for (const x of [1.15, -1.1]) {                                                         // ring thrusters
    parts.push(box(x - 0.34, x + 0.34, 0.16, 0.3, -0.42, 0.42, TRIM));
    parts.push(box(x - 0.27, x + 0.27, 0.13, 0.17, -0.34, 0.34, THRUST));
  }
  parts.push(box(1.42, 1.48, 0.47, 0.53, -0.16, 0.16, HEAD));
  parts.push(box(-1.4, -1.33, 0.6, 0.66, -0.2, 0.2, TAIL));
  parts.push(...pair(-1.2, 1.3, 0.44, 0.47, 0.285, 0.295, ACCENT));
  DESIGNS.wasp = { dims: [1.48, 0.0, 0.5, 0.63], dims2: [0.42, 0.18, 0.13, 0.7] };
  return mergeGeometries(parts);
}

// ------------------------------------------------------------------------------------------------
// createTraffic
// ------------------------------------------------------------------------------------------------

export function createTraffic(parent, opts = {}) {
  const hi = (opts.quality || "high") !== "low";
  const R = mulberry32(0x5eed + 7);
  const rr = (a, b) => a + (b - a) * R();
  const group = new THREE.Group(); group.name = "traffic";
  parent.add(group);
  const trash = [];

  // sky / fog: the city's own uniform objects when given (opts.sky, so the two can't drift apart), else defaults
  const sky = opts.sky || {};
  const U = {
    uTime: { value: 0 },
    uI: { value: 0 },
    uSkyHorizon: sky.uSkyHorizon || { value: lin(0x1c1244) },
    uSkyHorizonC: sky.uSkyHorizonC || { value: lin(0x3a0c3c) },
    uSkyZenith: sky.uSkyZenith || { value: lin(0x030207) },
    uSkyBelow: sky.uSkyBelow || { value: lin(0x3a1438) },
    uFogDensity: sky.uFogDensity || { value: 0.0052 },
    uFr: { value: new THREE.Vector4(1.22, 4.44, Math.tan(THREE.MathUtils.degToRad(19)), -0.023) },
    uBand: { value: new THREE.Vector4(0, 0, 0, 0) },
    uWin: { value: new THREE.Vector4(0.1, 1.0, 1, 0) },
  };
  const shader = (o) => {
    const m = new THREE.ShaderMaterial({
      uniforms: { ...U, ...(o.uniforms || {}) },
      vertexShader: COMMON + o.vs, fragmentShader: COMMON + o.fs,
      transparent: !!o.additive, depthWrite: !o.additive,
      blending: o.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      side: o.side ?? THREE.FrontSide,
    });
    trash.push(m);
    return m;
  };
  const add = (obj, order) => { obj.renderOrder = order; obj.frustumCulled = false; group.add(obj); if (obj.geometry) trash.push(obj.geometry); return obj; };

  // ---------------------------------------------------------------- lanes and cars
  // paint (linear): mostly near-black with a tint, so the neon reflections carry the shape; a few cab yellows / pale
  const PAINT = [lin(0x17181f), lin(0x1d1228), lin(0x0e1c22), lin(0x26111a), lin(0x121214), lin(0x2a2c36)];
  const CAB = lin(0xe6a414), PALE = lin(0x8c8e9c); // (light paints get a self-lit term in the shader so they read)
  const GLOW = [lin(PALETTE.pink), lin(0x3ee8ff), lin(PALETTE.green), lin(0xa64dff), lin(0x2f7dff)];
  const ACC = [lin(PALETTE.pink), lin(0x3ee8ff), lin(PALETTE.green), lin(0xffc93d)];

  const cars = { kestrel: [], mule: [], wasp: [] };
  const all = [];
  // lane: { y, z, yaw, speed (signed), span, wiggle, n cars, design?, lc?, hs? }
  const lanes = [];
  // the span must be long enough that a car wraps around well outside the frame (the frame is ~1.2 x depth wide
  // on a 16:10 screen; the camera also drifts / moves to other perches), and it sets how often a car comes by
  const spanFor = (z, extra) => Math.max(2.8 * (Math.abs(z) + 10), 0) + extra;
  // Heights of the close / near lanes are planned in SCREEN terms (lane class 1..4, aLane.x = the screen fraction where
  // the car's top edge lands, 0 = top) and turned into metres in the vertex shader from the live shot (setFraming:
  // main.js passes its master shot on every resize / cut), so they hold on every aspect, phone included. In the
  // landscape master her hair fills the top centre, the bubble sits right of her head at f ~0.27-0.38 and the stairwell
  // hut / water tank tops sit at f ~0.42-0.46, so: an upper band (top edge f 0.08-0.15, just under the header, the car's body stays above the
  // bubble) and a lower band (top edge f 0.41-0.45: under the bubble, over the hut / tank tops). Fly-bys run in the upper
  // band and only cross the open right part of the frame (uWin: outside it they dive in from / climb back to the sky).
  // Portrait phones shift the bands (uBand) and turn the fly-bys / hero off (they'd only cross her hair).
  const TOP = { kestrel: 1.42, mule: 2.15, wasp: 1.5 }; // top edge above the car's origin, incl. the bob
  // 0) a rare hero pass just past the roof's end (z -8.5), banked toward the camera (roof + canopy show), its top just
  // above the bubble: ~a third of the screen is car for ~1.5 s, every ~40 s
  lanes.push({ f: 0.17, cls: 4, z: -8.5, yaw: 0, speed: -10, span: 400, wiggle: 0.3, n: 1, design: "kestrel", lc: 0, hs: 0, hero: true });
  // 1) close fly-bys, 12-20 m behind her: one car each, slow enough to read (~2 s on screen), one every ~11-17 s;
  // they travel right-to-left / left-to-right inside the window so the nose leads into open sky
  const fly = [
    { z: -12, f: 0.09, speed: -7, T: 11.5, design: "kestrel", lc: 0.8 },
    { z: -16, f: 0.1, speed: 8.5, T: 13, design: "mule", lc: 0 },
    { z: -20, f: 0.085, speed: -10, T: 17, design: "wasp", lc: 0.6 },
  ].slice(0, hi ? 3 : 2);
  for (const L of fly) lanes.push({ f: L.f, cls: 3, z: L.z, yaw: rr(-0.04, 0.04), speed: L.speed, span: Math.abs(L.speed) * L.T, wiggle: 0.3, n: 1, fly: true, design: L.design, lc: L.lc, hs: 0 });
  // 2) the near band, 24-48 m: big enough on screen to read every shape; both directions, alternating bands, 2-3 cars
  // per lane so a couple are always crossing an open patch of sky
  const nearZ = hi ? [-24, -28, -33, -38, -44, -48, -30, -41] : [-26, -36, -45, -31];
  const nearF = hi ? [0.09, 0.42, 0.12, 0.44, 0.08, 0.41, 0.15, 0.45] : [0.09, 0.43, 0.08, 0.42];
  nearZ.forEach((z, i) => {
    const speed = rr(11, 18) * (i % 2 ? 1 : -1);
    lanes.push({ f: nearF[i], cls: nearF[i] < 0.3 ? 1 : 2, z, yaw: rr(-0.1, 0.1), speed, span: spanFor(z, rr(50, 120)), wiggle: rr(0.4, 1.0), n: hi ? 3 : 2, lc: R() < 0.6 ? rr(1.0, 1.8) : 0, hs: R() < 0.4 ? rr(0.6, 1.2) : 0 });
  });
  // 3) mid band, 55-140 m: denser streams, higher up, a bit faster
  const midZ = hi ? [-58, -70, -84, -100, -118, -138] : [-65, -95, -130];
  midZ.forEach((z, i) => {
    const speed = rr(14, 26) * (i % 2 ? -1 : 1);
    lanes.push({ y: rr(1, 4) + Math.abs(z) * rr(0.0, 0.07), z, yaw: rr(-0.1, 0.1), speed, span: spanFor(z, rr(30, 90)), wiggle: rr(0.5, 1.6), n: 3, lc: R() < 0.5 ? rr(1.5, 3) : 0, hs: R() < 0.3 ? rr(1, 2.5) : 0 });
  });

  // the design mix: ~25% vans, at least 2 bikes per quality, the rest coupes (fixed per lane where given)
  const mixed = lanes.filter((L) => !L.design).reduce((n, L) => n + L.n, 0);
  const deck = [];
  const nMule = Math.round(mixed * 0.22), nWasp = Math.max(2, Math.round(mixed * 0.14));
  for (let i = 0; i < mixed; i++) deck.push(i < nMule ? "mule" : i < nMule + nWasp ? "wasp" : "kestrel");
  for (let i = deck.length - 1; i > 0; i--) { const j = (R() * (i + 1)) | 0; [deck[i], deck[j]] = [deck[j], deck[i]]; }
  for (const L of lanes) {
    const ph0 = L.hero ? 0.05 : R(); // (hero: first pass ~18 s after load, then every 40 s)
    for (let j = 0; j < L.n; j++) {
      const design = L.design || deck.pop() || "kestrel";
      const isBike = design === "wasp";
      let paint = PAINT[(R() * PAINT.length) | 0];
      if (design === "mule" && R() < 0.45) paint = CAB;
      else if (R() < 0.08) paint = PALE;
      const car = {
        design,
        // (a bike rides a parallel line 3 m nearer at the lane's speed, so it never runs through a car in its lane;
        // screen-planned lanes jitter their fraction instead of metres)
        lane: [L.cls ? L.f + (L.fly || L.hero ? 0 : rr(-0.015, 0.015)) + (isBike && !L.fly ? 0.02 : 0) : L.y + rr(-0.6, 0.6) - (isBike ? 0.8 : 0),
          L.z + (L.fly || L.hero ? 0 : rr(-1.2, 1.2)) + (isBike && !L.fly ? 3 : 0), L.speed, (ph0 + j / L.n + (L.hero ? 0 : rr(-0.08, 0.08)) + 1) % 1],
        lane2: [L.yaw, L.span, L.wiggle * (isBike ? 1.6 : 1), R()],
        lane3: [L.lc || 0, L.hs || 0, L.hero || L.fly ? 0 : rr(2, 6), (L.cls || 0) + (L.cls ? Math.min(TOP[design], 9.9) / 10 : 0)],
        paint, glow: GLOW[(R() * GLOW.length) | 0], acc: design === "mule" && paint === CAB ? ACC[3] : ACC[(R() * 3) | 0],
        head: R() < 0.7 ? lin(0xfff1dc) : lin(0xbfe6ff),
        cabin: R(),
      };
      cars[design].push(car); all.push(car);
    }
  }

  // ---------------------------------------------------------------- bodies: one InstancedMesh per design
  const bodyMat = shader({
    vs: CAR_FRAME + /* glsl */ `
      attribute float aPart;
      attribute vec3 aPaint; attribute vec3 aGlow; attribute vec3 aAcc; attribute vec4 aHead;
      varying vec3 vW; varying vec3 vN; varying vec3 vL; varying float vPart;
      varying vec3 vPaint; varying vec3 vGlow; varying vec3 vAcc; varying vec4 vHead; varying vec3 vF;
      void main(){
        vec3 O, F, Up, Rt; carFrame(O, F, Up, Rt);
        vec3 p = O + F * position.x + Up * position.y + Rt * position.z;
        vN = normalize(F * normal.x + Up * normal.y + Rt * normal.z);
        vW = p; vL = position; vPart = aPart; vF = F;
        vPaint = aPaint; vGlow = aGlow; vAcc = aAcc; vHead = aHead;
        gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
      }`,
    fs: /* glsl */ `
      varying vec3 vW; varying vec3 vN; varying vec3 vL; varying float vPart;
      varying vec3 vPaint; varying vec3 vGlow; varying vec3 vAcc; varying vec4 vHead; varying vec3 vF;
      // the fixed "neon city" the paint reflects
      vec3 env(vec3 r, vec3 wp){
        vec3 c = mix(vec3(0.32, 0.03, 0.15), vec3(0.01, 0.012, 0.04), smoothstep(-0.3, 0.12, r.y)); // city glow below, dark sky above
        c += vec3(0.06, 0.16, 0.5) * smoothstep(0.25, 0.85, r.y);                                   // blue upper sky
        c += vec3(1.0, 0.4, 0.65) * 0.8 * exp(-abs(r.y + 0.03) * 45.0);                             // the horizon streak
        // neon signs the car passes: bands fixed in the world, so they slide over the body as it moves
        float sg = sin(wp.x * 0.33 + wp.z * 0.21) * sin(wp.x * 0.071 + 1.7);
        float band = smoothstep(0.55, 0.8, sg) * smoothstep(-0.3, 0.1, r.y) * (1.0 - smoothstep(0.3, 0.6, r.y));
        c += mix(vec3(1.0, 0.04, 0.3), vec3(0.15, 0.85, 1.0), step(0.0, sin(wp.x * 0.05 + wp.z * 0.13))) * band * 1.4;
        return c;
      }
      void main(){
        vec3 N = normalize(vN);
        vec3 V = normalize(cameraPosition - vW);
        float ndv = clamp(dot(N, V), 0.0, 1.0);
        float fres = pow(1.0 - ndv, 3.0);
        vec3 r = reflect(-V, N);
        float part = vPart;
        vec3 col;
        if (part > 6.5) {
          // cabin light / lit window edge: warm or dashboard cyan, brightest seen side-on
          col = mix(vec3(0.35, 0.9, 1.0), vec3(1.0, 0.62, 0.3), step(0.6, vHead.w)) * (1.1 + 0.4 * uI);
        } else if (part < 0.5 || part > 5.5) {
          bool glass = part > 5.5;
          float lightPaint = glass ? 0.0 : smoothstep(0.08, 0.25, dot(vPaint, vec3(0.3, 0.55, 0.15)));
          // glass roof panels facing straight up read as body (the canopy has a painted spine anyway)
          vec3 base = glass ? vec3(0.002, 0.0025, 0.005) : vPaint;
          // reflections fade toward the lower body (a two-tone read even on one flat facet)
          float hk = glass ? 1.0 : 0.3 + 0.7 * smoothstep(0.25, 1.0, vL.y);
          col = base * (0.08 + 0.3 * max(N.y, 0.0));                                   // soft sky fill
          col += base * uSkyBelow * 2.0 * max(-N.y, 0.0);                               // city light from below
          col += vGlow * 0.22 * max(-N.y, 0.0) * smoothstep(0.75, 0.2, vL.y);           // its own underglow on the belly
          // light paints (cab yellow / pale) are lit by the city: a self-lit term, and less of the pink env washing them
          col += base * lightPaint * (0.2 + 0.16 * max(N.y, 0.0));
          // near cars: cap the env reflection (up close it washed the body out to a flat teal) and darken the lower body /
          // belly toward suit black, so the silhouette and the neon trim carry the read (Capri's flat cel contrast)
          float near = 1.0 - smoothstep(16.0, 46.0, length(cameraPosition - vW));
          col += env(r, vW) * mix(glass ? 0.03 : 0.03, glass ? 0.3 : mix(0.9, 0.45, lightPaint), fres) * hk * mix(1.0, glass ? 0.8 : 0.55, near);
          if (!glass) {
            col *= mix(1.0, mix(0.4, 1.0, smoothstep(0.2, 0.8, vL.y)), near);
            col *= 1.0 - near * 0.55 * smoothstep(-0.1, -0.7, N.y);
            // light paints: a vertical value gradient, two dark door / panel gaps on the flanks, a darker roof box
            float flank = step(0.6, abs(N.z));
            float gap = max(1.0 - smoothstep(0.012, 0.026, abs(vL.x - 0.3)), 1.0 - smoothstep(0.012, 0.026, abs(vL.x + 0.85))) * flank * step(0.3, vL.y) * step(vL.y, 1.45);
            col *= mix(1.0, mix(0.5, 1.0, smoothstep(0.25, 1.3, vL.y)) * (1.0 - 0.75 * gap) * (vL.y > 1.58 ? 0.5 : 1.0), lightPaint);
          }
          if (glass) {
            // a sliding diagonal reflection streak on the glass
            float st = smoothstep(0.86, 1.0, sin((vW.x + vW.y * 1.7 - vW.z * 0.6) * 1.1));
            col += vec3(0.75, 0.55, 0.9) * st * 0.18 * (0.3 + fres);
          }
          // fixed rim lights: hot pink from behind-left of the city, cyan from the right
          float rimK = glass ? 0.35 : 1.0;
          col += vec3(1.0, 0.05, 0.35) * pow(1.0 - ndv, 2.0) * max(dot(N, normalize(vec3(-0.7, 0.25, -0.5))), 0.0) * 0.55 * rimK;
          col += vec3(0.1, 0.7, 1.0) * pow(1.0 - ndv, 2.0) * max(dot(N, normalize(vec3(0.8, 0.35, 0.3))), 0.0) * 0.4 * rimK;
          if (glass) {
            // a dim cabin light behind the glass (dashboard cyan or warm), stronger when seen straight on
            vec3 cab = mix(vec3(0.1, 0.55, 0.75), vec3(0.9, 0.45, 0.2), step(0.6, vHead.w));
            col += cab * 0.09 * (0.4 + 0.6 * ndv);
          }
        } else if (part < 1.5) {
          col = vHead.rgb * 3.2;                                                         // headlights
        } else if (part < 2.5) {
          col = vec3(1.0, 0.03, 0.07) * 2.4;                                             // tail lights
        } else if (part < 3.5) {
          col = vAcc * 1.6;                                                              // neon trim
        } else if (part < 4.5) {
          col = vGlow * (2.4 + 0.8 * uI);                                                // thruster emitters
        } else {
          col = vPaint * 0.25 * (0.2 + 0.5 * max(N.y, 0.0)) + vGlow * 0.12 * max(-N.y, 0.0) // dark trim / pods
              + env(r, vW) * mix(0.02, 0.35, fres);
        }
        gl_FragColor = vec4(fogMix(col, vW), 1.0);
        ${OUT}
      }`,
  });

  const instAttrs = (geo, list) => {
    const n = list.length;
    const lane = new Float32Array(n * 4), lane2 = new Float32Array(n * 4), lane3 = new Float32Array(n * 4);
    const paint = new Float32Array(n * 3), glow = new Float32Array(n * 3), acc = new Float32Array(n * 3), head = new Float32Array(n * 4);
    list.forEach((c, i) => {
      lane.set(c.lane, i * 4); lane2.set(c.lane2, i * 4); lane3.set(c.lane3, i * 4);
      paint.set([c.paint.r, c.paint.g, c.paint.b], i * 3);
      glow.set([c.glow.r, c.glow.g, c.glow.b], i * 3);
      acc.set([c.acc.r, c.acc.g, c.acc.b], i * 3);
      head.set([c.head.r, c.head.g, c.head.b, c.cabin], i * 4);
    });
    geo.setAttribute("aLane", new THREE.InstancedBufferAttribute(lane, 4));
    geo.setAttribute("aLane2", new THREE.InstancedBufferAttribute(lane2, 4));
    geo.setAttribute("aLane3", new THREE.InstancedBufferAttribute(lane3, 4));
    geo.setAttribute("aPaint", new THREE.InstancedBufferAttribute(paint, 3));
    geo.setAttribute("aGlow", new THREE.InstancedBufferAttribute(glow, 3));
    geo.setAttribute("aAcc", new THREE.InstancedBufferAttribute(acc, 3));
    geo.setAttribute("aHead", new THREE.InstancedBufferAttribute(head, 4));
  };

  const builders = { kestrel, mule, wasp };
  const bodies = {};
  let tris = 0;
  for (const name of Object.keys(builders)) {
    const geo = builders[name]();
    if (!cars[name].length) continue;
    instAttrs(geo, cars[name]);
    bodies[name] = add(new THREE.InstancedMesh(geo, bodyMat, cars[name].length), 2);
    tris += (geo.attributes.position.count / 3) * cars[name].length;
  }

  // ink outline: an inverted hull per design (back faces, inflated per axis away from the car's centre so the faceted
  // geometry doesn't crack, ~2-4 px wide at any distance), near-black, fogged with the body: the cars get the same hard
  // dark contour as Capri's paper outline instead of melting into the haze. +3 draw calls.
  const inkMat = shader({
    side: THREE.BackSide,
    vs: CAR_FRAME + /* glsl */ `
      varying vec3 vW;
      void main(){
        vec3 O, F, Up, Rt; carFrame(O, F, Up, Rt);
        vec3 p0 = O + F * position.x + Up * position.y + Rt * position.z;
        float w = 0.018 + 0.0011 * length(cameraPosition - p0);
        vec3 q = position + sign(position - vec3(0.0, 0.72, 0.0)) * w;
        vec3 p = O + F * q.x + Up * q.y + Rt * q.z;
        vW = p;
        gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
      }`,
    fs: /* glsl */ `
      varying vec3 vW;
      void main(){
        gl_FragColor = vec4(fogMix(vec3(0.006, 0.004, 0.012), vW), 1.0);
        ${OUT}
      }`,
  });
  for (const name of Object.keys(bodies)) add(new THREE.InstancedMesh(bodies[name].geometry, inkMat, cars[name].length), 2);

  // ---------------------------------------------------------------- light effects for the near cars (one draw call)
  // Each car carries a set of quads; aQ.x = kind, aQ.yzw = per-kind params. Kinds:
  //   0 underglow card (flat, under the car)        1 headlight flare (billboard, one per side)
  //   2 tail-light flare (billboard, one per side)  3 headlight beam (axial billboard, forward, in the rain)
  //   4 thruster plume / haze under the car (axial) 5 light trail (axial ribbon behind)
  {
    const quads = [[0, 0], [1, 1], [1, -1], [2, 1], [2, -1], [3, 1], [3, -1], [4, 0], [5, 1], [5, -1]];
    const pos = [], q = [], idx = [];
    quads.forEach(([k, s], i) => {
      pos.push(-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0);
      for (let v = 0; v < 4; v++) q.push(k, s, 0, 0);
      idx.push(i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3);
    });
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("aQ", new THREE.Float32BufferAttribute(q, 4));
    geo.setIndex(idx);
    const list = all;
    instAttrs(geo, list);
    const dims = new Float32Array(list.length * 4), dims2 = new Float32Array(list.length * 4);
    list.forEach((c, i) => { dims.set(DESIGNS[c.design].dims, i * 4); dims2.set(DESIGNS[c.design].dims2, i * 4); });
    geo.setAttribute("aDim", new THREE.InstancedBufferAttribute(dims, 4));
    geo.setAttribute("aDim2", new THREE.InstancedBufferAttribute(dims2, 4));
    geo.instanceCount = list.length;
    const mat = shader({
      additive: true, side: THREE.DoubleSide,
      uniforms: { uBeams: { value: hi ? 1 : 0 } },
      vs: CAR_FRAME + /* glsl */ `
        attribute vec4 aQ; attribute vec4 aDim; attribute vec4 aDim2;
        attribute vec3 aPaint; attribute vec3 aGlow; attribute vec3 aAcc; attribute vec4 aHead;
        uniform float uBeams;
        varying vec2 vUv; varying vec3 vCol; varying float vKind; varying vec3 vW;
        void main(){
          vec3 O, F, Up, Rt; carFrame(O, F, Up, Rt);
          float k = aQ.x, sd = aQ.y;
          vec3 toCam = normalize(cameraPosition - O);
          vec3 camR = normalize(cross(toCam, vec3(0.0, 1.0, 0.0)));
          vec3 camU = cross(camR, toCam);
          float facing = dot(F, toCam);       // > 0: nose toward the viewer
          float hl = aDim.x, hw = aDim2.x;
          vec3 p; vec3 col;
          vUv = position.xy * 2.0;            // -1..1
          if (k < 0.5) {                      // underglow card
            p = O + Up * (aDim2.z - 0.06) + F * (position.x * hl * 2.5) + Rt * (position.y * hw * 3.4);
            col = aGlow * (0.22 + 0.12 * uI);
          } else if (k < 1.5) {               // headlight flare: wide anamorphic streak, strong only when facing us
            vec3 a = O + F * (hl + 0.05) + Up * aDim.z + Rt * (aDim.y * sd);
            float sz = mix(0.3, 0.8, smoothstep(-0.1, 0.8, facing));
            p = a + camR * position.x * sz * 2.4 + camU * position.y * sz * 0.6 + toCam * 0.3;
            col = aHead.rgb * mix(0.2, 1.2, smoothstep(-0.2, 0.7, facing)) * step(0.01, aDim.y + 0.5 - abs(sd) * 0.5);
            if (aDim.y < 0.01 && sd < 0.0) col *= 0.0;   // single headlight (bike)
          } else if (k < 2.5) {               // tail-light flare
            vec3 a = O - F * (hl + 0.05) + Up * aDim.w + Rt * (aDim2.y * sd);
            float sz = mix(0.25, 0.6, smoothstep(-0.1, 0.8, -facing));
            p = a + camR * position.x * sz * 2.2 + camU * position.y * sz * 0.6 + toCam * 0.3;
            col = vec3(1.0, 0.04, 0.08) * mix(0.25, 1.1, smoothstep(-0.2, 0.7, -facing));
          } else if (k < 3.5) {               // headlight beam: a long soft cone forward, faced toward the camera
            float Lb = 5.5 * aDim2.w, along = position.x + 0.5;
            vec3 a = O + F * (hl + 0.1) + Up * (aDim.z - 0.05) + Rt * (aDim.y * sd);
            vec3 side = normalize(cross(F, toCam));
            p = a + F * along * Lb - Up * along * Lb * 0.05 + side * position.y * mix(0.25, 1.4, along);
            // tinted toward the scene's cyan, and only when seen side-on (head-on it would be a lone grey smear)
            col = mix(aHead.rgb, vec3(0.35, 0.85, 1.0), 0.45) * 0.075 * uBeams * (1.0 - smoothstep(0.55, 0.9, abs(facing)));
            if (aDim.y < 0.01 && sd < 0.0) col *= 0.0;
          } else if (k < 4.5) {               // underglow haze: an elongated soft glow below the car, faced to us
            vec3 side = normalize(cross(F, toCam));
            vec3 up2 = normalize(cross(side, F));
            p = O + Up * (aDim2.z - 0.25) + F * position.x * hl * 3.0 + up2 * position.y * 1.4;
            col = aGlow * (0.16 + 0.1 * uI);
          } else {                            // light trail: thin ribbon behind the tail lights
            float s = abs(aLane.z), Lt = s * 0.35 + 1.0, along = position.x + 0.5; // 0 = at the car
            vec3 a = O - F * (hl + 0.1) + Up * aDim.w + Rt * (aDim2.y * sd);
            vec3 side = normalize(cross(F, toCam));
            p = a - F * along * Lt + side * position.y * 0.12;
            col = mix(vec3(1.0, 0.03, 0.08), aGlow, 0.35) * 0.16 * uBeams;
          }
          vCol = col; vKind = k; vW = p;
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fs: /* glsl */ `
        varying vec2 vUv; varying vec3 vCol; varying float vKind; varying vec3 vW;
        void main(){
          float k = vKind, a;
          vec2 u = vUv;
          if (k < 0.5) {
            float d = length(u); a = pow(max(1.0 - d, 0.0), 3.0);
          } else if (k < 2.5) {
            // a tight core + a horizontal streak
            float core = exp(-dot(u * vec2(3.2, 1.0), u * vec2(3.2, 1.0)) * 6.0);
            float streak = exp(-u.y * u.y * 30.0) * pow(max(1.0 - abs(u.x), 0.0), 2.0) * 0.35;
            a = core * 1.6 + streak;
          } else if (k < 3.5) {
            float along = u.x * 0.5 + 0.5;
            a = (1.0 - smoothstep(0.0, 1.0, abs(u.y))) * pow(1.0 - along, 1.6) * smoothstep(0.0, 0.08, along);
          } else if (k < 4.5) {
            a = pow(max(1.0 - length(u), 0.0), 2.0);
          } else {
            float along = u.x * 0.5 + 0.5;
            a = (1.0 - abs(u.y)) * pow(1.0 - along, 2.0);
          }
          gl_FragColor = vec4(vCol * a * fogKeep(vW), 1.0);
          ${OUT}
        }`,
    });
    add(new THREE.Mesh(geo, mat), 7);
  }

  // ---------------------------------------------------------------- far field: head / tail light pairs (one draw call)
  {
    const n = hi ? 34 : 16;
    const geo = new THREE.PlaneGeometry(1, 1);
    const lane = new Float32Array(n * 4), lane2 = new Float32Array(n * 4), hc = new Float32Array(n * 4);
    const trails = [lin(0xff2a3a), lin(0xff2a3a), lin(PALETTE.pink), lin(0x3ee8ff), lin(0xffb070)];
    for (let i = 0; i < n; i++) {
      const z = rr(-150, -300);
      const speed = rr(16, 40) * (R() < 0.5 ? -1 : 1);
      lane.set([rr(-12, 18 + Math.abs(z) * 0.1), z, speed, R()], i * 4);
      const tc = trails[(R() * trails.length) | 0];
      lane2.set([tc.r, tc.g, tc.b, rr(0.4, 0.75)], i * 4);
      const h = R() < 0.7 ? lin(0xfff2e0) : lin(0x9fdcff);
      hc.set([h.r, h.g, h.b, Math.abs(z) * 2.4 + 160], i * 4);
    }
    geo.setAttribute("aLane", new THREE.InstancedBufferAttribute(lane, 4));
    geo.setAttribute("aVc", new THREE.InstancedBufferAttribute(lane2, 4));
    geo.setAttribute("aHc", new THREE.InstancedBufferAttribute(hc, 4));
    const mat = shader({
      additive: true, side: THREE.DoubleSide,
      vs: /* glsl */ `attribute vec4 aLane; attribute vec4 aVc; attribute vec4 aHc;
        varying vec2 vUv; varying vec3 vTrail; varying vec3 vHead; varying float vL; varying float vSize; varying vec3 vW;
        void main(){
          float s = abs(aLane.z), dir = sign(aLane.z), span = aHc.w;
          float x = (mod(aLane.w * span + uTime * s, span) - span * 0.5) * dir;
          float L = s * 0.3 + 1.5, along = position.x + 0.5, size = aVc.w;
          vec3 head = vec3(x, aLane.x + sin(uTime * 0.4 + aLane.w * 40.0) * 0.6, aLane.y);
          vec3 p = head + vec3(dir * ((along - 1.0) * L + size * 1.5 * along), position.y * size * 2.0, 0.0);
          vUv = vec2(along, position.y * 2.0); vTrail = aVc.rgb; vHead = aHc.rgb; vL = L + size * 1.5; vSize = size; vW = p;
          gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
        }`,
      fs: /* glsl */ `varying vec2 vUv; varying vec3 vTrail; varying vec3 vHead; varying float vL; varying float vSize; varying vec3 vW;
        void main(){
          float dHead = (1.0 - vUv.x) * vL - vSize * 0.75;
          float r = abs(vUv.y);
          float head = exp(-(dHead * dHead) / (vSize * vSize * 0.25)) * exp(-r * r * 6.0);
          float trail = exp(-max(dHead, 0.0) / (vL * 0.35)) * smoothstep(0.0, 0.3, dHead) * (1.0 - smoothstep(0.0, 0.5, r));
          float tl = exp(-pow(dHead - vSize * 1.6, 2.0) / (vSize * vSize * 0.12)) * exp(-r * r * 8.0);
          vec3 col = vHead * head * 2.2 + vec3(1.0, 0.06, 0.08) * tl * 1.4 + vTrail * trail * 0.7;
          gl_FragColor = vec4(col * fogKeep(vW), 1.0);
          ${OUT}
        }`,
    });
    add(new THREE.InstancedMesh(geo, mat, n), 7);
  }

  // ---------------------------------------------------------------- API
  let target = 0, I = 0;
  function update(dt, t) {
    dt = Math.min(dt || 0, 0.1);
    I += (target - I) * (1 - Math.exp(-dt * 3.5));
    U.uTime.value = t; // (not wrapped: every lane would pop at the wrap; float32 holds cm steps for days at these speeds)
    U.uI.value = I;
  }
  function setIntensity(x) { target = THREE.MathUtils.clamp(+x || 0, 0, 1); }
  // setFraming({ pos, target, fov, aspect }): the shot the screen-planned lanes are laid out for (main.js: its master
  // shot on resize, the new shot on a camera cut). Not per frame, so the parallax drift still moves them.
  function setFraming(o) {
    if (!o || !o.pos || !o.target) return;
    const dz = Math.max(0.1, Math.hypot(o.pos.x - o.target.x, o.pos.z - o.target.z));
    U.uFr.value.set(o.pos.y, o.pos.z, Math.tan(THREE.MathUtils.degToRad((o.fov || 38) / 2)), (o.target.y - o.pos.y) / dz);
    const a = o.aspect || 1.6;
    if (a < 0.8) { U.uBand.value.set(0.25, 0.08, 0, 0); U.uWin.value.set(-1, 1, 0, 0); }        // phones: upper band below the bubble's zone, no fly-bys / hero
    else if (a < 1.25) { U.uBand.value.set(0.02, 0.02, 0.02, 0.02); U.uWin.value.set(0.25, 1.0, 1, 0); }
    else { U.uBand.value.set(0, 0, 0, 0); U.uWin.value.set(0.12, 1.0, 1, 0); }
  }
  function dispose() {
    parent.remove(group);
    for (const o of trash) o.dispose && o.dispose();
    trash.length = 0;
  }
  return { update, setIntensity, setFraming, dispose, group, stats: { cars: all.length, drawCalls: group.children.length, tris: Math.round(tris) } };
}
