// capri.js: loads the Capri glTF and brings it to life with procedural animation.
//
// The model is a static, unlit, low-poly mesh: no skeleton, no morph targets. At load time we build a
// small skeleton by hand (pivots measured from the mesh, see ANATOMY), compute soft skin weights on the CPU
// once (geodesic region labels over the welded mesh + smoothstep blends across each joint), and turn the
// meshes into SkinnedMeshes. Per frame we only set ~30 bone transforms; the GPU does the deformation.
// The black inverted-hull outline copies its weights from the nearest body vertex, so it deforms identically
// and stays glued on. A small onBeforeCompile patch adds hair wind, the 'glitch' slice jitter, a talking
// rim glow, and the outline colour pulse.
//
// Face: two big eye discs (separate overlay meshes on the green face plate, so they can be squashed for
// blinks and nudged for saccades) and a white zig-zag grin line (Object_5). The mouth "opens" by rotating a
// soft jaw region (the black lower face) around a hinge behind the grin; a duplicate lower grin rides the jaw
// and a pink mouth-interior strip stretches between the two lines.
//
// Contract: see README.md. createCapri(scene, { url, position, rotationY, scale, camera? }).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PALETTE } from './palette.js';

const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const rand = (a, b) => a + Math.random() * (b - a);
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// ---------------------------------------------------------------------------------------------
// Anatomy, measured from scene.bin (model space = world space of the glTF: metres, Y up, body faces +Z).
// [name, parent, pivot, end (for region labelling / joint direction), seed radius, joint blend radius, radial limit]
// ---------------------------------------------------------------------------------------------
const SKELETON = [
  ['hips', null, [0.0, 0.84, -0.05], [-0.01, 1.0, -0.06], 0.12, 0, 0],
  ['spine', 'hips', [-0.01, 1.0, -0.06], [-0.03, 1.2, -0.04], 0.11, 0.08, 0.3],
  ['chest', 'spine', [-0.03, 1.2, -0.04], [-0.07, 1.40, 0.03], 0.11, 0.08, 0.3],
  ['neck', 'chest', [-0.07, 1.40, 0.03], [-0.13, 1.47, 0.08], 0.05, 0.04, 0.1],
  ['head', 'neck', [-0.13, 1.47, 0.08], [-0.15, 1.75, 0.10], 0.06, 0.03, 0.12],
  ['shoulderR', 'chest', [-0.28, 1.37, 0.0], [-0.43, 1.02, -0.15], 0.035, 0.05, 0.09],
  ['elbowR', 'shoulderR', [-0.43, 1.02, -0.15], [-0.60, 0.72, -0.03], 0.035, 0.04, 0.07],
  ['handR', 'elbowR', [-0.60, 0.72, -0.03], [-0.72, 0.60, 0.10], 0.05, 0.03, 0.08],
  ['shoulderL', 'chest', [0.17, 1.34, -0.05], [0.37, 1.0, -0.19], 0.035, 0.05, 0.09],
  ['elbowL', 'shoulderL', [0.37, 1.0, -0.19], [0.48, 0.75, -0.12], 0.035, 0.04, 0.07],
  ['handL', 'elbowL', [0.48, 0.75, -0.12], [0.62, 0.63, -0.05], 0.05, 0.03, 0.08],
  ['thighR', 'hips', [-0.10, 0.76, -0.03], [-0.24, 0.63, 0.40], 0.085, 0.06, 0.13],
  ['shinR', 'thighR', [-0.24, 0.63, 0.40], [-0.24, 0.33, -0.02], 0.06, 0.04, 0.09],
  ['footR', 'shinR', [-0.24, 0.33, -0.02], [-0.23, 0.04, 0.0], 0.06, 0.03, 0.08],
  ['thighL', 'hips', [0.05, 0.84, 0.0], [0.05, 0.99, 0.43], 0.09, 0.06, 0.14],
  ['shinL', 'thighL', [0.05, 0.99, 0.43], [-0.56, 0.50, 0.58], 0.05, 0.04, 0.09],
];

export const ANATOMY = {
  headPivot: [-0.13, 1.47, 0.08],      // head turns around this (top of the neck)
  neckPivot: [-0.07, 1.40, 0.03],
  headCenter: [-0.12, 1.62, 0.12],     // centre of the skull, under the hair mass
  faceCenter: [-0.25, 1.53, 0.19],     // between the eyes and the mouth
  eyeA: [-0.201, 1.524, 0.273],        // the eye nearer the front (+Z)
  eyeB: [-0.296, 1.571, 0.118],
  mouth: [-0.233, 1.458, 0.155],       // centre of the white zig-zag grin (Object_5)
  mouthEnds: [[-0.261, 1.479, 0.114], [-0.205, 1.437, 0.195]],
  faceDirection: [-0.86, -0.31, 0.44], // where her face looks at rest (her head is turned ~63deg to her right, tilted down)
  seatPoint: [-0.05, 0.675, 0.13],     // under the hips at the front edge of the seat; ledge top = y 0.675
  shoulderR: [-0.28, 1.37, 0.0], elbowR: [-0.43, 1.02, -0.15], wristR: [-0.60, 0.72, -0.03],
  shoulderL: [0.17, 1.34, -0.05], elbowL: [0.37, 1.0, -0.19], wristL: [0.48, 0.75, -0.12],
  kneeR: [-0.24, 0.63, 0.40], ankleR: [-0.24, 0.33, -0.02], kneeL: [0.05, 0.99, 0.43], hips: [0.0, 0.84, -0.05],
};

// ---------------------------------------------------------------------------------------------
// POSTURES: whole-body poses layered under the procedural animation (see README "Postures").
// The mesh's rest pose is 'seated' (on a ledge, left knee up). A posture adds, per bone, a local rotation
// (Euler YXZ, radians, model axes: +X her left, +Y up, +Z the way her body faces) applied before the
// procedural layers, moves the wrist targets (chest rest frame, metres), offsets the whole rig from the
// seat point, and scales the idle leg swing. Switched by setPosture(), normally while she is invisible
// mid-teleport. Poses come from her construct form in the show (S02E04 crouch, S02E06 float).
// ---------------------------------------------------------------------------------------------
export const POSTURES = {
  // the mesh as modelled: sitting on a ledge edge, left knee up, right leg hanging
  seated: { faceCamera: true, maxTurn: 0.5, pivot: [0, 0.84, -0.05], grip: [0.15, 0.15] },
  // gargoyle crouch: feet on the ledge top (toes over the edge), knees up, hunched, forearms over the knees
  crouch: {
    pivot: [0, 0.84, -0.05], at: [0, 0.94, -0.27], faceCamera: true, maxTurn: 0.5, legSwing: 0, hips: [0.12, 0, 0], spine: [0.22, 0, 0], chest: [0.15, 0, 0], neck: [-0.44, 0, 0],
    legR: { foot: [-0.26, 0.6, 0.24], pole: [-0.45, 0.7, 1] }, legL: { foot: [0.24, 0.56, 0.42], pole: [0.45, 0.7, 1] }, footR: [-0.5, 0, 0],
    armR: { wrist: [-0.184, 1.084, 0.376] }, armL: { wrist: [0.065, 1.246, 0.231] }, handR: [1.0, 0], handL: [1.0, 0], grip: [0.85, 0.75],
  },
  // zero-g float (as in the show, she ignores gravity): upside down, cross-legged, hands holding her feet, body square
  // to the camera, slowly rolling clockwise / counterclockwise and drifting in front of the ledge
  float: {
    pivot: [0, 1.0, 0.1], at: [0, 1.75, 0.45], rootRot: [0, 0, Math.PI], legSwing: 0, faceCamera: true, faceBy: 'body', lookYawMax: 1.2,
    // drift: only a slow clockwise / counterclockwise roll about the axis pointing at the camera (2D), plus a gentle
    // sideways and up-down float
    drift: { rot: [0, 0, 0.6], pos: [0.16, 0.1, 0], f: [0.21, 0.13, 0.15] },
    spine: [0.18, 0, 0], chest: [0.12, 0, 0], neck: [-0.25, 0, 0],
    legR: { foot: [0.1, 0.66, 0.3], pole: [-1, 0.3, 0.4] }, legL: { foot: [-0.3, 0.68, 0.36], pole: [1, 0.3, 0.4] }, footR: [0.6, 0, 0],
    armR: { wrist: [-0.24, 0.74, 0.4] }, armL: { wrist: [0.16, 0.74, 0.4] }, handR: [0.6, 0], handL: [0.6, 0], grip: [0.95, 0.95],
  },
  // side recline along the ledge, facing out: on her right side, propped on her right elbow with her head on her hand,
  // left hand on her hip, left knee bent forward. ledgeYaw 0: main.js turns her square to the edge for this one.
  // (rig turned 90deg about Z: rig +Y = along the ledge to her right, rig +X = up)
  lie: {
    ledgeYaw: 0, pivot: [0, 0.84, -0.05], at: [0.4, 0.84, 0], rootRot: [0, 0, Math.PI / 2], legSwing: 0, faceCamera: true, maxTurn: 0.35,
    spine: [0, 0, -0.5], chest: [0, 0, -0.45], neck: [0.05, 0, -0.25],
    legR: { foot: [-0.14, -0.06, -0.02], pole: [0, 0, 1] }, legL: { foot: [-0.02, 0.28, 0.3], pole: [0.2, 0, 1] },
    armR: { wrist: [0.061, 1.379, 0.173] }, armL: { wrist: [0.22, 0.95, 0.18] }, handR: [0.8, 0.3], handL: [0.3, 0], grip: [0.55, 0.35],
  },
};

// ---------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------
class Spring { // critically-ish damped second order follower; frame-rate independent via substeps
  constructor(w, z, x = 0) { this.w = w; this.z = z; this.x = x; this.v = 0; this.t = x; }
  step(dt) { const a = this.w * this.w * (this.t - this.x) - 2 * this.z * this.w * this.v; this.v += a * dt; this.x += this.v * dt; }
}

function weldMap(pos) { // vertex -> canonical vertex with the identical position
  const n = pos.count, map = new Int32Array(n), key = new Map();
  for (let i = 0; i < n; i++) {
    const k = Math.round(pos.getX(i) * 1e5) + ',' + Math.round(pos.getY(i) * 1e5) + ',' + Math.round(pos.getZ(i) * 1e5);
    const j = key.get(k); if (j === undefined) { key.set(k, i); map[i] = i; } else map[i] = j;
  }
  return map;
}

function adjacency(geo, weld) { // neighbour lists over welded vertices (with edge lengths)
  const n = geo.attributes.position.count, idx = geo.index.array, P = geo.attributes.position;
  const nb = Array.from({ length: n }, () => []);
  const add = (a, b) => { a = weld[a]; b = weld[b]; if (a === b) return; const d = Math.hypot(P.getX(a) - P.getX(b), P.getY(a) - P.getY(b), P.getZ(a) - P.getZ(b)); nb[a].push(b, d); nb[b].push(a, d); };
  for (let i = 0; i < idx.length; i += 3) { add(idx[i], idx[i + 1]); add(idx[i + 1], idx[i + 2]); add(idx[i + 2], idx[i]); }
  return nb;
}

function components(n, nb, weld) {
  const comp = new Int32Array(n).fill(-1); let c = 0;
  for (let s = 0; s < n; s++) {
    const r = weld[s]; if (comp[r] >= 0) continue;
    const st = [r]; comp[r] = c;
    while (st.length) { const a = st.pop(); const l = nb[a]; for (let k = 0; k < l.length; k += 2) { const b = l[k]; if (comp[b] < 0) { comp[b] = c; st.push(b); } } }
    c++;
  }
  for (let i = 0; i < n; i++) comp[i] = comp[weld[i]];
  return { comp, count: c };
}

function segDist(p, a, b, out) { // distance from p to segment ab; out.t = param
  const abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
  const t = clamp(((p.x - a.x) * abx + (p.y - a.y) * aby + (p.z - a.z) * abz) / (abx * abx + aby * aby + abz * abz), 0, 1);
  if (out) out.t = t;
  return Math.hypot(p.x - a.x - abx * t, p.y - a.y - aby * t, p.z - a.z - abz * t);
}

class MinHeap { // (key, value) binary heap for Dijkstra
  constructor() { this.k = []; this.v = []; }
  push(k, v) { const K = this.k, V = this.v; let i = K.length; K.push(k); V.push(v); while (i > 0) { const p = (i - 1) >> 1; if (K[p] <= k) break; K[i] = K[p]; V[i] = V[p]; i = p; } K[i] = k; V[i] = v; }
  pop() { const K = this.k, V = this.v, top = V[0], last = K.length - 1; const k = K[last], v = V[last]; K.pop(); V.pop();
    if (last > 0) { let i = 0; for (;;) { let c = 2 * i + 1; if (c >= last) break; if (c + 1 < last && K[c + 1] < K[c]) c++; if (K[c] >= k) break; K[i] = K[c]; V[i] = V[c]; i = c; } K[i] = k; V[i] = v; }
    return top; }
  get size() { return this.k.length; }
}

function packWeights(n, wlists) { // [{bone:w}] -> skinIndex/skinWeight attributes (top 4)
  const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const e = Object.entries(wlists[i]).map(([b, w]) => [+b, w]).filter((x) => x[1] > 1e-4).sort((a, b) => b[1] - a[1]).slice(0, 4);
    let s = 0; for (const x of e) s += x[1];
    e.forEach((x, k) => { si[i * 4 + k] = x[0]; sw[i * 4 + k] = x[1] / (s || 1); });
    if (!e.length) sw[i * 4] = 1;
  }
  return { skinIndex: new THREE.Uint16BufferAttribute(si, 4), skinWeight: new THREE.Float32BufferAttribute(sw, 4) };
}

// ---------------------------------------------------------------------------------------------
export async function createCapri(scene, opts = {}) {
  const url = opts.url || '/capri_project_capricorn/scene.gltf';
  const gltf = await new GLTFLoader().loadAsync(url);
  gltf.scene.updateMatrixWorld(true);
  let bodySrc, eyeSrc, outlineSrc;
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const nm = o.material.name;
    if (nm === 'Main') bodySrc = o; else if (nm === 'Paper_Outline_Black') outlineSrc = o; else eyeSrc = o;
  });
  const bake = (m) => { const g = m.geometry.clone(); g.applyMatrix4(m.matrixWorld); return g; };
  const bodyGeo = bake(bodySrc), outlineGeo = bake(outlineSrc), mouthGeo = bake(eyeSrc);
  const MOUTH_SCALE = 1.3; // the grin is a few px wide at the default framing: enlarge it in the face plane

  // ----- root / rig -----
  const root = new THREE.Group(); root.name = 'capri';
  if (opts.position) root.position.copy(opts.position);
  root.rotation.y = opts.rotationY || 0;
  root.scale.setScalar(opts.scale || 1);
  scene.add(root);
  const rig = new THREE.Group(); root.add(rig);

  // ----- bones -----
  const bones = [], B = {}, rest = {};
  const mkBone = (name, parentName, pivot) => {
    const b = new THREE.Bone(); b.name = name; const p = v3(pivot);
    rest[name] = { pivot: p.clone() };
    if (parentName) { b.position.copy(p).sub(rest[parentName].worldPos); B[parentName].add(b); } else { b.position.copy(p); rig.add(b); }
    rest[name].worldPos = p.clone(); rest[name].pos = b.position.clone();
    B[name] = b; b.userData.index = bones.length; bones.push(b);
    return b;
  };
  for (const [n, p, piv] of SKELETON) mkBone(n, p, piv);
  const segs = SKELETON.map(([n, p, piv, end, seedR, blendR, radial]) => ({ name: n, parent: p, a: v3(piv), b: v3(end), seedR, blendR, radial, idx: B[n].userData.index }));

  // ----- analyse the body mesh -----
  const P = bodyGeo.attributes.position, N = bodyGeo.attributes.normal, nV = P.count;
  const weld = weldMap(P), nb = adjacency(bodyGeo, weld), { comp, count: nComp } = components(nV, nb, weld);
  const cInfo = Array.from({ length: nComp }, () => ({ n: 0, c: new THREE.Vector3(), min: new THREE.Vector3(1e9, 1e9, 1e9), max: new THREE.Vector3(-1e9, -1e9, -1e9), nrm: new THREE.Vector3(), verts: [] }));
  const tv = new THREE.Vector3();
  for (let i = 0; i < nV; i++) {
    const ci = cInfo[comp[i]]; tv.fromBufferAttribute(P, i); ci.n++; ci.c.add(tv); ci.min.min(tv); ci.max.max(tv); ci.verts.push(i);
    ci.nrm.x += N.getX(i); ci.nrm.y += N.getY(i); ci.nrm.z += N.getZ(i);
  }
  cInfo.forEach((ci) => { ci.c.divideScalar(ci.n); ci.diag = ci.min.distanceTo(ci.max); });
  const order = cInfo.map((_, i) => i).sort((a, b) => cInfo[b].n - cInfo[a].n);
  const role = new Array(nComp).fill('prop');
  role[order[0]] = 'main';
  const faceC = v3(ANATOMY.faceCenter), eyeA = v3(ANATOMY.eyeA), eyeB = v3(ANATOMY.eyeB);
  const irisComp = {};
  for (const ci of order.slice(1)) {
    const I = cInfo[ci];
    if (I.n >= 250 && I.diag > 0.4 && I.c.y > 1.55) role[ci] = I.c.x > 0 ? 'hairBack' : 'hairFront';
    else if (I.n >= 200 && I.diag < 0.2) { const k = I.c.distanceTo(eyeA) < I.c.distanceTo(eyeB) ? 'A' : 'B'; role[ci] = 'iris' + k; irisComp[k] = ci; }
  }
  for (const ci of order.slice(1)) {
    if (role[ci] !== 'prop') continue; const I = cInfo[ci];
    const dA = I.c.distanceTo(eyeA), dB = I.c.distanceTo(eyeB);
    if (I.n < 40 && I.diag < 0.3 && Math.min(dA, dB) < 0.05) role[ci] = 'lid' + (dA < dB ? 'A' : 'B');
    else if (I.c.distanceTo(faceC) < 0.2 && I.c.y > 1.4) role[ci] = 'face';
    else if (I.n > 40 && I.c.y > 0.7 && I.c.x < -0.35 && I.max.z < 0) role[ci] = 'shardR';
    else if (I.n > 40 && I.c.y > 0.7 && I.c.x > 0.3 && I.max.z < 0) role[ci] = 'shardL';
  }

  // ----- face frame (from the eye discs) -----
  const nA = cInfo[irisComp.A].nrm.clone().normalize(), nB = cInfo[irisComp.B].nrm.clone().normalize();
  const faceDir = v3(ANATOMY.faceDirection).normalize();
  if (nA.dot(faceDir) < 0) nA.negate(); if (nB.dot(faceDir) < 0) nB.negate();
  const eyeLine = cInfo[irisComp.A].c.clone().sub(cInfo[irisComp.B].c).normalize();
  const faceUp = new THREE.Vector3().crossVectors(eyeLine, faceDir).normalize(); if (faceUp.y < 0) faceUp.negate();
  const mouthA = v3(ANATOMY.mouthEnds[0]), mouthB = v3(ANATOMY.mouthEnds[1]), mouthC = v3(ANATOMY.mouth);
  const mouthLine = mouthB.clone().sub(mouthA).normalize();
  const mouthN = new THREE.Vector3(); { const MN = mouthGeo.attributes.normal; mouthN.set(MN.getX(0), MN.getY(0), MN.getZ(0)).normalize(); if (mouthN.dot(faceDir) < 0) mouthN.negate(); }
  const mouthUp = new THREE.Vector3().crossVectors(mouthLine, mouthN).normalize(); if (mouthUp.y < 0) mouthUp.negate();
  {
    const MP = mouthGeo.attributes.position, q = new THREE.Vector3();
    for (let i = 0; i < MP.count; i++) {
      q.fromBufferAttribute(MP, i).sub(mouthC);
      const n = q.dot(mouthN); q.addScaledVector(mouthN, -n).multiplyScalar(MOUTH_SCALE).addScaledVector(mouthN, n + 0.002).add(mouthC);
      MP.setXYZ(i, q.x, q.y, q.z);
    }
    MP.needsUpdate = true;
    mouthA.sub(mouthC).multiplyScalar(MOUTH_SCALE).add(mouthC); mouthB.sub(mouthC).multiplyScalar(MOUTH_SCALE).add(mouthC);
  }

  // extra bones: jaw, eyelids (oriented to the eye plane so scale.y squashes along the face's up), irises, hair masses, shards
  const headPivot = rest.head.worldPos;
  const jawHinge = mouthC.clone().addScaledVector(mouthN, -0.07).addScaledVector(mouthUp, 0.01);
  mkBone('jaw', 'head', jawHinge.toArray());
  const lidFrame = (cen, n) => {
    const up = faceUp.clone().addScaledVector(n, -n.dot(faceUp)).normalize();
    const right = new THREE.Vector3().crossVectors(up, n).normalize();
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, n));
  };
  for (const k of ['A', 'B']) {
    const I = cInfo[irisComp[k]];
    const lid = mkBone('lid' + k, 'head', I.c.toArray()); lid.quaternion.copy(lidFrame(I.c, k === 'A' ? nA : nB)); rest['lid' + k].quat = lid.quaternion.clone();
    const iris = new THREE.Bone(); iris.name = 'iris' + k; lid.add(iris); B['iris' + k] = iris; iris.userData.index = bones.length; bones.push(iris);
  }
  const lidUp = { A: new THREE.Vector3(0, 1, 0).applyQuaternion(rest.lidA.quat), B: new THREE.Vector3(0, 1, 0).applyQuaternion(rest.lidB.quat) };
  mkBone('hairBack', 'head', [0.02, 1.72, -0.02]);
  mkBone('hairFront', 'head', [-0.18, 1.70, 0.14]);
  for (const s of ['shardR', 'shardL']) { const ci = role.indexOf(s); mkBone(s, 'chest', ci >= 0 ? cInfo[ci].c.toArray() : [0, 1, 0]); }
  const bi = (n) => B[n].userData.index;

  // ----- geodesic region labels on the main body -----
  const label = new Int16Array(nV).fill(-1), dist = new Float32Array(nV).fill(Infinity), heap = new MinHeap();
  const tmp = {};
  for (let i = 0; i < nV; i++) {
    if (role[comp[i]] !== 'main' || weld[i] !== i) continue;
    tv.fromBufferAttribute(P, i);
    let best = -1, bd = Infinity;
    for (let s = 0; s < segs.length; s++) { const d = segDist(tv, segs[s].a, segs[s].b, tmp); if (d < bd) { bd = d; best = s; } }
    if (bd < segs[best].seedR) { label[i] = best; dist[i] = 0; heap.push(0, i); }
  }
  while (heap.size) {
    const a = heap.pop(); const l = nb[a];
    for (let k = 0; k < l.length; k += 2) { const b = l[k], d = dist[a] + l[k + 1]; if (d < dist[b]) { dist[b] = d; label[b] = label[a]; heap.push(d, b); } }
  }
  const segIndex = {}; segs.forEach((s, i) => (segIndex[s.name] = i));
  const children = segs.map((s) => segs.filter((c) => c.parent === s.name));
  const jointBlend = (p, child) => { // 0 = parent side, 1 = child side
    const dir = tv.copy(child.b).sub(child.a).normalize();
    const t = (p.x - child.a.x) * dir.x + (p.y - child.a.y) * dir.y + (p.z - child.a.z) * dir.z;
    const f = smooth(-child.blendR, child.blendR, t);
    // radial falloff so the blend only applies near the limb axis
    const rx = p.x - child.a.x - dir.x * t, ry = p.y - child.a.y - dir.y * t, rz = p.z - child.a.z - dir.z * t;
    const r = Math.hypot(rx, ry, rz); const radial = 1 - smooth(child.radial, child.radial * 1.8, r);
    return { f, radial };
  };

  // ----- fingers: the hands are low poly, each finger a spike of a few vertices. Walk the surface out from the wrist
  // (geodesic distance inside the hand region), cut the palm off at the radius that leaves the most separate pieces
  // (up to 5), and give each piece a bone at its knuckle, so the hand can curl (grip). Curl axis: across the finger,
  // towards the palm (the mesh rests with its palms on the ledge, so the palm side is the downward one).
  const FINGERS = { R: [], L: [] }, fingerW = new Map();
  for (const side of ['R', 'L']) {
    const L = segIndex['hand' + side], s = segs[L], H = [];
    for (let i = 0; i < nV; i++) if (weld[i] === i && label[i] === L) H.push(i);
    const inH = new Set(H), gd = new Map(), hq = new MinHeap();
    const near = H.map((i) => [i, Math.hypot(P.getX(i) - s.a.x, P.getY(i) - s.a.y, P.getZ(i) - s.a.z)]).sort((a, b) => a[1] - b[1]).slice(0, 6);
    for (const [i] of near) { gd.set(i, 0); hq.push(0, i); }
    while (hq.size) {
      const a = hq.pop(), l = nb[a];
      for (let k = 0; k < l.length; k += 2) { const b = l[k]; if (!inH.has(b)) continue; const d = gd.get(a) + l[k + 1]; if (d < (gd.get(b) ?? Infinity)) { gd.set(b, d); hq.push(d, b); } }
    }
    const piecesAt = (r) => {
      const far = H.filter((i) => (gd.get(i) ?? 0) > r), fs = new Set(far), seen = new Set(), out = [];
      for (const v of far) {
        if (seen.has(v)) continue; const st = [v], c = []; seen.add(v);
        while (st.length) { const a = st.pop(); c.push(a); const l = nb[a]; for (let k = 0; k < l.length; k += 2) { const b = l[k]; if (fs.has(b) && !seen.has(b)) { seen.add(b); st.push(b); } } }
        if (c.length >= 2) out.push(c);
      }
      return out;
    };
    let pieces = [], cut = 0;
    for (let r = 0.05; r <= 0.161; r += 0.01) { const c = piecesAt(r); if (c.length > pieces.length && c.length <= 5) { pieces = c; cut = r; } }
    if (pieces.length < 3) continue; // not a hand we can read: leave it rigid
    const fv = (i) => new THREE.Vector3().fromBufferAttribute(P, i);
    const info = pieces.map((c) => {
      const inC = new Set(c), ring = [];
      for (const a of c) { const l = nb[a]; for (let k = 0; k < l.length; k += 2) if (inH.has(l[k]) && !inC.has(l[k])) ring.push(l[k]); }
      const knuckle = new THREE.Vector3();
      if (ring.length) { for (const i of ring) knuckle.add(fv(i)); knuckle.divideScalar(ring.length); }
      else knuckle.copy(fv(c.reduce((m, i) => (gd.get(i) < gd.get(m) ? i : m), c[0])));
      const tip = fv(c.reduce((m, i) => (gd.get(i) > gd.get(m) ? i : m), c[0]));
      return { verts: c, knuckle, tip };
    });
    const wrist = s.a.clone(), kc = new THREE.Vector3(); info.forEach((f) => kc.add(f.knuckle)); kc.divideScalar(info.length);
    const fwd = kc.clone().sub(wrist).normalize();
    let span = new THREE.Vector3(), best = -1; // widest pair of knuckles = across the hand
    for (const a of info) for (const b of info) { const d = b.knuckle.clone().sub(a.knuckle); d.addScaledVector(fwd, -d.dot(fwd)); if (d.length() > best) { best = d.length(); span = d.clone(); } }
    const palm = new THREE.Vector3().crossVectors(span.normalize(), fwd).normalize(); if (palm.y > 0) palm.negate();
    info.sort((a, b) => a.knuckle.dot(span) - b.knuckle.dot(span)); // k = 0..n-1 across the hand, side to side
    info.forEach((f, k) => {
      f.palm = palm.clone();
      const dir = f.tip.clone().sub(f.knuckle).normalize();
      f.axis = new THREE.Vector3().crossVectors(dir, palm).normalize();
      f.bone = mkBone(`finger${side}${k}`, 'hand' + side, f.knuckle.toArray());
      for (const i of f.verts) fingerW.set(i, { bi: f.bone.userData.index, w: smooth(cut - 0.005, cut + 0.02, gd.get(i)) });
    });
    FINGERS[side] = info;
  }

  const wl = new Array(nV), hairTip = new Float32Array(nV);
  const headC = v3(ANATOMY.headCenter), pp = new THREE.Vector3();
  const jawWeight = (p) => {
    const s = pp.copy(p).sub(mouthC).dot(mouthUp);
    const fwd = pp.copy(p).sub(headC).dot(faceDir);
    return (1 - smooth(-0.012, 0.004, s)) * smooth(0.0, 0.06, fwd) * (1 - smooth(0.07, 0.11, p.distanceTo(mouthC)));
  };
  for (let i = 0; i < nV; i++) {
    const p = new THREE.Vector3().fromBufferAttribute(P, i), r = role[comp[i]];
    let w = {};
    if (r === 'main') {
      const L = label[weld[i]] >= 0 ? label[weld[i]] : 0, s = segs[L];
      w[s.idx] = 1;
      if (s.parent) { // blend with parent near own joint
        const { f, radial } = jointBlend(p, s); const g = lerp(1, f, radial);
        w[s.idx] = g; w[segs[segIndex[s.parent]].idx] = 1 - g;
      }
      for (const c of children[L]) { // blend with children near their joints
        const { f, radial } = jointBlend(p, c); const g = f * radial;
        if (g > 0) { w[c.idx] = (w[c.idx] || 0) + g * w[s.idx]; w[s.idx] *= 1 - g; }
      }
      const fw = fingerW.get(weld[i]);
      if (fw && fw.w > 0) { for (const k in w) w[k] *= 1 - fw.w; w[fw.bi] = (w[fw.bi] || 0) + fw.w; }
    } else if (r === 'hairBack' || r === 'hairFront') {
      const d = p.distanceTo(headC); const h = smooth(0.10, 0.42, d);
      w[bi('head')] = 1 - h; w[bi(r)] = h; hairTip[i] = smooth(0.25, 0.8, d);
    } else if (r === 'irisA' || r === 'irisB') { w[bi('iris' + r.slice(4))] = 1; }
    else if (r === 'lidA' || r === 'lidB') { w[bi(r)] = 1; }
    else if (r === 'face') { const j = jawWeight(p); w[bi('head')] = 1 - j; if (j > 0) w[bi('jaw')] = j; }
    else if (r === 'shardR' || r === 'shardL') { w[bi(r)] = 1; }
    wl[i] = w;
  }
  // props (floating bits / spikes): copy the weights of the nearest main-body vertex to the prop
  for (let ci = 0; ci < nComp; ci++) {
    if (role[ci] !== 'prop') continue;
    let bestD = Infinity, bestV = 0;
    const vs = cInfo[ci].verts; const step = Math.max(1, Math.floor(vs.length / 30));
    for (let k = 0; k < vs.length; k += step) {
      tv.fromBufferAttribute(P, vs[k]);
      for (let j = 0; j < nV; j++) { if (role[comp[j]] !== 'main') continue; const d = (P.getX(j) - tv.x) ** 2 + (P.getY(j) - tv.y) ** 2 + (P.getZ(j) - tv.z) ** 2; if (d < bestD) { bestD = d; bestV = j; } }
    }
    for (const k of vs) wl[k] = { ...wl[bestV] };
  }
  const bodyPack = packWeights(nV, wl);
  bodyGeo.setAttribute('skinIndex', bodyPack.skinIndex); bodyGeo.setAttribute('skinWeight', bodyPack.skinWeight);
  bodyGeo.setAttribute('aHair', new THREE.BufferAttribute(hairTip, 1));

  // ----- outline: copy from the nearest body vertex (outline = body inflated ~5mm along the normal) -----
  {
    const OP = outlineGeo.attributes.position, ON = outlineGeo.attributes.normal, n = OP.count;
    const cell = 0.03, grid = new Map(), key = (x, y, z) => Math.floor(x / cell) + ',' + Math.floor(y / cell) + ',' + Math.floor(z / cell);
    for (let i = 0; i < nV; i++) { const k = key(P.getX(i), P.getY(i), P.getZ(i)); let a = grid.get(k); if (!a) grid.set(k, (a = [])); a.push(i); }
    const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4), ht = new Float32Array(n);
    const BSI = bodyPack.skinIndex.array, BSW = bodyPack.skinWeight.array;
    for (let i = 0; i < n; i++) {
      const x = OP.getX(i) + ON.getX(i) * 0.005, y = OP.getY(i) + ON.getY(i) * 0.005, z = OP.getZ(i) + ON.getZ(i) * 0.005;
      const gx = Math.floor(x / cell), gy = Math.floor(y / cell), gz = Math.floor(z / cell);
      let best = -1, bd = Infinity;
      for (let R = 1; R <= 3 && best < 0; R++) {
        for (let dx = -R; dx <= R; dx++) for (let dy = -R; dy <= R; dy++) for (let dz = -R; dz <= R; dz++) {
          const a = grid.get((gx + dx) + ',' + (gy + dy) + ',' + (gz + dz)); if (!a) continue;
          for (const j of a) { const d = (P.getX(j) - x) ** 2 + (P.getY(j) - y) ** 2 + (P.getZ(j) - z) ** 2; if (d < bd) { bd = d; best = j; } }
        }
      }
      if (best < 0) best = 0;
      for (let k = 0; k < 4; k++) { si[i * 4 + k] = BSI[best * 4 + k]; sw[i * 4 + k] = BSW[best * 4 + k]; }
      ht[i] = hairTip[best];
    }
    outlineGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4));
    outlineGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
    outlineGeo.setAttribute('aHair', new THREE.BufferAttribute(ht, 1));
  }

  // ----- mouth: upper grin (head), lower grin (jaw, corners pinned), interior strip between them -----
  const mouthU = (p) => clamp(pp.copy(p).sub(mouthA).dot(mouthLine) / mouthA.distanceTo(mouthB), 0, 1);
  const cornerPin = (u) => Math.pow(Math.sin(Math.PI * u), 0.6);
  const rigid = (geo, bone) => {
    const n = geo.attributes.position.count, si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) { si[i * 4] = bi(bone); sw[i * 4] = 1; }
    geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4)); geo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
    geo.setAttribute('aHair', new THREE.BufferAttribute(new Float32Array(n), 1));
  };
  rigid(mouthGeo, 'head');
  const lowerGeo = mouthGeo.clone();
  {
    const LP = lowerGeo.attributes.position, n = LP.count, si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) { const u = mouthU(tv.fromBufferAttribute(LP, i)), j = cornerPin(u); si[i * 4] = bi('jaw'); sw[i * 4] = j; si[i * 4 + 1] = bi('head'); sw[i * 4 + 1] = 1 - j; }
    lowerGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4)); lowerGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
  }
  const holeGeo = new THREE.BufferGeometry();
  {
    const cols = 10, pos = [], si = [], sw = [], idx = [];
    for (let c = 0; c <= cols; c++) {
      const u = c / cols, p = mouthA.clone().lerp(mouthB, u).addScaledVector(mouthN, -0.0015);
      for (let row = 0; row < 2; row++) {
        pos.push(p.x, p.y, p.z);
        const j = row === 1 ? cornerPin(u) : 0;
        si.push(bi('jaw'), bi('head'), 0, 0); sw.push(j, 1 - j, 0, 0);
      }
      if (c < cols) { const a = c * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
    holeGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    holeGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4));
    holeGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
    holeGeo.setAttribute('aHair', new THREE.BufferAttribute(new Float32Array(cols * 2 + 2), 1));
    holeGeo.setIndex(idx);
  }

  // ----- materials + shader patch -----
  const U = {
    uTime: { value: 0 }, uGlitch: { value: 0 }, uWind: { value: 1 }, uTalkGlow: { value: 0 },
    uTint: { value: new THREE.Color(0, 0, 0) }, uPink: { value: new THREE.Color(PALETTE.pink) }, uGreen: { value: new THREE.Color(PALETTE.green) },
    // teleport (see TELEPORT below): she is cut into TP_BANDS horizontal slices of her rest-pose height
    uTpOn: { value: 0 },        // 1 while any of the below is non-zero
    uTpShift: { value: 0 },     // view-space sideways slice displacement (m)
    uTpDir: { value: 0 },       // 0 = slices scatter both ways, 1 = all slide in from one side (staggered)
    uTpSide: { value: 1 },      // that side (+1 / -1)
    uTpJit: { value: 0 },       // per-slice overshoot jitter (m, signed per slice)
    uTpJx: { value: 0 },        // whole-body sideways jump (m) for the pre-glitch flicker
    uTpGone: { value: 0 },      // fraction of slices removed (1 = invisible)
    uTpSweep: { value: 0 },     // 0 = slices drop out in random order, 1 = top first
    uTpSquash: { value: 0 },    // 0..1: each slice thins to a line at its centre
    uTpStretch: { value: 0 },   // sideways stretch of the slices about her centre (0 = none)
    uTpHot: { value: 0 },       // colour blow-out: 0 = her flat look, 1 = pink/cyan slice tints, 2 = white
    uTpFade: { value: 0 },      // reduced motion: ordered-dither fade (1 = gone)
    uTpSeed: { value: 0 },
    uTpC: { value: new THREE.Vector3() }, // her centre in view space (the stretch pivot)
  };
  const TP_BANDS = '24.0';
  // extra = per-material uniforms (the after-image / RGB-split ghosts)
  const patch = (mat, kind, extra) => {
    const ghost = kind === 'ghost';
    mat.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, U, extra || {});
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', `#include <common>
          ${ghost ? '#define CAP_GHOST\n uniform mat4 uGhostX; uniform float uGhostShift; uniform float uGhostTp; uniform float uGhostRise;' : ''}
          attribute float aHair; uniform float uTime; uniform float uGlitch; uniform float uWind;
          uniform float uTpOn, uTpShift, uTpDir, uTpSide, uTpJit, uTpJx, uTpStretch, uTpSeed; uniform vec3 uTpC;
          varying vec3 vCapPos; varying vec3 vCapN; varying vec3 vCapV;
          float capHash(float n){ return fract(sin(n * 91.345) * 43758.5453); }`)
        .replace('#include <skinning_vertex>', `#include <skinning_vertex>
          // hair: slow wind sway, weighted by distance from the scalp
          if (aHair > 0.0) {
            float ph = position.x * 3.1 + position.y * 2.3 + position.z * 1.7;
            // a slow, large primary sway (the whole mass breathes in the wind) + a quicker flutter at the tips
            float gust = 0.65 + 0.35 * sin(uTime * 0.31) * sin(uTime * 0.17 + 1.0);
            vec3 sway = vec3(sin(uTime * 0.55 + ph * 0.5) * 1.0 + 0.3 * sin(uTime * 2.1 + ph * 2.0),
                             0.3 * sin(uTime * 0.8 + ph * 1.5),
                             cos(uTime * 0.45 + ph * 0.4) * 0.8 + 0.3 * sin(uTime * 1.7 + ph * 2.6));
            transformed += sway * aHair * aHair * 0.05 * gust * uWind;
          }
          // glitch: horizontal slice displacement
          if (uGlitch > 0.001) {
            float band = floor(position.y * 26.0) + floor(uTime * 22.0) * 13.0;
            float h = capHash(band);
            float on = step(1.0 - uGlitch * 0.55, h);
            transformed.x += (capHash(band + 7.0) - 0.5) * 0.14 * on * uGlitch;
            transformed.z += (capHash(band + 3.0) - 0.5) * 0.06 * on * uGlitch;
          }
          vCapPos = position;`)
        .replace('#include <project_vertex>', `#include <project_vertex>
          #ifdef CAP_GHOST
            // after-image: the live skinned pose, mapped back to where she was (uGhostX = oldRoot * inverse(root))
            { vec4 gwp = uGhostX * (modelMatrix * vec4(transformed, 1.0)); gwp.y += uGhostRise; mvPosition = viewMatrix * gwp; }
            float capUseTp = uGhostTp;
          #else
            float capUseTp = 1.0;
          #endif
          // teleport: horizontal slices slide sideways in view space (so "horizontal" is the screen's), stretch out
          if (uTpOn > 0.5 && capUseTp > 0.5) {
            float tb = floor(position.y * ${TP_BANDS});
            float th = capHash(tb * 1.37 + uTpSeed);
            float scatter = (th < 0.5 ? -1.0 : 1.0) * (0.3 + 1.4 * abs(th - 0.5)) * step(0.28, capHash(tb + uTpSeed + 2.2));
            float slide = uTpSide * (0.45 + 1.1 * th);
            mvPosition.x += mix(scatter, slide, uTpDir) * uTpShift + uTpJit * (th - 0.5) * 2.0 + uTpJx;
            float hs = capHash(tb * 3.1 + uTpSeed); // (most slices stretch ~1.3-1.6x; ~1 in 10 'hero' slices overshoot ~2.5x)
            mvPosition.x = uTpC.x + (mvPosition.x - uTpC.x) * (1.0 + uTpStretch * (hs > 0.9 ? 1.4 : 0.25 + 0.3 * hs));
          }
          #ifdef CAP_GHOST
            mvPosition.x += uGhostShift; mvPosition.z -= 0.03 * uGhostTp; // split layers sit just behind her: only fringes show
          #endif
          gl_Position = projectionMatrix * mvPosition;
          vCapV = -mvPosition.xyz;
          #ifdef USE_SKINNING
            vCapN = normalize(transformedNormal);
          #else
            vCapN = vec3(0.0, 0.0, 1.0);
          #endif`);
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>
          ${ghost ? '#define CAP_GHOST\n uniform vec3 uGhostCol; uniform vec3 uGhostRim; uniform float uGhostA; uniform float uGhostTp;' : ''}
          uniform float uTime; uniform float uGlitch; uniform float uTalkGlow; uniform vec3 uTint; uniform vec3 uPink; uniform vec3 uGreen;
          uniform float uTpOn, uTpGone, uTpSweep, uTpSquash, uTpHot, uTpFade, uTpSeed, uTpShift;
          varying vec3 vCapPos; varying vec3 vCapN; varying vec3 vCapV;
          float capHash(float n){ return fract(sin(n * 91.345) * 43758.5453); }
          float capB2(vec2 a){ a = floor(a); return fract(dot(a, vec2(0.5, a.y * 0.75))); }
          float capBayer(vec2 a){ return capB2(0.5 * a) * 0.25 + capB2(a); } // 4x4 ordered dither, 0..1`)
        .replace('#include <opaque_fragment>', `#include <opaque_fragment>
          ${kind === 'outline' ? 'gl_FragColor.rgb += uTint;' : ''}
          ${kind === 'body' ? `
          {
            float rim = 1.0 - abs(dot(normalize(vCapN), normalize(vCapV)));
            rim = smoothstep(0.55, 1.0, rim);
            gl_FragColor.rgb += uPink * rim * uTalkGlow * 0.55;
          }` : ''}
          if (uGlitch > 0.001) {
            float band = floor(vCapPos.y * 46.0) + floor(uTime * 22.0) * 13.0;
            float h = capHash(band);
            float on = step(1.0 - uGlitch * 0.4, h);
            vec3 tint = mix(uPink, uGreen, step(0.5, capHash(band + 1.7)));
            gl_FragColor.rgb = mix(gl_FragColor.rgb, tint, on * 0.85);
            float scan = 0.5 + 0.5 * sin(gl_FragCoord.y * 2.2 + uTime * 80.0);
            gl_FragColor.rgb += vec3(0.08, 0.0, 0.05) * scan * uGlitch;
          }
          #ifdef CAP_GHOST
            float capUseTp = uGhostTp;
          #else
            float capUseTp = 1.0;
          #endif
          // teleport: slices drop out, thin to scanlines, tint pink / cyan per slice, blow out to white
          if (uTpOn > 0.5 && capUseTp > 0.5) {
            float yb = vCapPos.y * ${TP_BANDS}; float tb = floor(yb); float f = fract(yb);
            float th = capHash(tb * 1.37 + uTpSeed);
            float key = mix(capHash(tb * 2.71 + uTpSeed + 5.1), clamp(vCapPos.y / 2.4, 0.0, 1.0), uTpSweep);
            if (key < uTpGone * 1.02 - 0.01) discard;
            float thick = 1.0 - uTpSquash * (0.82 + 0.16 * capHash(tb + 4.4));
            if (abs(f - 0.5) > 0.5 * thick) discard;
            if (f > 1.0 - 0.16 * clamp(uTpShift * 6.0, 0.0, 1.0)) discard; // thin slits between the sliding slices
            vec3 cy = vec3(0.12, 0.85, 1.0);
            float flick = step(0.4, capHash(tb + 9.0 + floor(uTime * 30.0)));
            // (scaled by the material colour, so main.js's darkened bloom pre-render darkens these too: no white blob)
            float capK = max(diffuse.r, max(diffuse.g, diffuse.b));
            vec3 tint = (th < 0.3 ? cy : uPink) * capK;
            vec3 c = mix(gl_FragColor.rgb, tint, clamp(uTpHot, 0.0, 1.0) * (0.5 + 0.5 * flick));
            c = mix(c, vec3(0.85, 0.74, 0.8) * capK, clamp(uTpHot - 1.0, 0.0, 1.0) * (0.6 + 0.4 * flick));
            gl_FragColor.rgb = c;
          }
          if (uTpFade > 0.001 && capBayer(gl_FragCoord.xy) < uTpFade * 1.02 - 0.01) discard;
          #ifdef CAP_GHOST
          {
            float rim = 1.0 - abs(dot(normalize(vCapN), normalize(vCapV)));
            vec3 col = mix(uGhostCol, uGhostRim, smoothstep(0.35, 0.9, rim));
            float a = uGhostA;
            if (uGhostTp < 0.5) {
              // the after-image: a frozen-looking silhouette breaking into drifting slices as it fades
              float gb = floor(vCapPos.y * 30.0);
              a *= step(capHash(gb * 1.9 + uTpSeed), 0.15 + 1.1 * uGhostA) * step(0.25, fract(vCapPos.y * 30.0));
              a *= 0.45 + 0.55 * smoothstep(0.2, 0.85, rim);
              a *= 0.7 + 0.3 * sin(gl_FragCoord.y * 1.7);
            }
            gl_FragColor = vec4(col * a * max(diffuse.r, max(diffuse.g, diffuse.b)), 1.0); // (darkened in the bloom pre-render)
          }
          #endif`);
    };
    mat.customProgramCacheKey = () => 'capri-' + kind;
    return mat;
  };
  const bodyMat = patch(bodySrc.material.clone(), 'body');
  const outlineMat = patch(outlineSrc.material.clone(), 'outline');
  const teethMat = patch(eyeSrc.material.clone(), 'teeth');
  teethMat.polygonOffset = true; teethMat.polygonOffsetFactor = -2; teethMat.polygonOffsetUnits = -4;
  const holeMat = patch(new THREE.MeshBasicMaterial({ color: new THREE.Color(PALETTE.pink).multiplyScalar(0.55), side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }), 'hole');

  const skeleton = new THREE.Skeleton(bones);
  const mk = (geo, mat, name, order) => {
    const m = new THREE.SkinnedMesh(geo, mat); m.name = name; m.frustumCulled = false; m.renderOrder = order || 0; rig.add(m); return m;
  };
  const bodyMesh = mk(bodyGeo, bodyMat, 'capri-body');
  const outlineMesh = mk(outlineGeo, outlineMat, 'capri-outline');
  const holeMesh = mk(holeGeo, holeMat, 'capri-mouth-hole', 1);
  const upperMesh = mk(mouthGeo, teethMat, 'capri-teeth-upper', 2);
  const lowerMesh = mk(lowerGeo, teethMat, 'capri-teeth-lower', 2);
  const meshes = [bodyMesh, outlineMesh, holeMesh, upperMesh, lowerMesh];
  // teleport ghosts: two extra additive renders of her body (hidden unless a teleport runs). During the jump-out they
  // follow her slices, shifted left / right in view space (a real RGB split of her silhouette); afterwards one stays
  // behind as the after-image at the old spot (uGhostX maps the live pose back there).
  const mkGhost = (col, rim) => {
    const gu = {
      uGhostX: { value: new THREE.Matrix4() }, uGhostShift: { value: 0 }, uGhostTp: { value: 1 }, uGhostRise: { value: 0 },
      uGhostCol: { value: new THREE.Color(col) }, uGhostRim: { value: new THREE.Color(rim) }, uGhostA: { value: 0 },
    };
    const mat = patch(new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }), 'ghost', gu);
    const m = mk(bodyGeo, mat, 'capri-tp-ghost', 3); m.visible = false; m.userData.u = gu; return m;
  };
  const ghostA = mkGhost(0xff1f4f, 0xffb0c8); // red-pink layer
  const ghostB = mkGhost(0x1fd8ff, 0xff3a78); // cyan, pink rim (the after-image)
  root.updateMatrixWorld(true);
  for (const m of meshes) m.bind(skeleton);
  ghostA.bind(skeleton); ghostB.bind(skeleton);
  for (const m of [gltf.scene]) m.traverse((o) => { if (o.geometry) o.geometry.dispose(); });

  // anchor for the speech bubble tail (rides the head): the grin's corner on the bubble side (her left, screen right),
  // nudged a little further out along the mouth line and down, so the tail ends at her mouth, clear of her eyes
  const faceAnchor = new THREE.Object3D();
  faceAnchor.position.copy(mouthC).lerp(mouthB, 0.75).addScaledVector(mouthLine, 0.012).addScaledVector(mouthUp, -0.006).sub(headPivot);
  B.head.add(faceAnchor);
  const restBox = new THREE.Box3().setFromBufferAttribute(P);
  let lastCamera = opts.camera || null;
  bodyMesh.onBeforeRender = (r, s, cam) => { if (cam && cam.isCamera && !opts.camera) lastCamera = cam; };

  // ---------------------------------------------------------------------------------------------
  // animation state
  // ---------------------------------------------------------------------------------------------
  const CH = { // channel: [omega, zeta]
    headYaw: [9, 0.8], headPitch: [11, 0.7], headRoll: [8, 0.8],
    chestPitch: [5, 0.9], chestRoll: [4, 0.9], chestYaw: [4, 0.9],
    rx: [8, 0.85], ry: [8, 0.85], rz: [8, 0.85], lx: [8, 0.85], ly: [8, 0.85], lz: [8, 0.85],
    handRp: [10, 0.7], handRr: [10, 0.7], handLp: [10, 0.7], handLr: [10, 0.7],
    shUpR: [9, 0.8], shUpL: [9, 0.8], shin: [6, 0.6], legL: [6, 0.7],
    jaw: [42, 0.9], widen: [14, 0.8], eyeX: [40, 0.9], eyeY: [40, 0.9],
  };
  const S = {}; for (const k in CH) S[k] = new Spring(CH[k][0], CH[k][1]);
  const zero = () => { const o = {}; for (const k in CH) o[k] = 0; o.lookW = 0; o.lid = 0; o.glitch = 0; o.lookPitchBias = 0; o.lookYawBias = 0; o.hairKick = 0; return o; };
  const hairLag = { yaw: new Spring(5, 0.32), pitch: new Spring(5, 0.32), roll: new Spring(5, 0.32) };

  let state = 'idle';
  const W = { idle: 1, listening: 0, thinking: 0, talking: 0 };
  let lookTarget = null; // world Vector3 or null (= camera)
  const clips = [];
  let time = 0;
  // idle scheduling
  let nextBlink = rand(1, 3), blinkT = -1, doubleBlink = false;
  let nextSaccade = rand(0.5, 2), sacc = { x: 0, y: 0 };
  // glances are relative to the look direction; the eyes lead the head by ~80 ms
  let nextGlance = rand(5, 10), glance = { yaw: 0, pitch: 0, start: 0, until: 0 };
  // speech
  let talkRecent = 0, wordLen = 0, wordVowels = 0, prevVowel = false, charsSinceGesture = 0, totalChars = 0;
  let gesturesThisReply = 0, lastEndAt = -10, bubbleGlanceUntil = 0, lastCharAt = -10, ahead = 30;
  // jaw: one open-then-close beat per syllable (not per character), at most ~5 per second
  let pulseT = -10, pulseAmp = 0, closeT = -10, closeTo = 0, pendingAmp = 0, jawEnvV = 0;
  const SYL_MIN = 0.19;    // s between jaw beats; a vowel onset that comes sooner is merged into the next beat
  // phrase-level head pose: picked at each phrase start, held until the next punctuation (reads as emphasis)
  let phraseStart = true, sentenceGate = false, lastGestureSide = 'L';
  const phrasePose = { yaw: 0, roll: 0, pitch: 0 }, phraseNow = { yaw: 0, roll: 0, pitch: 0 };
  // listening
  let lastTypeAt = -10, typeNodDone = true, listenSince = 0, caret = 0.5, nextMm = 0, lookUpUntil = -10, nextLookUp = 0;
  const EYE_R = 0.028;     // eye disc radius (m)
  const IRIS = 0.011;      // how far the iris can travel in the disc (m)
  const BUBBLE_SIDE = 1;   // +eyeX / +headYaw = toward the city and the bubble (her left, screen right)

  const env = (p, a, r) => smooth(0, a, p) * (1 - smooth(1 - r, 1, p));
  const addClip = (dur, fn, tag) => { if (tag) for (let i = clips.length - 1; i >= 0; i--) if (clips[i].tag === tag) clips.splice(i, 1); clips.push({ t: 0, dur, fn, tag }); };
  const kick = (ch, v) => { S[ch].v += v; };
  const restWR = v3(ANATOMY.wristR), restWL = v3(ANATOMY.wristL);
  const handsOnLedge = 0.07; // lift the wrists so the fingertips rest on the ledge top instead of sinking in

  // ----- gestures (wrist targets are model-space; we convert to offsets from the rest wrist) -----
  const offR = (x, y, z) => [x - restWR.x, y - restWR.y - handsOnLedge, z - restWR.z];
  const offL = (x, y, z) => [x - restWL.x, y - restWL.y - handsOnLedge, z - restWL.z];
  const GESTURES = {
    explainR(p, o) { const e = env(p, 0.25, 0.4), d = offR(-0.56, 1.02, 0.26); o.rx += d[0] * e; o.ry += d[1] * e; o.rz += d[2] * e; o.handRp += -0.5 * e; o.handRr += 0.6 * e; o.headRoll += -0.04 * e; },
    explainL(p, o) { const e = env(p, 0.25, 0.4), d = offL(0.42, 1.0, 0.2); o.lx += d[0] * e; o.ly += d[1] * e; o.lz += d[2] * e; o.handLp += -0.5 * e; o.handLr += -0.6 * e; o.headRoll += 0.04 * e; },
    palmR(p, o) { const e = env(p, 0.2, 0.45), d = offR(-0.66, 0.9, 0.18); o.rx += d[0] * e; o.ry += d[1] * e; o.rz += d[2] * e; o.handRr += 0.9 * e; o.shUpR += 0.01 * e; },
    flickR(p, o) { const e = env(p, 0.15, 0.6), d = offR(-0.62, 0.92, 0.12); o.rx += d[0] * e; o.ry += d[1] * e; o.rz += d[2] * e; o.handRp += -0.9 * Math.sin(Math.PI * clamp(p * 2.2, 0, 1)) ; },
    flickL(p, o) { const e = env(p, 0.15, 0.6), d = offL(0.55, 0.92, 0.08); o.lx += d[0] * e; o.ly += d[1] * e; o.lz += d[2] * e; o.handLp += -0.9 * Math.sin(Math.PI * clamp(p * 2.2, 0, 1)); },
    palmL(p, o) { const e = env(p, 0.2, 0.45), d = offL(0.60, 0.92, 0.16); o.lx += d[0] * e; o.ly += d[1] * e; o.lz += d[2] * e; o.handLr += -0.9 * e; o.shUpL += 0.01 * e; o.headRoll += 0.05 * e; },
    bothOpen(p, o) { const e = env(p, 0.22, 0.4), dr = offR(-0.62, 0.95, 0.2), dl = offL(0.56, 0.95, 0.14); o.rx += dr[0] * e; o.ry += dr[1] * e; o.rz += dr[2] * e; o.lx += dl[0] * e; o.ly += dl[1] * e; o.lz += dl[2] * e; o.handRr += 0.7 * e; o.handLr += -0.7 * e; o.chestPitch += 0.04 * e; },
    countL(p, o) { const e = env(p, 0.3, 0.35), d = offL(0.36, 1.12, 0.22); o.lx += d[0] * e; o.ly += d[1] * e; o.lz += d[2] * e; o.handLp += (-0.3 + 0.25 * Math.sin(p * 18)) * e; },
  };
  const startGesture = (name, dur) => addClip(dur || rand(1.3, 1.9), GESTURES[name], 'gesture');

  let settleDir = 1;
  const REACTS = {
    greet: [2.4, (p, o) => {
      // hand out to HER right side (screen left) at shoulder height, forearm against the city rather than the dark
      // hair; two and a half slow, wide side-to-side waves; head tilt toward it and a smirk
      const e = env(p, 0.14, 0.26), d = offR(-0.98, 1.30, 0.30);
      const wave = Math.sin((p - 0.12) * 2 * Math.PI * 2.6) * smooth(0.1, 0.24, p) * (1 - smooth(0.72, 0.88, p));
      o.rx += (d[0] + wave * 0.18) * e; o.ry += (d[1] - Math.abs(wave) * 0.03) * e; o.rz += d[2] * e;
      o.handRr += (1.0 + wave * 0.5) * e; o.handRp += (-0.2 + wave * 0.3) * e; o.shUpR += 0.03 * e;
      o.headRoll += 0.16 * e; o.headPitch += 0.05 * e; o.headYaw += -0.04 * e; o.chestRoll += -0.06 * e; o.chestPitch += 0.04 * e;
      o.jaw += 0.18 * env(p, 0.1, 0.5); o.lid += 0.22 * e; o.widen += 0.2 * e;
    }],
    laugh: [1.6, (p, o) => {
      const e = env(p, 0.08, 0.35), b = Math.abs(Math.sin(p * Math.PI * 9)) * (1 - p * 0.6);
      o.headPitch += (0.2 + 0.1 * b) * e; o.chestPitch += (-0.07 + 0.05 * b) * e; o.shUpR += 0.035 * b * e; o.shUpL += 0.035 * b * e; o.headYaw += 0.08 * e;
      o.jaw += (0.45 + 0.45 * b) * e; o.lid += 0.6 * e; o.headRoll += 0.06 * e; o.ry += 0.04 * b * e; o.ly += 0.04 * b * e; o.hairKick += 0.5 * b * e;
    }],
    shrug: [1.5, (p, o) => {
      const e = env(p, 0.18, 0.35), dr = offR(-0.84, 1.02, 0.20), dl = offL(0.78, 1.0, 0.16);
      o.rx += dr[0] * e; o.ry += dr[1] * e; o.rz += dr[2] * e; o.lx += dl[0] * e; o.ly += dl[1] * e; o.lz += dl[2] * e;
      o.handRr += 1.0 * e; o.handLr += -1.0 * e; o.handRp += -0.5 * e; o.handLp += -0.5 * e; o.shUpR += 0.065 * e; o.shUpL += 0.065 * e; o.headRoll += 0.22 * e; o.headPitch += -0.06 * e; o.chestRoll += -0.04 * e; o.lid += 0.3 * e; o.jaw += 0.12 * e;
    }],
    nod: [1.0, (p, o) => { const e = env(p, 0.1, 0.2); o.headPitch += -0.2 * Math.sin(p * Math.PI * 4) * e - 0.05 * e; o.chestPitch += 0.03 * e; }],
    // a small "mm-hm" while listening
    mm: [0.6, (p, o) => { const e = Math.sin(Math.PI * p); o.headPitch += -0.09 * e; o.chestPitch += 0.015 * e; }],
    surprised: [1.5, (p, o) => {
      // mostly face and head: a jolt back, eyes wide, mouth open, hair kicks; the hands only flinch up a little,
      // forward and out to the sides (kept below / in front of the hair so they read against her suit and the city)
      const e = smooth(0, 0.05, p) * (1 - smooth(0.55, 1, p));
      const recoil = Math.sin(Math.PI * clamp(p / 0.35, 0, 1)) * (1 - smooth(0.3, 0.6, p));
      o.headPitch += 0.15 * e + 0.12 * recoil; o.chestPitch += -0.1 * e - 0.1 * recoil; o.widen += 1.1 * e; o.jaw += 0.75 * e;
      // (her left arm stays down: raised, its black upper arm merges with the hair's dark underside into a slab)
      const dr = offR(-0.70, 0.86, 0.2), f = e * 0.85;
      o.rx += dr[0] * f; o.ry += dr[1] * f; o.rz += dr[2] * f; o.lz += 0.05 * e; o.ly += 0.02 * e;
      o.handRp += -0.6 * e; o.handLp += -0.4 * e; o.handRr += 0.5 * e;
      o.shUpR += 0.05 * e; o.shUpL += 0.05 * e; o.shin += 0.35 * e; o.hairKick += recoil;
    }],
    // end of an utterance: a small settling nod and a smirky head tilt, so a reply doesn't just stop
    settle: [1.1, (p, o) => {
      const e = env(p, 0.15, 0.45);
      o.headPitch += -0.12 * Math.sin(Math.PI * clamp(p / 0.45, 0, 1)) - 0.03 * e; o.headRoll += 0.14 * e * settleDir;
      o.jaw += 0.12 * e; o.lid += 0.18 * e; o.chestPitch += -0.02 * e;
    }],
    glitch: [0.8, (p, o) => {
      const e = env(p, 0.05, 0.35), step = Math.floor(time * 18);
      const h = (n) => { const x = Math.sin(n * 127.1 + step * 311.7) * 43758.5453; return x - Math.floor(x); };
      o.glitch += e * (0.55 + 0.45 * h(1)); o.headYaw += (h(2) - 0.5) * 0.25 * e; o.headRoll += (h(3) - 0.5) * 0.2 * e; o.eyeX += (h(4) - 0.5) * 0.006 * e;
    }],
  };

  // ---------------------------------------------------------------------------------------------
  // per-state targets
  // ---------------------------------------------------------------------------------------------
  const stateTargets = (s, t, o, w) => {
    if (w <= 0.001) return;
    const sw = (k, v) => (o[k] += v * w);
    if (s === 'idle') {
      sw('lookW', 0.75);
      sw('headYaw', 0.10 * Math.sin(t * 0.23) + 0.05 * Math.sin(t * 0.61 + 1));
      sw('headPitch', 0.04 * Math.sin(t * 0.31 + 2));
      sw('headRoll', 0.05 * Math.sin(t * 0.19 + 0.5) + 0.04);
      sw('chestRoll', 0.025 * Math.sin(t * 0.17));
      sw('chestYaw', 0.04 * Math.sin(t * 0.13 + 1.3));
      sw('shin', 0.16 * Math.sin(t * 2.0) + 0.05 * Math.sin(t * 0.7));
      sw('legL', 0.04 * Math.sin(t * 2.0 + 1.2));
      sw('rx', 0.01 * Math.sin(t * 0.4)); sw('lx', 0.01 * Math.sin(t * 0.37 + 2));
    } else if (s === 'listening') {
      // reading what you type: leaning in more the longer you type, eyes down on the prompt bar and following the
      // caret, slow breathing, head tilted; 'mm' bobs and an occasional glance up at you are scheduled in update()
      const lt = clamp((t - listenSince) / 4, 0, 1);
      const up = t < lookUpUntil ? smooth(0, 0.15, lookUpUntil - t) * smooth(0, 0.15, t - (lookUpUntil - 0.9)) : 0;
      sw('lookW', 1);
      sw('headRoll', 0.24 + 0.04 * Math.sin(t * 0.7)); sw('headPitch', -0.03 + 0.02 * Math.sin(t * 1.25) + 0.14 * up);
      sw('chestPitch', 0.1 + 0.1 * lt + 0.015 * Math.sin(t * 1.25)); sw('chestYaw', -0.05); sw('chestRoll', -0.04 + 0.01 * Math.sin(t * 0.5));
      sw('shin', 0.05 * Math.sin(t * 1.4)); sw('legL', 0.02 * Math.sin(t * 1.4));
      sw('rz', 0.07 + 0.03 * lt); sw('lz', 0.06 + 0.03 * lt); sw('ry', 0.03);
      sw('widen', 0.25 * (1 - up));
      sw('eyeY', -IRIS * 0.85 * (1 - up)); sw('eyeX', (caret - 0.5) * 2 * IRIS * 0.75 * (1 - up));
    } else if (s === 'thinking') {
      // "hmm": a small turn toward the city (face stays readable), eyes up and to the side, a slow head bob,
      // knuckles to the near cheek in front of the green face plate; foot swinging
      sw('lookW', 0.6);
      sw('headYaw', lookYaw * 0.4 + 0.22 * BUBBLE_SIDE + 0.04 * Math.sin(t * 0.5)); sw('headPitch', lookPitch * 0.4 + 0.1 + 0.035 * Math.sin(t * 1.3)); sw('headRoll', -0.16 + 0.03 * Math.sin(t * 0.9));
      sw('chestPitch', 0.02); sw('chestRoll', 0.04);
      const d = offR(-0.40, 1.30, 0.46);
      sw('rx', d[0]); sw('ry', d[1]); sw('rz', d[2]); sw('handRp', -1.1 + 0.25 * Math.max(0, Math.sin(t * 6))); sw('handRr', 0.7);
      sw('shin', 0.22 * Math.sin(t * 2.6) + 0.05);
      sw('eyeY', IRIS * 1.0); sw('eyeX', IRIS * 0.9 * BUBBLE_SIDE); sw('lid', 0.12);
    } else if (s === 'talking') {
      // a talking body layer under the per-syllable beats: leaning in, slow torso sway; the phrase pose (picked at
      // each phrase start) adds a held head turn / tilt per phrase
      sw('lookW', 1);
      sw('headYaw', 0.06 * Math.sin(t * 0.9) + phraseNow.yaw); sw('headPitch', 0.04 * Math.sin(t * 1.7 + 1) + phraseNow.pitch); sw('headRoll', 0.06 + 0.05 * Math.sin(t * 0.9) + phraseNow.roll);
      sw('chestPitch', 0.08 + 0.025 * Math.sin(t * 1.3)); sw('chestYaw', 0.07 * Math.sin(t * 0.6) + phraseNow.yaw * 0.4); sw('chestRoll', 0.04 * Math.sin(t * 0.47 + 1) + phraseNow.roll * 0.3);
      sw('shin', 0.12 * Math.sin(t * 2.4)); sw('legL', 0.04 * Math.sin(t * 2.4 + 0.8));
      sw('lx', 0.012 * Math.sin(t * 0.8)); sw('ly', 0.01 * Math.sin(t * 1.1 + 1));
    }
  };

  // ---------------------------------------------------------------------------------------------
  // speech: per character -> syllable envelope, beats, gestures
  // ---------------------------------------------------------------------------------------------
  const OPEN = { a: 1, o: 0.9, e: 0.72, i: 0.5, u: 0.6, y: 0.5, 'á': 1, 'é': 0.72, 'í': 0.5, 'ó': 0.9, 'ú': 0.6 };
  const hairKick = (k) => { hairLag.yaw.v += rand(-1, 1) * k; hairLag.pitch.v += -k * rand(0.6, 1.2); hairLag.roll.v += rand(-1, 1) * k * 0.7; };
  // jaw beats: a vowel onset starts a short open-then-close pulse (~60 ms up, ~110 ms down to ~0.05). Onsets closer
  // than SYL_MIN to the last one are merged (one pending beat fires when the gap allows), so the jaw shuts between
  // beats instead of fluttering half-open; m/b/p (and punctuation) snap it shut, f/v nearly
  const onset = (amp) => { pulseT = time; pulseAmp = amp; pendingAmp = 0; S.jaw.v += 3 * amp; };
  const closeMouth = (to) => { closeT = time; closeTo = to; pendingAmp = 0; };
  const jawEnv = () => {
    const s0 = time - pulseT;
    let e = s0 < 0.06 ? s0 / 0.06 : s0 < 0.085 ? 1 : 1 - (s0 - 0.085) / 0.11;
    e = clamp(e, 0, 1); e = e * e * (3 - 2 * e);
    let j = pulseAmp * (0.05 + 0.95 * e) * (s0 < 0.6 ? 1 : 0);
    if (closeT > pulseT - 0.001) { const k = Math.exp(-(time - closeT) * 45); j = Math.min(j, closeTo + Math.max(0, j - closeTo) * k); }
    return j;
  };
  const PHRASE_GESTURES = { R: ['explainR', 'palmR', 'flickR', 'explainR'], L: ['explainL', 'countL', 'palmL', 'flickL'] };
  const pickGesture = (both) => {
    if (both && Math.random() < 0.25) return 'bothOpen';
    const side = lastGestureSide === 'R' ? 'L' : 'R'; lastGestureSide = side;
    const l = PHRASE_GESTURES[side]; return l[Math.floor(Math.random() * l.length)];
  };
  const endWord = () => {
    if (wordLen > 0) {
      const amp = (wordLen >= 6 ? 1 : wordLen >= 4 ? 0.7 : 0.4) * rand(0.7, 1.15);
      kick('headPitch', -2.2 * amp); kick('headYaw', rand(-0.7, 0.7) * amp); kick('headRoll', rand(-0.5, 0.5) * amp);
      hairKick(0.5 * amp);
      // stressed long word: a small forward lean, like pressing the point
      if (wordLen > 6) addClip(0.55, (p, o) => { o.chestPitch += 0.08 * env(p, 0.3, 0.55); o.headPitch += -0.04 * env(p, 0.3, 0.55); }, 'lean');
      const free = !clips.some((c) => c.tag === 'gesture');
      // the first gesture early; a fresh one at most sentence starts (alternating hands) while there's enough left
      // to say; now and then one mid-sentence
      let pr = 0;
      if (gesturesThisReply === 0 && totalChars > 14) pr = 0.9;
      else if (sentenceGate && ahead > 15) pr = 0.7;
      else if (charsSinceGesture > 28) pr = 0.3;
      sentenceGate = false;
      if (free && charsSinceGesture > 8 && Math.random() < pr) {
        startGesture(pickGesture(gesturesThisReply > 0), rand(1.2, 1.7));
        charsSinceGesture = 0; gesturesThisReply++;
      }
    }
    wordLen = 0; wordVowels = 0; prevVowel = false;
  };
  // at phrase boundaries the eyes flick toward her bubble for a moment, then back to the viewer
  const phraseGlance = () => { if (Math.random() < 0.45) { bubbleGlanceUntil = time + rand(0.35, 0.6); } };
  const newPhrase = () => {
    phrasePose.yaw = rand(-0.12, 0.12); phrasePose.roll = rand(-0.12, 0.12); phrasePose.pitch = rand(-0.05, 0.04);
    phraseStart = false;
  };
  const endPhrase = (sentence) => { phraseStart = true; if (sentence) sentenceGate = true; };
  const speakChar = (ch) => {
    const c = ch.toLowerCase();
    talkRecent = 0.4; totalChars++; charsSinceGesture++; lastCharAt = time;
    const isLetter = /\p{L}|[0-9]/u.test(c);
    if (isLetter && phraseStart) newPhrase();
    if (OPEN[c] !== undefined || /[0-9]/.test(c)) {
      const o = OPEN[c] !== undefined ? OPEN[c] : 0.6;
      if (!prevVowel) { // syllable onset: a jaw beat (or a merged one), the head dips; stressed ones bob the shoulders
        wordVowels++;
        const stressed = wordVowels === 1 && wordLen >= 2;
        kick('headPitch', stressed ? -1.0 : -0.4);
        if (stressed) { kick('headYaw', rand(-0.4, 0.4)); kick('headRoll', rand(-0.3, 0.3)); kick('shUpR', 0.35); kick('shUpL', 0.3); kick('chestPitch', 0.25); }
        const amp = o * rand(0.85, 1.12);
        if (time - pulseT >= SYL_MIN && time - closeT >= 0.045) onset(amp); else pendingAmp = Math.max(pendingAmp, amp);
      }
      prevVowel = true; wordLen++;
    } else if (/\p{L}/u.test(c)) {
      prevVowel = false; wordLen++;
      if ('mbp'.includes(c)) closeMouth(0);
      else if ('fv'.includes(c)) closeMouth(0.1);
    } else if (c === ' ' || c === '\n' || c === '\t') { endWord(); }
    else if (',;:'.includes(c)) { closeMouth(0); endWord(); endPhrase(false); kick('headPitch', -1.0); phraseGlance(); }
    else if (c === '.' || c === '…') { closeMouth(0); endWord(); endPhrase(true); addClip(0.6, (p, o) => { o.headPitch += -0.09 * env(p, 0.2, 0.6); }, 'settle'); phraseGlance(); }
    else if (c === '!') {
      closeMouth(0); endWord(); endPhrase(true); kick('headPitch', -4); kick('headRoll', rand(-1.2, 1.2)); hairKick(1.6);
      addClip(0.7, (p, o) => { o.widen += 0.6 * env(p, 0.1, 0.6); }, 'bang');
      if (Math.random() < 0.5 && !clips.some((c2) => c2.tag === 'gesture')) { startGesture(lastGestureSide === 'R' ? 'flickL' : 'flickR', 0.7); lastGestureSide = lastGestureSide === 'R' ? 'L' : 'R'; }
    } else if (c === '?') {
      closeMouth(0); endWord(); endPhrase(true);
      const dir = Math.random() < 0.7 ? 1 : -1;
      addClip(1.4, (p, o) => { const e = env(p, 0.18, 0.4); o.headRoll += 0.2 * dir * e; o.headPitch += 0.06 * e; o.widen += 0.35 * e; o.chestRoll += 0.02 * dir * e; }, 'question');
    } else if (c === '-' || c === '—' || c === '–') { closeMouth(0); endWord(); endPhrase(false); }
  };

  // ---------------------------------------------------------------------------------------------
  // arm IK (two bones, in the chest's rest frame)
  // ---------------------------------------------------------------------------------------------
  const arm = (side) => {
    const S0 = v3(ANATOMY['shoulder' + side]), E0 = v3(ANATOMY['elbow' + side]), W0 = v3(ANATOMY['wrist' + side]);
    const l1 = S0.distanceTo(E0), l2 = E0.distanceTo(W0);
    const sw = W0.clone().sub(S0).normalize();
    const restPole = E0.clone().sub(S0); restPole.addScaledVector(sw, -restPole.dot(sw)).normalize();
    const gesturePole = new THREE.Vector3(side === 'R' ? -0.55 : 0.55, -0.55, -0.6).normalize();
    return { S0, E0, W0, l1, l2, restPole, gesturePole, up1: E0.clone().sub(S0).normalize(), fore: W0.clone().sub(E0).normalize() };
  };
  const ARMS = { R: arm('R'), L: arm('L') };
  const _w = new THREE.Vector3(), _t = new THREE.Vector3(), _u = new THREE.Vector3(), _pole = new THREE.Vector3(), _e = new THREE.Vector3(), _d = new THREE.Vector3(), _q = new THREE.Quaternion(), _qi = new THREE.Quaternion();
  const solveArm = (side, target, shoulderOffset, qShoulder, qElbow) => {
    const A = ARMS[side]; const S0 = _e.copy(A.S0).add(shoulderOffset);
    _t.copy(target).sub(S0); let d = _t.length();
    d = clamp(d, Math.abs(A.l1 - A.l2) + 0.01, A.l1 + A.l2 - 0.002); _u.copy(_t).normalize();
    const moved = target.distanceTo(A.W0);
    _pole.copy(A.restPole).lerp(A.gesturePole, smooth(0.0, 0.2, moved));
    _pole.addScaledVector(_u, -_pole.dot(_u)).normalize();
    const a = (A.l1 * A.l1 - A.l2 * A.l2 + d * d) / (2 * d), h = Math.sqrt(Math.max(0, A.l1 * A.l1 - a * a));
    const elbow = _d.copy(_u).multiplyScalar(a).addScaledVector(_pole, h); // relative to shoulder
    qShoulder.setFromUnitVectors(A.up1, _w.copy(elbow).normalize());
    const fore = _w.copy(_u).multiplyScalar(d).sub(elbow).normalize(); // wrist - elbow
    _qi.copy(qShoulder).invert(); fore.applyQuaternion(_qi);
    qElbow.setFromUnitVectors(A.fore, fore);
  };

  // ---------------------------------------------------------------------------------------------
  const Y = new THREE.Vector3(0, 1, 0), X = new THREE.Vector3(1, 0, 0), Z = new THREE.Vector3(0, 0, 1);
  const lateral = new THREE.Vector3().crossVectors(faceDir, Y).normalize();
  const faceYaw0 = Math.atan2(faceDir.x, faceDir.z), facePitch0 = Math.asin(faceDir.y);
  const eyeCenter = eyeA.clone().add(eyeB).multiplyScalar(0.5);
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qc = new THREE.Quaternion();
  const headRot = (out, yaw, pitch, roll) => {
    qa.setFromAxisAngle(Y, yaw); qb.setFromAxisAngle(lateral, pitch); qc.setFromAxisAngle(faceDir, roll);
    return out.copy(qa).multiply(qb).multiply(qc);
  };
  const _eu = new THREE.Euler(0, 0, 0, 'YXZ');
  const eulerQ = (out, x, y, z) => out.setFromEuler(_eu.set(x, y, z, 'YXZ'));
  const invRoot = new THREE.Matrix4(), camM = new THREE.Vector3(), wristT = new THREE.Vector3(), shOff = new THREE.Vector3();
  const shinRest = -0.75, footRest = 0.3; // swing the dangling shin forward so it hangs clear of a ledge face
  const qS = new THREE.Quaternion(), qE = new THREE.Quaternion();
  let lookYaw = 0, lookPitch = 0;
  const override = {}; // debug: force channel values
  const grip = [0.2, 0.2]; // per hand (R, L): 0 = fingers as modelled (open), 1 = fist
  const fingerFx = [{ greet: 0, splay: 0 }, { greet: 0, splay: 0 }], _fq = new THREE.Quaternion();
  // ----- posture layer (POSTURES above) -----
  // def: { hips|spine|chest|neck: [x,y,z] Euler, legR|legL: { foot: [x,y,z], pole: [x,y,z] } (rig space: the foot /
  //   shin end goes there, the knee bends toward pole), footR: [x,y,z], armR|armL: { wrist: [x,y,z] } (rig space) or
  //   [dx,dy,dz] (offset from the rest wrist), handR|handL: [pitch, roll], root: [x,y,z], rootRot: [x,y,z], legSwing, bob }
  let postureName = 'seated', PD = {};
  const posePtsCache = {};
  const _pq = new THREE.Quaternion(), _pq2 = new THREE.Quaternion(), _pe = new THREE.Euler(0, 0, 0, 'YXZ');
  const _pv = new THREE.Vector3(), _pv2 = new THREE.Vector3(), _pv3 = new THREE.Vector3(), _pm = new THREE.Matrix4(), _pm2 = new THREE.Matrix4();
  const prePosture = (bone, e) => { if (e && (e[0] || e[1] || e[2])) bone.quaternion.premultiply(_pq.setFromEuler(_pe.set(e[0], e[1], e[2], 'YXZ'))); };
  const LEG = { R: { thigh: 'thighR', shin: 'shinR', end: v3(ANATOMY.ankleR) }, L: { thigh: 'thighL', shin: 'shinL', end: v3([-0.56, 0.50, 0.58]) } };
  for (const k of ['R', 'L']) {
    const L = LEG[k], hip = rest[L.thigh].worldPos, knee = rest[L.shin].worldPos;
    L.hip = hip; L.l1 = hip.distanceTo(knee); L.l2 = knee.distanceTo(L.end);
    L.d1 = knee.clone().sub(hip).normalize(); L.d2 = L.end.clone().sub(knee).normalize();
  }
  const hipsPivot = rest.hips.worldPos;
  const solveLeg = (k, foot, pole) => { // two-bone IK in the hips' frame (after the posture's hips rotation)
    const L = LEG[k], qh = _pq2.copy(B.hips.quaternion).invert();
    const f = _pv.fromArray(foot).sub(hipsPivot).applyQuaternion(qh).add(hipsPivot).sub(L.hip);
    const pl = _pv2.fromArray(pole || [0, 0, 1]).applyQuaternion(qh);
    let d = f.length(); d = clamp(d, Math.abs(L.l1 - L.l2) + 0.01, L.l1 + L.l2 - 0.002); const u = f.normalize();
    pl.addScaledVector(u, -pl.dot(u)).normalize();
    const a = (L.l1 * L.l1 - L.l2 * L.l2 + d * d) / (2 * d), h = Math.sqrt(Math.max(0, L.l1 * L.l1 - a * a));
    const kneeV = _pv3.copy(u).multiplyScalar(a).addScaledVector(pl, h);
    const qT = B[L.thigh].quaternion.setFromUnitVectors(L.d1, kneeV.clone().normalize());
    const shinDir = u.multiplyScalar(d).sub(kneeV).normalize();
    const cur = L.d2.clone().applyQuaternion(qT), R = new THREE.Quaternion().setFromUnitVectors(cur, shinDir);
    B[L.shin].quaternion.copy(qT).invert().multiply(R).multiply(qT);
  };
  // whole-rig placement: rootRot (+ a slow zero-g drift) about PD.pivot (rig space), which lands at PD.at (root
  // space); without a pivot the rig turns about its origin and moves by PD.root. bob: a vertical sine (m).
  const _rq = new THREE.Quaternion(), _rv = new THREE.Vector3(), _re = new THREE.Euler(0, 0, 0, 'YXZ');
  const driftSin = (a, i, t) => (a && a[i] ? a[i] * Math.sin(t * PD.drift.f[i] + i * 1.7) : 0);
  let faceYaw = 0, snapFace = true, poseFaceYaw = 0;
  const _mq = new THREE.Quaternion(), _mq2 = new THREE.Quaternion(), _mv = new THREE.Vector3();
  const facePoseYaw = (def) => { // yaw of her face direction once the posture's whole-rig rotation is applied
    const r = def.rootRot || [0, 0, 0];
    const f = faceDir.clone().applyEuler(new THREE.Euler(r[0], r[1], r[2], 'YXZ'));
    return Math.atan2(f.x, f.z);
  };
  const placeRig = (t) => {
    const prr = PD.rootRot || [0, 0, 0], D = PD.drift, bob = (PD.bob || 0) * Math.sin(t * 1.7);
    const rot = [0, 1, 2].map((i) => prr[i] + (D ? driftSin(D.rot, i, t) : 0));
    if (PD.faceCamera && lastCamera) {
      // turn her about her pivot so her FACE points at the camera (the face sits ~63deg to her right of her body,
      // and a flipped pose mirrors that), eased; ledge poses only turn up to maxTurn so their feet stay on the ledge
      root.updateWorldMatrix(true, false); lastCamera.getWorldPosition(_rv); root.worldToLocal(_rv);
      const at = PD.at || PD.pivot || [0, 0, 0];
      // faceBy 'body': square her body to the camera instead (her head still turns to look); else aim the face
      let want = wrapPi(Math.atan2(_rv.x - at[0], _rv.z - at[2]) - (PD.faceBy === 'body' ? 0 : poseFaceYaw));
      if (PD.maxTurn !== undefined) want = clamp(want, -PD.maxTurn, PD.maxTurn);
      faceYaw += wrapPi(want - faceYaw) * (snapFace ? 1 : 0.05); snapFace = false;
      rot[1] += faceYaw;
    }
    const dp = [0, 1, 2].map((i) => (D ? driftSin(D.pos, i, t + 3.1) : 0));
    _rq.setFromEuler(_re.set(rot[0], rot[1], rot[2], 'YXZ')); rig.quaternion.copy(_rq);
    if (PD.pivot) {
      _rv.fromArray(PD.pivot).applyQuaternion(_rq);
      rig.position.fromArray(PD.at || PD.pivot).sub(_rv);
    } else rig.position.fromArray(PD.root || [0, 0, 0]);
    rig.position.x += dp[0]; rig.position.y += dp[1] + bob; rig.position.z += dp[2];
  };
  const chestRest = () => rest.chest.worldPos;
  const toChestRest = (p, out) => { // rig-space point -> the chest's rest frame (what the arm IK expects)
    B.chest.updateWorldMatrix(true, false); rig.updateWorldMatrix(true, false);
    _pm.copy(rig.matrixWorld).invert().multiply(B.chest.matrixWorld); _pm2.copy(_pm).invert();
    return out.fromArray(p).applyMatrix4(_pm2).add(chestRest());
  };
  const OUTLINE_BASE = new THREE.Color(0x2a0618); // added to the black hull (sRGB hex -> linear)

  function update(dt, t) {
    dt = clamp(dt || 0, 0, 0.1); time = t !== undefined ? t : time + dt;
    // state blend
    for (const k in W) W[k] += ((k === state ? 1 : 0) - W[k]) * (1 - Math.exp(-dt * 5));
    const O = zero();

    // look target -> yaw/pitch relative to the face's rest direction
    root.updateMatrixWorld();
    // the look target in the chest's rest frame (the frame the neck / head rotations are written in), so the head
    // still finds the camera when a posture rolls or flips her torso (recline, upside-down float)
    B.chest.updateWorldMatrix(true, false); invRoot.copy(B.chest.matrixWorld).invert();
    let haveTarget = false;
    if (lookTarget) { camM.copy(lookTarget).applyMatrix4(invRoot).add(rest.chest.worldPos); haveTarget = true; }
    else if (lastCamera) { lastCamera.getWorldPosition(camM); camM.applyMatrix4(invRoot).add(rest.chest.worldPos); haveTarget = true; }
    if (haveTarget) {
      const d = camM.sub(eyeCenter);
      const ly = PD.lookYawMax || 0.8; // how far her head may turn to find the camera (float: body square, head does the rest)
      lookYaw = clamp(wrapPi(Math.atan2(d.x, d.z) - faceYaw0), -ly, ly);
      lookPitch = clamp(Math.atan2(d.y, Math.hypot(d.x, d.z)) - facePitch0, -0.35, 0.45);
    } else { lookYaw = 0; lookPitch = 0; }
    // phrase pose: eases toward the pose picked at the phrase start (slow, so it reads as a held attitude)
    const pk = 1 - Math.exp(-dt * 3.2), talkingNow = state === 'talking';
    for (const k of ['yaw', 'roll', 'pitch']) phraseNow[k] += ((talkingNow ? phrasePose[k] : 0) - phraseNow[k]) * pk;
    for (const k in W) stateTargets(k, time, O, W[k]);
    // occasional glances away while idle: relative to the look direction, mostly toward the city / bubble side
    // (turning the other way swings her face into the hair mass), eyes first, the head ~80 ms later
    if (time > nextGlance) {
      glance = { yaw: rand(-0.15, 0.35) * BUBBLE_SIDE, pitch: rand(-0.08, 0.16), start: time + 0.08, until: time + rand(0.8, 1.5) };
      nextGlance = time + rand(5, 11);
    }
    const inGlance = time < glance.until;
    const gl = inGlance && time > glance.start ? W.idle : 0;
    O.headYaw += (lookYaw + O.lookYawBias) * O.lookW + glance.yaw * gl;
    O.headPitch += (lookPitch + O.lookPitchBias) * O.lookW + glance.pitch * gl;
    if (inGlance) { O.eyeX += clamp(glance.yaw * 0.03, -IRIS, IRIS) * W.idle; O.eyeY += glance.pitch * 0.03 * W.idle; }
    // talking: a quick look toward her bubble at phrase boundaries
    if (time < bubbleGlanceUntil) { O.eyeX += IRIS * 0.9 * BUBBLE_SIDE * W.talking; O.headYaw += 0.06 * BUBBLE_SIDE * W.talking; }
    // listening: a nod when the typing pauses; small 'mm' bobs while you type; now and then a glance up at you
    if (state === 'listening') {
      if (!typeNodDone && time - lastTypeAt > 0.7) { typeNodDone = true; api.react('nod'); nextMm = time + rand(1.6, 2.2); }
      else if (time > nextMm && time - lastTypeAt < 0.5) { addClip(REACTS.mm[0], REACTS.mm[1], 'react-mm'); nextMm = time + rand(1.5, 2.1); }
      if (time > nextLookUp && time - lastTypeAt < 0.6 && time - listenSince > 2) { lookUpUntil = time + 0.9; blinkT = 0; doubleBlink = false; nextLookUp = time + rand(4.5, 7); }
    }
    // talking, but the text has run dry for a moment (the model is slower than her speech): mouth shut, eyes on her bubble
    if (state === 'talking' && time - lastCharAt > 0.35) { const k = smooth(0.35, 0.6, time - lastCharAt); O.eyeX += IRIS * 0.75 * BUBBLE_SIDE * k * W.talking; O.headYaw += 0.04 * BUBBLE_SIDE * k * W.talking; }

    // saccades (big enough to see at the default framing)
    if (time > nextSaccade) {
      const m = W.talking > 0.5 ? 0.6 : 1;
      sacc = { x: rand(-1, 1) * IRIS * 0.8 * m, y: rand(-0.6, 0.6) * IRIS * 0.6 * m };
      nextSaccade = time + rand(0.6, 2.6); kick('headYaw', sacc.x * 25);
    }
    O.eyeX += sacc.x; O.eyeY += sacc.y;
    O.eyeX = clamp(O.eyeX, -IRIS * 1.1, IRIS * 1.1); O.eyeY = clamp(O.eyeY, -IRIS, IRIS);

    // clips
    for (let i = clips.length - 1; i >= 0; i--) { const c = clips[i]; c.t += dt; const p = c.t / c.dur; if (p >= 1) { clips.splice(i, 1); continue; } c.fn(p, O); }

    // speech envelope: per-syllable jaw beats (a merged onset fires once the minimum gap has passed)
    talkRecent = Math.max(0, talkRecent - dt);
    if (pendingAmp > 0 && talkRecent > 0 && time - pulseT >= SYL_MIN && time - closeT >= 0.045) onset(pendingAmp);
    if (talkRecent <= 0) pendingAmp = 0;
    jawEnvV = jawEnv();
    O.jaw += jawEnvV * 1.15;
    O.jaw = clamp(O.jaw, 0, 1.25);
    O.ry += handsOnLedge; O.ly += handsOnLedge;
    // never turn the face past her profile into the hair mass (her face starts ~63deg to her right)
    if (O.glitch < 0.05) O.headYaw = Math.max(O.headYaw, 0.12);

    // springs
    for (const k in CH) S[k].t = O[k];
    const n = Math.max(1, Math.ceil(dt / (1 / 120))), h = dt / n;
    for (let s = 0; s < n; s++) for (const k in CH) S[k].step(h);
    for (const k in override) if (S[k]) S[k].x = override[k];

    // blinks
    let blink = 0;
    if (time > nextBlink && blinkT < 0) { blinkT = 0; doubleBlink = Math.random() < 0.2; }
    if (blinkT >= 0) {
      blinkT += dt; const bt = blinkT;
      blink = bt < 0.07 ? bt / 0.07 : bt < 0.1 ? 1 : bt < 0.22 ? 1 - (bt - 0.1) / 0.12 : 0;
      if (bt > 0.22) { if (doubleBlink) { blinkT = 0; doubleBlink = false; } else { blinkT = -1; nextBlink = time + rand(2.2, 5.5); } }
    }
    // blinks squash the disc to a dark lid line (never to nothing); happy squints (laugh / smirk) squash it toward its
    // TOP edge so the eyes read as "^ ^" closed-happy rather than gone
    const widenK = 1 + 0.35 * S.widen.x;
    const blinkOpen = clamp((1 - blink) * widenK, 0.14, 1.45);
    const happy = clamp(O.lid, 0, 0.8), happyOpen = 1 - happy;
    const lidOpen = Math.max(0.14, Math.min(blinkOpen, happyOpen * widenK));
    const lidShift = EYE_R * (1 - happyOpen) * (blinkOpen > happyOpen * widenK ? 1 : 0.4);

    // ---- apply to bones ----
    headRot(B.neck.quaternion, S.headYaw.x * 0.35, S.headPitch.x * 0.35, S.headRoll.x * 0.35);
    headRot(B.head.quaternion, S.headYaw.x * 0.65, S.headPitch.x * 0.65, S.headRoll.x * 0.65);
    const breath = Math.sin(time * 1.6) * (W.talking > 0.5 ? 0.6 : 1);
    eulerQ(B.spine.quaternion, S.chestPitch.x * 0.4 + 0.006 * breath, S.chestYaw.x * 0.4, S.chestRoll.x * 0.4);
    eulerQ(B.chest.quaternion, S.chestPitch.x * 0.6 - 0.01 * breath, S.chestYaw.x * 0.6, S.chestRoll.x * 0.6);
    B.jaw.quaternion.setFromAxisAngle(mouthLine, jawSign * 0.5 * S.jaw.x);
    for (const k of ['A', 'B']) {
      const lid = B['lid' + k];
      lid.scale.set(1 + 0.08 * S.widen.x, lidOpen, 1);
      lid.position.copy(rest['lid' + k].pos).addScaledVector(lidUp[k], lidShift);
      B['iris' + k].position.set(S.eyeX.x, S.eyeY.x / Math.max(lidOpen, 0.3), 0);
    }
    // posture: torso and whole-rig offset (legs below, after the procedural leg swing)
    B.hips.quaternion.identity();
    prePosture(B.hips, PD.hips); prePosture(B.spine, PD.spine); prePosture(B.chest, PD.chest); prePosture(B.neck, PD.neck);
    placeRig(time);
    // arms
    for (const side of ['R', 'L']) {
      const l = side === 'R' ? 'r' : 'l';
      const up = S['shUp' + side].x + 0.004 * (1 + breath);
      shOff.set(0, up, 0);
      B['shoulder' + side].position.copy(rest['shoulder' + side].pos).add(shOff);
      const pa = PD['arm' + side];
      if (pa && pa.wrist) toChestRest(pa.wrist, wristT); else { wristT.copy(ARMS[side].W0); if (Array.isArray(pa)) wristT.add(_t.fromArray(pa)); }
      // a posture that places this hand (chin, knee, feet) keeps it there: talking gestures, reactions and idle sway
      // move it only 15% as far (they're written for the seated pose, where the hands rest on the ledge)
      const keep = PD['busy' + side] ?? (pa && pa.wrist ? 0.85 : 0), mv = 1 - keep;
      wristT.add(_t.set(S[l + 'x'].x * mv, (S[l + 'y'].x - handsOnLedge) * mv + handsOnLedge, S[l + 'z'].x * mv)).add(shOff);
      solveArm(side, wristT, shOff, qS, qE);
      B['shoulder' + side].quaternion.copy(qS); B['elbow' + side].quaternion.copy(qE);
      const ph = PD['hand' + side] || [0, 0];
      eulerQ(B['hand' + side].quaternion, S['hand' + side + 'p'].x + ph[0], 0, S['hand' + side + 'r'].x + ph[1]);
      // grip: ease toward the posture's grip (relaxed claws by default), opening up while the hand gestures
      const off = Math.hypot(S[l + 'x'].x, S[l + 'y'].x - handsOnLedge, S[l + 'z'].x) * mv;
      const gi = side === 'R' ? 0 : 1, want = (PD.grip ? PD.grip[gi] : 0.2) * (1 - 0.85 * smooth(0.05, 0.16, off));
      grip[gi] += (want - grip[gi]) * (1 - Math.exp(-dt * 8));
      // finger life, per finger so they never move as one block: a loose rolling curl while the hand gestures, small
      // curls on speech beats, a flutter in the greeting wave, a splay (fan out, uncurl) when surprised or shrugging,
      // and a slow idle drift. Each finger also fans out from the middle by FX.spread.
      const fx = fingerFx[gi], on = (tag) => clips.some((c) => c.tag === tag), ease = 1 - Math.exp(-dt * 10);
      fx.greet += ((side === 'R' && on('react-greet') ? 1 : 0) - fx.greet) * ease;
      fx.splay += ((on('react-surprised') || on('react-shrug') ? 1 : 0) - fx.splay) * ease;
      const gk = smooth(0.05, 0.16, off), talk = W.talking * jawEnvV, n = FINGERS[side].length, mid = (n - 1) / 2;
      const spread = 0.04 + 0.1 * gk + 0.3 * fx.splay + 0.06 * fx.greet * Math.sin(time * 6);
      FINGERS[side].forEach((f, k) => {
        let c = grip[gi] * 1.5 * (0.85 + 0.06 * k);
        c += gk * (0.2 + 0.14 * Math.sin(time * 2.3 + k * 0.8 + gi * 1.7));            // loose, rolling curl in gestures
        c += talk * (0.1 + 0.08 * Math.sin(k * 1.3 + time * 7)) * (0.4 + 0.6 * gk);      // speech beats
        c += fx.greet * 0.32 * Math.sin(time * 11 + k * 1.1);                            // greeting flutter
        c += 0.045 * Math.sin(time * 0.7 + k * 1.9 + gi * 3);                            // idle drift
        c *= 1 - 0.75 * fx.splay;                                                        // splay: uncurl
        f.bone.quaternion.setFromAxisAngle(f.axis, clamp(c, -0.15, 1.6));
        f.bone.quaternion.premultiply(_fq.setFromAxisAngle(f.palm, spread * (k - mid) * 0.35 * (side === 'R' ? 1 : -1)));
      });
    }
    // legs
    const ls = PD.legSwing ?? 1;
    B.shinR.quaternion.setFromAxisAngle(X, shinRest - S.shin.x * ls);
    B.footR.quaternion.setFromAxisAngle(X, footRest + 0.4 * S.shin.x * ls);
    B.shinL.quaternion.setFromAxisAngle(X, S.legL.x * ls);
    B.thighR.quaternion.identity(); B.thighL.quaternion.identity();
    for (const k of ['R', 'L']) if (PD['leg' + k]) solveLeg(k, PD['leg' + k].foot, PD['leg' + k].pole);
    prePosture(B.footR, PD.footR);
    // hair: lags behind the head, overshoots a little, drifts
    hairLag.yaw.t = S.headYaw.x + S.chestYaw.x; hairLag.pitch.t = S.headPitch.x + S.chestPitch.x; hairLag.roll.t = S.headRoll.x + S.chestRoll.x;
    if (O.hairKick) { hairLag.pitch.v -= O.hairKick * dt * 14; hairLag.roll.v += O.hairKick * dt * 6 * Math.sin(time * 9); }
    for (let s = 0; s < n; s++) { hairLag.yaw.step(h); hairLag.pitch.step(h); hairLag.roll.step(h); }
    const LAG = 1.5; // head beats ripple visibly through the spikes
    const hy = clamp((hairLag.yaw.x - hairLag.yaw.t) * LAG, -0.3, 0.3), hp = clamp((hairLag.pitch.x - hairLag.pitch.t) * LAG, -0.3, 0.3), hr = clamp((hairLag.roll.x - hairLag.roll.t) * LAG, -0.3, 0.3);
    const drift = 0.03 * Math.sin(time * 0.6), drift2 = 0.025 * Math.sin(time * 0.43 + 1);
    headRot(B.hairBack.quaternion, hy + drift2, -hp * 0.8 + drift * 0.5, hr + drift);
    headRot(B.hairFront.quaternion, hy * 1.2 - drift2, -hp + drift * 0.4, hr * 1.2 - drift * 0.6);
    // floating shards bob
    if (B.shardR) { B.shardR.position.copy(rest.shardR.pos); B.shardR.position.y += 0.02 * Math.sin(time * 1.3); B.shardR.rotation.set(0.05 * Math.sin(time * 0.9), 0.1 * Math.sin(time * 0.6), 0); }
    if (B.shardL) { B.shardL.position.copy(rest.shardL.pos); B.shardL.position.y += 0.02 * Math.sin(time * 1.1 + 2); B.shardL.rotation.set(0.05 * Math.sin(time * 0.8 + 1), 0.1 * Math.sin(time * 0.5 + 2), 0); }

    // shader uniforms
    U.uTime.value = time;
    U.uGlitch.value = clamp(O.glitch, 0, 1);
    const talkGlow = W.talking * (0.55 + 0.45 * S.jaw.x);
    U.uTalkGlow.value = talkGlow * 0.35;
    // (the outline is a very dark magenta, not pure black, so her silhouette separates from the blue-black night)
    U.uTint.value.setHex(PALETTE.pink).multiplyScalar(0.22 * talkGlow + 0.5 * U.uGlitch.value).add(OUTLINE_BASE);
    // measure where her face points with the posture on (chest frame: posture and rig turn included, the head's own
    // look-at excluded), so faceCamera aims the real face, not an estimate; used from the next frame on
    if (PD.faceCamera) {
      B.chest.updateWorldMatrix(true, false); root.getWorldQuaternion(_mq); B.chest.getWorldQuaternion(_mq2);
      _mv.copy(faceDir).applyQuaternion(_mq.invert().multiply(_mq2));
      poseFaceYaw = wrapPi(Math.atan2(_mv.x, _mv.z) - faceYaw);
    }
    tpUpdate(dt);
  }

  // ---------------------------------------------------------------------------------------------
  // TELEPORT: a digital-construct jump (the show's Capri glitches in and out of existence).
  //   out (~0.32 s): two pre-glitch flickers (slice jitter + a sideways hop) -> slices slide apart both ways with an
  //     RGB split (the ghost layers) while her flat colours blow out to pink / cyan, crackling red arcs off her
  //     fingertips -> the slices thin to scanlines, stretch sideways and blink out on a hard pink-white flash
  //     (+ a thin vertical scan line through her height, the flash's light on the ledge) -> 'gone'.
  //   gone: invisible; a cyan / pink after-image of her silhouette lingers at the old spot, breaking up (~0.6 s).
  //   in (~0.4 s): second flash; white scanlines snap in from one side, top first, thicken back into slices,
  //     overshoot-jitter, the colours settle from white through pink back to her flat look; a last tick of glitch.
  // Reduced motion: an ordered-dither crossfade, no flashes / arcs / ghosts.
  // All timing runs on update()'s clock (so a virtual clock can step it frame by frame).
  // ---------------------------------------------------------------------------------------------
  const fx = createTeleportFx(scene);
  const I4 = new THREE.Matrix4();
  const tp = {
    phase: null, t: 0, dur: 0, resolve: null, promise: null, reduced: false, side: 1,
    flashAt: -10, flashSeedT: -10, surface: { point: new THREE.Vector3(), yaw: 0, w: 1.5 }, burstPos: new THREE.Vector3(), burstH: 2.4,
    ghostT: -1, oldM: new THREE.Matrix4(), arcNext: 0, waits: [], seed: 0, k: 1,
    out: { phase: null, flash: 0, glitch: 0, visible: 1, active: false },
  };
  const seg = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
  const eOut = (x) => 1 - (1 - x) * (1 - x);
  const flashEnv = (d) => (d < -0.045 ? 0 : d < 0 ? Math.pow(1 + d / 0.045, 2) : Math.exp(-d / 0.07));
  const lineEnv = (d) => (d < -0.06 ? 0 : d < 0 ? (1 + d / 0.06) : Math.exp(-d / 0.035)); // (gone ~0.1 s after each flash)
  const _tbc = new THREE.Vector3(), _thp = new THREE.Vector3(), _tep = new THREE.Vector3(), _tcp = new THREE.Vector3(), _tq = new THREE.Vector3(), _tm = new THREE.Matrix4();
  const reducedNow = () => { try { return matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; } };
  function defaultSurface(out) {
    root.updateMatrixWorld();
    out.point.copy(v3(ANATOMY.seatPoint)).applyMatrix4(rig.matrixWorld);
    out.yaw = root.rotation.y - 0.71; // her yaw on a ledge is ~ledge yaw + 0.71 (see main.js / city perches)
    return out;
  }
  function setSurface(s) {
    if (s && s.point) { tp.surface.point.copy(s.point); tp.surface.yaw = s.yaw || 0; tp.surface.w = s.w || 1.5; } else { defaultSurface(tp.surface); tp.surface.w = 1.5; }
  }
  const _tbox = new THREE.Box3();
  function bodyCenter(out) {
    root.updateMatrixWorld();
    const b = _tbox.copy(restBox).applyMatrix4(rig.matrixWorld); b.getCenter(out); tp.burstH = b.max.y - b.min.y; return out;
  }
  function fingertip(side, out) {
    const h = B['hand' + side], e = B['elbow' + side];
    h.getWorldPosition(out); e.getWorldPosition(_tep);
    return out.addScaledVector(_tq.copy(out).sub(_tep).normalize(), 0.13 * root.scale.x);
  }
  function crackle(k, hero) {
    // rebuild the arcs: from her fingertips (the show's red lightning) and a couple of body points, outward / down.
    // hero: one long, thick bolt from her right fingertip down to the ledge (fired just before the first flash, while
    // she's still recognisable, so for a frame or two it is the brightest thing on screen)
    const s = root.scale.x, n = hero ? 3 : 3 + ((Math.random() * 3) | 0);
    bodyCenter(_tbc);
    for (let a = 0; a < fx.arcList.length; a++) {
      const arc = fx.arcList[a];
      if (a >= n) { arc.on = false; continue; }
      if (a < 2) fingertip(a === 0 ? 'R' : 'L', _thp);
      else { const bn = ['chest', 'head', 'hips', 'spine', 'hairBack'][(Math.random() * 5) | 0]; B[bn].getWorldPosition(_thp); _thp.x += rand(-0.12, 0.12) * s; _thp.y += rand(-0.1, 0.15) * s; _thp.z += rand(-0.1, 0.1) * s; }
      if (hero && a === 0) {
        // down past the ledge front, out to her side
        _tep.copy(tp.surface.point).add(_tq.set(rand(-0.9, -0.5), rand(-0.55, -0.25), rand(0.05, 0.3)).applyAxisAngle(Y, tp.surface.yaw).multiplyScalar(s));
        fx.setArc(arc, _thp, _tep, 0.26, 0.042 * s, 1.7);
        continue;
      }
      if (a === 1 && Math.random() < 0.5) _tep.copy(tp.surface.point).add(_tq.set(rand(-0.5, 0.5), 0, rand(-0.3, 0)).multiplyScalar(s));
      else {
        _tq.copy(_thp).sub(_tbc).setY(0); if (_tq.lengthSq() < 1e-4) _tq.set(rand(-1, 1), 0, rand(-1, 1)); _tq.normalize();
        _tq.x += rand(-0.6, 0.6); _tq.z += rand(-0.6, 0.6); _tq.y = rand(-0.9, 0.5); _tq.normalize();
        _tep.copy(_thp).addScaledVector(_tq, rand(0.3, 0.8) * s);
      }
      fx.setArc(arc, _thp, _tep, 0.32, rand(0.018, 0.03) * s, k * rand(0.6, 1.1));
    }
  }
  function startPhase(phase, dur, opts) {
    if (tp.resolve) { const r = tp.resolve; tp.resolve = null; r(); }
    tp.phase = phase; tp.t = 0; tp.dur = dur; tp.reduced = opts.reduced !== undefined ? !!opts.reduced : reducedNow();
    tp.k = clamp(1.48 * (opts.fx ?? 1), 0.5, 1);
    setSurface(opts.surface);
    tp.arcNext = 0; tp.heroDone = false;
    tp.promise = new Promise((res) => { tp.resolve = res; });
    return tp.promise;
  }
  function tpUpdate(dt) {
    let shift = 0, dir = 0, jit = 0, jx = 0, gone = 0, sweep = 0, squash = 0, stretch = 0, hot = 0, fade = 0, glitch = 0;
    let splitA = 0, splitOff = 0, arcK = 0, heroArc = false, calm = false;
    for (let i = tp.waits.length - 1; i >= 0; i--) if (time >= tp.waits[i].at) { const w = tp.waits.splice(i, 1)[0]; w.res(); }
    if (tp.phase === 'out') {
      tp.t += dt; const t = tp.t;
      if (tp.reduced) { fade = smooth(0, tp.dur, t); hot = 0.35 * fade; }
      else {
        // anticipation: two crisp sideways hops of her normal flat look (a little slice jitter, no colour blow-out,
        // hardly any screen glitch), then the shred starts at ~0.1 s
        // (0-0.1 s) calm: two crisp sideways hops of her flat look; (0.1-0.21) her slices jump to held offsets in two
        // discrete steps (re-shuffled per step, ~2 frames each) still in her own flat colours, thin red / cyan fringes
        // only; (0.21-0.25) the pink / white blow-out, thinning to scanlines, then the flash at 0.25 and the blink-out
        const f1 = t < 0.035, f2 = t > 0.065 && t < 0.095;
        glitch = f1 ? 0.12 : f2 ? 0.18 : t > 0.1 ? 0.22 + 0.2 * seg(t, 0.18, 0.25) : 0;
        jx = f1 ? 0.04 * tp.side : f2 ? -0.055 * tp.side : 0;
        jit = f1 || f2 ? 0.025 : 0; calm = t < 0.1;
        const st = t >= 0.1 && t < 0.21 ? Math.min(1, Math.floor((t - 0.1) / 0.055)) : -1;
        if (st >= 0) { shift = [0.13, 0.25][st]; U.uTpSeed.value = tp.seed + st * 7.31; }
        else if (t >= 0.21) { shift = 0.25 + 0.17 * seg(t, 0.21, 0.27); U.uTpSeed.value = tp.seed + 14.62; }
        hot = seg(t, 0.205, 0.24) + seg(t, 0.235, 0.265);
        gone = 0.3 * seg(t, 0.14, 0.24) + 0.7 * seg(t, 0.25, 0.32);
        squash = 0.88 * eOut(seg(t, 0.21, 0.26));
        stretch = 1.1 * Math.pow(seg(t, 0.23, 0.32), 2);
        sweep = 0.2;
        splitA = seg(t, 0.1, 0.14) * (1 - seg(t, 0.27, 0.32)); splitOff = 0.012 + 0.02 * seg(t, 0.1, 0.25);
        arcK = t > 0.11 && t < 0.27 ? 0.6 + 0.4 * seg(t, 0.11, 0.25) : 0;
        heroArc = t > 0.15 && t < 0.21;
      }
      if (t >= tp.dur) {
        tp.phase = 'gone'; root.updateMatrixWorld(); tp.oldM.copy(root.matrixWorld);
        if (!tp.reduced) tp.ghostT = 0;
        if (tp.resolve) { const r = tp.resolve; tp.resolve = null; r(); }
      }
    }
    if (tp.phase === 'gone') { if (tp.reduced) fade = 1; else gone = 1; }
    if (tp.phase === 'in') {
      tp.t += dt; const t = tp.t;
      if (tp.reduced) { fade = 1 - smooth(0, tp.dur, t); hot = 0.35 * fade; }
      else {
        // (slices converge from a moderate spread, about her own width: the arrival is a burst around her, not the frame)
        gone = 1 - seg(t, 0.0, 0.08); sweep = 0.65;
        squash = 0.88 * (1 - eOut(seg(t, 0.05, 0.2)));
        stretch = 0.6 * (1 - eOut(seg(t, 0.0, 0.16)));
        dir = 1; shift = 0.26 * (1 - eOut(seg(t, 0.0, 0.15)));
        jit = t > 0.12 ? 0.06 * Math.exp(-(t - 0.12) * 11) * Math.sin((t - 0.12) * 58) : 0;
        // (peak 1.6, not pure white: in a close shot she fills much of the frame)
        hot = t < 0.04 ? 1.4 : t < 0.12 ? 1.4 - 0.4 * seg(t, 0.04, 0.12) : 1 - eOut(seg(t, 0.12, 0.34));
        glitch = t > 0.26 && t < 0.29 ? 0.4 : 0;
        splitA = 1 - seg(t, 0.08, 0.3); splitOff = 0.04 * (1 - seg(t, 0.0, 0.3));
        arcK = t > 0.03 && t < 0.33 ? 1 - 0.6 * seg(t, 0.15, 0.33) : 0;
      }
      if (t >= tp.dur) { tp.phase = null; if (tp.resolve) { const r = tp.resolve; tp.resolve = null; r(); } }
    }
    // slices (scaled down when she is big on screen: opts.fx from main.js, so the break-up stays about her own size)
    shift *= tp.k; stretch *= tp.k; splitOff *= tp.k;
    U.uTpShift.value = shift; U.uTpDir.value = dir; U.uTpSide.value = tp.side; U.uTpJit.value = jit; U.uTpJx.value = jx;
    U.uTpGone.value = gone; U.uTpSweep.value = sweep; U.uTpSquash.value = squash; U.uTpStretch.value = stretch;
    U.uTpHot.value = hot; U.uTpFade.value = fade;
    const on = tp.phase !== null && !(tp.phase === 'gone' && false);
    U.uTpOn.value = on && !tp.reduced ? 1 : 0;
    if (glitch > U.uGlitch.value) { U.uGlitch.value = glitch; U.uTint.value.setHex(PALETTE.pink).multiplyScalar(0.5 * glitch).add(OUTLINE_BASE); }
    if (U.uTpOn.value && lastCamera) {
      lastCamera.updateMatrixWorld();
      bodyCenter(_tbc); U.uTpC.value.copy(_tbc).applyMatrix4(lastCamera.matrixWorldInverse);
    }
    // flash (burst + vertical scan line + light on the ledge)
    const df = time - tp.flashAt;
    const fI = tp.reduced ? 0 : flashEnv(df), lI = tp.reduced ? 0 : lineEnv(df);
    if (tp.phase === 'out' && df < 0 && df > -0.2) bodyCenter(tp.burstPos); // (in: set by teleportIn, root already moved)
    fx.burst.visible = fI > 0.01 || lI > 0.02;
    if (fx.burst.visible && lastCamera) {
      lastCamera.getWorldPosition(_tcp);
      fx.burst.position.copy(tp.burstPos).addScaledVector(_tq.copy(_tcp).sub(tp.burstPos).normalize(), 0.35);
      fx.burst.quaternion.copy(lastCamera.quaternion);
      // about her silhouette's size (it used to be a 7 m soft bloom): a hard horizontal blink-out line, a few
      // scanline bars, the vertical scan line; kept out of the bloom pre-render (main.js, teleportFx.bloomSkip)
      const s = root.scale.x, bw = 1.6 * s * (lastCamera && lastCamera.aspect < 0.8 ? 0.8 : 1);
      fx.burstU.uSize.value.set(bw, tp.burstH * 1.2); fx.burst.scale.set(bw, tp.burstH * 1.2, 1);
      fx.burstU.uH.value = tp.burstH * 1.02; fx.burstU.uCoreY.value = 0.12 * s;
      fx.burstU.uI.value = fI; fx.burstU.uLine.value = lI;
      if (tp.flashSeedT !== tp.flashAt) { tp.flashSeedT = tp.flashAt; fx.burstU.uSeed.value = Math.random() * 100; }
    }
    // arcs
    if (arcK > 0 && !tp.reduced) {
      if (heroArc && !tp.heroDone) { crackle(arcK, true); tp.heroDone = true; tp.arcNext = time + 0.05; }
      else if (time >= tp.arcNext) { crackle(arcK); tp.arcNext = time + rand(0.03, 0.055); }
      if (lastCamera) { lastCamera.getWorldPosition(_tcp); fx.buildArcs(_tcp); }
      fx.arcs.visible = true;
    } else fx.arcs.visible = false;
    // the flash's light on the ledge (+ a flicker while the arcs crackle)
    const dI = tp.reduced ? 0 : fI * 0.95 + (df > -0.05 ? 0.16 * Math.exp(-Math.max(0, df) / 0.12) : 0) + arcK * 0.12 * Math.random();
    fx.decal.visible = dI > 0.004;
    if (fx.decal.visible) { fx.decal.position.copy(tp.surface.point); fx.decal.rotation.set(0, tp.surface.yaw, 0); fx.decal.scale.set(root.scale.x * Math.min(1, (tp.surface.w || 1.5) / 1.5), root.scale.x, root.scale.x); fx.decalU.uI.value = dI; }
    // ghosts: RGB-split layers while she breaks up / reassembles, then the after-image
    const gA = ghostA.userData.u, gB = ghostB.userData.u;
    gA.uGhostTp.value = 1; gA.uGhostX.value.copy(I4); gA.uGhostRise.value = 0;
    gA.uGhostShift.value = -splitOff * tp.side; gA.uGhostA.value = splitA * 0.5;
    ghostA.visible = splitA > 0.003 && !tp.reduced;
    if (tp.ghostT >= 0) {
      tp.ghostT += dt;
      const g = tp.ghostT, a = 0.6 * Math.pow(Math.max(0, 1 - g / 0.6), 1.4) * seg(g, 0, 0.03);
      root.updateMatrixWorld(); _tm.copy(root.matrixWorld).invert();
      gB.uGhostTp.value = 0; gB.uGhostX.value.multiplyMatrices(tp.oldM, _tm);
      gB.uGhostRise.value = 0.1 * g * root.scale.x; gB.uGhostShift.value = 0;
      gB.uGhostCol.value.setHex(0x1fd8ff); gB.uGhostRim.value.setHex(0xff3a78);
      gB.uGhostA.value = a; ghostB.visible = a > 0.003;
      if (g > 0.6) { tp.ghostT = -1; ghostB.visible = false; }
    } else {
      gB.uGhostTp.value = 1; gB.uGhostX.value.copy(I4); gB.uGhostRise.value = 0;
      gB.uGhostCol.value.setHex(0x1fd8ff); gB.uGhostRim.value.setHex(0x1fd8ff);
      gB.uGhostShift.value = splitOff * tp.side; gB.uGhostA.value = tp.phase === 'out' ? splitA * 0.5 : 0;
      ghostB.visible = gB.uGhostA.value > 0.003 && !tp.reduced;
    }
    // for main.js: screen flash / glitch suggestions and how visible she is
    const o = tp.out;
    o.phase = tp.phase; o.active = tp.phase !== null || tp.ghostT >= 0 || fI > 0.01;
    o.flash = fI; o.line = lI; o.glitch = calm ? glitch * 0.25 : Math.max(glitch, fI);
    o.visible = tp.reduced ? 1 - fade : 1 - gone;
  }
  const after = (s) => new Promise((res) => tp.waits.push({ at: time + s, res }));

  let api = null;
  // jaw rotation sign: opening must move the mouth centre down the face
  let jawSign = 1;
  { const q = new THREE.Quaternion().setFromAxisAngle(mouthLine, 0.2); const p = mouthC.clone().sub(jawHinge).applyQuaternion(q).add(jawHinge); if (p.sub(mouthC).dot(mouthUp) > 0) jawSign = -1; }

  update(0, 0);

  // eslint-disable-next-line prefer-const
  api = {
    root,
    update,
    setState(s) {
      if (!(s in W) || s === state) return;
      const prev = state; state = s;
      if (s === 'talking') {
        charsSinceGesture = 12; gesturesThisReply = 0; totalChars = 0; phraseStart = true; sentenceGate = false;
        // snapping back from "hmm" to the viewer on the first word
        if (prev === 'thinking') { kick('headPitch', -2.5); kick('headYaw', -1.5 * BUBBLE_SIDE); sacc = { x: 0, y: 0 }; hairKick(1.2); }
        if (Math.random() < 0.4) setTimeout(() => { if (state === 'talking' && !clips.some((c) => c.tag === 'gesture')) { startGesture(Math.random() < 0.5 ? 'explainR' : 'palmR', 1.4); gesturesThisReply++; } }, 300);
      }
      if (prev === 'talking' && s !== 'talking') api.endUtterance();
      if (s === 'listening') { listenSince = time; nextMm = time + rand(1.2, 1.8); nextLookUp = time + rand(2.5, 4); if (prev === 'idle') { kick('headPitch', 1.2); kick('headRoll', 1.0); } typeNodDone = true; }
      if (s === 'thinking') hairKick(0.6);
    },
    // optional extras used by main.js: a keystroke in the prompt (small head kick; a nod when typing pauses),
    // and the end of a reply (settling nod + smirk). Both safe to call any time.
    // typing(caretFrac?): caretFrac 0..1 = where the caret sits along the prompt bar (her eyes follow it)
    typing(caretFrac) {
      if (typeof caretFrac === 'number' && isFinite(caretFrac)) caret = clamp(caretFrac, 0, 1);
      if (state !== 'listening' && state !== 'idle') return;
      if (time - lastTypeAt > 0.35) { kick('headPitch', -0.8); kick('headRoll', rand(-0.4, 0.4)); }
      lastTypeAt = time; typeNodDone = false;
    },
    endUtterance() {
      if (time - lastEndAt < 1.5) return;
      lastEndAt = time; settleDir = Math.random() < 0.7 ? 1 : -1;
      addClip(REACTS.settle[0], REACTS.settle[1], 'react-settle');
    },
    // speak(text, aheadChars?): aheadChars = how much of the reply is still queued after this text (optional; it
    // lets her start a fresh gesture at a sentence start only when there's enough left to say)
    speak(text, aheadChars) { if (!text) return; if (typeof aheadChars === 'number') ahead = aheadChars; for (const ch of String(text)) speakChar(ch); },
    react(kind) {
      const r = REACTS[kind]; if (!r) return;
      addClip(r[0], r[1], 'react-' + kind);
      if (kind === 'surprised' || kind === 'laugh') hairKick(kind === 'surprised' ? 2 : 1);
      // the greet / shrug / surprised poses own the arms: drop a gesture that would fight them
      if (kind === 'greet' || kind === 'shrug' || kind === 'surprised') for (let i = clips.length - 1; i >= 0; i--) if (clips[i].tag === 'gesture') clips.splice(i, 1);
    },
    lookAt(v) { lookTarget = v ? v.clone() : null; },
    getHeadWorldPosition(out = new THREE.Vector3()) { faceAnchor.updateWorldMatrix(true, false); return faceAnchor.getWorldPosition(out); },
    getBounds() { root.updateMatrixWorld(); return restBox.clone().applyMatrix4(rig.matrixWorld); },
    seatPoint: v3(ANATOMY.seatPoint),
    // ---- optional extras (not required by main.js) ----
    faceDirection: faceDir.clone(),   // model-space direction her face points at rest; see README
    get state() { return state; },
    setCamera(cam) { lastCamera = cam; },
    // ---- teleport (see TELEPORT above / README) ----
    // teleportOut({ surface?: { point: Vector3, yaw }, reduced? }) -> Promise, resolves once she's gone (~0.32 s)
    teleportOut(opts = {}) {
      if (tp.phase === 'out') return tp.promise;
      if (tp.phase === 'gone') return Promise.resolve();
      tp.side = Math.random() < 0.5 ? -1 : 1; tp.seed = Math.random() * 50;
      const p = startPhase('out', opts.reduced ?? reducedNow() ? 0.35 : 0.32, opts);
      tp.flashAt = tp.reduced ? -10 : time + 0.25;
      bodyCenter(tp.burstPos);
      return p;
    },
    // teleportIn({ surface?, reduced? }) -> Promise, resolves once she has settled (~0.4 s). Move root first.
    teleportIn(opts = {}) {
      if (tp.phase === 'in') return tp.promise;
      const p = startPhase('in', 0.4, opts);
      tp.flashAt = tp.reduced ? -10 : time + 0.03;
      bodyCenter(tp.burstPos); // at the new spot, whatever the frame rate (root was moved before this call)
      return p;
    },
    // the camera hard-cut while she was gone: the after-image would hang where the new shot no longer matches
    clearAfterImage() { tp.ghostT = -1; ghostB.visible = false; },
    // for main.js: FX meshes to warm up (compile) at boot, and the ones kept out of the bloom pre-render
    teleportFx: { warm: [ghostA, ghostB, fx.burst, fx.arcs, fx.decal], bloomSkip: [fx.burst, ghostA, ghostB] },
    // teleport({ seat | position, rotationY?, scale? }, { fromSurface?, toSurface?, gap = 0.12, onMove?, reduced? }):
    // out -> wait `gap` -> move root (seat: root = seat - seatPoint*scale, rotated) -> onMove() -> in
    async teleport(to = {}, opts = {}) {
      await api.teleportOut({ surface: opts.fromSurface, reduced: opts.reduced });
      await after(opts.gap ?? 0.12);
      if (to.rotationY !== undefined) root.rotation.y = to.rotationY;
      if (to.scale) root.scale.setScalar(to.scale);
      if (to.seat) root.position.copy(to.seat).sub(v3(ANATOMY.seatPoint).multiplyScalar(root.scale.x).applyAxisAngle(Y, root.rotation.y));
      else if (to.position) root.position.copy(to.position);
      root.updateMatrixWorld(true);
      try { opts.onMove?.(); } catch (e) { console.error(e); }
      await api.teleportIn({ surface: opts.toSurface, reduced: opts.reduced });
    },
    wait: after,                       // Promise resolved after s seconds of update() time
    get teleportState() { return tp.out; }, // { phase: null|'out'|'gone'|'in', active, flash 0..1, line, glitch 0..1, visible 0..1 }
    getLocalBounds() { return restBox.clone(); }, // model-space rest bounds (rig space; root applies position / yaw / scale)
    // model-space sample points of her rest silhouette (every k-th body vertex; main.js fits shots to these, which is
    // much tighter than the box: its corners overshoot her round hair / angled limbs by ~15% of her height)
    // (+ the extreme vertex along each of ~160 directions, so hair-spike tips and fingertips are always in)
    getRestPoints(maxN = 220) {
      const k = Math.max(1, Math.floor(P.count / maxN)), out = [], v = new THREE.Vector3(), seen = new Set();
      for (let i = 0; i < P.count; i += k) { out.push(new THREE.Vector3().fromBufferAttribute(P, i)); seen.add(i); }
      const nd = 160, ga = Math.PI * (3 - Math.sqrt(5));
      for (let d = 0; d < nd; d++) {
        const y = 1 - (2 * (d + 0.5)) / nd, r = Math.sqrt(1 - y * y), dx = Math.cos(ga * d) * r, dz = Math.sin(ga * d) * r;
        let best = -1, bv = -1e9;
        for (let i = 0; i < P.count; i++) { v.fromBufferAttribute(P, i); const q = v.x * dx + v.y * y + v.z * dz; if (q > bv) { bv = q; best = i; } }
        if (best >= 0 && !seen.has(best)) { seen.add(best); out.push(new THREE.Vector3().fromBufferAttribute(P, best)); }
      }
      return out;
    },
    // setPosture(name | def): switch the whole-body pose (POSTURES) at once; main.js does it while she is invisible
    setPosture(p) {
      const def = typeof p === 'string' ? POSTURES[p] : p; if (!def) return false;
      postureName = typeof p === 'string' ? p : 'custom'; PD = def; snapFace = true; poseFaceYaw = facePoseYaw(def); faceYaw = 0; return true;
    },
    get posture() { return postureName; },
    POSTURES,
    // getPosturePoints(name): like getRestPoints, but her silhouette in that posture (root space: rig offset included),
    // so main.js can fit a camera to her crouching / floating / lying. Cached per posture.
    getPosturePoints(name, maxN = 220) {
      if (!POSTURES[name] || name === 'seated') return api.getRestPoints(maxN);
      if (posePtsCache[name]) return posePtsCache[name];
      const savePD = PD, saveName = postureName;
      PD = POSTURES[name];
      const pts = [], v = new THREE.Vector3(), k = Math.max(1, Math.floor(P.count / maxN)), all = [];
      const samples = PD.drift ? 8 : 1, t0 = time; // a drifting pose: her silhouette over a whole drift cycle
      for (let si = 0; si < samples; si++) {
        update(0, t0 + si * 4.1); rig.updateMatrixWorld(true);
        for (let i = 0; i < P.count; i += si ? 3 : 1) all.push(bodyMesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(rig.matrix));
      }
      time = t0;
      for (let i = 0; i < P.count; i += k) pts.push(all[i]); // (the first sample holds every vertex in order)
      const nd = 160, ga = Math.PI * (3 - Math.sqrt(5));
      for (let d = 0; d < nd; d++) { // + the extreme vertex along ~160 directions (hair tips, fingertips)
        const y = 1 - (2 * (d + 0.5)) / nd, r = Math.sqrt(1 - y * y), dx = Math.cos(ga * d) * r, dz = Math.sin(ga * d) * r;
        let best = null, bv = -1e9;
        for (const q of all) { const t = q.x * dx + q.y * y + q.z * dz; if (t > bv) { bv = t; best = q; } }
        if (best) pts.push(best);
      }
      PD = savePD; postureName = saveName; update(0);
      return (posePtsCache[name] = pts);
    },
    _debug: { fingers: FINGERS, grip, bones: B, skeleton, meshes, role, comp, label, segs, weld, nb, positions: P, ANATOMY, uniforms: U, springs: S, override, clips, get mouthTarget() { return jawEnvV; } },
  };
  return api;
}

// ---------------------------------------------------------------------------------------------
// Teleport FX (created once, reused; all additive, hidden while idle): 3 draw calls while a flash runs.
//   burst: a camera-facing quad over her (about her silhouette's size): a hard horizontal blink-out line, a few
//          hard-edged scanline bars, a thin vertical scan line through her full height
//   arcs:  crackling red-pink lightning ribbons off her fingertips / body (rebuilt on the CPU every ~40 ms)
//   decal: the flash's light on the surface she sits on (top + front face of the ledge, in the ledge's frame)
// ---------------------------------------------------------------------------------------------
function createTeleportFx(scene) {
  const group = new THREE.Group(); group.name = 'capri-tp-fx'; scene.add(group);
  const pink = new THREE.Color(PALETTE.pink);
  const vs = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }';

  // ---- burst ----
  const burstU = { uI: { value: 0 }, uLine: { value: 0 }, uSeed: { value: 0 }, uSize: { value: new THREE.Vector2(7, 4.4) }, uCoreY: { value: 0 }, uH: { value: 2.4 }, uPink: { value: pink } };
  const burst = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShaderMaterial({
    uniforms: burstU, vertexShader: vs,
    fragmentShader: `uniform float uI, uLine, uSeed, uCoreY, uH; uniform vec2 uSize; uniform vec3 uPink; varying vec2 vUv;
      float h(float n){ return fract(sin(n * 91.345 + uSeed * 7.13) * 43758.5453); }
      void main(){
        // crisp, digital: a hard horizontal blink-out line (a CRT switching off), hard-edged scanline bars across her
        // height, a thin vertical scan line; only a small soft halo (no big radial bloom)
        vec2 p = (vUv - 0.5) * uSize, hs = uSize * 0.5;
        vec2 q = p - vec2(0.0, uCoreY);
        float ex = 1.0 - smoothstep(hs.x * 0.55, hs.x, abs(p.x));
        float ay = abs(p.y);
        float span = 1.0 - smoothstep(uH * 0.5, uH * 0.5 + 0.12, ay);
        float hline = (1.0 - smoothstep(0.012, 0.018, abs(q.y))) * exp(-abs(q.x) * 1.2) * ex;
        float halo = exp(-abs(q.y) * 40.0) * exp(-abs(q.x) * 2.2) * ex;
        float r2 = dot(q * vec2(0.6, 1.0), q * vec2(0.6, 1.0));
        float hot = exp(-r2 * 160.0);
        float bi = floor(p.y * 9.0);
        float bx = (h(bi + 2.0) - 0.5) * 0.8, bwid = 0.12 + 0.6 * h(bi + 1.0);
        float bar = step(0.7, h(bi)) * step(abs(p.x - bx), bwid) * step(0.6, fract(p.y * 9.0)) * span * ex;
        vec3 barC = mix(uPink * 1.4, mix(vec3(0.3, 0.95, 1.0), vec3(1.0, 0.9, 0.95), step(0.5, h(bi + 4.0))), step(0.55, h(bi + 3.0)));
        float vline = (1.0 - smoothstep(0.004, 0.008, abs(p.x))) * span;
        float vhalo = exp(-abs(p.x) * 30.0) * span;
        vec3 white = vec3(1.0, 0.9, 0.95);
        // (the vertical scan line is only a short tick through her middle now: full height it read as a crosshair)
        float mid = 1.0 - smoothstep(uH * 0.12, uH * 0.2, abs(p.y));
        vec3 col = (white * (hline * 1.6 + hot * 1.0) + uPink * halo * 0.45 + barC * bar * 0.7) * uI
                 + (white * vline * 0.9 + uPink * vhalo * 0.15) * uLine * mid;
        gl_FragColor = vec4(col, 1.0);
      }`,
    transparent: true, depthWrite: false, depthTest: true, blending: THREE.AdditiveBlending, fog: false, toneMapped: true,
  }));
  burst.frustumCulled = false; burst.renderOrder = 12; burst.visible = false; group.add(burst);

  // ---- lightning arcs: up to MAXA ribbons of SEG segments ----
  const MAXA = 6, SEG = 10, NP = SEG + 1, NV = MAXA * NP * 2;
  const aPos = new Float32Array(NV * 3), aV = new Float32Array(NV), aA = new Float32Array(NV), idx = [];
  for (let a = 0; a < MAXA; a++) for (let i = 0; i < SEG; i++) { const o = (a * NP + i) * 2; idx.push(o, o + 1, o + 2, o + 1, o + 3, o + 2); }
  for (let k = 0; k < NV; k++) aV[k] = k % 2 ? 1 : -1;
  const ag = new THREE.BufferGeometry();
  ag.setAttribute('position', new THREE.BufferAttribute(aPos, 3).setUsage(THREE.DynamicDrawUsage));
  ag.setAttribute('aV', new THREE.BufferAttribute(aV, 1));
  ag.setAttribute('aA', new THREE.BufferAttribute(aA, 1).setUsage(THREE.DynamicDrawUsage));
  ag.setIndex(idx);
  const arcs = new THREE.Mesh(ag, new THREE.ShaderMaterial({
    uniforms: { uPink: { value: pink } },
    vertexShader: 'attribute float aV, aA; varying float vV, vA; void main(){ vV = aV; vA = aA; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `uniform vec3 uPink; varying float vV, vA;
      void main(){ float x = abs(vV); float core = 1.0 - smoothstep(0.0, 0.35, x); float glow = 1.0 - smoothstep(0.2, 1.0, x);
        vec3 red = vec3(1.0, 0.06, 0.16);
        gl_FragColor = vec4((red * glow * 1.6 + vec3(1.0, 0.75, 0.85) * core * 2.2) * vA, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, side: THREE.DoubleSide,
  }));
  arcs.frustumCulled = false; arcs.renderOrder = 11; arcs.visible = false; group.add(arcs);
  const arcList = []; // { pts: Vector3[NP], w, a }
  for (let a = 0; a < MAXA; a++) arcList.push({ pts: Array.from({ length: NP }, () => new THREE.Vector3()), w: 0.02, a: 0, on: false });
  const _d = new THREE.Vector3(), _p1 = new THREE.Vector3(), _p2 = new THREE.Vector3(), _t = new THREE.Vector3(), _s = new THREE.Vector3(), _c = new THREE.Vector3();
  function setArc(arc, from, to, jag, w, a) {
    _d.copy(to).sub(from); const len = _d.length() || 1e-3; _d.divideScalar(len);
    _p1.set(0, 1, 0).cross(_d); if (_p1.lengthSq() < 1e-4) _p1.set(1, 0, 0); _p1.normalize(); _p2.crossVectors(_d, _p1);
    let ox = 0, oy = 0;
    for (let i = 0; i < NP; i++) {
      const u = i / SEG, env = Math.sin(Math.PI * u);
      ox = ox * 0.35 + (Math.random() - 0.5) * jag * len; oy = oy * 0.35 + (Math.random() - 0.5) * jag * len;
      arc.pts[i].copy(from).lerp(to, u).addScaledVector(_p1, ox * env * 1.6).addScaledVector(_p2, oy * env * 1.6);
    }
    arc.w = w; arc.a = a; arc.on = true;
  }
  function buildArcs(camPos) {
    for (let a = 0; a < MAXA; a++) {
      const arc = arcList[a];
      for (let i = 0; i < NP; i++) {
        const k = (a * NP + i) * 2;
        if (!arc.on) { for (let j = 0; j < 2; j++) { aA[k + j] = 0; aPos.set([0, -999, 0], (k + j) * 3); } continue; }
        const p = arc.pts[i];
        _t.copy(arc.pts[Math.min(SEG, i + 1)]).sub(arc.pts[Math.max(0, i - 1)]).normalize();
        _c.copy(camPos).sub(p).normalize();
        _s.crossVectors(_t, _c).normalize();
        const w = arc.w * (1 - 0.65 * (i / SEG)); // tapers toward the tip
        aPos.set([p.x - _s.x * w, p.y - _s.y * w, p.z - _s.z * w], k * 3);
        aPos.set([p.x + _s.x * w, p.y + _s.y * w, p.z + _s.z * w], (k + 1) * 3);
        const fa = arc.a * (1 - 0.5 * (i / SEG));
        aA[k] = fa; aA[k + 1] = fa;
      }
    }
    ag.attributes.position.needsUpdate = true; ag.attributes.aA.needsUpdate = true;
  }

  // ---- surface light decal (ledge frame: +Z = out of the ledge front, top surface at y 0, front face at z 0) ----
  const dPos = [], dUv = [], dIdx = [];
  const quad = (a, b, c, d, ua, ub, uc, ud) => { const o = dPos.length / 3; dPos.push(...a, ...b, ...c, ...d); dUv.push(...ua, ...ub, ...uc, ...ud); dIdx.push(o, o + 1, o + 2, o, o + 2, o + 3); };
  // top: x -1.5..1.5, z -1..0 (uv = (x, z)); front face: x -1.5..1.5, y -1.3..0 (uv = (x, y))
  quad([-1.5, 0.004, 0], [1.5, 0.004, 0], [1.5, 0.004, -1], [-1.5, 0.004, -1], [-1.5, 0], [1.5, 0], [1.5, -1], [-1.5, -1]);
  quad([-1.5, -1.3, 0.006], [1.5, -1.3, 0.006], [1.5, 0, 0.006], [-1.5, 0, 0.006], [-1.5, -1.3], [1.5, -1.3], [1.5, 0], [-1.5, 0]);
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.Float32BufferAttribute(dPos, 3));
  dg.setAttribute('uv', new THREE.Float32BufferAttribute(dUv, 2));
  dg.setIndex(dIdx);
  const decalU = { uI: { value: 0 }, uPink: { value: pink } };
  const decal = new THREE.Mesh(dg, new THREE.ShaderMaterial({
    uniforms: decalU,
    vertexShader: vs,
    fragmentShader: `uniform float uI; uniform vec3 uPink; varying vec2 vUv;
      void main(){ vec2 q = vUv * vec2(0.8, 1.6); float r2 = dot(q, q);
        float a = exp(-r2 * 2.2) * (1.0 - smoothstep(1.1, 1.5, abs(vUv.x)));
        vec3 col = mix(uPink * 1.2, vec3(1.0, 0.85, 0.92), exp(-r2 * 14.0));
        gl_FragColor = vec4(col * a * uI, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  }));
  decal.frustumCulled = false; decal.renderOrder = 3; decal.visible = false; group.add(decal);

  return {
    group, burst, burstU, arcs, arcList, setArc, buildArcs, decal, decalU,
    dispose() { scene.remove(group); burst.geometry.dispose(); burst.material.dispose(); ag.dispose(); arcs.material.dispose(); dg.dispose(); decal.material.dispose(); },
  };
}
