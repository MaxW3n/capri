# Capri 3D frontend

`server.py` serves this page at `/` (the old flat chat is at `/classic`). It's plain ES modules with
three.js r169 vendored in `web/vendor/three/`, so there's no build step and nothing comes from a CDN.
URLs are absolute from the repo root (`/web/3d/...`, `/web/vendor/three/...`, `/capri_project_capricorn/...`),
so `python3 -m http.server` run from the repo root serves the page too. Without the model server
it runs in **demo mode** and replies from `demo_lines.json` (real dose-1.0 replies from the 27B test set, vector `capri_s2_27b`).

The 3D model is "Capri (Project Capricorn)" by [0x00001337](https://sketchfab.com/0x00001337), from
[Sketchfab](https://sketchfab.com/3d-models/capri-project-capricorn-2a779d03af574268bae518dd7c6321e1), licensed under
[CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/). The page credits it in the bottom bar. See `CREDITS.md`.

## Files and contracts

| file | exports | job |
|---|---|---|
| `index.html` | — | importmap, CSS, HTML overlay (prompt bar at the bottom, header, settings) |
| `main.js` | — | renderer, camera, post-processing, render loop; wires everything |
| `capri.js` | `createCapri(scene, opts)` | loads the glTF and animates it procedurally |
| `city.js` | `createCity(scene, renderer, opts)` | the cyberpunk environment (+ perches Capri can teleport to) |
| `traffic.js` | `createTraffic(parent, opts)` | the flying hover cars (city.js creates it inside its group) |
| `bubble.js` | `createBubble(scene, opts)` | the 3D speech-bubble plane her replies stream into |
| `chat.js` | `createChat(opts)` | `/chat` SSE client, history, demo fallback |
| `voice.js` | `createVoice()` | WebAudio "talk blips", synced to the text |
| `palette.js` | `PALETTE` | the shared colours |

### `palette.js`
```js
export const PALETTE = {
  pink: 0xfe0353,  // her hair / suit, sampled from the model texture
  green: 0x19f267, // acid green accents
  blue: 0x1961c5,  // her eye
  ink: 0x1b1b1b,   // suit black
  void: 0x07040a,  // sky / background
  white: 0xebe9ea,
  css: { pink: "#fe0353", green: "#19f267", blue: "#1961c5", ink: "#1b1b1b", void: "#07040a", white: "#ebe9ea" },
};
```

### `capri.js`: `await createCapri(scene, { url, position, rotationY, scale })` →
```
{
  root: THREE.Group,                      // already added to the scene
  update(dt, t),                          // call every frame
  setState(s),                            // 'idle' | 'listening' (user typing) | 'thinking' (waiting for first token) | 'talking'
  speak(text, aheadChars?),               // a chunk of streamed reply text: drives mouth/head/gesture beats per syllable/punctuation
                                          // (optional aheadChars = how much is still queued after it; gates sentence-start gestures)
  react(kind),                            // one-shot beats: 'greet' | 'laugh' | 'shrug' | 'nod' | 'surprised' | 'glitch'
  lookAt(worldVec3 | null),               // head/eye target (null = back to the camera)
  getHeadWorldPosition(outVec3),          // where the bubble's tail should point
  getBounds(): THREE.Box3,                // world-space bounds
  seatPoint: THREE.Vector3                // MODEL-space point (under her hips, at the front of where she sits) that
                                          // main.js puts on city.seat: root.position = seat - seatPoint*scale
}
```
Optional extras (additive; main.js calls them with `?.`): `opts.camera` (otherwise she grabs whatever camera renders
her, via `onBeforeRender`, so `lookAt(null)` = look at the viewer); `faceDirection: THREE.Vector3` (model space, see
below); `setCamera(cam)`; `state` getter; `typing(caretFrac?)` (a keystroke in the prompt: small head kick, a nod
when typing pauses; optional caretFrac 0..1 = the caret's position along the prompt bar, her eyes follow it); `endUtterance()` (settling nod + smirk at the end of a reply; also fires on talking -> idle); reacts also
include `'settle'` and `'mm'` (a small listening bob); and `export const ANATOMY` (measured pivots).

Speech: the jaw is driven per **syllable**, not per character: each vowel onset starts a short open-then-close beat
(~60 ms up, ~110 ms down to ~0.05); onsets closer than ~0.19 s are merged so it shuts between beats; m/b/p and
punctuation snap it shut. Each phrase (up to the next `,` `.` `!` `?` `—`) gets its own held head turn / tilt; a fresh
gesture is likely at each sentence start (alternating hands) when `aheadChars` > 15; long stressed words add a small
forward lean. While 'talking' with no new text for > 0.35 s (a slow model) she holds a pause: mouth shut, eyes on the bubble.

The model (`capri_project_capricorn/scene.gltf`) is a static, unlit, low-poly mesh: **no skeleton, no morph
targets**. capri.js builds a 25-bone skeleton by hand at load (pivots measured from the mesh), computes soft skin
weights once on the CPU (geodesic region labels over the welded mesh + smoothstep blends across each joint) and turns
the meshes into `SkinnedMesh`es, so the GPU does the deformation and per frame we only set bone transforms. The black
inverted-hull outline (`Object_6`, `Paper_Outline_Black`) copies the weights of the nearest body vertex so it stays
glued on. An `onBeforeCompile` patch adds hair wind, the 'glitch' slice jitter, a faint pink rim and the outline glow
while talking. 5 draw calls (body, outline, upper grin, lower grin, mouth interior).

Anatomy (mesh coords = model space: metres, Y up, the **body** faces +Z, floor ~y 0.04, hair top y 2.37):
- **Her head is turned ~63° to her right and tilted down**: the face looks along `faceDirection` ≈ (-0.86, -0.31, 0.44).
  The head can turn ~±45° toward a look target, so **orient her so the camera sits between +Z and faceDirection**:
  e.g. `rotationY = cameraYaw + 0.6` where `cameraYaw = atan2(camDir.x, camDir.z)` of the direction from her to the
  camera. A camera slightly below her eye line suits the downward tilt (she sits above the viewer).
- Face: green face plate, two big eye discs at (-0.20, 1.52, 0.27) and (-0.30, 1.57, 0.12) (separate overlay meshes:
  blinks squash them, saccades nudge them), white zig-zag grin (`Object_5`, the 24-vert "white" mesh) centred at
  (-0.233, 1.458, 0.155), enlarged 1.3x in the face plane at load. The mouth opens by rotating the black lower face
  around a hinge behind the grin; a copy of the grin rides the jaw with a pink interior between. Head pivot
  (-0.13, 1.47, 0.08); bubble anchor = the grin's corner on the bubble side (her left, screen right), returned by
  `getHeadWorldPosition`, so the tail ends at her mouth and never on her eyes. The black outline hull is tinted a very
  dark magenta (#2a0618) so her silhouette separates from the night.
- Sitting: `seatPoint` = (-0.05, 0.675, 0.13): ledge top at y 0.675, ledge front face at z 0.13. The dangling (black)
  shin is swung forward at load so it hangs in front of the ledge face; the hands rest on the ledge top beside her
  (fingertips reach out to x -0.75 / +0.62, so a ledge at least ~1.6 m wide looks right). The crossed (pink) foot
  reaches z 0.62 and the thin black rod near it z 0.92, both in front of the ledge.
- Arms: shoulders (-0.28, 1.37, 0.0) / (0.17, 1.34, -0.05), elbows (-0.43, 1.02, -0.15) / (0.37, 1.0, -0.19),
  wrists (-0.60, 0.72, -0.03) / (0.48, 0.75, -0.12); posed by 2-bone IK toward wrist targets.
- Two floating paper shards beside her arms (green at x -0.5, pink at x +0.45) bob gently.

#### Teleport (capri.js effect, main.js scheduling)
In the show Capri is a construct on a biochip: she glitches in and out of existence. The jump is a *digital* one,
cut into horizontal slices (24 bands of her rest-pose height), all in the existing `onBeforeCompile` patch so the body,
outline, grin and mouth meshes break up identically:
- **out** (~0.32 s): a calm beat (0-0.1 s, 3 frames at 30 fps) of two crisp sideways hops of her normal flat look (a
  little slice jitter, no colour blow-out, hardly any screen glitch) -> (0.1-0.21 s) her slices jump to held offsets in
  two discrete steps (~2 frames each, re-shuffled per step) *still in her own flat colours*, the RGB split only as thin
  red / cyan fringes (two additive ghost renders of her body just behind her, a couple of cm apart); red-pink lightning
  crackles off her fingertips (one long, thick white-cored bolt from her right fingertip down past the ledge at
  ~0.15-0.21 s, while she is still recognisable) -> (0.21-0.25 s) only now the per-slice pink / white blow-out, the slices
  thin to scanlines and stretch -> the flash at 0.25 s and the blink-out: a burst quad about her silhouette's size (a hard
  horizontal blink-out line ~1.2x her width, a few hard-edged scanline bars, a short vertical tick through her middle
  that is gone ~0.1 s after the flash; no big soft bloom, kept out of the bloom pre-render) + the flash's light on the
  ledge top / front. Slice sizes are clamped to her own size: shifts are ~0.1-0.4 m in view space, most slices stretch
  ~1.3-1.6x about her centre, only ~1 in 10 "hero" slices overshoot ~2.5x.
- **gone**: invisible; a cyan / pink-rimmed after-image of her silhouette stays at the old spot, breaking into
  drifting slices (~0.6 s). Most jumps keep the camera still (main.js's master shot), so it stays visible while she
  snaps in at the new spot; main.js calls `clearAfterImage()` only when it hard-cuts the camera.
- **in** (~0.4 s): the second flash (its burst is placed when `teleportIn()` is called, so it lands at the new spot at
  any frame rate); white scanlines snap in from one side (from a moderate spread, about her own width), top first,
  thicken back into slices with an overshoot jitter, colours settle white -> pink -> her flat look, arcs crackle, a last
  glitch tick.
- Reduced motion: an ordered-dither crossfade (no flashes, arcs, ghosts, screen glitch). At most two bright flashes
  (~0.2 s apart); the full-frame lift is low (uFlash <= 0.1).
FX: 2 ghost SkinnedMeshes (share the body geometry + skeleton) + burst quad + arc ribbons (CPU-rebuilt every ~40 ms)
+ surface decal, created once, all hidden otherwise. The burst and ghosts are skipped in the bloom pre-render
(`teleportFx.bloomSkip`), the arcs and decal feed it: measured peak **82 calls (+7) while a flash runs**. Their
programs are compiled at boot (`teleportFx.warm`, `renderer.compile` into the composer's target behind the boot
overlay), so the first jump doesn't hitch.
All timing runs on `update()`'s clock, so a virtual clock can step it frame by frame.
```
capri.teleportOut({ surface?: { point: Vector3, yaw /* ledge yaw */, w? /* half-width */ }, reduced? }) -> Promise   // resolves when gone
capri.teleportIn({ surface?, reduced?, fx? }) -> Promise   // move root first; resolves when settled
                      // (fx 0.45..1, main.js's TP.fxK: shrinks the slice shift / stretch when she is big on screen)
capri.teleport({ seat | position, rotationY?, scale? }, { fromSurface?, toSurface?, gap = 0.12, onMove?, reduced? }) -> Promise
capri.teleportState   // { phase: null|'out'|'gone'|'in', active, flash 0..1, line, glitch 0..1, visible 0..1 } (read per frame)
capri.wait(s)         // Promise resolved after s seconds of update() time
capri.getLocalBounds()// model-space rest bounds
capri.getRestPoints(n?)// model-space sample points of her rest silhouette (+ extreme vertices): main.js fits shots to these
capri.clearAfterImage()// end the after-image now (main.js: on a hard camera cut)
capri.teleportFx      // { warm: Mesh[], bloomSkip: Mesh[] } for main.js
```
`surface` defaults to her seatPoint with yaw = rotationY - 0.71. `reduced` defaults to prefers-reduced-motion.

main.js owns *when*: `teleportTo(i?, why?)` (also `window.__capri.teleport(i?, why?)` -> Promise<boolean>; `__capri.perches`,
`__capri.perchIdx`). While idle every ~25-60 s (not within 6 s of typing, 8 s after a line, 20 s of the last jump), and
~25% of sends (0.5 s after the nod, before her reply starts; the speech clock holds the reply until she has landed and the bubble is back, at most ~1 s).
Never mid-sentence, never the current perch, the previous perch only rarely (home a bit more often), and perches inside
the master shot ~5x more often than ones that need a camera cut; portrait phones skip `portrait: false` perches (and jump
home if the phone turns, once she has finished the line). Sequence: `voice.zap('out')` -> teleportOut -> 0.11 s -> root
to the new perch, contact shadows / backlight card re-placed -> camera -> (only after a cut: the new spot is held empty
for 0.1 s) -> `voice.zap('in')` -> teleportIn.
**Robustness**: if anything throws between teleportOut and teleportIn (or a jump hangs for 3 s: the watchdog) she is put
back on a valid perch (the target if she got there, else where she came from), re-framed and brought back in, so she can
never stay invisible; `voice.zap` can't break a jump.
**Camera: one master shot.** `frame` = the per-aspect base shot (fitted to perches[0]; landscape ~7% wider than a tight
fit so the foreground corners and the fore perch are in it). A jump keeps the camera still whenever the new perch fits
that shot (`inMaster(i)`), so you see her flash out at one spot (the after-image stays there) and snap in at the other:
on desktop all four perches are in it. Fitting uses her real rest silhouette (`capri.getRestPoints()`: ~220 body
vertices + the extreme vertex in ~160 directions, much tighter than the box) against the safe rect: under the header,
landscape keeps ~40% of one side for the bubble, never behind the prompt bar or on the chips; her legs may run off the
frame's bottom edge beside the dock, and a perch's `cropY` (model-space height) lets the part below it go behind a near
prop. Only when a perch doesn't fit the master shot (phones, odd aspects) does the camera hard-cut while she is gone
(the perch's `view` / `viewPortrait`, else the smallest pan + dolly that fits; reduced motion too, never a glide); a tiny
reframe (< 0.35 m) eases over 0.45 s. After a cut the after-image is cleared (from a new angle it reads as a duplicate).
**Bubble**: faded / glitched across the jump (its uniforms). After an *idle* jump it stays down (`bubble.hide()` while
she's gone: her old line must not re-appear at the new spot as if she said it again) until she next speaks; after a
*send* jump it comes back with her last arrival tick (~0.12 s after she is whole again). main.js also feeds
`bubble.setAvoid(x)`: the right edge of her silhouette around head height, so the panel keeps ~20 px clear of her hair.
**Screen FX** scale with how much of the frame she fills (`TP.fxK` = 0.42 / her height fraction, 0.45..1): the post pass
gets a glitch burst (<= 0.6 x k), extra RGB split (`uSplit`) and a low pink-white lift (`uFlash` <= 0.1 x k) on each
flash, so the fore-perch arrival is a burst around her, not a frame-wide wash; the backlight card pulses with the flash.
Without `city.perches` (city failed to load) main.js uses only the default seat, so she never teleports.
STOP also drops a send-triggered jump that hasn't started yet (`TP.pendingAt`); a jump already running finishes.
See `city.perches` below for the perch list and per-perch cameras.

#### Postures

The mesh is modelled sitting on a ledge. `POSTURES` in `capri.js` adds three more whole-body poses, taken from her
construct form in the show: `crouch` (gargoyle squat, feet on the ledge, forearms over the knees), `float` (upright,
legs together, arms hanging, a slow bob) and `lie` (on her front on a roof, head at the edge, propped on her elbows).
A posture is a layer under the procedural animation: per-bone rotations (hips, spine, chest, neck), two-bone leg IK to
foot targets, wrist targets for the arm IK, and an offset / rotation of the whole rig from the seat point. Talking,
gestures, blinks and teleports run on top unchanged.

Fingers: the hands are low poly, each finger a spike of a few vertices. At load `capri.js` walks the surface out from
each wrist, cuts the palm off where the most separate pieces remain (5 per hand on this mesh) and gives each piece a
knuckle bone. A grip per hand (posture `grip: [right, left]`, 0 = as modelled, 1 = fist; 0.2 by default) curls them
towards the palm, and opens while that hand gestures. Postures can also turn her to the camera (`faceCamera`, with
`faceBy: 'body'` or her face, limited by `maxTurn`), and the head finds the camera from her posed chest, so it still
looks at you when a posture rolls or flips her.

`capri.setPosture(name)` switches at once; `capri.getPosturePoints(name)` returns her silhouette in that pose (cached)
so the camera fits it. `main.js` keeps one posture per perch (`PERCH_POSTURES`: parapet seated or float, crates
crouch, stair hut lie, water tank float or lie), sets it while she is invisible mid-teleport, and rolls a new one for
the perch she just left.

### `city.js`: `createCity(scene, renderer, { quality })` →
```
{ update(dt, t), seat: THREE.Vector3 /* where Capri's hips go */, seatRotationY: number,
  setIntensity(x /* 0..1, how "loud" the scene is: she's talking = higher */), dispose(),
  setAspect(a),             // optional extra: viewport aspect; portrait (< 0.8) shows an extra emblem hologram beside her head
  group: THREE.Group,       // optional extra: everything the city added (already in the scene)
  recommendedFar: number,   // optional extra: 1500. The skyline sits ~450 m out, so camera.far must be >= ~600
  perches: Perch[],         // places Capri can sit / teleport to (below); perches[0] is the parapet (= seat)
  floorY: number }          // -0.62: the lower roof terrace in front of the parapet (camera side)
```
`seat` is (0, 0.8, 0): the top surface of the parapet coping is at y 0.8 and its front edge at z 0; her legs hang
in front of it (+Z), over a soft pink/blue wash on the parapet face so her black shin reads. `quality: 'low'` halves the buildings, passes `quality: 'low'` to traffic.js and uses 2.2k rain drops instead of 6.5k.
Flying traffic comes from `traffic.js`: `createCity` calls `createTraffic(group, { quality })` and forwards
`update(dt, t)`, `setIntensity(x)` and `dispose()` to it (the old instanced head/tail-light "orb" traffic is gone).
The city sets `scene.fog` (FogExp2, only noticeable up close) and `scene.background`, and adds a hemisphere light
and four small point lights for its own props (`dispose()` restores fog/background). The billboard tower sits at the right edge with the billboard high up (above the bubble band), and
roofs are kept under the horizon in two "vista" wedges (left third; and an open canyon from just left of her head across
the bubble band, which also holds for the far field), so the sky shows behind her: a magenta-to-void gradient with two
drifting smog layers lit pink from below near the skyline, violet / blue higher up. Hot pink is kept mostly for her and
the bubble (the blade sign beside her is blue).

**Depth layers.** Foreground (near the camera, dark and rim-lit so they read as the closest, slightly hazy layer):
bottom-left the fore perch's crate stack (slatted pallet, two crates of different sizes / offsets / yaws, a small box, a
strap, a hazard tag and an original chevron stencil) with a narrow utility cabinet in front of it close to the lens (lit
lid edge, status LED); the chain-link fence at the left edge; the utility pole just off-frame left with its cables sagging
across the top-left; bottom-right a staggered crate pair, a small box and a cable reel (cyan corner tag, pink tag); and
a few big soft out-of-focus rain streaks drawn in view space right in front of the lens (they follow any camera). Crate
faces get per-board value variation, wider grooves and a little violet / blue key light so they read as crates from the
master shot. Every
perch has a lit backing behind her dangling legs (parapet washes; an LED strip under the top crate's lip + a pink wash;
the hut lamp + wash; a pink strip under the tank lid + a small violet wash, a lit lid edge and leg rims so the tank reads).
Midground: her parapet; wet tops (coping, hut roof) with hard-edged puddles that mirror the nearest neon as rippled
streaks; in front of the parapet (+Z) a lower roof terrace at `floorY` (-0.62; a curb at the wall's foot with the pink
tube) with its own puddle shader (only seen from low cameras). Behind her on the roof: a stairwell hut (left) and a
steel water tank on a stand (centre-right), both perches. The foreground keeps out of the prompt-bar band except the
near cabinet, which sits mostly behind the bar's left end on purpose.
The depth layer adds 4 draw calls (silhouettes, fence, floor, lens rain); its other props are merged into the existing
concrete / metal / neon meshes. The city is ~18 draw calls + traffic.js's 8.

**Budget (measured, 1280x800 high quality, `renderer.info` with autoReset off over one frame):** 81 draw calls /
~122k triangles for the whole frame (main scene + the half-res bloom pre-render + post + bubble), idle (iPhone 13 / low:
81 calls, ~65k tris); a teleport flash adds ~7 calls.

**`city.perches`** (each a fresh copy; `city.setFraming(o)` forwards main.js's current shot to traffic.js):
```
{ id, seat: Vector3,      // world point for Capri's seatPoint (top surface y, front edge) - root.position = seat - R(rotationY)*seatPoint*scale
  rotationY: number,      // her final root.rotation.y there (= camera yaw from the seat + 0.6, the README rule;
                          //   perches[0] gives 0.71, what main.js computes today from seatRotationY 0)
  ledgeYaw: number,       // the edge's outward normal (atan2(nx, nz)); her body is turned ~0.6 off it like on the parapet
  ledgeW?: number,        // half-width of the surface (m): sizes the flash's light on it (parapet 1.5, crates 0.55)
  scale?: number,         // (unset = 1)
  depth: 'fore'|'mid'|'back', portrait: boolean,
  view: {pos, target, fov} | null,          // suggested desktop camera for this perch (null = main.js's default shot)
  cropY?: number,                           // model-space height: only the part of her above it must fit the frame
  viewPortrait?: {pos, target, fov} }       // suggested portrait-phone camera (absent = the default portrait shot works)
```
| id | depth | seat | rotationY | camera |
|---|---|---|---|---|
| `parapet` | mid | (0, 0.8, 0) | 0.71 | the master shot (default shots) |
| `crates` | fore | (-0.6, 0.45, 1.15): top crate of a stack on the terrace ~1 m in front of the parapet, left of her legs; ledgeYaw 0.3 | 0.9 | **in the master shot** on desktop (~1.3x the parapet size at 1280x800, more on wider screens), `cropY: 0.6`: her shins hang behind a utility cabinet close to the lens (bottom-left), never under the prompt bar; portrait phones cut to `viewPortrait` pos (0.1, 1.45, 5.0) -> (-0.55, 1.75, 1.12), fov 46 |
| `stair-hut` | back | (-1.12, 1.45, -2.42), roof slab over the hut door | 0.83 | in the master shot (desktop, and portrait phones); `view` kept as a fallback for odd aspects |
| `water-tank` | back | (0.6, 1.75, -4.6), lid rim of the tank | 0.57 | in the master shot (desktop, and portrait phones); portrait fallback view |

On desktop every jump happens inside one shot. The fore stack's top crate is only as wide as her hands need (its right
end stays clear of the parapet perch's shins in the master shot). Foreground of the master shot: the fore crate stack and
the near utility cabinet bottom-left, the chain-link fence at the left edge, the pole just off-frame left (its cables
cross the top-left), a staggered crate pair + cable reel bottom-right (planks with per-board value variation and a
little violet / blue key from the city, so they read as crates), and the lens rain. Wet tops (the coping, the hut roof):
hard-edged puddles mirror the GOAT sign, the billboard, the corner glow and the hologram as rippled vertical streaks.
Checked at 1280x800, 1366x768, 1440x900, 1920x1080 and iPhone 13 with her rest silhouette: inside the safe rect, the
bubble has room.

Bloom: city emissives are tuned for UnrealBloomPass threshold ~0.2, but Capri's unlit texture (hot pink, white)
is above that, so either exclude her from bloom or keep the threshold >= ~0.3 / darken her slightly.

### `traffic.js`: `createTraffic(parent, { quality, sky? })` →
```
{ update(dt, t), setIntensity(x /* 0..1, like city's: brightens underglow / thrusters a little */), dispose(),
  setFraming({ pos, target, fov, aspect }),   // the shot the close lanes are laid out for (city.setFraming forwards it)
  group: THREE.Group /* already added to parent */, stats: { cars, drawCalls, tris } }
```
Flying traffic as actual vehicles (it replaces city.js's old head/tail-light "flying traffic" block). Three original
low-poly hover designs, each a few extruded side profiles + boxes merged into one geometry (flat-shaded facets like
Capri): **kestrel** (a fastback coupe on four hover pods, glass canopy, slanted headlight slits, full-width tail bar,
side neon line), **mule** (a van with a car's side silhouette: low nose wedge, raked windscreen, stepped roof with a
raised cargo box, a dark glass band with a lit cabin line, pods hanging below the body; ~45% cab yellow with a self-lit
term, a vertical value gradient, two dark door / panel gaps and a darker roof box so a partial view still reads as a
van; ~22% of the mix, at least 2 bikes per quality) and **wasp** (a hover bike with a hunched rider, ring thrusters).
Model space: +X nose, +Y up, +Z the car's right, metres (coupe 4.9 m long).
8 draw calls: one InstancedMesh per design (opaque, renderOrder 2), one **ink outline** per design (an inverted hull:
back faces of the same instanced geometry, inflated per axis away from the car's centre so the faceted mesh doesn't crack,
~0.02 m + 0.0011 x distance, so ~2-4 px at any range; near-black, fogged like the body: the cars get Capri's hard dark
contour instead of melting into the haze), one additive mesh for every light effect of the near cars (anamorphic head /
tail flares, headlight beams in the rain, underglow card + haze, short light trails; renderOrder 7) and one for the far
field (150-300 m: head/tail light pairs only). 46 detailed cars (high), 20 (low; low also drops beams and trails).
Motion is all in the vertex shader from `uTime` (`update` only sets it; **not wrapped**, so nothing pops after an hour):
instance attributes `aLane = (y or screen fraction, z, signed speed, phase)`, `aLane2 = (lane yaw, loop span, weave
amplitude, seed)`, `aLane3 = (lane-change amp, altitude-step amp, surge, class + top/10)`; cars weave gently, change
lanes / step altitude with a ~15-20 deg bank / pitch, surge +-1 m/s, bob, and point their nose along the direction of travel.
**Lane heights in screen terms.** Class 0 lanes (mid band, 58-138 m) have absolute heights. Classes 1-4 (near upper,
near lower, fly-by, hero) store the screen fraction where the car's **top edge** lands (so a tall van can't poke up into
the bubble) and the vertex shader turns it into metres from `uFr` (the current shot: camera y / z, tan(fov/2), view
slope), set by `setFraming` (main.js: on resize and on a camera cut, not per frame, so the parallax drift still moves
them). Landscape master: upper band top edge at f 0.08-0.15 (just under the header; the body stays above the bubble), lower band at f 0.41-0.45
(under the bubble, over the hut / tank tops), 8 near lanes at z -24..-48 with 3 cars each (high). Fly-bys (z -12 / -16 /
-20, coupe / van / bike, one every ~11-17 s, 7-10 m/s) run in the upper band and only cross the open right part of the
frame (`uWin`, ndc x 0.12..1): outside it they dive in from / climb back to the sky, nose leading into open space. The
hero coupe (z -8.5, every ~40 s, first ~18 s after load) flies with its top just above the bubble and banks ~17 deg
toward the camera so its roof and canopy show. Portrait phones (`uBand`): bands shifted down (upper band below the
bubble's zone), fly-bys and hero off (they'd only cross her hair); squareish screens shift a little.
Shading is a fake, no lights: dark paint, fresnel reflection of a fixed neon environment (pink city glow below the
horizon, a horizon streak, blue sky above, world-fixed sign bands that slide over the body as it moves), pink / cyan rim,
city underlight + own underglow on the belly, emissive light strips. Near cars (< ~45 m) cap that reflection and darken
the lower body / belly toward suit black, so the silhouette and the neon trim carry the read. Fog (`fogMix` / `fogKeep`)
and the sky colours use city.js's own uniform objects (`opts.sky`, falling back to copies), so far cars fade like the
buildings; output = the same `tonemapping_fragment` / `colorspace_fragment` OUT snippet. Integration: city.js does
`createTraffic(city.group, { quality })`, forwards `update(dt, t)` / `setIntensity(x)` / `setFraming(o)` / `dispose()`.

### `bubble.js`: `createBubble(bubbleScene, { maxWidth })` →
`bubbleScene` is a separate THREE.Scene that main.js renders *after* bloom and the post pass (same camera,
no depth test against the city), so the text stays crisp.
```
{ group: THREE.Group,
  update(dt, t, camera),                  // billboard toward the camera, animate
  begin(),                                // new reply: clear and pop open, show "..." thinking state
  append(text),                           // already-paced text (see "Speech clock"); draw it at once, with a cursor
  end(),                                  // reply finished
  setError(message),                      // show an error state in the bubble
  setAnchor(worldVec3),                   // the point the tail points at (her head)
  hide() }
```
`maxWidth` is the panel's max width in **CSS px** (default 440). The bubble lays itself out in screen space:
each frame it projects the anchor, sits right of her head (else left, else above, e.g. on phones), springs
toward that spot and is placed in 3D at about her depth, scaled so 1 local unit = 1 CSS px, so the text is
always ~17.5px (16px under 560px wide) whatever the framing. It billboards with lag, a ~10° yaw toward her (a real
perspective keystone), velocity tilt, a float bob, and only half-follows the camera's turning (pointer parallax swings
it). The tail stops ~24 px short of the anchor; a tail out of the bottom edge (bubble above her head) stops at the top
of her hair. Faint rain streaks slide across it. Clipped scroll edges fade; the scroll target snaps to whole lines. One draw call (SDF shader + a CanvasTexture for the text). Short replies
shrink-wrap; long ones grow to a max line count, then scroll. `**bold**` is rendered bold.
Optional extras (additive): `opts.renderer` (viewport size from its canvas + anisotropy; else window size),
`opts.insets {top,bottom,left,right}` (CSS px kept clear for the header / prompt bar; default 64/124/16/16),
`opts.label` ('CAPRI'), `opts.channel` (a dim tag after the label; none by default), `opts.fontSize`, `opts.maxLines`, `opts.side`,
`opts.scene` (the main scene: adds a soft additive pink light spill ~0.9 m behind the panel, one extra draw call),
`opts.injectFonts` (default true: adds a Google Fonts link for Chakra Petch + Atkinson Hyperlegible unless the
page already has one; system fallbacks offline). Methods `setInsets(obj)`, `setViewport(w, h)` (CSS px; no args =
auto), `setSide('auto'|'right'|'left'|'top')`, `clear()`, `scrollBy(dyCssPx) -> moved?` (scrolling back pauses
auto-follow until the bottom is reached again), `getScreenRect() -> {x,y,w,h} | null` (the panel in CSS px, for
hit-testing wheel / touch); getters `state` ('hidden'|'thinking'|'talking'|'done'|'error'), `text`, `side`,
`scrollable`, `pageHeight`. `append()` while hidden auto-begins. Text is ~17.5px up to ~1500px wide, scaling to 22px
(and the max width with it) on big screens; while streaming it re-lays-out at most ~16x/s. main.js should render it with
`renderer.autoClear = false; composer.render(); renderer.clearDepth(); renderer.render(bubbleScene, camera)`.

### `chat.js`: `createChat({ onToken, onStart, onDone, onError, onMode })` →
```
{ send(text, dose) -> Promise, stop(), history: [{role, content}], mode: 'live' | 'demo', info /* from /info */ }
```
Optional extras: `send` resolves to `{ reply, aborted, error, restore }` (`restore` = the user text when nothing came back);
`ready` (Promise of the first mode), `lastTps`, `busy`, `reset()` (new conversation: aborts the running request and
empties the history; a reply still in flight from before can't write into the new one, and its send resolves with
`stale: true`). `onDone({ reply, aborted, mode, tps, finish, tokens })`. A reply cut off by a server `{error}` is not
kept in the history (the turn is undone and `restore` gives the text back). In demo mode a question that shares fewer
than two content words with any recorded question gets an in-character deflection instead of an unrelated line.
A network failure in live mode switches to demo mode (`onMode('demo')`) and answers that message from the demo lines;
in demo mode `/info` is re-probed every 20 s (and right before each send) and the page switches to live when the server
appears. The first probe allows 5 s and retries once, since single-threaded server.py may still be busy serving the page.
A `/info` 404 (a plain static server) backs re-probing off to every 60 s (and when the tab comes back). Only the newest
send may fire callbacks or touch the history: a superseded one resolves `{ stale: true }` quietly and removes its own
user turn. A stream that ends without `{done}` (server.py died mid-reply) counts as a network failure: the turn is
undone, the text restored and the mode drops to demo.

### `main.js` (no exports)
Selective bloom: a half-res render of the scene with Capri's material colours darkened feeds UnrealBloomPass, and the
bloom is added onto the full-res render in the single post pass (bloom add + glitch/CA/scanlines/grain + Neutral tone
mapping + sRGB), so her unlit acid green doesn't glow like a sign and keeps its saturation (ACES bleached it to mint).
The scene renders into a 4x MSAA half-float target (`EffectComposer(renderer, rt)`; the renderer itself has
`antialias: false`). Pixel ratio is capped at 1.5, steps down if frames keep running slow and back up after ~5 s of fast
frames (the first ~4 s after boot / a context restore are ignored). Contact shadows (one draw call) sit under her hips
and hands; a soft additive backlight card sits behind her torso and hair.
Framing: per-aspect base shots (desktop / tablet / portrait phone), then, once Capri has loaded, the camera slides along
that shot's direction so her rest-pose bounds fit between the header and the prompt bar (portrait phones keep the top
~quarter free for the bubble). STOP trims the model history to what she actually said before the cut. NEW CHAT bumps a
conversation id and calls `chat.reset()`, so a reply still in flight can't leak into the new chat. The revealed text is
scanned for cues that fire `capri.react()` (ha/heh -> laugh, dunno/whatever -> shrug, a leading yeah/sure -> nod,
what?/wait/?! -> surprised, sometimes a laugh after a short line ending in '!'; at most one per 2.5 s); she nods when you
send and looks at the prompt bar (`lookAt`) while you type. The user's last line is echoed above the prompt bar until
her reply ends (left-aligned on a dark chamfered backing, clear of her dangling foot; a gradient scrim behind the dock
fades the scene out under the UI). While she's talking she stays in 'talking' as long as the reply is still streaming in
(no switch to 'thinking' mid-sentence). Enter while she's replying sends nothing; the STOP button pulses and a hint
says how to cut her off. The settings popover closes when focus tabs out of it, and the bubble keeps clear of it; in
demo mode its dose slider is dimmed with a note (demo transcript lines are labelled `capri · demo`). The transcript
drawer traps Tab while open. Server errors show an in-character line in the bubble; the technical message goes to a toast and a SYS
transcript entry. The bubble scrolls back with the wheel / a drag over it / PageUp-PageDown.
Testing hooks: `window.__capri = { send, stop, capri, bubble, city, chat, voice, renderer, camera, composer, state, frame, computeFrame,
mode, frameMs, frames, glitch(), teleport(i?), perches, perchIdx, TP, shotInfo(i) }` (`shotInfo(i)`: perch i's suggested view vs the safe rect), and `?fps=N` caps the frame rate (handy for software-GL screenshots).

### `voice.js`: `createVoice()` →
```
{ resume() /* call from a user gesture */, blip(char), zap('out'|'in') /* teleport sound, only when enabled */, setEnabled(bool), enabled }
```

## Speech clock (main.js)

Tokens arrive from the model in bursts. main.js queues them and reveals the reply at speaking pace
(~30 chars/s, longer pauses after `,` `.` `!` `?` and newlines, faster if the queue backs up so she
never lags far behind the model). Each revealed character goes to `bubble.append(ch)`, `capri.speak(ch)`
and `voice.blip(ch)` together, so mouth, head beats, sound and text stay in sync. `capri.setState('talking')`
runs while the clock has characters left, not while the network is busy.
