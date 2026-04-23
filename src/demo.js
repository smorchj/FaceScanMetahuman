// End-to-end one-shot identity warp demo.
//
// Loads a MetaHuman GLB and its anchor JSON, runs MediaPipe on the
// webcam, and on button click fits the MH neutral to the user's face
// proportions via rigid Procrustes + biharmonic RBF. No texture, no
// live update. The goal is to see "that looks like me-ish" and move on.

import * as THREE from 'three';
import { mount } from './viewer.js';

import { MP_INDICES } from './mp_indices.js';
import { solveWarp, applyWarp, procrustesAlign, applyRigid } from './warp.js';

const MP_BUNDLE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs';
const MP_WASM   = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm';
const MP_MODEL  = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export async function start(opts) {
  const { stage, video, overlay, captureBtn, resetBtn, statusEl,
          glbUrl, anchorsUrl } = opts;

  const setStatus = (s) => { statusEl.textContent = s; };

  setStatus('fetching GLB and materials...');
  const three = await mount(stage, {
    glbUrl,
    mappingUrl: opts.mappingUrl,
    autoRotate: false,
    interactive: true,
    characterId: opts.characterId,
  });

  // Optional starter anchors: kept for backward compat. If it loads
  // we seed the overlay with it. First Capture rebuilds from scratch
  // at the estimated FOV, which typically supersedes this file.
  let anchorsData = null;
  if (anchorsUrl) {
    try {
      const r = await fetch(anchorsUrl);
      if (r.ok) anchorsData = await r.json();
    } catch (_) { /* fine, we'll build on first capture */ }
  }

  const preferredMesh = anchorsData && anchorsData.anchors[0]
                        && anchorsData.anchors[0].meshName;
  const skin = findSkinMesh(three.gltf.scene, preferredMesh);
  if (!skin) throw new Error('could not find a face/head mesh in GLB');

  // Head bone for visible pose matching: during capture we rotate
  // this bone each tick so Ada follows the user's head rotation in
  // the main view, which also makes our MP detect pass (rendered
  // head-on) see her at the same pose as the user.
  const headBone = findHeadBone(skin);
  const headBoneRestQuat = headBone ? headBone.quaternion.clone() : null;
  console.log('[demo] head bone:', headBone ? headBone.name : '(not found)');

  // Accumulating face texture:
  //   - Seed a canvas with Ada's existing skin base-color map.
  //   - Each tick, paint the webcam chunk of every MP triangle that
  //     is currently front-facing into the canvas at the UV positions
  //     of that triangle's three anchors. Triangles not facing the
  //     camera (user is turned away from them) are skipped, so their
  //     previously-baked pixels stay untouched. Over a few seconds
  //     of turning the user builds a complete face texture.
  const faceTex = initFaceTexture(skin);
  console.log('[demo] face texture size:', faceTex ? faceTex.size : '(skipped)');
  // Webcam snapshot canvas reused every tick.
  const sampleCanvas = document.createElement('canvas');
  const sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: false });

  // Snapshot rest positions for every head-adjacent mesh so the warp
  // can carry eyeballs, teeth, lashes, brows, and hair along with the
  // skin. Each entry: { mesh, rest (Float32Array), attr }. Reset
  // button restores all of them.
  const warpTargets = collectHeadMeshes(three.gltf.scene);
  if (!warpTargets.some((t) => t.mesh === skin)) {
    // Ensure the skin itself is in there even if the name heuristic missed it.
    warpTargets.push({ mesh: skin });
  }
  for (const t of warpTargets) {
    t.attr = t.mesh.geometry.attributes.position;
    t.rest = new Float32Array(t.attr.array);
  }
  console.log('[demo] warp targets:', warpTargets.map((t) => t.mesh.name));

  // Seed anchors from the optional file. Empty array otherwise; the
  // first Capture rebuilds from the current webcam FOV anyway.
  const anchors = [];
  if (anchorsData) {
    const seen = new Set();
    for (const a of anchorsData.anchors) {
      const mpIdx = (a.mpIndex !== undefined) ? a.mpIndex : MP_INDICES[a.name];
      if (mpIdx === undefined) continue;
      if (seen.has(a.vertexIndex)) continue;
      seen.add(a.vertexIndex);
      anchors.push({
        name: a.name,
        mhIdx: a.vertexIndex,
        mhRest: a.restPosition.slice(),
        mpIdx,
      });
    }
    console.log('[demo] seed anchors from file:', anchors.length);
  }

  frameHead(three.gltf.scene, three.camera, three.controls);

  setStatus('loading MediaPipe face landmarker...');
  const vision = await import(MP_BUNDLE);
  const { FaceLandmarker, FilesetResolver } = vision;
  const resolver = await FilesetResolver.forVisionTasks(MP_WASM);
  // Two landmarker instances: one for live webcam (VIDEO mode), one
  // for on-demand Ada renders (IMAGE mode). MP does not let a single
  // instance switch between modes.
  const landmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
  });
  const imgLandmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
    runningMode: 'IMAGE',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  setStatus('requesting webcam...');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
  });
  video.srcObject = stream;
  await new Promise((r) => video.addEventListener('loadeddata', r, { once: true }));
  video.play();

  // Match overlay canvas to video intrinsic size.
  const overlayCtx = overlay.getContext('2d');
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  // Declare anchor cache before the tick loop so the overlay can
  // show whichever anchor set is currently driving the warp.
  let cachedAnchors = null;
  let cachedFovDeg = null;

  let latest = null;
  let latestMatrix = null;
  (function tick() {
    if (video.readyState >= 2) {
      const res = landmarker.detectForVideo(video, performance.now());
      if (res.faceLandmarks && res.faceLandmarks[0]) {
        latest = res.faceLandmarks[0];
        latestMatrix = res.facialTransformationMatrixes
                    && res.facialTransformationMatrixes[0]
                    && res.facialTransformationMatrixes[0].data;
        drawOverlay(overlayCtx, overlay, latest, cachedAnchors || anchors);
      } else {
        latest = null;
        latestMatrix = null;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
    requestAnimationFrame(tick);
  })();

  captureBtn.disabled = false;
  setStatus('ready. look at the camera and click capture.');

  // Pose-matched multi-frame capture. Start toggles; Stop freezes.
  // Each sample renders Ada with the detect camera rotated to match
  // the user's current head rotation (from MP's facial transformation
  // matrix), runs MP on her render, and uses those pose-matched
  // ada landmarks as the Procrustes source. The alignment thus
  // factors out pose geometry; identity is what remains.
  //
  // FOV is hardcoded at 60° during live capture to keep the loop
  // fast. Swap to FOV detection later if needed.
  const HARDCODED_FOV = 60;
  const SAMPLE_INTERVAL_MS = 150;
  const LIVE_UPDATE_EVERY_N = 3;

  let captureLoopId = null;
  let cachedMpToMh = null;           // rigid+scale MP canonical -> MH rest
  let cachedAdaLmRaw = null;         // Ada's 478 landmarks (head-on)
  let cachedMpTriangles = null;      // MP's canonical triangulation: [[mpA, mpB, mpC], ...]
  let cachedVertexContain = null;    // per face-mesh vertex: {tri, a, b, c} or null
  const USER_SMOOTH_N = 6;           // running-average window for user lattice
  let userRecent = [];               // array of raw user 478-landmark arrays
  let userQuatRef = null;            // user head quat at start, = "head-on" baseline

  async function startCapture() {
    if (!latest) { setStatus('no face detected'); return; }
    captureBtn.textContent = 'Stop capture';
    if (!cachedAnchors || Math.abs(cachedFovDeg - HARDCODED_FOV) > 0.01) {
      setStatus('building anchor map on Ada at FOV ' + HARDCODED_FOV + '°...');
      const built = await buildAnchorsAtFov(three, skin, imgLandmarker, HARDCODED_FOV);
      cachedAnchors    = built.anchors;
      cachedMpToMh     = built.mpToMh;
      cachedAdaLmRaw   = built.adaLandmarksRaw;
      cachedFovDeg     = HARDCODED_FOV;
      // Build MP's canonical triangulation from its edge list, then
      // keep only triangles whose 3 MP indices all exist as anchors.
      const tess = imgLandmarker.constructor.FACE_LANDMARKS_TESSELATION
                || landmarker.constructor.FACE_LANDMARKS_TESSELATION;
      cachedMpTriangles = buildMpTriangles(tess, cachedAnchors);
      // Precompute per-face-vertex containment in MP triangle space.
      setStatus('precomputing triangle containment...');
      cachedVertexContain = precomputeVertexContainment(
        skin, cachedAnchors, cachedMpTriangles,
      );
      const covered = cachedVertexContain.filter(Boolean).length;
      setStatus('anchors: ' + cachedAnchors.length
        + ', MP triangles: ' + cachedMpTriangles.length
        + ', covered face verts: ' + covered + '. capturing live...');
    }
    userRecent = [];
    userQuatRef = null;
    captureLoopId = setInterval(() => captureTick(), SAMPLE_INTERVAL_MS);
  }

  function stopCapture() {
    if (captureLoopId !== null) {
      clearInterval(captureLoopId);
      captureLoopId = null;
    }
    if (headBone && headBoneRestQuat) {
      headBone.quaternion.copy(headBoneRestQuat);
    }
    restoreOriginalMap();
    captureBtn.textContent = 'Start capture';
    setStatus('capture stopped.');
  }

  async function captureTick() {
    if (!latest || !latestMatrix) return;
    try {
      userRecent.push(latest);
      if (userRecent.length > USER_SMOOTH_N) userRecent.shift();
      const userAvg = averageLandmarks(userRecent);

      // Convert MP's facial-transformation matrix to a three.js
      // quaternion, then apply the standard "selfie mirror" flip.
      // In mirror-space a rotation (axis, angle) becomes (mirrored
      // axis, -angle) which is equivalent to negating the quaternion
      // components parallel to the mirror plane. For a horizontal
      // webcam mirror (x <-> -x) that means y <-> -y, z <-> -z on
      // the quaternion. Subtract a baseline captured at Start so the
      // user's first frame is treated as rest.
      const mpMat = new THREE.Matrix4().fromArray(latestMatrix);
      const rawQuat = new THREE.Quaternion().setFromRotationMatrix(mpMat);
      // Selfie-mirror: negate y and z.
      rawQuat.y = -rawQuat.y;
      rawQuat.z = -rawQuat.z;

      if (!userQuatRef) userQuatRef = rawQuat.clone();
      const userDeltaQuat = rawQuat.clone().multiply(userQuatRef.clone().invert());

      if (headBone && headBoneRestQuat) {
        headBone.quaternion.copy(headBoneRestQuat).multiply(userDeltaQuat);
      }

      // Render Ada head-on from detect camera (her head bone is now
      // rotated so MP sees her at the user's pose).
      const adaLmPose = await renderAdaHeadOn(
        three, HARDCODED_FOV, imgLandmarker,
      );
      if (!adaLmPose) return;

      const targets = latticeTargetsPoseMatched(
        userAvg, adaLmPose, userDeltaQuat, cachedAnchors, cachedMpToMh,
      );
      const stats = runWarpFromTargets(targets, cachedAnchors, warpTargets);

      // Accumulating texture paint: for every MP triangle currently
      // facing the camera, warp the webcam region bounded by its
      // three MP landmarks into the UV region bounded by its three
      // anchor UVs on a persistent canvas. Ada's original texture
      // stays in the canvas where triangles haven't been painted yet,
      // acting as the fallback.
      paintFaceTextureMpTriangles(
        faceTex, video, sampleCanvas, sampleCtx,
        latest, userDeltaQuat,
        cachedAnchors, cachedMpTriangles, cachedAdaLmRaw,
      );

      setStatus('capturing (pose-matched lattice)\n'
        + 'smoothed frames: ' + userRecent.length + '\n'
        + 'mean residual: ' + stats.meanResidual.toFixed(5));
    } catch (err) {
      console.error('[demo] captureTick error:', err);
    }
  }

  captureBtn.addEventListener('click', () => {
    if (captureLoopId === null) startCapture();
    else stopCapture();
  });

  resetBtn.addEventListener('click', () => {
    for (const t of warpTargets) {
      t.attr.array.set(t.rest);
      t.attr.needsUpdate = true;
      t.mesh.geometry.computeBoundingSphere();
    }
    setStatus('reset to rest pose');
  });
}

// All head-adjacent meshes that should ride along with the warped face.
// Keyword match covers MH face sub-meshes, hair cards, lash/brow cards,
// teeth, and tongue. Skips body, clothing, and anything further away.
//
// `category`:
//   'skin' -- distance-weighted falloff from face anchors (face skin,
//             teeth, tongue, occlusion — things that sit in the head and
//             should follow the face shape but respect the falloff).
//   'attached' -- full warp (w = 1) regardless of distance. Hair, lashes,
//                 and brows sit OFFSET from the scalp and would otherwise
//                 float because their verts fall outside the falloff
//                 radius.
const HEAD_MESH_HINTS = {
  face:    'skin',
  head:    'skin',
  teeth:   'skin',
  tongue:  'skin',
  saliva:  'skin',
  occlusion: 'skin',
  hair:    'attached',
  lash:    'attached',
  brow:    'attached',
};

function collectHeadMeshes(root) {
  const out = [];
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    const n = (obj.name || '').toLowerCase();
    let category = null;
    for (const [hint, cat] of Object.entries(HEAD_MESH_HINTS)) {
      if (n.includes(hint)) { category = cat; break; }
    }
    if (category) out.push({ mesh: obj, category });
  });
  return out;
}

// --- core warp step ---

// Extract the rotation-only quaternion from a 16-element column-major
// 4x4 matrix (MediaPipe facial transformation format).
function rotationFromMatrix16(m) {
  const mat = new THREE.Matrix4().fromArray(m);
  const q = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  mat.decompose(pos, q, scl);
  return q;
}

// Set up a dynamic diffuse texture on the face skin material: copy
// Ada's current base-color map into a canvas, wrap that canvas as a
// CanvasTexture, and swap it in as material.map. The canvas is then
// painted live during capture. Returns null if the material has no
// existing map to seed from.
function initFaceTexture(skin) {
  const mat = Array.isArray(skin.material) ? skin.material[0] : skin.material;
  if (!mat) return null;
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });

  const src = mat.map && mat.map.image;
  if (src && src.width) {
    ctx.drawImage(src, 0, 0, size, size);
  } else {
    // Fallback seed. Skin-ish so the face is recognisable before any
    // capture has painted onto it.
    ctx.fillStyle = '#b08066';
    ctx.fillRect(0, 0, size, size);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = mat.map ? mat.map.flipY : false;
  texture.wrapS = mat.map ? mat.map.wrapS : THREE.ClampToEdgeWrapping;
  texture.wrapT = mat.map ? mat.map.wrapT : THREE.ClampToEdgeWrapping;
  mat.map = texture;
  mat.needsUpdate = true;
  return { canvas, ctx, texture, size };
}

// Visibility check: rotate each anchor's canonical +Z normal (proxied
// by the anchor's own canonical position direction from face centre)
// by the user's head rotation. If the resulting +Z is positive, the
// anchor faces the camera and its landmark is sampleable.
function anchorFacesCamera(anchor, userQuat, cachedAdaLmRaw) {
  const l = cachedAdaLmRaw[anchor.mpIdx];
  if (!l) return true;
  // Canonical-ish direction: offset from face centre, normalised.
  const dx = l.x - 0.5, dy = -(l.y - 0.5), dz = -l.z;
  // Rotate by the user's head rotation (delta from head-on).
  const v = new THREE.Vector3(dx, dy, dz).applyQuaternion(userQuat);
  return v.z > 0.01;
}

// Enumerate triangles from MediaPipe's tesselation edge list. Keep
// only those whose 3 MP indices all appear as anchors (so we have
// a valid MH correspondence for every vertex of the triangle).
// Returns array of [mpA, mpB, mpC, anchorA, anchorB, anchorC].
function buildMpTriangles(tessConnections, anchors) {
  if (!tessConnections) return [];
  const mpToAnchorIdx = new Map();
  anchors.forEach((a, i) => mpToAnchorIdx.set(a.mpIdx, i));

  const adj = new Map();
  for (const c of tessConnections) {
    const s = c.start, e = c.end;
    if (!adj.has(s)) adj.set(s, new Set());
    if (!adj.has(e)) adj.set(e, new Set());
    adj.get(s).add(e);
    adj.get(e).add(s);
  }
  const seen = new Set();
  const out = [];
  for (const v of adj.keys()) {
    const neigh = Array.from(adj.get(v));
    for (let i = 0; i < neigh.length; i++) {
      const a = neigh[i]; if (a <= v) continue;
      for (let j = i + 1; j < neigh.length; j++) {
        const b = neigh[j]; if (b <= a) continue;
        if (!adj.get(a).has(b)) continue;
        const ai = mpToAnchorIdx.get(v);
        const bi = mpToAnchorIdx.get(a);
        const ci = mpToAnchorIdx.get(b);
        if (ai === undefined || bi === undefined || ci === undefined) continue;
        const key = v + ',' + a + ',' + b;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push([v, a, b, ai, bi, ci]);
      }
    }
  }
  return out;
}

// For each face-mesh vertex close enough to an anchor, assign an MP
// triangle: either the one that contains the vertex (XY projection),
// or the nearest by centroid distance. Nearest-triangle assignment
// extrapolates barycentric slightly outside [0,1] for silhouette
// verts, which avoids the "fragmented texture" artifact where a
// triangle drawn by the renderer has one vertex with webcam UV and
// another with Ada's original UV.
//
// Verts farther than FACE_REACH metres from any anchor keep their
// original UV (null). That is the implicit face mask.
function precomputeVertexContainment(skin, anchors, mpTriangles) {
  const posAttr = skin.geometry.attributes.position;
  const n = posAttr.count;
  const FACE_REACH = 0.05;
  const reachSq = FACE_REACH * FACE_REACH;
  const result = new Array(n);

  // Precompute triangle centroids (XY) for the nearest-triangle query.
  const centroids = mpTriangles.map((tri) => {
    const A = anchors[tri[3]].mhRest;
    const B = anchors[tri[4]].mhRest;
    const C = anchors[tri[5]].mhRest;
    return [(A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3];
  });

  for (let i = 0; i < n; i++) {
    const px = posAttr.getX(i);
    const py = posAttr.getY(i);
    const pz = posAttr.getZ(i);

    // Face mask: skip verts far from any anchor.
    let minAnchorSq = Infinity;
    for (let k = 0; k < anchors.length; k++) {
      const r = anchors[k].mhRest;
      const dx = px - r[0], dy = py - r[1], dz = pz - r[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < minAnchorSq) minAnchorSq = d;
    }
    if (minAnchorSq > reachSq) { result[i] = null; continue; }

    // Prefer a triangle that contains the vertex; fall back to the
    // nearest one by centroid distance.
    let bestInside = null, bestInsideBias = Infinity;
    let bestNearest = null, bestNearestDist = Infinity;
    for (let t = 0; t < mpTriangles.length; t++) {
      const tri = mpTriangles[t];
      const A = anchors[tri[3]].mhRest;
      const B = anchors[tri[4]].mhRest;
      const C = anchors[tri[5]].mhRest;
      const bary = bary2d(px, py, A[0], A[1], B[0], B[1], C[0], C[1]);
      if (!bary) continue;
      const minAxis = Math.min(bary.a, bary.b, bary.c);
      if (minAxis >= 0) {
        const bias = -minAxis;
        if (bias < bestInsideBias) {
          bestInsideBias = bias;
          bestInside = { tri: t, a: bary.a, b: bary.b, c: bary.c };
        }
      } else {
        const cx = centroids[t][0], cy = centroids[t][1];
        const dx = px - cx, dy = py - cy;
        const dist = dx * dx + dy * dy;
        if (dist < bestNearestDist) {
          bestNearestDist = dist;
          bestNearest = { tri: t, a: bary.a, b: bary.b, c: bary.c };
        }
      }
    }
    result[i] = bestInside || bestNearest;
  }
  return result;
}

function bary2d(px, py, ax, ay, bx, by, cx, cy) {
  const denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(denom) < 1e-14) return null;
  const a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom;
  const b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom;
  const c = 1 - a - b;
  return { a, b, c };
}

// Paint webcam into the persistent face-texture canvas, one MP
// triangle at a time. Only triangles whose canonical normal, when
// rotated by the user's head delta, points toward the camera
// (+Z after rotation) are drawn. Accumulates across frames: as the
// user turns, freshly-visible triangles overwrite; turned-away
// triangles keep the pixels baked earlier.
function paintFaceTextureMpTriangles(
  faceTex, video, sampleCanvas, sampleCtx,
  userLm, userDeltaQuat,
  anchors, triangles, adaLmRaw,
) {
  if (!faceTex || !video.videoWidth || !triangles || !adaLmRaw) return;
  if (sampleCanvas.width !== video.videoWidth
   || sampleCanvas.height !== video.videoHeight) {
    sampleCanvas.width = video.videoWidth;
    sampleCanvas.height = video.videoHeight;
  }
  sampleCtx.drawImage(video, 0, 0);
  const { ctx, size, texture } = faceTex;
  const W = sampleCanvas.width, H = sampleCanvas.height;

  const n1 = new THREE.Vector3();
  const n2 = new THREE.Vector3();
  const cross = new THREE.Vector3();

  for (const tri of triangles) {
    const mpA = tri[0], mpB = tri[1], mpC = tri[2];
    const ai = tri[3], bi = tri[4], ci = tri[5];
    const la = userLm[mpA], lb = userLm[mpB], lc = userLm[mpC];
    if (!la || !lb || !lc) continue;

    // Visibility via canonical normal rotated by user head delta.
    const A = adaLmRaw[mpA], B = adaLmRaw[mpB], C = adaLmRaw[mpC];
    n1.set(B.x - A.x, -(B.y - A.y), -(B.z - A.z));
    n2.set(C.x - A.x, -(C.y - A.y), -(C.z - A.z));
    cross.crossVectors(n1, n2).applyQuaternion(userDeltaQuat);
    if (cross.z <= 0) continue;

    // Secondary check: positive signed area in webcam (back-facing
    // user landmarks have reversed winding).
    const sx1 = la.x * W, sy1 = la.y * H;
    const sx2 = lb.x * W, sy2 = lb.y * H;
    const sx3 = lc.x * W, sy3 = lc.y * H;
    const signedArea = (sx2 - sx1) * (sy3 - sy1) - (sx3 - sx1) * (sy2 - sy1);
    if (signedArea <= 2) continue;

    const uvA = anchors[ai].uv, uvB = anchors[bi].uv, uvC = anchors[ci].uv;
    if (!uvA || !uvB || !uvC) continue;
    const dx1 = uvA[0] * size, dy1 = (1 - uvA[1]) * size;
    const dx2 = uvB[0] * size, dy2 = (1 - uvB[1]) * size;
    const dx3 = uvC[0] * size, dy3 = (1 - uvC[1]) * size;

    drawTriangleWarp(ctx, sampleCanvas,
      sx1, sy1, sx2, sy2, sx3, sy3,
      dx1, dy1, dx2, dy2, dx3, dy3);
  }
  texture.needsUpdate = true;
}

// Each frame: for each face vertex inside an MP triangle, set its UV
// to the barycentric blend of that triangle's 3 MP landmark webcam
// positions. Verts outside every MP triangle keep their original UV.
function updateFaceUVsFromMpTriangles(uvAttr, origUv, triangles, containment, userLm) {
  if (!uvAttr || !origUv || !triangles || !containment) return;
  const arr = uvAttr.array;
  for (let i = 0; i < containment.length; i++) {
    const c = containment[i];
    if (!c) {
      arr[i * 2 + 0] = origUv[i * 2 + 0];
      arr[i * 2 + 1] = origUv[i * 2 + 1];
      continue;
    }
    const tri = triangles[c.tri];
    const la = userLm[tri[0]];
    const lb = userLm[tri[1]];
    const lc = userLm[tri[2]];
    arr[i * 2 + 0] = c.a * la.x + c.b * lb.x + c.c * lc.x;
    arr[i * 2 + 1] = c.a * la.y + c.b * lb.y + c.c * lc.y;
  }
  uvAttr.needsUpdate = true;
}

// Legacy RBF-based UV update, kept for comparison. Not called.
function updateFaceUVsFromLattice(skin, uvAttr, origUv, anchors, userLm) {
  if (!uvAttr || !origUv) return;
  const FACE_REACH = 0.04;
  const reachSq = FACE_REACH * FACE_REACH;

  // Build RBF: source = anchor 3D rest positions, target = anchor
  // MP 2D image positions (z packed as 0). Output x,y are the
  // webcam UVs; z we ignore.
  const sources = anchors.map((a) => a.mhRest);
  const targets = anchors.map((a) => {
    const l = userLm[a.mpIdx];
    return [l.x, l.y, 0];
  });
  const warp = solveWarp(sources, targets);

  const posAttr = skin.geometry.attributes.position;
  const arr = uvAttr.array;
  const n = posAttr.count;
  const pt = [0, 0, 0];
  const out = [0, 0, 0];

  for (let i = 0; i < n; i++) {
    pt[0] = posAttr.getX(i);
    pt[1] = posAttr.getY(i);
    pt[2] = posAttr.getZ(i);

    // Nearest anchor distance
    let minSq = Infinity;
    for (let k = 0; k < anchors.length; k++) {
      const a = anchors[k].mhRest;
      const dx = pt[0] - a[0], dy = pt[1] - a[1], dz = pt[2] - a[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < minSq) minSq = d;
    }
    if (minSq > reachSq) {
      arr[i * 2 + 0] = origUv[i * 2 + 0];
      arr[i * 2 + 1] = origUv[i * 2 + 1];
      continue;
    }

    applyWarp(warp, pt, out);
    // MP normalized coords are [0,1]. Clamp so any overshoot does
    // not wrap to weird parts of the webcam texture.
    arr[i * 2 + 0] = Math.max(0, Math.min(1, out[0]));
    arr[i * 2 + 1] = Math.max(0, Math.min(1, out[1]));
  }
  uvAttr.needsUpdate = true;
}

// Project webcam pixels into the dynamic face texture, one affine
// triangle at a time. For each triangle of the anchor Delaunay
// triangulation we warp the corresponding webcam region into the UV
// region bounded by the triangle's three UV corners. Fills the whole
// face in one pass; blank regions only remain where triangles are
// back-facing (skipped) or outside the face mesh (no anchors there).
function projectWebcamTriangles(
  faceTex, video, sampleCanvas, sampleCtx,
  userLm, anchors, triangles,
) {
  if (!video.videoWidth) return;
  if (sampleCanvas.width !== video.videoWidth
   || sampleCanvas.height !== video.videoHeight) {
    sampleCanvas.width = video.videoWidth;
    sampleCanvas.height = video.videoHeight;
  }
  sampleCtx.drawImage(video, 0, 0);
  const { ctx, size, texture } = faceTex;
  const W = sampleCanvas.width, H = sampleCanvas.height;

  for (const [ia, ib, ic] of triangles) {
    const a = anchors[ia], b = anchors[ib], c = anchors[ic];
    const la = userLm[a.mpIdx], lb = userLm[b.mpIdx], lc = userLm[c.mpIdx];
    if (!la || !lb || !lc) continue;

    const sx1 = la.x * W, sy1 = la.y * H;
    const sx2 = lb.x * W, sy2 = lb.y * H;
    const sx3 = lc.x * W, sy3 = lc.y * H;

    // Signed area in source image: positive for front-facing
    // triangles, negative when the user rotates away and MP's
    // extrapolated landmarks flip the winding.
    const signedArea = (sx2 - sx1) * (sy3 - sy1) - (sx3 - sx1) * (sy2 - sy1);
    if (signedArea <= 0) continue;

    // MH mesh UVs put V=0 at the BOTTOM of the texture image; canvas
    // coords put y=0 at the TOP. Flip V when converting to canvas.
    const dx1 = a.uv[0] * size, dy1 = (1 - a.uv[1]) * size;
    const dx2 = b.uv[0] * size, dy2 = (1 - b.uv[1]) * size;
    const dx3 = c.uv[0] * size, dy3 = (1 - c.uv[1]) * size;

    drawTriangleWarp(ctx, sampleCanvas,
      sx1, sy1, sx2, sy2, sx3, sy3,
      dx1, dy1, dx2, dy2, dx3, dy3);
  }
  texture.needsUpdate = true;
}

// Affine warp of an arbitrary triangle in `src` to an arbitrary
// triangle in `ctx`. Uses setTransform with the unique affine matrix
// mapping (s1,s2,s3) to (d1,d2,d3), clipped to the destination
// triangle so only pixels inside are drawn.
function drawTriangleWarp(
  ctx, src,
  sx1, sy1, sx2, sy2, sx3, sy3,
  dx1, dy1, dx2, dy2, dx3, dy3,
) {
  // Solve for affine M such that M * [sx_i; sy_i; 1] = [dx_i; dy_i].
  // Two 3x3 systems, one per output axis, sharing the same design
  // matrix. Inline to avoid allocations.
  const det = sx1 * (sy2 - sy3) - sy1 * (sx2 - sx3) + (sx2 * sy3 - sx3 * sy2);
  if (Math.abs(det) < 1e-9) return;
  const inv = 1 / det;
  const a = (dx1 * (sy2 - sy3) + dx2 * (sy3 - sy1) + dx3 * (sy1 - sy2)) * inv;
  const b = (dx1 * (sx3 - sx2) + dx2 * (sx1 - sx3) + dx3 * (sx2 - sx1)) * inv;
  const e = (dx1 * (sx2 * sy3 - sx3 * sy2)
           + dx2 * (sx3 * sy1 - sx1 * sy3)
           + dx3 * (sx1 * sy2 - sx2 * sy1)) * inv;
  const c = (dy1 * (sy2 - sy3) + dy2 * (sy3 - sy1) + dy3 * (sy1 - sy2)) * inv;
  const d = (dy1 * (sx3 - sx2) + dy2 * (sx1 - sx3) + dy3 * (sx2 - sx1)) * inv;
  const f = (dy1 * (sx2 * sy3 - sx3 * sy2)
           + dy2 * (sx3 * sy1 - sx1 * sy3)
           + dy3 * (sx1 * sy2 - sx2 * sy1)) * inv;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(dx1, dy1);
  ctx.lineTo(dx2, dy2);
  ctx.lineTo(dx3, dy3);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, c, b, d, e, f);
  ctx.drawImage(src, 0, 0);
  ctx.restore();
}

function dist3(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Find the head bone on a MH skinned mesh. MH skeletons expose it as
// "head" but lowercase vs capitalized varies by export.
function findHeadBone(skin) {
  if (!skin.skeleton || !skin.skeleton.bones) return null;
  const bones = skin.skeleton.bones;
  for (const b of bones) if ((b.name || '').toLowerCase() === 'head') return b;
  for (const b of bones) if ((b.name || '').toLowerCase().includes('head')) return b;
  return null;
}

// Render Ada head-on from a detect camera at the current scene
// framing. Ada's bone state (including any head-bone rotation the
// caller has set) is applied during rendering, so MP sees her in
// whatever pose the skeleton currently encodes.
async function renderAdaHeadOn(three, fovDeg, landmarker) {
  const DETECT = 512;
  const cam = new THREE.PerspectiveCamera(fovDeg, 1, 0.01, 20);
  const target = three.controls ? three.controls.target.clone()
                                : new THREE.Vector3(0, 1.5, 0);
  const restDist = three.camera.position.distanceTo(target);
  cam.position.set(target.x, target.y, target.z + restDist);
  cam.up.set(0, 1, 0);
  cam.lookAt(target);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();

  const rt = new THREE.WebGLRenderTarget(DETECT, DETECT, {
    colorSpace: THREE.SRGBColorSpace,
  });
  three.renderer.setRenderTarget(rt);
  three.renderer.render(three.scene, cam);
  three.renderer.setRenderTarget(null);
  const pixels = new Uint8Array(DETECT * DETECT * 4);
  three.renderer.readRenderTargetPixels(rt, 0, 0, DETECT, DETECT, pixels);
  rt.dispose();

  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = DETECT;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(DETECT, DETECT);
  const rowBytes = DETECT * 4;
  for (let y = 0; y < DETECT; y++) {
    const src = (DETECT - 1 - y) * rowBytes;
    img.data.set(pixels.subarray(src, src + rowBytes), y * rowBytes);
  }
  ctx.putImageData(img, 0, 0);

  const result = landmarker.detect(canvas);
  if (!result.faceLandmarks || !result.faceLandmarks[0]) return null;
  return result.faceLandmarks[0];
}

// Render Ada with the detect camera rotated to match the user's head
// rotation. Run MP on the rendered frame. Returns the 478 landmarks
// (in MP normalized image space) or null if detection failed.
async function renderAdaPoseMatched(three, userRotation, fovDeg, landmarker) {
  const DETECT = 512; // smaller than one-shot anchor build; live loop
  // Detect camera sits at Ada's face looking at it, then is orbited
  // around Ada's face center by the user's rotation. With Ada at rest
  // and the camera rotated by userRotation^-1 around the face pivot,
  // the view appears as if Ada rotated her head by userRotation.
  const cam = new THREE.PerspectiveCamera(fovDeg, 1, 0.01, 20);

  // Face center comes from the three.camera's current target (set at
  // frameHead); here we reuse its pos/target relationship.
  const target = three.controls ? three.controls.target.clone()
                                : new THREE.Vector3(0, 1.5, 0);
  const restDist = three.camera.position.distanceTo(target);
  const restOffset = new THREE.Vector3(0, 0, restDist);

  // Rotate rest offset by the INVERSE of user's rotation around the face.
  const invRotation = userRotation.clone().invert();
  restOffset.applyQuaternion(invRotation);
  cam.position.copy(target).add(restOffset);
  cam.up.set(0, 1, 0);
  cam.lookAt(target);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();

  const rt = new THREE.WebGLRenderTarget(DETECT, DETECT, {
    colorSpace: THREE.SRGBColorSpace,
  });
  three.renderer.setRenderTarget(rt);
  three.renderer.render(three.scene, cam);
  three.renderer.setRenderTarget(null);
  const pixels = new Uint8Array(DETECT * DETECT * 4);
  three.renderer.readRenderTargetPixels(rt, 0, 0, DETECT, DETECT, pixels);
  rt.dispose();

  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = DETECT;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(DETECT, DETECT);
  const rowBytes = DETECT * 4;
  for (let y = 0; y < DETECT; y++) {
    const src = (DETECT - 1 - y) * rowBytes;
    img.data.set(pixels.subarray(src, src + rowBytes), y * rowBytes);
  }
  ctx.putImageData(img, 0, 0);

  const result = landmarker.detect(canvas);
  if (!result.faceLandmarks || !result.faceLandmarks[0]) return null;
  return result.faceLandmarks[0];
}

// Z-boost for the per-anchor MH delta. MP's z coordinate encodes less
// shape identity per unit than x/y do in practice; multiplying the
// z-component of the delta pulls depth back into the warp. Tune in
// testing; 1.0 = raw MP z, higher = exaggerated depth.
const Z_BOOST = 3.0;

// Delta magnitude stats, logged once per computeAligned call so we
// can see whether z is just small or truly zero.
let _deltaStatsSample = 0;

// Per-frame aligned targets where the Procrustes rigid-align runs
// against pose-matched Ada landmarks (not Ada-at-rest). The 2D MP
// positions of user and Ada are both viewed from the same virtual
// angle, so their difference is pure identity. After Procrustes
// alignment we map each aligned-user 2D position into 3D MH space
// using a fixed scale derived from anchor rest positions once.
function computeAlignedTargetsPoseMatched(userLm, adaLm, anchors) {
  // User and Ada landmarks in MP's 3D-ish space.
  const userPts = anchors.map((a) => {
    const l = userLm[a.mpIdx];
    return [l.x - 0.5, -(l.y - 0.5), -l.z];
  });
  const adaPts = anchors.map((a) => {
    const l = adaLm[a.mpIdx];
    return [l.x - 0.5, -(l.y - 0.5), -l.z];
  });
  // Rigid-align user MP points to Ada MP points (pose-matched) so
  // translation / uniform scale cancel out. Identity is the non-rigid
  // residual after alignment.
  const userToAda = procrustesAlign(adaPts, userPts);
  const userAlignedToAdaMp = userPts.map((p) => applyRigid(userToAda, p, [0, 0, 0]));

  // Deltas in MP space per anchor.
  const deltas = userAlignedToAdaMp.map((p, i) => [
    p[0] - adaPts[i][0],
    p[1] - adaPts[i][1],
    (p[2] - adaPts[i][2]) * Z_BOOST,
  ]);

  // Diagnostic: per-axis delta magnitudes once every ~1s.
  if (_deltaStatsSample++ % 7 === 0) {
    let mx = 0, my = 0, mz = 0;
    for (const d of deltas) {
      mx += Math.abs(d[0]); my += Math.abs(d[1]); mz += Math.abs(d[2]);
    }
    const n = deltas.length;
    console.log('[demo] mean |delta|  x=' + (mx/n).toFixed(4)
                + '  y=' + (my/n).toFixed(4)
                + '  z=' + (mz/n).toFixed(4) + '  (z already x' + Z_BOOST + ')');
  }

  // Scale MP-space delta into MH-space. Compute scale once by Procrustes
  // between Ada MP positions and MH rest positions.
  const mhRestPts = anchors.map((a) => a.mhRest);
  const mpToMh = procrustesAlign(mhRestPts, adaPts);
  const scale = mpToMh.scale;
  const R = mpToMh.R;

  // Target = MH rest + rotated scaled delta.
  return mhRestPts.map((rest, i) => {
    const d = deltas[i];
    return [
      rest[0] + scale * (R[0]*d[0] + R[1]*d[1] + R[2]*d[2]),
      rest[1] + scale * (R[3]*d[0] + R[4]*d[1] + R[5]*d[2]),
      rest[2] + scale * (R[6]*d[0] + R[7]*d[1] + R[8]*d[2]),
    ];
  });
}

// Average a list of 478-landmark frames element-wise. Returns a new
// array of {x,y,z} objects.
function averageLandmarks(frames) {
  const n = frames[0].length;
  const out = new Array(n);
  const k = frames.length;
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0, sz = 0;
    for (const f of frames) {
      const l = f[i];
      sx += l.x; sy += l.y; sz += l.z;
    }
    out[i] = { x: sx / k, y: sy / k, z: sz / k };
  }
  return out;
}

// Pose-matched lattice warp.
//
//   userLm:        user's 478 landmarks at current head rotation R
//   adaPoseMatched: 478 landmarks on Ada, rendered at the same R
//   userQuat:      the rotation R itself (canonical -> camera)
//
// Both lattices are at the same pose, so Procrustes residual is close
// to pure identity at that pose. We then un-rotate by R^-1 to bring
// the delta back into canonical frame, scale via mpToMh, and add to
// each anchor's MH rest position.
function latticeTargetsPoseMatched(userLm, adaPoseMatched, userQuat, anchors, mpToMh) {
  const userPts = userLm.map(mpToCanonical);
  const adaPts  = adaPoseMatched.map(mpToCanonical);
  const rigid = procrustesAlign(adaPts, userPts);
  const userAligned = userPts.map((p) => applyRigid(rigid, p, [0, 0, 0]));

  // R^-1 as a row-major 3x3 for un-rotating the residual.
  const invMat = new THREE.Matrix4()
    .makeRotationFromQuaternion(userQuat.clone().invert()).elements;
  const iR = [
    invMat[0], invMat[4], invMat[8],
    invMat[1], invMat[5], invMat[9],
    invMat[2], invMat[6], invMat[10],
  ];
  const s = mpToMh.scale, mR = mpToMh.R;

  return anchors.map((a) => {
    const i = a.mpIdx;
    const rx = userAligned[i][0] - adaPts[i][0];
    const ry = userAligned[i][1] - adaPts[i][1];
    const rz = userAligned[i][2] - adaPts[i][2];
    // Un-rotate to canonical.
    const cx = iR[0]*rx + iR[1]*ry + iR[2]*rz;
    const cy = iR[3]*rx + iR[4]*ry + iR[5]*rz;
    const cz = iR[6]*rx + iR[7]*ry + iR[8]*rz;
    return [
      a.mhRest[0] + s * (mR[0]*cx + mR[1]*cy + mR[2]*cz),
      a.mhRest[1] + s * (mR[3]*cx + mR[4]*cy + mR[5]*cz),
      a.mhRest[2] + s * (mR[6]*cx + mR[7]*cy + mR[8]*cz),
    ];
  });
}

// Multi-view triangulation of the per-anchor 3D identity delta.
//
// For frame k with user head rotation R_k (canonical -> camera space)
// and observed per-anchor 2D delta d_k = [dx, dy]:
//
//   d_k ~= (top 2 rows of R_k) * D
//
// where D is the 3D identity delta in MP canonical space, constant
// across frames. Stacking 2N rows gives an overdetermined system
// solved by normal equations:
//
//   (AᵀA) D = Aᵀ b
//
// 3x3 symmetric solve per anchor. Cheap.
function triangulateAnchorDeltas(samples, numAnchors) {
  // Tikhonov ridge on the normal equations. Scales with the number of
  // samples so a typical well-posed solve is untouched but the z
  // direction cannot blow up when the user hasn't rotated much yet.
  const ridge = 0.02 * samples.length;
  // Anchors whose magnitude exceeds this are clamped. MP canonical
  // face spans ~0.5 in normalized image units; identity deltas
  // should be a small fraction of that.
  const MAX_MAG = 0.15;
  const out = new Array(numAnchors);
  for (let i = 0; i < numAnchors; i++) {
    let A00=0, A01=0, A02=0, A11=0, A12=0, A22=0;
    let b0=0, b1=0, b2=0;
    for (const s of samples) {
      const R = s.R, d = s.deltas2D[i];
      const r00=R[0], r01=R[1], r02=R[2];
      const r10=R[3], r11=R[4], r12=R[5];
      A00 += r00*r00 + r10*r10;
      A01 += r00*r01 + r10*r11;
      A02 += r00*r02 + r10*r12;
      A11 += r01*r01 + r11*r11;
      A12 += r01*r02 + r11*r12;
      A22 += r02*r02 + r12*r12;
      b0  += r00*d[0] + r10*d[1];
      b1  += r01*d[0] + r11*d[1];
      b2  += r02*d[0] + r12*d[1];
    }
    A00 += ridge; A11 += ridge; A22 += ridge;
    const det = A00*(A11*A22 - A12*A12)
              - A01*(A01*A22 - A12*A02)
              + A02*(A01*A12 - A11*A02);
    if (Math.abs(det) < 1e-9) { out[i] = [0, 0, 0]; continue; }
    const inv00 =  (A11*A22 - A12*A12) / det;
    const inv01 = -(A01*A22 - A12*A02) / det;
    const inv02 =  (A01*A12 - A11*A02) / det;
    const inv11 =  (A00*A22 - A02*A02) / det;
    const inv12 = -(A00*A12 - A01*A02) / det;
    const inv22 =  (A00*A11 - A01*A01) / det;
    let dx = inv00*b0 + inv01*b1 + inv02*b2;
    let dy = inv01*b0 + inv11*b1 + inv12*b2;
    let dz = inv02*b0 + inv12*b1 + inv22*b2;
    const m = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (m > MAX_MAG) {
      const k = MAX_MAG / m;
      dx *= k; dy *= k; dz *= k;
    }
    out[i] = [dx, dy, dz];
  }
  return out;
}

// Convert triangulated MP-space deltas into MH-space warp targets by
// applying the MP->MH rigid+scale computed once from Ada at rest.
function buildTargetsFromDeltas3D(deltas3D, anchors, mpToMh) {
  const s = mpToMh.scale, R = mpToMh.R;
  return anchors.map((a, i) => {
    const d = deltas3D[i];
    return [
      a.mhRest[0] + s * (R[0]*d[0] + R[1]*d[1] + R[2]*d[2]),
      a.mhRest[1] + s * (R[3]*d[0] + R[4]*d[1] + R[5]*d[2]),
      a.mhRest[2] + s * (R[6]*d[0] + R[7]*d[1] + R[8]*d[2]),
    ];
  });
}

// Per-anchor average of aligned targets across N frames. Each sample
// is an array of [x,y,z] per anchor. Output is same-shape averaged.
function averageSamples(samples) {
  const n = samples[0].length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0, sz = 0;
    for (const s of samples) {
      sx += s[i][0]; sy += s[i][1]; sz += s[i][2];
    }
    const k = samples.length;
    out[i] = [sx / k, sy / k, sz / k];
  }
  return out;
}

// Compute the rigid-aligned MP target positions for one frame. Target
// i is "where the RBF should pull rest_anchor i to match this frame
// of the user." Multi-frame capture averages these across N frames
// before solving the warp.
function computeAlignedTargets(mpLandmarks, anchors) {
  // MediaPipe normalized landmarks vs MH world:
  //   MP raw +X = image right  = subject-left  = MH +X   (no flip)
  //   MP raw +Y = image down                   = MH -Y   (flip)
  //   MP raw +Z = into screen  = away from cam = MH -Z   (flip)
  const mpAnchors = anchors.map((a) => {
    const l = mpLandmarks[a.mpIdx];
    return [l.x - 0.5, -(l.y - 0.5), -l.z];
  });
  const mhAnchors = anchors.map((a) => a.mhRest);
  const rigid = procrustesAlign(mhAnchors, mpAnchors);
  return mpAnchors.map((p) => applyRigid(rigid, p, [0, 0, 0]));
}

function runWarp(mpLandmarks, anchors, warpTargets) {
  const alignedTargets = computeAlignedTargets(mpLandmarks, anchors);
  return runWarpFromTargets(alignedTargets, anchors, warpTargets);
}

function runWarpFromTargets(alignedTargets, anchors, warpTargets) {
  const mhAnchors = anchors.map((a) => a.mhRest);

  // Solve the RBF from MH rest -> aligned MP targets.
  const warp = solveWarp(mhAnchors, alignedTargets);

  // Sanity: anchor residuals after warp.
  let residual = 0;
  const tmp = [0, 0, 0];
  for (let i = 0; i < anchors.length; i++) {
    applyWarp(warp, mhAnchors[i], tmp);
    const dx = tmp[0] - alignedTargets[i][0];
    const dy = tmp[1] - alignedTargets[i][1];
    const dz = tmp[2] - alignedTargets[i][2];
    residual += Math.sqrt(dx*dx + dy*dy + dz*dz);
  }
  residual /= anchors.length;

  // Per-vertex distance-weighted blend: full warp on the face, tapers
  // to zero past `FALLOFF_OUTER` metres so the back of the head,
  // neck, and shoulders keep the rest geometry. Biharmonic RBF
  // extrapolation otherwise drifts unboundedly with distance.
  const FALLOFF_INNER = 0.02;      // inside this, weight = 1.0
  const FALLOFF_OUTER = 0.09;      // past this, weight = 0.0
  const innerSq = FALLOFF_INNER * FALLOFF_INNER;
  const outerSq = FALLOFF_OUTER * FALLOFF_OUTER;
  const anchorRest = mhAnchors; // reuse array-of-3 already built above

  let delta = 0, totalVerts = 0;
  const point = [0, 0, 0];
  const out = [0, 0, 0];
  for (const t of warpTargets) {
    const arr = t.attr.array;
    const n = arr.length / 3;
    // Hair / lashes / brows get full warp so they stay glued to the
    // scalp. Skin-category meshes use distance-weighted falloff.
    const forceFull = t.category === 'attached';
    for (let i = 0; i < n; i++) {
      const rx = t.rest[i * 3 + 0];
      const ry = t.rest[i * 3 + 1];
      const rz = t.rest[i * 3 + 2];

      let w;
      if (forceFull) {
        w = 1;
      } else {
        // Nearest-anchor squared distance.
        let minSq = Infinity;
        for (let k = 0; k < anchorRest.length; k++) {
          const ax = anchorRest[k][0], ay = anchorRest[k][1], az = anchorRest[k][2];
          const dx = rx - ax, dy = ry - ay, dz = rz - az;
          const d = dx*dx + dy*dy + dz*dz;
          if (d < minSq) minSq = d;
        }
        if (minSq <= innerSq) w = 1;
        else if (minSq >= outerSq) w = 0;
        else {
          const dist = Math.sqrt(minSq);
          const u = (dist - FALLOFF_INNER) / (FALLOFF_OUTER - FALLOFF_INNER);
          w = 1 - (3 * u * u - 2 * u * u * u); // inverted smoothstep
        }
      }

      if (w > 0) {
        point[0] = rx; point[1] = ry; point[2] = rz;
        applyWarp(warp, point, out);
        arr[i * 3 + 0] = rx + w * (out[0] - rx);
        arr[i * 3 + 1] = ry + w * (out[1] - ry);
        arr[i * 3 + 2] = rz + w * (out[2] - rz);
      } else {
        arr[i * 3 + 0] = rx;
        arr[i * 3 + 1] = ry;
        arr[i * 3 + 2] = rz;
      }
      const ddx = arr[i*3+0] - rx, ddy = arr[i*3+1] - ry, ddz = arr[i*3+2] - rz;
      delta += Math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    }
    totalVerts += n;
    t.attr.needsUpdate = true;
    t.mesh.geometry.computeBoundingSphere();
  }
  delta /= Math.max(totalVerts, 1);

  return { n: anchors.length, meanResidual: residual, meanDelta: delta };
}

// --- plumbing ---

function findSkinMesh(root, preferredName) {
  // First, try the name that appears in the anchors file. If that
  // mesh is not present (e.g. loading a different MH GLB), fall back
  // to the largest face-like mesh by vertex count.
  let byName = null;
  const candidates = [];
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    if (obj.name === preferredName) byName = obj;
    const n = (obj.name || '').toLowerCase();
    if ((n.includes('face') || n.includes('head'))) candidates.push(obj);
  });
  if (byName) return byName;
  candidates.sort((a, b) =>
    b.geometry.attributes.position.count - a.geometry.attributes.position.count);
  return candidates[0] || null;
}

function frameHead(root, camera, controls) {
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const headY = box.max.y - size.y * 0.12;
  const target = new THREE.Vector3(center.x, headY, center.z);
  camera.position.set(target.x, target.y, target.z + Math.max(size.y, size.x) * 0.6);
  controls.target.copy(target);
  controls.update();
}

// Derive the user's webcam FOV from a single MediaPipe detection.
//
// MP returns each landmark in normalized image coords and a rigid
// 4x4 transformation matrix (column-major) placing MP's canonical
// face in camera space. The face's projected width in the image is
// related to its real metric width, its distance from the camera
// (tz), and the camera's focal length. Solve for FOV.
//
// MP's canonical face model is roughly 15 units wide across the
// outer-ear landmarks (indices 234 and 454), in whatever unit the
// matrix translation uses. We measure those two landmarks' image
// distance and back out FOV.
function estimateWebcamFovDeg(landmarks, matrix) {
  const tz = Math.abs(matrix[14]);
  // Distance between MP's "ear trace" landmarks; these live on the
  // canonical face's widest cross-section and stay at consistent
  // relative position regardless of expression.
  const L = 234, R = 454;
  const dx = landmarks[L].x - landmarks[R].x;
  const dy = landmarks[L].y - landmarks[R].y;
  const faceWidthImage = Math.sqrt(dx * dx + dy * dy);
  // Empirical canonical width in matrix units. If estimates look
  // systematically high or low, tune this constant.
  const CANONICAL_WIDTH = 15;
  const fovRad = 2 * Math.atan(CANONICAL_WIDTH / (2 * tz * faceWidthImage));
  // Clamp to plausible range so a garbage detection does not wreck
  // the anchor rebuild.
  const degs = THREE.MathUtils.radToDeg(fovRad);
  return Math.min(110, Math.max(35, degs));
}

// Render Ada at a given FOV, run MP on that render, raycast each of
// the 478 landmarks back into the mesh. Returns the anchor list in
// the same shape the warp expects.
async function buildAnchorsAtFov(three, skin, landmarker, fovDeg) {
  const DETECT_SIZE = 1024;
  const cam = three.camera.clone();
  cam.fov = fovDeg;
  cam.aspect = 1;
  cam.updateProjectionMatrix();

  const rt = new THREE.WebGLRenderTarget(DETECT_SIZE, DETECT_SIZE, {
    colorSpace: THREE.SRGBColorSpace,
  });
  three.renderer.setRenderTarget(rt);
  three.renderer.render(three.scene, cam);
  three.renderer.setRenderTarget(null);
  const pixels = new Uint8Array(DETECT_SIZE * DETECT_SIZE * 4);
  three.renderer.readRenderTargetPixels(rt, 0, 0, DETECT_SIZE, DETECT_SIZE, pixels);
  rt.dispose();

  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = DETECT_SIZE;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(DETECT_SIZE, DETECT_SIZE);
  const rowBytes = DETECT_SIZE * 4;
  for (let y = 0; y < DETECT_SIZE; y++) {
    const src = (DETECT_SIZE - 1 - y) * rowBytes;
    img.data.set(pixels.subarray(src, src + rowBytes), y * rowBytes);
  }
  ctx.putImageData(img, 0, 0);

  const result = landmarker.detect(canvas);
  if (!result.faceLandmarks || !result.faceLandmarks[0]) {
    throw new Error('MP did not detect Ada at FOV ' + fovDeg.toFixed(1));
  }
  const landmarks = result.faceLandmarks[0];

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const out = [];
  const seenVerts = new Set();
  const wp = new THREE.Vector3();
  const rest = new THREE.Vector3();
  const posAttr = skin.geometry.attributes.position;

  for (let i = 0; i < landmarks.length; i++) {
    const l = landmarks[i];
    ndc.set(l.x * 2 - 1, -(l.y * 2 - 1));
    raycaster.setFromCamera(ndc, cam);
    const hits = raycaster.intersectObject(skin, false);
    if (!hits.length) continue;
    const hit = hits[0];
    const face = hit.face;
    let bestIdx = face.a, bestDist = Infinity;
    for (const idx of [face.a, face.b, face.c]) {
      if (skin.isSkinnedMesh && skin.getVertexPosition) {
        skin.getVertexPosition(idx, wp);
      } else {
        wp.fromBufferAttribute(posAttr, idx);
      }
      wp.applyMatrix4(skin.matrixWorld);
      const d = wp.distanceToSquared(hit.point);
      if (d < bestDist) { bestDist = d; bestIdx = idx; }
    }
    rest.fromBufferAttribute(posAttr, bestIdx);
    if (rest.z < 0.02) continue;
    if (seenVerts.has(bestIdx)) continue;
    seenVerts.add(bestIdx);
    out.push({
      name: 'mp_' + i,
      mhIdx: bestIdx,
      mhRest: [rest.x, rest.y, rest.z],
      mpIdx: i,
    });
  }

  // One-shot MP-canonical -> MH-rest rigid+scale from Ada's own
  // landmarks; used every frame to convert user-identity deltas
  // into MH-space offsets.
  const adaPtsRest = out.map((a) => mpToCanonical(landmarks[a.mpIdx]));
  const mhRestPts = out.map((a) => a.mhRest);
  const mpToMh = procrustesAlign(mhRestPts, adaPtsRest);

  // Keep the full 478 raw landmarks too so capture-time Procrustes
  // can rigid-align the user's full lattice to Ada's full lattice,
  // not just the subset we anchor.
  return { anchors: out, mpToMh, adaLandmarksRaw: landmarks };
}

// Convert one MP normalized-image landmark to the "canonical-ish"
// 3D frame we use everywhere else (y up, z forward toward camera).
function mpToCanonical(l) {
  return [l.x - 0.5, -(l.y - 0.5), -l.z];
}

function drawOverlay(ctx, canvas, landmarks, anchors) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // All 478 as faint dots.
  ctx.fillStyle = 'rgba(150,150,150,0.4)';
  for (let i = 0; i < landmarks.length; i++) {
    const l = landmarks[i];
    ctx.fillRect(l.x * canvas.width - 1, l.y * canvas.height - 1, 2, 2);
  }
  // Anchored ones highlighted.
  ctx.fillStyle = '#60d0ff';
  for (const a of anchors) {
    const l = landmarks[a.mpIdx];
    if (!l) continue;
    ctx.beginPath();
    ctx.arc(l.x * canvas.width, l.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}
