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

  // Per-FOV cache of auto-generated anchors is declared above the
  // tick loop so the overlay can use whichever set is active.
  captureBtn.addEventListener('click', async () => {
    if (!latest) { setStatus('no face detected right now'); return; }
    if (!latestMatrix) { setStatus('no facial matrix yet; wait a beat and retry'); return; }
    captureBtn.disabled = true;
    try {
      const fovDeg = estimateWebcamFovDeg(latest, latestMatrix);
      setStatus('estimated webcam FOV: ' + fovDeg.toFixed(1) + '°');
      if (!cachedAnchors || Math.abs(fovDeg - cachedFovDeg) > 3) {
        setStatus('FOV changed to ' + fovDeg.toFixed(1) + '°; rebuilding anchors on Ada...');
        cachedAnchors = await buildAnchorsAtFov(three, skin, imgLandmarker, fovDeg);
        cachedFovDeg = fovDeg;
        setStatus('anchors built at FOV ' + fovDeg.toFixed(1) + '° ('
                  + cachedAnchors.length + ' mapped). Warping...');
      }
      const warpStats = runWarp(latest, cachedAnchors, warpTargets);
      setStatus(
        'warp applied.\n' +
        'FOV: ' + cachedFovDeg.toFixed(1) + '°\n' +
        'anchors: ' + warpStats.n + '\n' +
        'mean anchor residual: ' + warpStats.meanResidual.toFixed(5) + ' m\n' +
        'mean vertex delta:    ' + warpStats.meanDelta.toFixed(5) + ' m'
      );
    } catch (err) {
      console.error(err);
      setStatus('warp failed: ' + (err.message || err));
    }
    captureBtn.disabled = false;
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
const HEAD_MESH_HINTS = ['face', 'head', 'hair', 'lash', 'brow', 'teeth', 'tongue', 'saliva', 'occlusion'];

function collectHeadMeshes(root) {
  const out = [];
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    const n = (obj.name || '').toLowerCase();
    if (HEAD_MESH_HINTS.some((h) => n.includes(h))) out.push({ mesh: obj });
  });
  return out;
}

// --- core warp step ---

function runWarp(mpLandmarks, anchors, warpTargets) {
  // MediaPipe normalized landmarks vs MH world:
  //   MP raw +X = image right  = subject-left  = MH +X   (no flip)
  //   MP raw +Y = image down                   = MH -Y   (flip)
  //   MP raw +Z = into screen  = away from cam = MH -Z   (flip)
  // The video element's CSS mirror does not change MP's input pixels.
  const mpAnchors = anchors.map((a) => {
    const l = mpLandmarks[a.mpIdx];
    return [l.x - 0.5, -(l.y - 0.5), -l.z];
  });
  const mhAnchors = anchors.map((a) => a.mhRest);

  // Align MP into MH space (rigid + uniform scale).
  const rigid = procrustesAlign(mhAnchors, mpAnchors);
  const alignedTargets = mpAnchors.map((p) => applyRigid(rigid, p, [0, 0, 0]));

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

  // Apply to every vertex of every head-adjacent mesh.
  let delta = 0, totalVerts = 0;
  const point = [0, 0, 0];
  const out = [0, 0, 0];
  for (const t of warpTargets) {
    const arr = t.attr.array;
    const n = arr.length / 3;
    for (let i = 0; i < n; i++) {
      point[0] = t.rest[i * 3 + 0];
      point[1] = t.rest[i * 3 + 1];
      point[2] = t.rest[i * 3 + 2];
      applyWarp(warp, point, out);
      arr[i * 3 + 0] = out[0];
      arr[i * 3 + 1] = out[1];
      arr[i * 3 + 2] = out[2];
      const ddx = out[0] - point[0], ddy = out[1] - point[1], ddz = out[2] - point[2];
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
    if (seenVerts.has(bestIdx)) continue;
    seenVerts.add(bestIdx);
    rest.fromBufferAttribute(posAttr, bestIdx);
    out.push({
      name: 'mp_' + i,
      mhIdx: bestIdx,
      mhRest: [rest.x, rest.y, rest.z],
      mpIdx: i,
    });
  }
  return out;
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
