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

  setStatus('fetching GLB, materials, and anchors...');
  const [three, anchorsData] = await Promise.all([
    mount(stage, {
      glbUrl,
      mappingUrl: opts.mappingUrl,
      autoRotate: false,
      interactive: true,
      characterId: opts.characterId,
    }),
    fetch(anchorsUrl).then((r) => {
      if (!r.ok) throw new Error('anchors: ' + r.status + ' ' + anchorsUrl);
      return r.json();
    }),
  ]);

  const skin = findSkinMesh(three.gltf.scene, anchorsData.anchors[0].meshName);
  if (!skin) throw new Error('could not find skin mesh ' + anchorsData.anchors[0].meshName);

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

  // Pair up MH rest anchors with MediaPipe indices. Auto-generated
  // anchor files carry `mpIndex` directly; the legacy hand-picked file
  // relies on the name -> index lookup in mp_indices.js.
  // Dedup on MH vertex index: auto-map sometimes lands two MP
  // landmarks on the same MH vert (e.g. adjacent inner-lip indices),
  // which makes the RBF kernel singular. First pick wins.
  const seen = new Set();
  const anchors = [];
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
  console.log('[demo] anchors after dedup:', anchors.length,
              'of', anchorsData.anchors.length);

  frameHead(three.gltf.scene, three.camera, three.controls);

  setStatus('loading MediaPipe face landmarker...');
  const vision = await import(MP_BUNDLE);
  const { FaceLandmarker, FilesetResolver } = vision;
  const resolver = await FilesetResolver.forVisionTasks(MP_WASM);
  const landmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
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

  let latest = null;
  (function tick() {
    if (video.readyState >= 2) {
      const res = landmarker.detectForVideo(video, performance.now());
      if (res.faceLandmarks && res.faceLandmarks[0]) {
        latest = res.faceLandmarks[0];
        drawOverlay(overlayCtx, overlay, latest, anchors);
      } else {
        latest = null;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
    requestAnimationFrame(tick);
  })();

  captureBtn.disabled = false;
  setStatus('ready. look at the camera and click capture.');

  captureBtn.addEventListener('click', () => {
    if (!latest) { setStatus('no face detected right now'); return; }
    try {
      const warpStats = runWarp(latest, anchors, warpTargets);
      setStatus(
        'warp applied.\n' +
        'anchors: ' + warpStats.n + '\n' +
        'mean anchor residual: ' + warpStats.meanResidual.toFixed(5) + ' m\n' +
        'mean vertex delta:    ' + warpStats.meanDelta.toFixed(5) + ' m'
      );
    } catch (err) {
      console.error(err);
      setStatus('warp failed: ' + (err.message || err));
    }
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
