// End-to-end one-shot identity warp demo.
//
// Loads a MetaHuman GLB and its anchor JSON, runs MediaPipe on the
// webcam, and on button click fits the MH neutral to the user's face
// proportions via rigid Procrustes + biharmonic RBF. No texture, no
// live update. The goal is to see "that looks like me-ish" and move on.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import { MP_INDICES } from './mp_indices.js';
import { solveWarp, applyWarp, procrustesAlign, applyRigid } from './warp.js';

const DRACO_DECODER = 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/';
const MP_BUNDLE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs';
const MP_WASM   = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm';
const MP_MODEL  = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export async function start(opts) {
  const { stage, video, overlay, captureBtn, resetBtn, statusEl,
          glbUrl, anchorsUrl } = opts;

  const setStatus = (s) => { statusEl.textContent = s; };

  setStatus('fetching GLB and anchors...');
  const [gltf, anchorsData] = await Promise.all([
    loadGlb(glbUrl),
    fetch(anchorsUrl).then((r) => {
      if (!r.ok) throw new Error('anchors: ' + r.status + ' ' + anchorsUrl);
      return r.json();
    }),
  ]);

  const three = buildScene(stage);
  three.scene.add(gltf.scene);

  const skin = findSkinMesh(gltf.scene, anchorsData.anchors[0].meshName);
  if (!skin) throw new Error('could not find skin mesh ' + anchorsData.anchors[0].meshName);

  // Snapshot the rest pose so every warp starts from identity.
  const posAttr = skin.geometry.attributes.position;
  const restPositions = new Float32Array(posAttr.array);

  // Pair up MH rest anchors with MediaPipe indices.
  const anchors = anchorsData.anchors.map((a) => {
    const mpIdx = MP_INDICES[a.name];
    if (mpIdx === undefined) {
      console.warn('[demo] no MP index for', a.name, '-- will skip');
    }
    return {
      name: a.name,
      mhIdx: a.vertexIndex,
      mhRest: a.restPosition.slice(),
      mpIdx,
    };
  }).filter((a) => a.mpIdx !== undefined);

  frameHead(gltf.scene, three.camera, three.controls);

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
      const warpStats = runWarp(latest, anchors, skin, restPositions);
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
    posAttr.array.set(restPositions);
    posAttr.needsUpdate = true;
    skin.geometry.computeBoundingSphere();
    setStatus('reset to rest pose');
  });
}

// --- core warp step ---

function runWarp(mpLandmarks, anchors, skin, restPositions) {
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

  // Apply to every vertex of the skin mesh.
  const posAttr = skin.geometry.attributes.position;
  const arr = posAttr.array;
  let delta = 0;
  const point = [0, 0, 0];
  const out = [0, 0, 0];
  const n = arr.length / 3;
  for (let i = 0; i < n; i++) {
    point[0] = restPositions[i * 3 + 0];
    point[1] = restPositions[i * 3 + 1];
    point[2] = restPositions[i * 3 + 2];
    applyWarp(warp, point, out);
    arr[i * 3 + 0] = out[0];
    arr[i * 3 + 1] = out[1];
    arr[i * 3 + 2] = out[2];
    const ddx = out[0] - point[0], ddy = out[1] - point[1], ddz = out[2] - point[2];
    delta += Math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
  }
  delta /= n;

  posAttr.needsUpdate = true;
  skin.geometry.computeBoundingSphere();

  return { n: anchors.length, meanResidual: residual, meanDelta: delta };
}

// --- plumbing ---

function loadGlb(url) {
  const draco = new DRACOLoader().setDecoderPath(DRACO_DECODER);
  const loader = new GLTFLoader().setDRACOLoader(draco);
  return new Promise((res, rej) => loader.load(url, res, undefined, rej));
}

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

function buildScene(stage) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d11);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  const camera = new THREE.PerspectiveCamera(
    32, stage.clientWidth / stage.clientHeight, 0.01, 20);
  camera.position.set(0, 1.6, 0.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();

  window.addEventListener('resize', () => {
    renderer.setSize(stage.clientWidth, stage.clientHeight);
    camera.aspect = stage.clientWidth / stage.clientHeight;
    camera.updateProjectionMatrix();
  });

  return { renderer, scene, camera, controls };
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
