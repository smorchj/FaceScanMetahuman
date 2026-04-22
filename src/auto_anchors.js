// Auto-generate a dense MP-to-MH landmark correspondence.
//
// Instead of asking a human to click each landmark on the MH, we ask
// MediaPipe where each landmark SHOULD be on the MH's own rendered
// face. Then we raycast each MP landmark back into the 3D scene to
// find the matching MH skin vertex. One pass gives us ~400 self-
// consistent correspondences vs the 41 we could hand-pick.

import * as THREE from 'three';
import { mount } from './viewer.js';

const MP_BUNDLE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs';
const MP_WASM   = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm';
const MP_MODEL  = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// Render at this resolution for the MP detection pass. Higher gives
// more precise landmarks at the cost of a slightly slower one-shot.
const DETECT_SIZE = 1024;

export async function start(opts) {
  const { stage, thumb, runBtn, downloadBtn, statusEl,
          glbUrl, mappingUrl, characterId } = opts;

  const setStatus = (s) => { statusEl.textContent = s; };

  setStatus('loading MH GLB and materials...');
  const three = await mount(stage, {
    glbUrl,
    mappingUrl,
    autoRotate: false,
    interactive: true,
    characterId,
  });
  const skin = findSkinMesh(three.gltf.scene);
  if (!skin) throw new Error('no face/head mesh found in GLB');
  frameHead(three.gltf.scene, three.camera, three.controls);

  setStatus('loading MediaPipe face landmarker...');
  const vision = await import(MP_BUNDLE);
  const { FaceLandmarker, FilesetResolver } = vision;
  const resolver = await FilesetResolver.forVisionTasks(MP_WASM);
  const landmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
    runningMode: 'IMAGE',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  runBtn.disabled = false;
  setStatus('ready. orbit Ada until her face looks forward, then click Detect.');

  let anchorsPayload = null;

  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true;
    setStatus('rendering Ada at detection resolution...');

    // Render with the SAME renderer that mounted the scene so we keep
    // all the compiled MH materials + PMREM environment. Draw into an
    // offscreen render target, read pixels back, blit into a 2D canvas
    // (flipped Y to match canvas coords), hand that to MediaPipe.
    const detectCamera = three.camera.clone();
    detectCamera.aspect = 1;
    detectCamera.updateProjectionMatrix();
    const rt = new THREE.WebGLRenderTarget(DETECT_SIZE, DETECT_SIZE, {
      colorSpace: THREE.SRGBColorSpace,
    });
    three.renderer.setRenderTarget(rt);
    three.renderer.render(three.scene, detectCamera);
    three.renderer.setRenderTarget(null);
    const pixels = new Uint8Array(DETECT_SIZE * DETECT_SIZE * 4);
    three.renderer.readRenderTargetPixels(rt, 0, 0, DETECT_SIZE, DETECT_SIZE, pixels);
    rt.dispose();

    const offCanvas = document.createElement('canvas');
    offCanvas.width = offCanvas.height = DETECT_SIZE;
    const offCtx = offCanvas.getContext('2d');
    const img = offCtx.createImageData(DETECT_SIZE, DETECT_SIZE);
    // WebGL origin is bottom-left, canvas is top-left. Flip rows.
    const rowBytes = DETECT_SIZE * 4;
    for (let y = 0; y < DETECT_SIZE; y++) {
      const src = (DETECT_SIZE - 1 - y) * rowBytes;
      const dst = y * rowBytes;
      img.data.set(pixels.subarray(src, src + rowBytes), dst);
    }
    offCtx.putImageData(img, 0, 0);

    // Show the thumbnail first so the operator can see what MP was
    // actually given, regardless of whether detection succeeds.
    drawThumb(thumb, offCanvas, null);
    // Sanity: are the pixels non-black?
    const tctx = thumb.getContext('2d');
    const sample = tctx.getImageData(thumb.width / 2, thumb.height / 2, 1, 1).data;
    const midPixel = '(' + sample[0] + ',' + sample[1] + ',' + sample[2] + ')';

    setStatus('running MediaPipe on Ada... (centre pixel rgb ' + midPixel + ')');
    const result = landmarker.detect(offCanvas);
    if (!result.faceLandmarks || !result.faceLandmarks[0]) {
      setStatus('MediaPipe did not detect a face.\n'
                + 'Centre pixel rgb: ' + midPixel + '\n'
                + 'Thumbnail shows exactly what MP was given. '
                + 'If it looks blank / monochrome, the render is the problem. '
                + 'If it looks like Ada, MP is rejecting her.');
      runBtn.disabled = false;
      return;
    }
    const landmarks = result.faceLandmarks[0];

    // Redraw with landmark overlay on top.
    drawThumb(thumb, offCanvas, landmarks);

    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const anchors = [];
    let hits = 0, misses = 0;
    for (let i = 0; i < landmarks.length; i++) {
      const l = landmarks[i];
      ndc.set(l.x * 2 - 1, -(l.y * 2 - 1));
      raycaster.setFromCamera(ndc, detectCamera);
      const intersects = raycaster.intersectObject(skin, false);
      if (!intersects.length) { misses++; continue; }
      const hit = intersects[0];
      const pick = vertexFromHit(skin, hit);
      anchors.push({
        name: 'mp_' + String(i).padStart(3, '0'),
        mpIndex: i,
        meshName: skin.name,
        vertexIndex: pick.vertexIndex,
        restPosition: pick.restPosition,
      });
      hits++;
    }

    anchorsPayload = {
      version: 2,
      characterId,
      source: 'auto-anchors (MediaPipe on MH render)',
      skinMesh: skin.name,
      anchors,
    };

    downloadBtn.disabled = false;
    setStatus(
      'mapped ' + hits + ' / ' + landmarks.length + ' MP landmarks to MH verts'
      + '\n' + misses + ' landmarks missed the face (iris / edge / hidden).'
      + '\nClick download to save anchors_auto.json.'
    );
    runBtn.disabled = false;
  });

  downloadBtn.addEventListener('click', () => {
    if (!anchorsPayload) return;
    const blob = new Blob([JSON.stringify(anchorsPayload, null, 2)],
                          { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'anchors_auto.json';
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

function vertexFromHit(skin, hit) {
  const geom = skin.geometry;
  const pos = geom.attributes.position;
  const face = hit.face;
  // Find the triangle vertex whose rendered world position is closest
  // to the hit point. Using skinnedWorldPos avoids the bind-pose vs
  // world-pose scale trap that bit the manual picker.
  const candidates = [face.a, face.b, face.c];
  let bestIdx = candidates[0];
  let bestDist = Infinity;
  const wp = new THREE.Vector3();
  for (const idx of candidates) {
    skinnedWorldPos(skin, idx, wp);
    const d = wp.distanceToSquared(hit.point);
    if (d < bestDist) { bestDist = d; bestIdx = idx; }
  }
  const rest = new THREE.Vector3().fromBufferAttribute(pos, bestIdx);
  return {
    vertexIndex: bestIdx,
    restPosition: [rest.x, rest.y, rest.z],
  };
}

function skinnedWorldPos(mesh, idx, out) {
  if (mesh.isSkinnedMesh && typeof mesh.getVertexPosition === 'function') {
    mesh.getVertexPosition(idx, out);
  } else {
    out.fromBufferAttribute(mesh.geometry.attributes.position, idx);
  }
  return out.applyMatrix4(mesh.matrixWorld);
}

function findSkinMesh(root) {
  const candidates = [];
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    const n = (obj.name || '').toLowerCase();
    if (n.includes('face') || n.includes('head')) candidates.push(obj);
  });
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

function drawThumb(thumb, source, landmarks) {
  thumb.width = source.width;
  thumb.height = source.height;
  const ctx = thumb.getContext('2d');
  ctx.drawImage(source, 0, 0);
  if (!landmarks) return;
  ctx.fillStyle = 'rgba(0,255,180,0.85)';
  for (const l of landmarks) {
    ctx.fillRect(l.x * thumb.width - 1.5, l.y * thumb.height - 1.5, 3, 3);
  }
}
