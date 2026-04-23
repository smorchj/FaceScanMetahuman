# FaceScanMetahuman

Turn a face scan into a MetaHuman in the browser. No pixel streaming,
no cloud GPU, no native app install. Classical geometry and texture
projection, wired to MediaPipe face tracking.

## Live demo

**https://smorchj.github.io/FaceScanMetahuman/demo.html**

Grant webcam access, click Start capture, slowly turn your head. Ada
will warp toward your proportions and the webcam projects onto her
face. Stop to freeze the current snapshot. Works on desktop and
mobile (iOS Safari and Android Chrome both support the required
getUserMedia + WebGL over the Pages HTTPS).

Other pages in the deployment:

- **[/](https://smorchj.github.io/FaceScanMetahuman/)** — anchor
  picker for building a per-character MH↔MP correspondence by hand.
- **[/auto-anchors.html](https://smorchj.github.io/FaceScanMetahuman/auto-anchors.html)**
  — automated variant that renders the MH, runs MediaPipe on the
  render, and raycasts every landmark back into the mesh.

## Where this is going

A single web page where a visitor points their phone or webcam at
their face, turns their head through a short guided capture, and
watches a neutral MetaHuman warp into a likeness of them in real time.
The output is a plain GLB that loads in three.js, Babylon, PlayCanvas,
Unity WebGL, or anything that speaks glTF.

Building blocks, in rough order of how complete they are:

1. **Anchor picker** — click anatomical landmarks on a MetaHuman face
   mesh and export a vertex-index map.
2. **Auto anchor map** — render the MH, run MediaPipe on that render,
   raycast each of the 478 landmarks back into the skin mesh. Produces
   a dense `anchors_auto.json`.
3. **Identity warp** — pose-matched lattice warp. User's 478 MP
   landmarks are rigid-aligned to Ada's pose-matched lattice, the
   residual becomes the identity delta, unrotated to canonical and
   scaled into MH metres by a one-shot MP→MH Procrustes. Applied to
   every head-adjacent mesh with distance-weighted falloff so back of
   head, neck, and ears keep their rest geometry.
4. **Texture projection** — live webcam projected onto Ada's face via
   per-vertex UV rewrite. Each face vertex samples the webcam pixel at
   its corresponding MP landmark position via precomputed triangle
   containment. On Stop the current frame is frozen into a
   CanvasTexture so Ada keeps the snapshot while you orbit around.
5. **Export** — freeze the deformed mesh and baked texture, emit a
   GLB. Not done yet.

## Run locally

You need a MetaHuman GLB (the companion
[metahuman-to-glb](https://github.com/smorchj/metahuman-to-glb)
pipeline produces one). Drop it at `samples/ada/ada.glb` together
with its `mh_materials.json` and `textures/` folder (or pass paths
via URL params).

```bash
python -m http.server 8000
# then open http://localhost:8000/demo.html
```

URL params on `demo.html`:

- `?glb=path/to/character.glb` (default `samples/ada/ada.glb`)
- `?materials=path/to/mh_materials.json` (default `samples/ada/mh_materials.json`)
- `?anchors=path/to/anchors.json` (optional seed file; first Start
  rebuilds a dense map anyway)
- `?id=character_name` (default `ada`)

## Picker output schema

```json
{
  "version": 1,
  "characterId": "ada",
  "anchors": [
    {
      "name": "nose_tip",
      "meshName": "Ada_FaceMesh_LOD0_1",
      "vertexIndex": 16268,
      "restPosition": [-0.0001, 1.4607, 0.1188]
    }
  ]
}
```

`restPosition` is the geometry-local position of the vertex in the
mesh's bind pose. It's a convenience for consumers; the load-of-truth
is `vertexIndex` into the named mesh's position attribute.

## Status

Mesh warp + live texture projection work end-to-end in the browser.
Quality is rough (RGB-only depth, no delighting, no temporal accumulation
of texture yet), but the pipeline is correct and the architecture
holds. Snapshot + export steps are next.

## License

MIT.
