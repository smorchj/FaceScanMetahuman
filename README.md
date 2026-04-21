# FaceScanMetahuman

Turn a face scan into a MetaHuman in the browser. No pixel streaming,
no cloud GPU, no native app install. Classical geometry and texture
projection, wired to MediaPipe face tracking.

## Where this is going

A single web page where a visitor points their phone or webcam at
their face, turns their head through a short guided capture, and
watches a neutral MetaHuman warp into a likeness of them in real time.
The output is a plain GLB that loads in three.js, Babylon, PlayCanvas,
Unity WebGL, or anything that speaks glTF.

Five building blocks, smallest to largest:

1. **Anchor picker** (this repo, today). A tool that lets an operator
   click anatomical landmarks on a MetaHuman face mesh and exports the
   vertex-index map. This is the MH-side half of the
   MediaPipe-to-MetaHuman correspondence.
2. **MediaPipe anchor map.** Same shape, different topology. For each
   anatomical landmark name, the corresponding MediaPipe FaceLandmarker
   index.
3. **One-shot identity warp.** RBF deformation of the MH neutral mesh
   from MediaPipe landmarks solved across a few captured angles. Apply
   the same delta to every blendshape so the rig still fires.
4. **Live texture bake.** Project webcam frames onto the MH face UV
   through the solved head pose. Accumulate across frames, weight by
   surface visibility.
5. **Export.** Freeze the deformed mesh and baked texture, emit a GLB.

## Run the anchor picker

You need a MetaHuman GLB (the companion
[metahuman-to-glb](https://github.com/smorchj/metahuman-to-glb)
pipeline produces one). Put it anywhere the static server can serve.

```bash
# Anywhere that serves static files works. Python is easiest:
python -m http.server 8000
```

Open `http://localhost:8000/?glb=path/to/metahuman.glb`. The picker
cycles through 25 anatomical landmarks (midline plus left-side);
the right-side twin is auto-mirrored across the face's local X=0 plane.

Output is `face_anchors.json` with 41 named entries (25 picked plus
16 mirrored). All entries reference a single skin mesh: the tool
constrains picks to the largest face sub-mesh it finds (the MH skin
in every version I've tested).

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
mesh's bind pose. It is a convenience for consumers; the load of
truth is `vertexIndex` into the named mesh's position attribute.

## Status

Stage 1 works. The picker has been validated against Ada from the
MetaHuman sample set. Stage 2 is next.

## License

MIT.
