// MediaPipe FaceLandmarker landmark indices for each anatomical name
// we already pick on the MH side. Paired with the MH vertex indices in
// face_anchors.json, these form the MediaPipe-to-MetaHuman landmark
// correspondence that drives the RBF identity warp.
//
// Handedness note: MediaPipe uses subject-relative "left" and "right"
// (subject's left is viewer's right when facing them head-on), which
// matches the convention we use in the anchor picker. No flip needed.
//
// Confidence ratings on each line:
//   H = canonical, high confidence from published references
//   M = commonly cited but with some variance across sources
//   L = best-effort spatial guess; eyeball with the debug overlay and
//       correct if the landmark lands on the wrong feature.
//
// If any of these turn out to be wrong during visual verification, edit
// in place. The rest of the pipeline depends only on the map existing,
// not on any specific index.

export const MP_INDICES = {
  // Midline, top to bottom
  forehead_center:       10,    // H
  nasion:                168,   // H  (between brows, top of bridge)
  nose_bridge_top:       6,     // H  (between eyes)
  nose_tip:              1,     // H
  subnasale:             2,     // H  (base of nose / top of philtrum)
  upper_lip_center:      0,     // H  (outer vermilion top midline)
  lower_lip_center:      17,    // H  (outer vermilion bottom midline)
  chin_upper:            200,   // M  (mental protuberance)
  chin_tip:              152,   // M

  // Left side (subject's left)
  left_temple:           356,   // L
  left_brow_outer:       300,   // M
  left_brow_middle:      334,   // M
  left_brow_inner:       336,   // H
  left_eye_outer_corner: 263,   // H
  left_eye_upper_lid:    386,   // H
  left_eye_inner_corner: 362,   // H
  left_eye_lower_lid:    374,   // H
  left_cheekbone:        347,   // L
  left_cheek_center:     411,   // L
  left_nostril:          326,   // M
  left_lip_corner:       291,   // H
  upper_lip_cupid_left:  310,   // M
  left_jaw_mid:          378,   // L
  left_jaw_corner:       361,   // M
  left_ear_base:         454,   // L  (MP mesh barely reaches ear; approximate)

  // Right side (subject's right; mirror of left_*)
  right_temple:          127,   // L
  right_brow_outer:      70,    // M
  right_brow_middle:     105,   // M
  right_brow_inner:      107,   // H
  right_eye_outer_corner: 33,   // H
  right_eye_upper_lid:   159,   // H
  right_eye_inner_corner: 133,  // H
  right_eye_lower_lid:   145,   // H
  right_cheekbone:       118,   // L
  right_cheek_center:    187,   // L
  right_nostril:         97,    // M
  right_lip_corner:      61,    // H
  upper_lip_cupid_right: 80,    // M
  right_jaw_mid:         149,   // L
  right_jaw_corner:      132,   // M
  right_ear_base:        234,   // L
};

// Number of total anchor correspondences (25 picked + 16 mirrored).
export const ANCHOR_COUNT = Object.keys(MP_INDICES).length;
