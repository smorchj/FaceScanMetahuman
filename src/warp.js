// 3D thin-plate-spline-style RBF warp for face identity deformation.
//
// Given N source anchors (MH rest-pose positions) and N target anchors
// (MediaPipe landmarks aligned into MH space), we solve for a smooth
// deformation function phi such that phi(source_i) = target_i for all i.
// Applying phi to every MH face vertex produces the identity-warped
// mesh that matches the user's proportions while preserving the rig.
//
// Kernel: biharmonic (K(r) = |r|) in 3D. Simple, smooth, singularity-
// free away from anchors, and behaves well for face-scale warps.
//
// System solved (regularized TPS form):
//
//   [ K  P ] [ w ]   [ Y ]
//   [ P' 0 ] [ a ] = [ 0 ]
//
//   K is N x N with K_ij = |source_i - source_j|
//   P is N x 4 with row i = [1, sx_i, sy_i, sz_i]
//   w is N x 3 of RBF weights (one set of three per anchor)
//   a is 4 x 3 of affine coefficients
//   Y is N x 3 of target positions
//
// The system is (N + 4) x (N + 4). For 41 anchors that's 45 x 45, which
// a plain in-place Gauss-Jordan solver handles in a millisecond.

const EPS = 1e-12;

export function solveWarp(sources, targets) {
  if (sources.length !== targets.length) {
    throw new Error('source/target count mismatch');
  }
  const n = sources.length;
  const dim = n + 4;

  // Build the augmented system [A | Y]. We solve for 3 right-hand
  // sides simultaneously (x, y, z of the combined [w; a] vector).
  const A = new Float64Array(dim * (dim + 3));
  const stride = dim + 3;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A[i * stride + j] = kernel(sources[i], sources[j]);
    }
    A[i * stride + n + 0] = 1;
    A[i * stride + n + 1] = sources[i][0];
    A[i * stride + n + 2] = sources[i][1];
    A[i * stride + n + 3] = sources[i][2];
  }
  for (let i = 0; i < n; i++) {
    A[(n + 0) * stride + i] = 1;
    A[(n + 1) * stride + i] = sources[i][0];
    A[(n + 2) * stride + i] = sources[i][1];
    A[(n + 3) * stride + i] = sources[i][2];
  }
  // RHS (Y, then zero block)
  for (let i = 0; i < n; i++) {
    A[i * stride + dim + 0] = targets[i][0];
    A[i * stride + dim + 1] = targets[i][1];
    A[i * stride + dim + 2] = targets[i][2];
  }

  gaussJordan(A, dim, dim + 3);

  const w = new Float64Array(n * 3);
  const a = new Float64Array(4 * 3);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      w[i * 3 + c] = A[i * stride + dim + c];
    }
  }
  for (let i = 0; i < 4; i++) {
    for (let c = 0; c < 3; c++) {
      a[i * 3 + c] = A[(n + i) * stride + dim + c];
    }
  }

  return { sources, weights: w, affine: a, n };
}

// Apply the solved warp to an arbitrary point.
export function applyWarp(warp, point, out) {
  const { sources, weights, affine, n } = warp;
  let x = affine[0 * 3 + 0] + affine[1 * 3 + 0] * point[0]
        + affine[2 * 3 + 0] * point[1] + affine[3 * 3 + 0] * point[2];
  let y = affine[0 * 3 + 1] + affine[1 * 3 + 1] * point[0]
        + affine[2 * 3 + 1] * point[1] + affine[3 * 3 + 1] * point[2];
  let z = affine[0 * 3 + 2] + affine[1 * 3 + 2] * point[0]
        + affine[2 * 3 + 2] * point[1] + affine[3 * 3 + 2] * point[2];
  for (let i = 0; i < n; i++) {
    const k = kernel(sources[i], point);
    x += weights[i * 3 + 0] * k;
    y += weights[i * 3 + 1] * k;
    z += weights[i * 3 + 2] * k;
  }
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}

// Rigid-plus-uniform-scale Procrustes. Aligns target anchors to source
// anchors before the RBF solves the residual non-rigid deformation.
// Returns a 4x4 column-major matrix that maps target -> source space.
// The warp solve then happens entirely in source (MH) space, so
// downstream consumers only see deltas relative to the rest pose.
export function procrustesAlign(sourcePts, targetPts) {
  const n = sourcePts.length;
  const sC = centroid(sourcePts);
  const tC = centroid(targetPts);
  const sCentered = sourcePts.map((p) => sub(p, sC));
  const tCentered = targetPts.map((p) => sub(p, tC));

  // Cross-covariance H = sum(tCentered_i * sCentered_iT)
  const H = [0,0,0, 0,0,0, 0,0,0];
  for (let i = 0; i < n; i++) {
    const s = sCentered[i], t = tCentered[i];
    H[0] += t[0]*s[0]; H[1] += t[0]*s[1]; H[2] += t[0]*s[2];
    H[3] += t[1]*s[0]; H[4] += t[1]*s[1]; H[5] += t[1]*s[2];
    H[6] += t[2]*s[0]; H[7] += t[2]*s[1]; H[8] += t[2]*s[2];
  }
  const { U, S, Vt } = svd3x3(H);
  // R = V * diag(1,1,det(V*Uᵀ)) * Uᵀ   (reflection guard)
  const VUt = mul3(transpose3(Vt), transpose3(U));
  let d = det3(VUt);
  const R = d >= 0
    ? mul3(transpose3(Vt), transpose3(U))
    : mul3(transpose3(Vt), mul3([1,0,0, 0,1,0, 0,0,-1], transpose3(U)));
  // Uniform scale (Umeyama 1991): the thing being scaled is the TARGET
  // argument (applyRigid maps target -> source), so the denominator is
  // the variance of the target points, not the source.
  let varT = 0;
  for (let i = 0; i < n; i++) {
    varT += tCentered[i][0]*tCentered[i][0]
          + tCentered[i][1]*tCentered[i][1]
          + tCentered[i][2]*tCentered[i][2];
  }
  const scale = varT > EPS ? (S[0] + S[1] + S[2]) / varT : 1;

  const translate = [
    sC[0] - scale * (R[0]*tC[0] + R[1]*tC[1] + R[2]*tC[2]),
    sC[1] - scale * (R[3]*tC[0] + R[4]*tC[1] + R[5]*tC[2]),
    sC[2] - scale * (R[6]*tC[0] + R[7]*tC[1] + R[8]*tC[2]),
  ];
  return { R, scale, translate };
}

export function applyRigid(rigid, p, out) {
  const { R, scale, translate } = rigid;
  out[0] = scale * (R[0]*p[0] + R[1]*p[1] + R[2]*p[2]) + translate[0];
  out[1] = scale * (R[3]*p[0] + R[4]*p[1] + R[5]*p[2]) + translate[1];
  out[2] = scale * (R[6]*p[0] + R[7]*p[1] + R[8]*p[2]) + translate[2];
  return out;
}

// -------- internals --------

function kernel(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

function centroid(pts) {
  let x = 0, y = 0, z = 0;
  for (const p of pts) { x += p[0]; y += p[1]; z += p[2]; }
  const n = pts.length;
  return [x / n, y / n, z / n];
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }

function transpose3(M) {
  return [M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]];
}
function mul3(A, B) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i*3 + j] = A[i*3+0]*B[0*3+j] + A[i*3+1]*B[1*3+j] + A[i*3+2]*B[2*3+j];
    }
  }
  return r;
}
function det3(M) {
  return M[0]*(M[4]*M[8]-M[5]*M[7])
       - M[1]*(M[3]*M[8]-M[5]*M[6])
       + M[2]*(M[3]*M[7]-M[4]*M[6]);
}

// Jacobi-based SVD for a 3x3 matrix. Slow but bulletproof for this
// size; Procrustes runs once per warp solve, so cost is irrelevant.
function svd3x3(A) {
  // Compute V from eigendecomposition of AᵀA, then U = A V / S.
  const AtA = mul3(transpose3(A), A);
  const { vectors, values } = jacobiSym3(AtA);
  // Sort descending by eigenvalue
  const order = [0, 1, 2].sort((i, j) => values[j] - values[i]);
  const S = order.map((i) => Math.sqrt(Math.max(values[i], 0)));
  const V = new Array(9);
  for (let col = 0; col < 3; col++) {
    const src = order[col];
    V[0*3 + col] = vectors[0*3 + src];
    V[1*3 + col] = vectors[1*3 + src];
    V[2*3 + col] = vectors[2*3 + src];
  }
  // U columns = (A * V col) / S
  const U = new Array(9);
  for (let col = 0; col < 3; col++) {
    const s = S[col] > EPS ? S[col] : 1;
    U[0*3 + col] = (A[0]*V[0*3+col] + A[1]*V[1*3+col] + A[2]*V[2*3+col]) / s;
    U[3 + col] = (A[3]*V[0*3+col] + A[4]*V[1*3+col] + A[5]*V[2*3+col]) / s;
    U[6 + col] = (A[6]*V[0*3+col] + A[7]*V[1*3+col] + A[8]*V[2*3+col]) / s;
  }
  return { U, S, Vt: transpose3(V) };
}

function jacobiSym3(M) {
  const a = M.slice();
  const v = [1,0,0, 0,1,0, 0,0,1];
  for (let iter = 0; iter < 50; iter++) {
    // Find off-diagonal element with largest absolute value
    let p = 0, q = 1, max = Math.abs(a[1]);
    if (Math.abs(a[2]) > max) { p = 0; q = 2; max = Math.abs(a[2]); }
    if (Math.abs(a[5]) > max) { p = 1; q = 2; max = Math.abs(a[5]); }
    if (max < 1e-12) break;
    const app = a[p*3 + p], aqq = a[q*3 + q], apq = a[p*3 + q];
    const theta = (aqq - app) / (2 * apq);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta*theta));
    const c = 1 / Math.sqrt(1 + t*t);
    const s = t * c;
    for (let r = 0; r < 3; r++) {
      const arp = a[r*3 + p], arq = a[r*3 + q];
      a[r*3 + p] = c*arp - s*arq;
      a[r*3 + q] = s*arp + c*arq;
    }
    for (let r = 0; r < 3; r++) {
      const apr = a[p*3 + r], aqr = a[q*3 + r];
      a[p*3 + r] = c*apr - s*aqr;
      a[q*3 + r] = s*apr + c*aqr;
    }
    for (let r = 0; r < 3; r++) {
      const vrp = v[r*3 + p], vrq = v[r*3 + q];
      v[r*3 + p] = c*vrp - s*vrq;
      v[r*3 + q] = s*vrp + c*vrq;
    }
  }
  return { vectors: v, values: [a[0], a[4], a[8]] };
}

// In-place Gauss-Jordan elimination with partial pivoting.
// Augmented matrix has `rows` rows and `cols` total columns (including rhs).
function gaussJordan(M, rows, cols) {
  for (let col = 0; col < rows; col++) {
    // Pivot: largest-magnitude row at or below col
    let pivotRow = col, pivotMag = Math.abs(M[col * cols + col]);
    for (let r = col + 1; r < rows; r++) {
      const v = Math.abs(M[r * cols + col]);
      if (v > pivotMag) { pivotMag = v; pivotRow = r; }
    }
    if (pivotMag < EPS) {
      throw new Error('singular RBF system (coincident anchors?)');
    }
    if (pivotRow !== col) {
      for (let c = 0; c < cols; c++) {
        const tmp = M[col * cols + c];
        M[col * cols + c] = M[pivotRow * cols + c];
        M[pivotRow * cols + c] = tmp;
      }
    }
    // Normalize pivot row
    const piv = M[col * cols + col];
    for (let c = 0; c < cols; c++) M[col * cols + c] /= piv;
    // Eliminate other rows
    for (let r = 0; r < rows; r++) {
      if (r === col) continue;
      const factor = M[r * cols + col];
      if (factor === 0) continue;
      for (let c = 0; c < cols; c++) {
        M[r * cols + c] -= factor * M[col * cols + c];
      }
    }
  }
}
