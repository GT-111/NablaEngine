"""MuJoCo Warp's general convex collision for box-box pairs (native CCD: GJK, EPA and multicontact by polygon
clipping), translated to CUDA for one thread per pair (mujoco_warp/_src/collision_gjk.py and collision_convex.py,
Copyright 2025 The Newton Developers, Apache License 2.0; their order of operations, so that the contacts agree with
MuJoCo Warp's up to float rounding). Used by the G1 engine for models with box-box pairs (g1_engine.model_hooks).

header(m): the iteration counts and tolerance (MuJoCo Warp: EPA 16 iterations when every convex pair is box-box,
the GJK ccd_iterations; MULTICCD always for box-box). DEVICE: the device code (needs make_frame from the engine).
"""


def header(m):
    """#defines of the box-box CCD (the model's ccd_iterations and ccd_tolerance)."""
    return (f"#define GJK_ITERATIONS {int(m.opt.ccd_iterations)}\n#define EPA_ITERATIONS 16\n"
            f"#define CCD_TOLERANCE {float(m.opt.ccd_tolerance):.9g}f\n#define NPOLYMAX 4\n#define NMESHDEG 3\n")


DEVICE = r"""
struct f3 { float x, y, z; };
__device__ __forceinline__ f3 mk3(float x, float y, float z) { f3 r; r.x = x; r.y = y; r.z = z; return r; }
__device__ __forceinline__ f3 ld3(const float* p) { return mk3(p[0], p[1], p[2]); }
__device__ __forceinline__ f3 operator+(f3 a, f3 b) { return mk3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ f3 operator-(f3 a, f3 b) { return mk3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ f3 operator-(f3 a) { return mk3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ f3 operator*(float s, f3 a) { return mk3(s * a.x, s * a.y, s * a.z); }
__device__ __forceinline__ f3 operator*(f3 a, float s) { return mk3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ f3 operator/(f3 a, float s) { return mk3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float dt3(f3 a, f3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ f3 cr3(f3 a, f3 b) { return mk3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ float nm3(f3 a) { return sqrtf(dt3(a, a)); }
__device__ __forceinline__ f3 nz3(f3 a) { float n = nm3(a); return n > 0.f ? a / n : mk3(0.f, 0.f, 0.f); }  // wp.normalize
__device__ __forceinline__ float cvsign(float x) { return x < 0.f ? -1.f : 1.f; }  // wp.sign
__device__ __forceinline__ float cget(f3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
__device__ __forceinline__ void cset(f3& a, int i, float v) { if (i == 0) a.x = v; else if (i == 1) a.y = v; else a.z = v; }
// R (row-major 3x3) times v, and its transpose
__device__ __forceinline__ f3 rmul(const float* R, f3 v) {
  return mk3(R[0] * v.x + R[1] * v.y + R[2] * v.z, R[3] * v.x + R[4] * v.y + R[5] * v.z, R[6] * v.x + R[7] * v.y + R[8] * v.z);
}
__device__ __forceinline__ f3 rtmul(const float* R, f3 v) {
  return mk3(R[0] * v.x + R[3] * v.y + R[6] * v.z, R[1] * v.x + R[4] * v.y + R[7] * v.z, R[2] * v.x + R[5] * v.y + R[8] * v.z);
}


// ---------------------------------------------------------------- convex collisions (MuJoCo Warp's collision_gjk.py)
#define CV_FLOAT_MAX 1e30f
#define CV_FLOAT_MIN (-1e30f)
#define CV_MINVAL 1e-15f
#define CV_MINVAL2 1e-30f
#define CV_MIN_DIST2 1e-10f
#define CV_MIN_DIST3 1e-10f
#define CV_MIN_DIST4 1e-17f
#define CV_MIN_EPATOL 1e-7f
#define CV_FACE_TOL 0.99605140f  // cos(0.0889)
#define CV_EDGE_TOL 0.08878295f  // sin(0.0889)
#define CV_INTERSECT_TOL 0.0000003f
#define EPA_NV (10 + 2 * EPA_ITERATIONS)  // polytope vertices (pairs: geom 1, geom 2)
#define EPA_NF (6 + 5 * EPA_ITERATIONS)   // faces (MJ_MAX_EPAFACES = 5)
#define EPA_NH 24                         // horizon edges (MJ_MAX_EPAHORIZON)
#define FACE_DELETED 0x80000000u
#define FACE_INVALID 0x40000000u

// a geom as MuJoCo Warp's collision_core.Geom (the fields the convex code reads); margins are 0
struct CGeom {
  f3 pos;
  float R[9];
  f3 size;
  int type;   // the engine's: 1 sphere, 2 capsule, 3 box
  int index;  // the support's cached vertex (MuJoCo Warp's geom.index)
};
struct SupportPoint { f3 point; int cached; int vidx; };

// support point of a geom in direction dir (MuJoCo Warp's support)
__device__ SupportPoint cv_support(const CGeom& g, f3 dir) {
  SupportPoint sp;
  sp.cached = -1;
  sp.vidx = -1;
  if (g.type == 1) { sp.point = g.pos + g.size.x * dir; return sp; }
  f3 ld = rtmul(g.R, dir);
  if (g.type == 3) {
    f3 tmp = mk3(cvsign(ld.x), cvsign(ld.y), cvsign(ld.z));
    f3 res = mk3(tmp.x * g.size.x, tmp.y * g.size.y, tmp.z * g.size.z);
    sp.point = rmul(g.R, res) + g.pos;
    sp.vidx = (tmp.x > 0.f ? 1 : 0) + (tmp.y > 0.f ? 2 : 0) + (tmp.z > 0.f ? 4 : 0);
    return sp;
  }
  if (g.type == 2) {
    f3 res = g.size.x * ld;
    res.z += cvsign(ld.z) * g.size.y;
    sp.point = rmul(g.R, res) + g.pos;
    return sp;
  }
  return sp;  // (no meshes in this module)
}

struct Simplex { f3 v[4]; };
struct GJKResult {
  bool separated;
  float dist;
  f3 x1, x2;
  int dim;
  f3 simplex[4], simplex1[4], simplex2[4];
  int sidx1[4], sidx2[4];
  int index1, index2;
};

__device__ __forceinline__ f3 lin_combine(int n, const float* l, const f3* m) {
  if (n == 1) return l[0] * m[0];
  if (n == 2) return l[0] * m[0] + l[1] * m[1];
  if (n == 3) return l[0] * m[0] + l[1] * m[1] + l[2] * m[2];
  return l[0] * m[0] + l[1] * m[1] + l[2] * m[2] + l[3] * m[3];
}
__device__ __forceinline__ float det3f(f3 a, f3 b, f3 c) { return dt3(a, cr3(b, c)); }
__device__ __forceinline__ int same_sign(float a, float b) {
  if (a > 0.f && b > 0.f) return 1;
  if (a < 0.f && b < 0.f) return -1;
  return 0;
}
__device__ __forceinline__ f3 proj_origin_line(f3 v1, f3 v2) {
  f3 diff = v2 - v1;
  float scl = -(dt3(v2, diff) / dt3(diff, diff));
  return v2 + scl * diff;
}
// returns ret (1: degenerate)
__device__ __forceinline__ int proj_origin_plane(f3 v1, f3 v2, f3 v3, f3* r) {
  f3 d21 = v2 - v1, d31 = v3 - v1, d32 = v3 - v2;
  f3 n = cr3(d32, d21);
  float nv = dt3(n, v2), nn = dt3(n, n);
  if (nn == 0.f) { *r = mk3(0.f, 0.f, 0.f); return 1; }
  if (nv != 0.f && nn > CV_MINVAL) { *r = (nv / nn) * n; return 0; }
  n = cr3(d21, d31);
  nv = dt3(n, v1); nn = dt3(n, n);
  if (nn == 0.f) { *r = mk3(0.f, 0.f, 0.f); return 1; }
  if (nv != 0.f && nn > CV_MINVAL) { *r = (nv / nn) * n; return 0; }
  n = cr3(d31, d32);
  nv = dt3(n, v3); nn = dt3(n, n);
  *r = (nv / nn) * n;
  return 0;
}
__device__ void S1D(f3 s1, f3 s2, float* l) {
  f3 p_o = proj_origin_line(s1, s2);
  float mu_max = s1.x - s2.x;
  int index = 0;
  float mu = s1.y - s2.y;
  if (fabsf(mu) >= fabsf(mu_max)) { mu_max = mu; index = 1; }
  mu = s1.z - s2.z;
  if (fabsf(mu) >= fabsf(mu_max)) { mu_max = mu; index = 2; }
  float C1 = cget(p_o, index) - cget(s2, index), C2 = cget(s1, index) - cget(p_o, index);
  if (same_sign(mu_max, C1) && same_sign(mu_max, C2)) { l[0] = C1 / mu_max; l[1] = C2 / mu_max; return; }
  l[0] = 0.f; l[1] = 1.f;
}
__device__ void S2D(f3 s1, f3 s2, f3 s3, float* l) {
  f3 p_o;
  if (proj_origin_plane(s1, s2, s3, &p_o)) { S1D(s1, s2, l); l[2] = 0.f; return; }
  float M_14 = s2.y * s3.z - s2.z * s3.y - s1.y * s3.z + s1.z * s3.y + s1.y * s2.z - s1.z * s2.y;
  float M_24 = s2.x * s3.z - s2.z * s3.x - s1.x * s3.z + s1.z * s3.x + s1.x * s2.z - s1.z * s2.x;
  float M_34 = s2.x * s3.y - s2.y * s3.x - s1.x * s3.y + s1.y * s3.x + s1.x * s2.y - s1.y * s2.x;
  float M_max, a1, a2, b1, b2, c1, c2, p1, p2;
  float mu1 = fabsf(M_14), mu2 = fabsf(M_24), mu3 = fabsf(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) { M_max = M_14; a1 = s1.y; a2 = s1.z; b1 = s2.y; b2 = s2.z; c1 = s3.y; c2 = s3.z; p1 = p_o.y; p2 = p_o.z; }
  else if (mu2 >= mu3) { M_max = M_24; a1 = s1.x; a2 = s1.z; b1 = s2.x; b2 = s2.z; c1 = s3.x; c2 = s3.z; p1 = p_o.x; p2 = p_o.z; }
  else { M_max = M_34; a1 = s1.x; a2 = s1.y; b1 = s2.x; b2 = s2.y; c1 = s3.x; c2 = s3.y; p1 = p_o.x; p2 = p_o.y; }
  float C31 = p1 * b2 + p2 * c1 + b1 * c2 - p1 * c2 - p2 * b1 - c1 * b2;
  float C32 = p1 * c2 + p2 * a1 + c1 * a2 - p1 * a2 - p2 * c1 - a1 * c2;
  float C33 = p1 * a2 + p2 * b1 + a1 * b2 - p1 * b2 - p2 * a1 - b1 * a2;
  int comp1 = same_sign(M_max, C31), comp2 = same_sign(M_max, C32), comp3 = same_sign(M_max, C33);
  if (comp1 && comp2 && comp3) { l[0] = C31 / M_max; l[1] = C32 / M_max; l[2] = C33 / M_max; return; }
  float dmin = CV_FLOAT_MAX, sub[2];
  l[0] = l[1] = l[2] = 0.f;
  if (!comp1) {
    S1D(s2, s3, sub);
    f3 x = sub[0] * s2 + sub[1] * s3;
    l[0] = 0.f; l[1] = sub[0]; l[2] = sub[1];
    dmin = dt3(x, x);
  }
  if (!comp2) {
    S1D(s1, s3, sub);
    f3 x = sub[0] * s1 + sub[1] * s3;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = 0.f; l[2] = sub[1]; dmin = d; }
  }
  if (!comp3) {
    S1D(s1, s2, sub);
    f3 x = sub[0] * s1 + sub[1] * s2;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = sub[1]; l[2] = 0.f; }
  }
}
__device__ void S3D(f3 s1, f3 s2, f3 s3, f3 s4, float* l) {
  float C41 = -det3f(s2, s3, s4), C42 = det3f(s1, s3, s4), C43 = -det3f(s1, s2, s4), C44 = det3f(s1, s2, s3);
  float m_det = C41 + C42 + C43 + C44;
  int comp1 = same_sign(m_det, C41), comp2 = same_sign(m_det, C42), comp3 = same_sign(m_det, C43),
      comp4 = same_sign(m_det, C44);
  if (comp1 && comp2 && comp3 && comp4) { l[0] = C41 / m_det; l[1] = C42 / m_det; l[2] = C43 / m_det; l[3] = C44 / m_det; return; }
  l[0] = l[1] = l[2] = l[3] = 0.f;
  float dmin = CV_FLOAT_MAX, sub[3];
  if (!comp1) {
    S2D(s2, s3, s4, sub);
    f3 x = sub[0] * s2 + sub[1] * s3 + sub[2] * s4;
    l[0] = 0.f; l[1] = sub[0]; l[2] = sub[1]; l[3] = sub[2];
    dmin = dt3(x, x);
  }
  if (!comp2) {
    S2D(s1, s3, s4, sub);
    f3 x = sub[0] * s1 + sub[1] * s3 + sub[2] * s4;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = 0.f; l[2] = sub[1]; l[3] = sub[2]; dmin = d; }
  }
  if (!comp3) {
    S2D(s1, s2, s4, sub);
    f3 x = sub[0] * s1 + sub[1] * s2 + sub[2] * s4;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = sub[1]; l[2] = 0.f; l[3] = sub[2]; dmin = d; }
  }
  if (!comp4) {
    S2D(s1, s2, s3, sub);
    f3 x = sub[0] * s1 + sub[1] * s2 + sub[2] * s3;
    float d = dt3(x, x);
    if (d < dmin) { l[0] = sub[0]; l[1] = sub[1]; l[2] = sub[2]; l[3] = 0.f; }
  }
}
__device__ __forceinline__ void subdistance(int n, const f3* s, float* l) {
  if (n == 4) { S3D(s[0], s[1], s[2], s[3], l); return; }
  if (n == 3) { S2D(s[0], s[1], s[2], l); l[3] = 0.f; return; }
  if (n == 2) { S1D(s[0], s[1], l); l[2] = l[3] = 0.f; return; }
  l[0] = 1.f; l[1] = l[2] = l[3] = 0.f;
}

__device__ __forceinline__ bool is_discrete(int t1, int t2) { return (t1 == 3 || t1 == 4) && (t2 == 3 || t2 == 4); }

// GJK (MuJoCo Warp's gjk), cutoff 0 or the capsule / sphere radii
__device__ void cv_gjk(float tol, int iters, CGeom& g1, CGeom& g2, f3 x1_0, f3 x2_0, float cutoff, bool disc,
                       GJKResult& r) {
  f3 simplex[4], simplex1[4], simplex2[4];
  int si1[4] = {0, 0, 0, 0}, si2[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) simplex[i] = simplex1[i] = simplex2[i] = mk3(0.f, 0.f, 0.f);
  int n = 0;
  float l[4] = {1.f, 0.f, 0.f, 0.f};
  float epsilon = disc ? 0.f : 0.5f * tol * tol, min_norm = disc ? CV_MINVAL : tol;
  f3 x_k = x1_0 - x2_0;
  float xnorm = sqrtf(dt3(x_k, x_k)), xnorm_prev = 0.f;
  for (int it = 0; it < iters; it++) {
    if (xnorm < min_norm || fabsf(xnorm_prev - xnorm) < CV_MINVAL) break;
    f3 dir_neg = x_k / xnorm;  // (_gjk_support: direction tuning for discrete geoms)
    if (disc && xnorm < 1e-4f) {
      if (n == 2) {
        f3 edge = simplex[1] - simplex[0];
        float e2 = dt3(edge, edge);
        if (e2 > CV_MINVAL2) {
          float proj = dt3(dir_neg, edge) / e2;
          dir_neg = dir_neg - proj * edge;
          float dn = nm3(dir_neg);
          if (dn > CV_MINVAL) dir_neg = dir_neg / dn;
        }
      } else if (n == 3) {
        f3 e1 = simplex[1] - simplex[0], e2 = simplex[2] - simplex[0], nrm = cr3(e1, e2);
        float nn = nm3(nrm);
        if (nn > CV_MINVAL) dir_neg = cvsign(dt3(dir_neg, nrm)) * nrm / nn;
      }
    }
    SupportPoint sp1 = cv_support(g1, -dir_neg), sp2 = cv_support(g2, dir_neg);
    simplex1[n] = sp1.point; g1.index = sp1.cached; si1[n] = sp1.vidx;
    simplex2[n] = sp2.point; g2.index = sp2.cached; si2[n] = sp2.vidx;
    simplex[n] = simplex1[n] - simplex2[n];
    if (dt3(x_k, x_k - simplex[n]) < epsilon) break;
    float lower = dt3(x_k, simplex[n]);
    if (cutoff == 0.f) {
      if (lower > 0.f) { r.separated = true; r.dim = 0; r.dist = CV_FLOAT_MAX; r.index1 = g1.index; r.index2 = g2.index; return; }
    } else if (cutoff < CV_FLOAT_MAX) {
      if (lower > 0.f && lower >= cutoff * xnorm) {
        r.separated = true; r.dim = 0; r.dist = CV_FLOAT_MAX; r.index1 = g1.index; r.index2 = g2.index; return;
      }
    }
    subdistance(n + 1, simplex, l);
    n = 0;
    for (int i = 0; i < 4; i++) {
      if (l[i] == 0.f) continue;
      simplex[n] = simplex[i]; simplex1[n] = simplex1[i]; simplex2[n] = simplex2[i];
      si1[n] = si1[i]; si2[n] = si2[i]; l[n] = l[i];
      n++;
    }
    if (n < 1) break;
    x_k = lin_combine(n, l, simplex);
    xnorm_prev = xnorm;
    xnorm = sqrtf(dt3(x_k, x_k));
    if (n == 4) break;
  }
  r.separated = false;
  r.x1 = n == 0 ? x1_0 : lin_combine(n, l, simplex1);
  r.x2 = n == 0 ? x2_0 : lin_combine(n, l, simplex2);
  if (xnorm > 0.f) {
    f3 dir = x_k / xnorm;
    SupportPoint sp1 = cv_support(g1, -dir), sp2 = cv_support(g2, dir);
    r.separated = dt3(x_k, sp1.point - sp2.point) > 0.f;
  }
  r.dist = (n == 4 && !r.separated) ? 0.f : xnorm;
  r.dim = n;
  for (int i = 0; i < 4; i++) {
    r.simplex[i] = simplex[i]; r.simplex1[i] = simplex1[i]; r.simplex2[i] = simplex2[i];
    r.sidx1[i] = si1[i]; r.sidx2[i] = si2[i];
  }
  r.index1 = g1.index;
  r.index2 = g2.index;
}

// ------------------------------------------------------------------------------------------------ EPA polytope
struct Polytope {
  f3 vert[EPA_NV];
  int vidx[EPA_NV];
  int nvert;
  f3 center;
  unsigned face[EPA_NF];
  f3 face_pr[EPA_NF];
  float face_norm2[EPA_NF];
  int nface;
  int horizon[EPA_NH];
  int nhorizon;
  int status;
};
__device__ __forceinline__ void face_verts(unsigned f, int* v) { v[0] = f & 0x3FF; v[1] = (f >> 10) & 0x3FF; v[2] = (f >> 20) & 0x3FF; }
__device__ __forceinline__ bool face_deleted(unsigned f) { return f & FACE_DELETED; }
__device__ __forceinline__ bool face_invalid(unsigned f) { return f & (FACE_DELETED | FACE_INVALID); }

__device__ float attach_face(Polytope& pt, int idx, int v1, int v2, int v3) {
  if (pt.nface == EPA_NF) return 0.f;
  f3 p1 = pt.vert[2 * v1] - pt.vert[2 * v1 + 1], p2 = pt.vert[2 * v2] - pt.vert[2 * v2 + 1],
     p3 = pt.vert[2 * v3] - pt.vert[2 * v3 + 1], r;
  if (proj_origin_plane(p3, p2, p1, &r)) return 0.f;
  if (dt3(r, p1 - pt.center) < 0.f) r = -r;
  pt.face[idx] = (unsigned)(v1 + (v2 << 10) + (v3 << 20));
  pt.face_pr[idx] = r;
  pt.face_norm2[idx] = dt3(r, r);
  return pt.face_norm2[idx];
}
__device__ void epa_support(Polytope& pt, int idx, CGeom& g1, CGeom& g2, f3 dir, int* i1, int* i2) {
  SupportPoint sp = cv_support(g1, dir);
  pt.vert[2 * idx] = sp.point; pt.vidx[2 * idx] = sp.vidx; *i1 = sp.cached;
  sp = cv_support(g2, -dir);
  pt.vert[2 * idx + 1] = sp.point; pt.vidx[2 * idx + 1] = sp.vidx; *i2 = sp.cached;
}
__device__ void replace_simplex3(const Polytope& pt, int v1, int v2, int v3, GJKResult& r) {
  const int v[3] = {v1, v2, v3};
  for (int i = 0; i < 3; i++) {
    r.simplex1[i] = pt.vert[2 * v[i]];
    r.simplex2[i] = pt.vert[2 * v[i] + 1];
    r.simplex[i] = r.simplex1[i] - r.simplex2[i];
    r.sidx1[i] = pt.vidx[2 * v[i]];
    r.sidx2[i] = pt.vidx[2 * v[i] + 1];
  }
}
__device__ void rotmat120(f3 axis, float* R) {
  float n = nm3(axis), u1 = axis.x / n, u2 = axis.y / n, u3 = axis.z / n;
  const float s = 0.86602540378f, c = -0.5f;
  R[0] = c + u1 * u1 * (1.f - c); R[1] = u1 * u2 * (1.f - c) - u3 * s; R[2] = u1 * u3 * (1.f - c) + u2 * s;
  R[3] = u2 * u1 * (1.f - c) + u3 * s; R[4] = c + u2 * u2 * (1.f - c); R[5] = u2 * u3 * (1.f - c) - u1 * s;
  R[6] = u1 * u3 * (1.f - c) - u2 * s; R[7] = u2 * u3 * (1.f - c) + u1 * s; R[8] = c + u3 * u3 * (1.f - c);
}
__device__ int ray_triangle(f3 v1, f3 v2, f3 v3, f3 v4, f3 v5) {
  float vol1 = det3f(v3 - v1, v4 - v1, v2 - v1), vol2 = det3f(v4 - v1, v5 - v1, v2 - v1), vol3 = det3f(v5 - v1, v3 - v1, v2 - v1);
  if (vol1 >= 0.f && vol2 >= 0.f && vol3 >= 0.f) return 1;
  if (vol1 <= 0.f && vol2 <= 0.f && vol3 <= 0.f) return -1;
  return 0;
}
__device__ f3 tri_affine(f3 v1, f3 v2, f3 v3, f3 p) {
  float M_14 = v2.y * v3.z - v2.z * v3.y - v1.y * v3.z + v1.z * v3.y + v1.y * v2.z - v1.z * v2.y;
  float M_24 = v2.x * v3.z - v2.z * v3.x - v1.x * v3.z + v1.z * v3.x + v1.x * v2.z - v1.z * v2.x;
  float M_34 = v2.x * v3.y - v2.y * v3.x - v1.x * v3.y + v1.y * v3.x + v1.x * v2.y - v1.y * v2.x;
  float M_max;
  int x, y;
  float mu1 = fabsf(M_14), mu2 = fabsf(M_24), mu3 = fabsf(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) { M_max = M_14; x = 1; y = 2; }
  else if (mu2 >= mu3) { M_max = M_24; x = 0; y = 2; }
  else { M_max = M_34; x = 0; y = 1; }
  float px = cget(p, x), py = cget(p, y), ax = cget(v1, x), ay = cget(v1, y), bx = cget(v2, x), by = cget(v2, y),
        cx = cget(v3, x), cy = cget(v3, y);
  float C31 = px * by + py * cx + bx * cy - px * cy - py * bx - cx * by;
  float C32 = px * cy + py * ax + cx * ay - px * ay - py * cx - ax * cy;
  float C33 = px * ay + py * bx + ax * by - px * by - py * ax - bx * ay;
  return mk3(C31 / M_max, C32 / M_max, C33 / M_max);
}
__device__ bool tri_point_intersect(f3 v1, f3 v2, f3 v3, f3 p) {
  f3 c = tri_affine(v1, v2, v3, p);
  if (c.x < 0.f || c.y < 0.f || c.z < 0.f) return false;
  f3 pr = mk3(v1.x * c.x + v2.x * c.y + v3.x * c.z, v1.y * c.x + v2.y * c.y + v3.y * c.z, v1.z * c.x + v2.z * c.y + v3.z * c.z);
  return nm3(pr - p) < CV_MINVAL;
}
__device__ bool same_side(f3 p0, f3 p1, f3 p2, f3 p3) {
  f3 n = cr3(p1 - p0, p2 - p0);
  float d1 = dt3(n, p3 - p0), d2 = dt3(n, -p0);
  return (d1 > 0.f && d2 > 0.f) || (d1 < 0.f && d2 < 0.f);
}
__device__ bool test_tetra(f3 p0, f3 p1, f3 p2, f3 p3) {
  return same_side(p0, p1, p2, p3) && same_side(p1, p2, p3, p0) && same_side(p2, p3, p0, p1) && same_side(p3, p0, p1, p2);
}
__device__ int add_edge(Polytope& pt, int e1, int e2) {
  int n = pt.nhorizon;
  if (n < 0) return -1;
  int edge = (min(e1, e2) << 10) | max(e1, e2);
  for (int i = 0; i < n; i++)
    if (edge == pt.horizon[i]) { pt.horizon[i] = pt.horizon[n - 1]; return n - 1; }
  if (n == EPA_NH) return -1;
  pt.horizon[n] = edge;
  return n + 1;
}

__device__ void polytope2(Polytope& pt, GJKResult& r, CGeom& g1, CGeom& g2) {
  f3 diff = r.simplex[1] - r.simplex[0];
  pt.center = 0.5f * (r.simplex[0] + r.simplex[1]);
  float value = CV_FLOAT_MAX;
  int index = 0;
  for (int i = 0; i < 3; i++)
    if (fabsf(cget(diff, i)) < value) { value = fabsf(cget(diff, i)); index = i; }
  f3 e = mk3(0.f, 0.f, 0.f);
  cset(e, index, 1.f);
  f3 d1 = cr3(e, diff);
  float R[9];
  rotmat120(diff, R);
  f3 d2 = rmul(R, d1), d3 = rmul(R, d2);
  pt.vert[0] = r.simplex1[0]; pt.vert[1] = r.simplex2[0]; pt.vert[2] = r.simplex1[1]; pt.vert[3] = r.simplex2[1];
  pt.vidx[0] = r.sidx1[0]; pt.vidx[1] = r.sidx2[0]; pt.vidx[2] = r.sidx1[1]; pt.vidx[3] = r.sidx2[1];
  int i1, i2;
  epa_support(pt, 2, g1, g2, d1 / nm3(d1), &i1, &i2);
  epa_support(pt, 3, g1, g2, d2 / nm3(d2), &i1, &i2);
  epa_support(pt, 4, g1, g2, d3 / nm3(d3), &i1, &i2);
  const int F[6][3] = {{0, 2, 3}, {0, 4, 2}, {0, 3, 4}, {1, 3, 2}, {1, 2, 4}, {1, 4, 3}};
  for (int f = 0; f < 6; f++)
    if (attach_face(pt, f, F[f][0], F[f][1], F[f][2]) < CV_MIN_DIST2) {
      pt.status = -1;
      replace_simplex3(pt, F[f][0], F[f][1], F[f][2], r);
      return;
    }
  f3 v2 = pt.vert[4] - pt.vert[5], v3 = pt.vert[6] - pt.vert[7], v4 = pt.vert[8] - pt.vert[9];
  if (!ray_triangle(r.simplex[0], r.simplex[1], v2, v3, v4)) { pt.status = 1; return; }
  pt.nvert = 5; pt.nface = 6; pt.status = 0;
}

__device__ void polytope3(Polytope& pt, const GJKResult& r, CGeom& g1, CGeom& g2) {
  pt.center = (r.simplex[0] + r.simplex[1] + r.simplex[2]) * (1.f / 3.f) ;
  f3 n = cr3(r.simplex[1] - r.simplex[0], r.simplex[2] - r.simplex[0]);
  float norm = nm3(n);
  if (norm < CV_MINVAL) { pt.status = 2; return; }
  n = n / norm;
  for (int i = 0; i < 3; i++) {
    pt.vert[2 * i] = r.simplex1[i]; pt.vert[2 * i + 1] = r.simplex2[i];
    pt.vidx[2 * i] = r.sidx1[i]; pt.vidx[2 * i + 1] = r.sidx2[i];
  }
  int i1, i2;
  epa_support(pt, 3, g1, g2, -n, &i1, &i2);
  epa_support(pt, 4, g1, g2, n, &i1, &i2);
  f3 v1 = r.simplex[0], v2 = r.simplex[1], v3 = r.simplex[2], v4 = pt.vert[6] - pt.vert[7], v5 = pt.vert[8] - pt.vert[9];
  if (tri_point_intersect(v1, v2, v3, v4)) { pt.status = 3; return; }
  if (tri_point_intersect(v1, v2, v3, v5)) { pt.status = 4; return; }
  if (r.dist > 1e-5f && !test_tetra(v1, v2, v3, v4) && !test_tetra(v1, v2, v3, v5)) { pt.status = 5; return; }
  const int F[6][3] = {{4, 0, 1}, {4, 2, 0}, {4, 1, 2}, {3, 1, 0}, {3, 0, 2}, {3, 2, 1}};
  for (int f = 0; f < 6; f++)
    if (attach_face(pt, f, F[f][0], F[f][1], F[f][2]) < CV_MIN_DIST3) { pt.status = 6 + f; return; }
  pt.nvert = 5; pt.nface = 6; pt.status = 0;
}


__device__ void polytope4(Polytope& pt, GJKResult& r) {
  pt.center = 0.25f * (r.simplex[0] + r.simplex[1] + r.simplex[2] + r.simplex[3]);
  for (int i = 0; i < 4; i++) {
    pt.vert[2 * i] = r.simplex1[i]; pt.vert[2 * i + 1] = r.simplex2[i];
    pt.vidx[2 * i] = r.sidx1[i]; pt.vidx[2 * i + 1] = r.sidx2[i];
  }
  const int F[4][3] = {{0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {3, 2, 1}};
  float dist[4];
  int idx = 0;
  for (int f = 0; f < 4; f++) {
    dist[f] = attach_face(pt, f, F[f][0], F[f][1], F[f][2]);
    if (dist[f] < CV_MIN_DIST4) { pt.status = -1; replace_simplex3(pt, F[f][0], F[f][1], F[f][2], r); return; }
    if (f == 1) idx = dist[0] < dist[1] ? 0 : 1;
    else if (f > 1) idx = dist[f] < dist[idx] ? f : idx;
  }
  if (!test_tetra(r.simplex[0], r.simplex[1], r.simplex[2], r.simplex[3])) {
    if (dist[idx] > CV_MINVAL) { pt.status = 12; return; }
    pt.status = -1;
    replace_simplex3(pt, F[idx][0], F[idx][1], F[idx][2], r);
    return;
  }
  pt.nvert = 4; pt.nface = 4; pt.status = 0;
}

// EPA (MuJoCo Warp's _epa): the closest face idx (-1: none) and the witness points
__device__ int cv_epa(float tol, int iters, Polytope& pt, CGeom& g1, CGeom& g2, bool disc) {
  float upper = CV_FLOAT_MAX, upper2 = CV_FLOAT_MAX;
  int idx = -1, pidx = -1;
  float epsilon = disc ? CV_MIN_EPATOL : tol;
  int nvalid = pt.nface;
  iters = min(iters, 1000);
  for (int it = 0; it < iters; it++) {
    pidx = idx;
    idx = -1;
    float lower2 = CV_FLOAT_MAX;
    for (int i = 0; i < pt.nface; i++)
      if (!face_invalid(pt.face[i]) && pt.face_norm2[i] < lower2) { idx = i; lower2 = pt.face_norm2[i]; }
    if (lower2 > upper2 || idx < 0) { idx = pidx; break; }
    if (lower2 <= 0.f) break;
    float lower = sqrtf(lower2);
    int wi = pt.nvert;
    if (2 * wi + 1 >= EPA_NV) break;  // (MuJoCo Warp's buffer holds iters + 5 vertices: not reached)
    f3 face_pr = pt.face_pr[idx];
    int i1, i2;
    epa_support(pt, wi, g1, g2, face_pr / lower, &i1, &i2);
    f3 w = pt.vert[2 * wi] - pt.vert[2 * wi + 1];
    g1.index = i1; g2.index = i2;
    pt.nvert++;
    float upper_k = dt3(face_pr, w) / lower;
    if (upper_k < upper) { upper = upper_k; upper2 = upper * upper; }
    if (upper - lower < epsilon) break;
    if (disc) {
      bool rep = false;
      for (int i = 0; i < pt.nvert - 1; i++)
        if (pt.vidx[2 * i] == pt.vidx[2 * wi] && pt.vidx[2 * i + 1] == pt.vidx[2 * wi + 1]) { rep = true; break; }
      if (rep) break;
    }
    nvalid--;
    pt.face[idx] |= FACE_DELETED;
    int fv[3];
    face_verts(pt.face[idx], fv);
    pt.nhorizon = add_edge(pt, fv[0], fv[1]);
    pt.nhorizon = add_edge(pt, fv[1], fv[2]);
    pt.nhorizon = add_edge(pt, fv[2], fv[0]);
    if (pt.nhorizon == -1) { idx = -1; break; }
    for (int i = 0; i < pt.nface; i++) {
      if (face_deleted(pt.face[i])) continue;
      if (dt3(pt.face_pr[i], w) - pt.face_norm2[i] > 1e-10f) {
        if (!face_invalid(pt.face[i])) nvalid--;
        pt.face[i] |= FACE_DELETED;
        face_verts(pt.face[i], fv);
        pt.nhorizon = add_edge(pt, fv[0], fv[1]);
        pt.nhorizon = add_edge(pt, fv[1], fv[2]);
        pt.nhorizon = add_edge(pt, fv[2], fv[0]);
        if (pt.nhorizon == -1) { idx = -1; break; }
      }
    }
    for (int i = 0; i < pt.nhorizon && idx != -1; i++) {
      int e0 = pt.horizon[i] & 0x3FF, e1 = (pt.horizon[i] >> 10) & 0x3FF;
      float d2 = attach_face(pt, pt.nface, wi, e0, e1);
      if (d2 == 0.f) { idx = -1; break; }
      pt.nface++;
      if (d2 >= lower2 && d2 <= upper2) nvalid++;
      else pt.face[pt.nface - 1] |= FACE_INVALID;
    }
    if (nvalid == 0 || idx == -1) break;
    pt.nhorizon = 0;
  }
  return idx;
}
// the witness points of face idx (MuJoCo Warp's _epa_witness without the height field case); returns the distance
__device__ float epa_witness(const Polytope& pt, int idx, f3* x1, f3* x2) {
  int fv[3];
  face_verts(pt.face[idx], fv);
  f3 v1 = pt.vert[2 * fv[0]] - pt.vert[2 * fv[0] + 1], v2 = pt.vert[2 * fv[1]] - pt.vert[2 * fv[1] + 1],
     v3 = pt.vert[2 * fv[2]] - pt.vert[2 * fv[2] + 1];
  f3 c = tri_affine(v1, v2, v3, pt.face_pr[idx]);
  v1 = pt.vert[2 * fv[0] + 1]; v2 = pt.vert[2 * fv[1] + 1]; v3 = pt.vert[2 * fv[2] + 1];
  *x2 = mk3(v1.x * c.x + v2.x * c.y + v3.x * c.z, v1.y * c.x + v2.y * c.y + v3.y * c.z, v1.z * c.x + v2.z * c.y + v3.z * c.z);
  v1 = pt.vert[2 * fv[0]]; v2 = pt.vert[2 * fv[1]]; v3 = pt.vert[2 * fv[2]];
  *x1 = mk3(v1.x * c.x + v2.x * c.y + v3.x * c.z, v1.y * c.x + v2.y * c.y + v3.y * c.z, v1.z * c.x + v2.z * c.y + v3.z * c.z);
  return -sqrtf(pt.face_norm2[idx]);
}

// ------------------------------------------------------------------------------------- multicontact (polygon clip)
struct MultiBuf {  // MuJoCo Warp's per-contact multiccd buffers (and the clipped points' provenance, for the VJP)
  f3 polygon[2 * NPOLYMAX], clipped[2 * NPOLYMAX], pnormal[NPOLYMAX];
  float pdist[NPOLYMAX];
  int idx1[NMESHDEG], idx2[NMESHDEG];
  f3 n1[NMESHDEG], n2[NMESHDEG], endvert[NMESHDEG], face1[NPOLYMAX], face2[NPOLYMAX];
  // provenance: each polygon vertex lies on two lines (tags: 0..7 the subject polygon's edge k, 8 + e the clipping
  // face's side plane e, 16 + i the subject segment's end i, 24 the subject segment's line), pin / cin: the line of
  // the polygon edge that leads into it
  unsigned char ptag[2 * NPOLYMAX][2], ctag[2 * NPOLYMAX][2], pin[2 * NPOLYMAX], cin[2 * NPOLYMAX];
  unsigned char otag[4][2];  // the output points' tags
  int mcase, mi, mj;  // multicontact's case (0 none, 1 face-face, 2 an edge of geom 2, 3 an edge of geom 1), i, j
};
__device__ int feature_dim(const int* fv, const Polytope& pt, int off, int* fi, f3* fx) {
  int v1i = pt.vidx[2 * fv[0] + off], v2i = pt.vidx[2 * fv[1] + off], v3i = pt.vidx[2 * fv[2] + off];
  fi[0] = v1i; fi[1] = v2i; fi[2] = v3i;
  fx[0] = pt.vert[2 * fv[0] + off]; fx[1] = pt.vert[2 * fv[1] + off]; fx[2] = pt.vert[2 * fv[2] + off];
  if (v1i != v2i) return (v3i == v1i || v3i == v2i) ? 2 : 3;
  fi[1] = v3i;
  fx[1] = pt.vert[2 * fv[2] + off];
  return v1i != v3i ? 2 : 1;
}
__device__ int aligned_faces(const f3* a, int la, const f3* b, int lb, int* res) {
  for (int i = 0; i < la; i++)
    for (int j = 0; j < lb; j++)
      if (dt3(a[i], b[j]) < -CV_FACE_TOL) { res[0] = i; res[1] = j; return 1; }
  return 0;
}
__device__ int aligned_face_edge(const f3* edge, int nedge, const f3* face, int nface, int* res) {
  for (int i = 0; i < nface; i++)
    for (int j = 0; j < nedge; j++)
      if (fabsf(dt3(edge[j], face[i])) < CV_EDGE_TOL) { res[0] = j; res[1] = i; return 1; }
  return 0;
}
__device__ __forceinline__ f3 box_fn(int i) {  // mat63 face normals: +x, -x, +y, -y, +z, -z
  return mk3(i == 0 ? 1.f : (i == 1 ? -1.f : 0.f), i == 2 ? 1.f : (i == 3 ? -1.f : 0.f), i == 4 ? 1.f : (i == 5 ? -1.f : 0.f));
}
__device__ int box_normals2(const float* R, f3 n, f3* nout, int* iout) {
  f3 ln = nz3(rtmul(R, n));
  for (int i = 0; i < 6; i++)
    if (dt3(ln, box_fn(i)) > CV_FACE_TOL) { nout[0] = rmul(R, box_fn(i)); iout[0] = i; return 1; }
  return 0;
}
__device__ int box_normals(int dim, const int* fi, const float* R, f3 dir, f3* nout, int* iout) {
  int v1 = fi[0], v2 = fi[1], v3 = fi[2];
  if (dim == 3) {
    int c = 0;
    float x = float((v1 & 1) && (v2 & 1) && (v3 & 1)) - float(!(v1 & 1) && !(v2 & 1) && !(v3 & 1));
    float y = float((v1 & 2) && (v2 & 2) && (v3 & 2)) - float(!(v1 & 2) && !(v2 & 2) && !(v3 & 2));
    float z = float((v1 & 4) && (v2 & 4) && (v3 & 4)) - float(!(v1 & 4) && !(v2 & 4) && !(v3 & 4));
    nout[0] = rmul(R, mk3(x, y, z));
    float sgn = x + y + z;
    if (x != 0.f) iout[c++] = 0;
    if (y != 0.f) iout[c++] = 2;
    if (z != 0.f) iout[c++] = 4;
    if (sgn == -1.f) iout[0] = iout[0] + 1;
    if (c == 1) return 1;
    return box_normals2(R, dir, nout, iout);
  }
  if (dim == 2) {
    int c = 0;
    float x = float((v1 & 1) && (v2 & 1)) - float(!(v1 & 1) && !(v2 & 1));
    float y = float((v1 & 2) && (v2 & 2)) - float(!(v1 & 2) && !(v2 & 2));
    float z = float((v1 & 4) && (v2 & 4)) - float(!(v1 & 4) && !(v2 & 4));
    if (x != 0.f) { nout[c] = rmul(R, mk3(x, 0.f, 0.f)); iout[c] = x > 0.f ? 0 : 1; c++; }
    if (y != 0.f) { nout[c] = rmul(R, mk3(0.f, y, 0.f)); iout[c] = y > 0.f ? 2 : 3; c++; }
    if (z != 0.f) { nout[c] = rmul(R, mk3(0.f, 0.f, z)); iout[c] = z > 0.f ? 4 : 5; c++; }
    if (c == 1 || c == 2) return c;
    return box_normals2(R, dir, nout, iout);
  }
  if (dim == 1) {
    float x = (v1 & 1) ? 1.f : -1.f, y = (v1 & 2) ? 1.f : -1.f, z = (v1 & 4) ? 1.f : -1.f;
    nout[0] = rmul(R, mk3(x, 0.f, 0.f)); nout[1] = rmul(R, mk3(0.f, y, 0.f)); nout[2] = rmul(R, mk3(0.f, 0.f, z));
    iout[0] = x > 0.f ? 0 : 1; iout[1] = y > 0.f ? 2 : 3; iout[2] = z > 0.f ? 4 : 5;
    return 3;
  }
  return 0;
}
__device__ int box_edge_normals(int dim, const CGeom& g, f3 v1, f3 v2, int v1i, f3* nout, f3* eout) {
  if (dim == 2) { eout[0] = v2; nout[0] = nz3(v2 - v1); return 1; }
  if (dim == 1) {
    float x = (v1i & 1) ? g.size.x : -g.size.x, y = (v1i & 2) ? g.size.y : -g.size.y, z = (v1i & 4) ? g.size.z : -g.size.z;
    eout[0] = rmul(g.R, mk3(-x, y, z)) + g.pos; nout[0] = nz3(eout[0] - v1);
    eout[1] = rmul(g.R, mk3(x, -y, z)) + g.pos; nout[1] = nz3(eout[1] - v1);
    eout[2] = rmul(g.R, mk3(x, y, -z)) + g.pos; nout[2] = nz3(eout[2] - v1);
    return 3;
  }
  return 0;
}
__device__ int box_face(const CGeom& g, int idx, f3* f) {
  const float sx = g.size.x, sy = g.size.y, sz = g.size.z;
  f3 c[4];
  if (idx == 0) { c[0] = mk3(sx, sy, sz); c[1] = mk3(sx, sy, -sz); c[2] = mk3(sx, -sy, -sz); c[3] = mk3(sx, -sy, sz); }
  else if (idx == 1) { c[0] = mk3(-sx, sy, -sz); c[1] = mk3(-sx, sy, sz); c[2] = mk3(-sx, -sy, sz); c[3] = mk3(-sx, -sy, -sz); }
  else if (idx == 2) { c[0] = mk3(-sx, sy, -sz); c[1] = mk3(sx, sy, -sz); c[2] = mk3(sx, sy, sz); c[3] = mk3(-sx, sy, sz); }
  else if (idx == 3) { c[0] = mk3(-sx, -sy, sz); c[1] = mk3(sx, -sy, sz); c[2] = mk3(sx, -sy, -sz); c[3] = mk3(-sx, -sy, -sz); }
  else if (idx == 4) { c[0] = mk3(-sx, sy, sz); c[1] = mk3(sx, sy, sz); c[2] = mk3(sx, -sy, sz); c[3] = mk3(-sx, -sy, sz); }
  else if (idx == 5) { c[0] = mk3(sx, sy, -sz); c[1] = mk3(-sx, sy, -sz); c[2] = mk3(-sx, -sy, -sz); c[3] = mk3(sx, -sy, -sz); }
  else return 0;
  for (int i = 0; i < 4; i++) f[i] = rmul(g.R, c[i]) + g.pos;
  return 4;
}
__device__ __forceinline__ float area4(f3 a, f3 b, f3 c, f3 d) { return 0.5f * nm3(cr3(a - c, b - d)); }
__device__ void polygon_quad(const f3* p, int np, int* q) {  // MuJoCo Warp's _polygon_quad
  int b = 1, c = 2, d = 3;
  q[0] = 0; q[1] = b; q[2] = c; q[3] = d;
  float m = area4(p[0], p[b], p[c], p[d]);
  for (int a = 0; a < np; a++) {
    while (true) {
      float m_next = area4(p[a], p[b], p[c], p[(d + 1) % np]);
      if (m_next <= m) break;
      m = m_next; d = (d + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      while (true) {
        m_next = area4(p[a], p[b], p[(c + 1) % np], p[d]);
        if (m_next <= m) break;
        m = m_next; c = (c + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      }
      while (true) {
        m_next = area4(p[a], p[(b + 1) % np], p[c], p[d]);
        if (m_next <= m) break;
        m = m_next; b = (b + 1) % np; q[0] = a; q[1] = b; q[2] = c; q[3] = d;
      }
    }
    if (b == a) {
      b = (b + 1) % np;
      if (c == b) {
        c = (c + 1) % np;
        if (d == c) d = (d + 1) % np;
      }
    }
  }
}
__device__ __forceinline__ void witness_on_face(f3 v, f3 p, f3 n, f3 dir, f3* w1, f3* w2, float* d) {
  float dist = dt3(v - p, n);
  *w1 = v - fabsf(dist) * dir;
  *w2 = v;
  *d = dist;
}
// clip polygon face2 against face1 (MuJoCo Warp's _polygon_clip): the contacts (witnesses, distances), count
__device__ int polygon_clip(MultiBuf& mb, const f3* face1, int nface1, const f3* face2, int nface2, f3 n, f3 dir,
                            f3* w1o, f3* w2o, float* dists) {
  if (nface1 < 3) return 0;
  for (int i = 0; i < nface1; i++) {
    f3 a = face1[i], b = face1[(i + 1) % nface1];
    f3 v3 = a + n, res = cr3(b - a, v3 - a);
    mb.pdist[i] = dt3(res, a);
    mb.pnormal[i] = res;
  }
  f3* poly = mb.polygon;
  f3* clip = mb.clipped;
  unsigned char (*ptag)[2] = mb.ptag;
  unsigned char (*ctag)[2] = mb.ctag;
  unsigned char* pin = mb.pin;
  unsigned char* cin = mb.cin;
  int npoly = nface2, nclip = 0;
  for (int i = 0; i < nface2; i++) {
    poly[i] = face2[i];
    if (nface2 == 2) { ptag[i][0] = 24; ptag[i][1] = (unsigned char)(16 + i); pin[i] = 24; }
    else { ptag[i][0] = (unsigned char)((i + nface2 - 1) % nface2); ptag[i][1] = (unsigned char)i;
           pin[i] = (unsigned char)((i + nface2 - 1) % nface2); }
  }
  for (int e = 0; e < nface1; e++) {
    for (int i = 0; i < npoly; i++) {
      const int iq = (i + 1) % npoly;
      f3 P = poly[i], Q = poly[iq];
      const unsigned char li = pin[iq];  // the line of edge P Q
      bool in1 = dt3(P - face1[e], mb.pnormal[e]) > -1e-10f, in2 = dt3(Q - face1[e], mb.pnormal[e]) > -1e-10f;
      if (!in1 && !in2) continue;
      if (in1 && in2) {
        if (nclip < 2 * NPOLYMAX) { clip[nclip] = Q; ctag[nclip][0] = ptag[iq][0]; ctag[nclip][1] = ptag[iq][1];
                                    cin[nclip] = li; nclip++; }
        continue;
      }
      float dot = dt3(mb.pnormal[e], Q - P), t = fabsf(dot) < 1e-10f ? CV_FLOAT_MAX : (mb.pdist[e] - dt3(mb.pnormal[e], P)) / dot;
      if (t > -CV_INTERSECT_TOL && t < 1.f + CV_INTERSECT_TOL) {
        t = fminf(fmaxf(t, 0.f), 1.f);
        if (nclip < 2 * NPOLYMAX) { clip[nclip] = P + t * (Q - P); ctag[nclip][0] = li;
                                    ctag[nclip][1] = (unsigned char)(8 + e); cin[nclip] = in1 ? li : (unsigned char)(8 + e);
                                    nclip++; }
      }
      if (in2 && nclip < 2 * NPOLYMAX) { clip[nclip] = Q; ctag[nclip][0] = ptag[iq][0]; ctag[nclip][1] = ptag[iq][1];
                                         cin[nclip] = li; nclip++; }
    }
    f3* tmp = poly; poly = clip; clip = tmp;
    unsigned char (*tt)[2] = ptag; ptag = ctag; ctag = tt;
    unsigned char* ti = pin; pin = cin; cin = ti;
    npoly = nclip;
    nclip = 0;
  }
  int m = npoly;
  npoly = 0;
  for (int i = 0; i < m; i++)
    if (dt3(poly[i] - face1[0], n) <= 0.f) {
      if (npoly != i) { poly[npoly] = poly[i]; ptag[npoly][0] = ptag[i][0]; ptag[npoly][1] = ptag[i][1]; }
      npoly++;
    }
  if (npoly < 1) return 0;
  if (nface2 == 2 && npoly > 2) {
    int b1 = 0, b2 = 1;
    float max_d = 0.f;
    for (int i = 0; i < npoly; i++)
      for (int j = i + 1; j < npoly; j++) {
        f3 df = poly[j] - poly[i];
        float d2 = dt3(df, df);
        if (d2 > max_d) { max_d = d2; b1 = i; b2 = j; }
      }
    witness_on_face(poly[b1], face1[0], n, dir, &w1o[0], &w2o[0], &dists[0]);
    witness_on_face(poly[b2], face1[0], n, dir, &w1o[1], &w2o[1], &dists[1]);
    mb.otag[0][0] = ptag[b1][0]; mb.otag[0][1] = ptag[b1][1]; mb.otag[1][0] = ptag[b2][0]; mb.otag[1][1] = ptag[b2][1];
    return 2;
  }
  if (npoly > 4) {
    int q[4];
    polygon_quad(poly, npoly, q);
    for (int i = 0; i < 4; i++) {
      witness_on_face(poly[q[i]], face1[0], n, dir, &w1o[i], &w2o[i], &dists[i]);
      mb.otag[i][0] = ptag[q[i]][0]; mb.otag[i][1] = ptag[q[i]][1];
    }
    return 4;
  }
  for (int i = 0; i < npoly; i++) {
    witness_on_face(poly[i], face1[0], n, dir, &w1o[i], &w2o[i], &dists[i]);
    mb.otag[i][0] = ptag[i][0]; mb.otag[i][1] = ptag[i][1];
  }
  return npoly;
}

// MuJoCo Warp's multicontact: up to 4 contacts (witnesses w1, w2, distances) from EPA face fidx; returns the count
// (1: the EPA witnesses x1, x2 alone)
__device__ int multicontact(MultiBuf& mb, const Polytope& pt, int fidx, f3 x1, f3 x2, const CGeom& g1,
                            const CGeom& g2, f3* w1, f3* w2, float* dists) {
  w1[0] = x1; w2[0] = x2;
  mb.mcase = 0;
  int fv[3], fi1[3], fi2[3];
  f3 fx1[3], fx2[3];
  face_verts(pt.face[fidx], fv);
  int nface1 = feature_dim(fv, pt, 0, fi1, fx1), nface2 = feature_dim(fv, pt, 1, fi2, fx2);
  f3 dir = x2 - x1, dir_neg = -dir;
  int nn1 = 0, nn2 = 0;
  nn1 = box_normals(nface1, fi1, g1.R, dir_neg, mb.n1, mb.idx1);  // (boxes only in this module)
  nn2 = box_normals(nface2, fi2, g2.R, dir, mb.n2, mb.idx2);
  int edge1 = 0, edge2 = 0, res[2] = {0, 0};
  int nres = aligned_faces(mb.n1, nn1, mb.n2, nn2, res);
  if (!nres) {
    if (nface1 < 3 && nface1 <= nface2) {
      nn1 = box_edge_normals(nface1, g1, fx1[0], fx1[1], fi1[0], mb.n1, mb.endvert);
      nres = aligned_face_edge(mb.n1, nn1, mb.n2, nn2, res);
      if (!nres) return 1;
      edge1 = 1;
    } else if (nface2 < 3) {
      nn2 = box_edge_normals(nface2, g2, fx2[0], fx2[1], fi2[0], mb.n2, mb.endvert);
      nres = aligned_face_edge(mb.n2, nn2, mb.n1, nn1, res);
      if (!nres) return 1;
      edge2 = 1;
    } else {
      return 1;
    }
  }
  int i = res[0], j = res[1], nf1, nf2;
  mb.mcase = edge1 ? 3 : (edge2 ? 2 : 1); mb.mi = i; mb.mj = j;
  if (edge1) { mb.face1[0] = pt.vert[2 * fv[0]]; mb.face1[1] = mb.endvert[i]; nf1 = 2; }
  else {
    int ind = edge2 ? mb.idx1[j] : mb.idx1[i];
    nf1 = box_face(g1, ind, mb.face1);
  }
  if (edge2) { mb.face2[0] = pt.vert[2 * fv[0] + 1]; mb.face2[1] = mb.endvert[i]; nf2 = 2; }
  else nf2 = box_face(g2, mb.idx2[j], mb.face2);
  if (edge1) {
    f3 c1[4], c2[4];
    int n = polygon_clip(mb, mb.face2, nf2, mb.face1, nf1, mb.n2[j], -mb.n2[j], c1, c2, dists);
    for (int k = 0; k < n; k++) { w1[k] = c2[k]; w2[k] = c1[k]; }  // (flipped back)
    return n;
  }
  if (edge2) return polygon_clip(mb, mb.face1, nf1, mb.face2, nf2, mb.n1[j], -mb.n1[j], w1, w2, dists);
  return polygon_clip(mb, mb.face1, nf1, mb.face2, nf2, mb.n1[i], mb.n2[j], w1, w2, dists);
}


// ----------------------------------------------------------------------------------------- box-box (ccd_kernel)
// A pair's contacts before they enter a world's table (up to 4: MuJoCo Warp's multicontact)
#ifndef CSINK_DEFINED
#define CSINK_DEFINED
#define SINK_MAX 8
struct CSink {
  int n, unsupported;
  int slot[SINK_MAX];
  float dist[SINK_MAX], pos[SINK_MAX][3], fr[SINK_MAX][9];
};
__device__ __forceinline__ void sink_put(CSink& o, int slot, float dist, f3 pc, const float* frame) {
  if (o.n >= SINK_MAX) { o.unsupported |= 4; return; }
  o.slot[o.n] = slot; o.dist[o.n] = dist;
  o.pos[o.n][0] = pc.x; o.pos[o.n][1] = pc.y; o.pos[o.n][2] = pc.z;
  for (int k = 0; k < 9; k++) o.fr[o.n][k] = frame[k];
  o.n++;
}
#endif

__device__ __forceinline__ CGeom cgeom_box(const float* p, const float* R, const float* s) {
  CGeom c;
  c.pos = ld3(p);
  for (int k = 0; k < 9; k++) c.R[k] = R[k];
  c.size = ld3(s);
  c.type = 3;
  c.index = -1;
  return c;
}

// How a contact depends on the boxes' poses (ccd_box_box's wit, for the adjoint's exact VJP): own 0 / 1 = geom 1 / 2;
// vectors in ccd_box_box's frame (geom 1's centre at the origin; c1, c2 the centres)
//   kind 1 (a clipped point): v the intersection of the planes g[0..2] . (x - x_r) = 0, each of a box (own[r]);
//   kind 2 (EPA, a vertex in a face): v a vertex of box own[0];
//   both: dist = nd . (v - xd) (nd, xd of box od), pos = v + dist dir / 2, the normal nsgn dir (dir of box odir);
//   kind 3 (EPA, edge-edge): x1 = a1 + s e1 on box 1, x2 = a2 + t e2 on box 2 closest, dist = -|x1 - x2|, the
//   normal normalize(x1 - x2), pos their midpoint; kind 4 (other EPA features): x1 = a1, x2 = a2 as points of their
//   boxes (approximate)
struct CcdWit {
  int kind;
  f3 v, g[3];
  int own[3];
  f3 nd, xd, dir;
  int od, odir;
  float nsgn;
  f3 a1, e1, a2, e2, c1, c2;
};

// the plane of tag t (MultiBuf's provenance) for kind-1 points: its normal and box
__device__ __forceinline__ void tag_plane(int t, int sel, const f3* subj, int nsubj, f3 nsub, int osub, const f3* cf,
                                          int ncf, f3 ncl, int ocl, const CGeom& gs, f3* g, int* own) {
  if (t < 8) { *g = cr3(subj[(t + 1) % nsubj] - subj[t], nsub); *own = osub; return; }
  if (t < 16) { const int e = t - 8; *g = cr3(cf[(e + 1) % ncf] - cf[e], ncl); *own = ocl; return; }
  if (t < 24) { *g = subj[1] - subj[0]; *own = osub; return; }
  // the segment's line: the subject box's two axes most perpendicular to it (sel 0 / 1)
  f3 d = subj[1] - subj[0];
  float c[3];
  for (int k = 0; k < 3; k++) c[k] = fabsf(gs.R[k] * d.x + gs.R[3 + k] * d.y + gs.R[6 + k] * d.z);
  int ax = (c[0] >= c[1] && c[0] >= c[2]) ? 0 : (c[1] >= c[2] ? 1 : 2);
  const int k = (ax + 1 + sel) % 3;
  *g = mk3(gs.R[k], gs.R[3 + k], gs.R[6 + k]);
  *own = osub;
}

// box-box as MuJoCo Warp's ccd_kernel (native CCD: GJK, EPA with EPA_ITERATIONS, then multicontact, which MuJoCo Warp
// runs for every box-box pair; margin and gap 0): the contacts with dist < 0 into o, their slots the multicontact's
// points (0 .. 3), their frame from the first one's witnesses. wit (null: none): the dependence of contact `slot`
// (the replay of the adjoint; the same arithmetic, so the same contacts)
__device__ __noinline__ void ccd_box_box(const float* p1, const float* R1, const float* s1, const float* p2,
                                         const float* R2, const float* s2, CSink& o, int slot = -1,
                                         CcdWit* wit = nullptr) {
  o.n = 0; o.unsupported = 0;
  CGeom g1 = cgeom_box(p1, R1, s1), g2 = cgeom_box(p2, R2, s2);
  const float tol = CCD_TOLERANCE;
  f3 origin = g1.pos;  // (translation invariance: geom 1 at the origin)
  g1.pos = mk3(0.f, 0.f, 0.f);
  g2.pos = g2.pos - origin;
  GJKResult r;
  cv_gjk(tol, GJK_ITERATIONS, g1, g2, g1.pos, g2.pos, 0.f, true, r);
  g1.index = r.index1; g2.index = r.index2;
  if (r.dist > tol || r.dim < 2 || r.separated) return;  // (no penetration depth to recover)
  Polytope pt;
  pt.nface = 0; pt.nvert = 0; pt.nhorizon = 0; pt.status = 0;
  if (r.dim == 2) { polytope2(pt, r, g1, g2); if (pt.status == -1) r.dim = 3; }
  else if (r.dim == 4) { polytope4(pt, r); if (pt.status == -1) r.dim = 3; }
  if (r.dim == 3) { pt.nface = 0; pt.nvert = 0; pt.nhorizon = 0; polytope3(pt, r, g1, g2); }
  if (pt.status) return;  // (origin on the boundary: dist = the GJK distance, not penetrating)
  const int fidx = cv_epa(tol, EPA_ITERATIONS, pt, g1, g2, true);
  if (fidx == -1) return;
  f3 x1, x2;
  const float dist = epa_witness(pt, fidx, &x1, &x2);
  if (!(dist < 0.f)) return;
  f3 w1[4], w2[4];
  float dists[4] = {0.f, 0.f, 0.f, 0.f};
  MultiBuf mb;
  int nc = multicontact(mb, pt, fidx, x1, x2, g1, g2, w1, w2, dists);
  const bool clipped = nc >= 1 && mb.mcase != 0;
  if (nc < 1) { nc = 1; w1[0] = x1; w2[0] = x2; }
  float nrm[3], fr[9];
  f3 nv = nz3(w1[0] - w2[0]);
  nrm[0] = nv.x; nrm[1] = nv.y; nrm[2] = nv.z;
  make_frame(nrm, fr);
  for (int i = 0; i < nc; i++) {
    const float di = nc > 1 ? dists[i] : dist;
    if (di < 0.f) sink_put(o, i, di, 0.5f * (w1[i] + w2[i]) + origin, fr);
  }
  if (!wit || slot < 0 || slot >= nc) return;
  CcdWit& W = *wit;
  W.c1 = g1.pos; W.c2 = g2.pos;
  if (clipped) {  // kind 1: the clipped point `slot`
    const int cs = mb.mcase, i = mb.mi, j = mb.mj;
    const f3* cf = cs == 3 ? mb.face2 : mb.face1;  // the clipping face
    const f3* sb = cs == 3 ? mb.face1 : mb.face2;  // the subject polygon / segment
    const int ocl = cs == 3 ? 1 : 0, osub = 1 - ocl, nsub = cs == 1 ? 4 : 2;
    const f3 ncl = cs == 1 ? mb.n1[i] : (cs == 2 ? mb.n1[j] : mb.n2[j]);
    const f3 nsb = cs == 1 ? mb.n2[j] : mk3(0.f, 0.f, 0.f);
    const CGeom& gs = osub ? g2 : g1;
    W.kind = 1;
    W.v = cs == 3 ? w1[slot] : w2[slot];
    const int t0 = mb.otag[slot][0], t1 = mb.otag[slot][1];
    if (cs == 1) {
      W.g[0] = nsb; W.own[0] = osub;
      tag_plane(t0, 0, sb, nsub, nsb, osub, cf, 4, ncl, ocl, gs, &W.g[1], &W.own[1]);
      tag_plane(t1, 0, sb, nsub, nsb, osub, cf, 4, ncl, ocl, gs, &W.g[2], &W.own[2]);
    } else {
      tag_plane(24, 0, sb, nsub, nsb, osub, cf, 4, ncl, ocl, gs, &W.g[0], &W.own[0]);
      tag_plane(24, 1, sb, nsub, nsb, osub, cf, 4, ncl, ocl, gs, &W.g[1], &W.own[1]);
      tag_plane(t0 == 24 ? t1 : t0, 0, sb, nsub, nsb, osub, cf, 4, ncl, ocl, gs, &W.g[2], &W.own[2]);
    }
    W.nd = ncl; W.xd = cf[0]; W.od = ocl;
    W.dir = cs == 1 ? mb.n2[j] : -ncl;
    W.odir = cs == 1 ? 1 : ocl;
    W.nsgn = cs == 3 ? 1.f : -1.f;
    return;
  }
  int fv[3], fi1[3], fi2[3];
  f3 fx1[3], fx2[3];
  face_verts(pt.face[fidx], fv);
  const int d1 = feature_dim(fv, pt, 0, fi1, fx1), d2 = feature_dim(fv, pt, 1, fi2, fx2);
  f3 nn[3];
  int ii[3];
  if (d1 == 3 && d2 == 1 && box_normals(3, fi1, g1.R, x1 - x2, nn, ii) >= 1) {  // a vertex of box 2 in a face of box 1
    W.kind = 2; W.v = fx2[0]; W.own[0] = 1; W.nd = nn[0]; W.xd = fx1[0]; W.od = 0;
    W.dir = -nn[0]; W.odir = 0; W.nsgn = -1.f;
  } else if (d1 == 1 && d2 == 3 && box_normals(3, fi2, g2.R, x2 - x1, nn, ii) >= 1) {  // a vertex of box 1 in a face of 2
    W.kind = 2; W.v = fx1[0]; W.own[0] = 0; W.nd = nn[0]; W.xd = fx2[0]; W.od = 1;
    W.dir = -nn[0]; W.odir = 1; W.nsgn = 1.f;
  } else if (d1 == 2 && d2 == 2) {  // edge-edge
    W.kind = 3; W.a1 = fx1[0]; W.e1 = fx1[1] - fx1[0]; W.a2 = fx2[0]; W.e2 = fx2[1] - fx2[0];
  } else {
    W.kind = 4; W.a1 = x1; W.a2 = x2;
  }
}
"""


ADJ_DEVICE = r"""
// ccd_box_box's contact `slot` again (dist, pos, fr: the forward's bits) and, with cot = (dbar, pbar[3], frbar[9]), its
// exact VJP onto the boxes' poses: gb = (xbar1, tbar1, xbar2, tbar2), forces on the centres and torques about them,
// from the contact's dependence (CcdWit): a point of a box p moves as dp = dx + dw x (p - c), a vector of it as dw x u.
// Kind 4 (EPA features other than vertex-face and edge-edge) takes its witnesses as points of their boxes (not exact).
// Returns the CcdWit kind (0: the slot is not a contact)
__device__ __noinline__ int ccd_box_box_vjp(const float* p1, const float* R1, const float* s1, const float* p2,
                                            const float* R2, const float* s2, int slot, const float* cot, float* gb,
                                            float* dist, float* pos, float* fr) {
  CSink o;
  CcdWit W;
  W.kind = 0;
  ccd_box_box(p1, R1, s1, p2, R2, s2, o, slot, &W);
  int c = -1;
  for (int i = 0; i < o.n; i++) if (o.slot[i] == slot) c = i;
  if (c < 0) return 0;
  *dist = o.dist[c];
  for (int k = 0; k < 3; k++) pos[k] = o.pos[c][k];
  for (int k = 0; k < 9; k++) fr[k] = o.fr[c][k];
  if (!cot) return W.kind;
  f3 F[2] = {mk3(0.f, 0.f, 0.f), mk3(0.f, 0.f, 0.f)}, T[2] = {mk3(0.f, 0.f, 0.f), mk3(0.f, 0.f, 0.f)};
  const f3 C[2] = {W.c1, W.c2};
  auto point = [&](int own, f3 x, f3 b) { F[own] = F[own] + b; T[own] = T[own] + cr3(x - C[own], b); };
  auto vec = [&](int own, f3 u, f3 b) { T[own] = T[own] + cr3(u, b); };
  float nb[3];
  make_frame_vjp(fr, cot + 4, nb);
  const f3 nbv = ld3(nb), pbv = ld3(cot + 1);
  const float dbv = cot[0], d = *dist;
  if (W.kind == 1 || W.kind == 2) {
    // pos = v + d dir / 2, the normal nsgn dir, dist = nd . (v - xd)
    f3 vbar = pbv, dirbar = (0.5f * d) * pbv + W.nsgn * nbv;
    const float dtot = dbv + 0.5f * dt3(W.dir, pbv);
    vbar = vbar + dtot * W.nd;
    vec(W.odir, W.dir, dirbar);
    vec(W.od, W.nd, dtot * (W.v - W.xd));
    point(W.od, W.xd, -dtot * W.nd);
    if (W.kind == 2) {
      point(W.own[0], W.v, vbar);
    } else {  // A v = b with the rows g_r (planes of their boxes): lambda = A^-T vbar, plane r gets lambda_r g_r at v
      const f3 a = W.g[0], b = W.g[1], cc = W.g[2];
      const f3 bc = cr3(b, cc), ca = cr3(cc, a), ab = cr3(a, b);
      const float det = dt3(a, bc);
      if (fabsf(det) > 1e-12f * nm3(a) * nm3(b) * nm3(cc)) {
        // A^-1 = [bc ca ab] / det (columns), so A^-T vbar = (bc . vbar, ca . vbar, ab . vbar) / det
        const float l0 = dt3(bc, vbar) / det, l1 = dt3(ca, vbar) / det, l2 = dt3(ab, vbar) / det;
        point(W.own[0], W.v, l0 * a);
        point(W.own[1], W.v, l1 * b);
        point(W.own[2], W.v, l2 * cc);
      } else {
        point(W.own[0], W.v, vbar);  // (degenerate planes: the point as one of its first plane's box)
      }
    }
  } else {
    f3 x1 = W.a1, x2 = W.a2, e1 = W.e1, e2 = W.e2;
    float s = 0.f, t = 0.f, A = 0.f, B = 0.f, Cc = 0.f;
    if (W.kind == 3) {  // the lines' closest points
      const f3 r = W.a1 - W.a2;
      A = dt3(e1, e1); B = dt3(e1, e2); Cc = dt3(e2, e2);
      const float den = B * B - A * Cc;
      s = (Cc * dt3(e1, r) - B * dt3(e2, r)) / den;
      t = (B * dt3(e1, r) - A * dt3(e2, r)) / den;
      x1 = W.a1 + s * e1; x2 = W.a2 + t * e2;
    }
    const f3 w = x1 - x2;
    const float L = nm3(w);
    const f3 n = L > 0.f ? w / L : mk3(0.f, 0.f, 0.f);
    // dist = -|w|, the normal w / |w|, pos = (x1 + x2) / 2
    const f3 wbar = (-dbv) * n + (L > 0.f ? (nbv - dt3(n, nbv) * n) / L : mk3(0.f, 0.f, 0.f));
    const f3 x1b = wbar + 0.5f * pbv, x2b = 0.5f * pbv - wbar;
    if (W.kind == 3) {
      // e1 . w = e2 . w = 0 at (s, t): mu = -J^-T (x1b . e1, x2b . e2), J = [[A, -B], [B, -C]]
      const float c1 = dt3(x1b, e1), c2 = dt3(x2b, e2), dj = -A * Cc + B * B;
      // J^T = [[A, B], [-B, -C]]: J^T y = c
      const float y1 = (-Cc * c1 - B * c2) / dj, y2 = (B * c1 + A * c2) / dj;
      const float m1 = -y1, m2 = -y2;
      const f3 q = m1 * e1 + m2 * e2;
      point(0, W.a1, x1b + q);
      vec(0, e1, s * x1b + m1 * w + s * q);
      point(1, W.a2, x2b - q);
      vec(1, e2, t * x2b + m2 * w - t * q);
    } else {
      point(0, x1, x1b);
      point(1, x2, x2b);
    }
  }
  gb[0] = F[0].x; gb[1] = F[0].y; gb[2] = F[0].z; gb[3] = T[0].x; gb[4] = T[0].y; gb[5] = T[0].z;
  gb[6] = F[1].x; gb[7] = F[1].y; gb[8] = F[1].z; gb[9] = T[1].x; gb[10] = T[1].y; gb[11] = T[1].z;
  return W.kind;
}
"""
