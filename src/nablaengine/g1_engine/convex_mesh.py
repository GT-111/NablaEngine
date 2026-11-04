"""MuJoCo 3.17 mesh narrowphase and specialized box-box contacts.

The CUDA and CPU geometry checks compile the same source.  MuJoCo is used here
only to read a compiled hull; collision never calls MuJoCo.  A candidate owns
one warp and caller-provided scratch.  The current scalar control path runs on
lane zero, so scratch must be in global or shared memory, not a thread stack.

The final EPA support face is the feature tape.  Its witness recovery and
native surface clipping are replayed with analytic directional derivatives.
Indices and branch choices are discrete; normals, witnesses and the complete
contact frame are differentiated on that branch.

Box-box uses the dedicated native separating-axis/clipping algorithm, including
all up-to-eight face contacts. Its feature records the selected axis, incident
face or supporting edges, and segment clamp choices. Boundary flags diagnose
selected axis ties, clipping/clamp equalities and ambiguous support signs; they
do not assert a global derivative at a change of contact features.

Third-party portions below are adapted from MuJoCo 3.17.0, engine_collision_gjk.c
and the mesh/box supports in engine_collision_convex.c. Copyright 2021, 2024
DeepMind Technologies Limited. Licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0 . See the retained notice in SOURCE.
Modified: compiled-hull tables, float32 geometry replay, iterative EPA horizon traversal,
caller-owned bounded scratch, and feature-based analytic contact derivatives.
The specialized box-box source is adapted from engine_collision_box.c,
Copyright 2016 Svetoslav Kolev, under the same Apache License, Version 2.0.
"""

import mujoco
import numpy as np


def _tables(m, geoms):
    meshes = sorted({int(m.geom_dataid[g]) for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH})
    mesh_index = {mesh: i for i, mesh in enumerate(meshes)}
    bounds = np.asarray(m.geom_size[geoms], np.float32).copy()
    for i, g in enumerate(geoms):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            bounds[i] = np.max(np.abs(np.asarray(m.mesh_vert[va:va + vn], np.float32)), axis=0)
    info, vert, graph, extrema, mapadr, mapnum, polymap = [], [], [], [], [], [], []
    polyadr, polynum, polyvert, normal = [], [], [], []
    for mesh in meshes:
        va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        pa, pn = int(m.mesh_polyadr[mesh]), int(m.mesh_polynum[mesh])
        ga = int(m.mesh_graphadr[mesh])
        if ga < 0 or pn == 0:
            raise ValueError("native convex contacts require the compiled mesh hull and polygon map")
        graph_end = min((int(a) for a in m.mesh_graphadr if a > ga), default=len(m.mesh_graph))
        info.extend((len(vert) // 3, vn, len(polyadr), pn, len(graph), len(extrema)))
        vert.extend(np.asarray(m.mesh_vert[va:va + vn]).ravel())
        graph.extend(m.mesh_graph[ga:graph_end])
        extrema.extend(np.asarray(m.mesh_extrema[mesh]).ravel())
        for v in range(va, va + vn):
            a, n = int(m.mesh_polymapadr[v]), int(m.mesh_polymapnum[v])
            mapadr.append(len(polymap))
            mapnum.append(n)
            polymap.extend(m.mesh_polymap[a:a + n])
        for p in range(pa, pa + pn):
            a, n = int(m.mesh_polyvertadr[p]), int(m.mesh_polyvertnum[p])
            polyadr.append(len(polyvert))
            polynum.append(n)
            polyvert.extend(m.mesh_polyvert[a:a + n])
        normal.extend(np.asarray(m.mesh_polynormal[pa:pa + pn]).ravel())
    return [
        ("cm_type", "i", m.geom_type[geoms]),
        ("cm_mesh", "i", [mesh_index.get(int(m.geom_dataid[g]), -1)
                            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else -1 for g in geoms]),
        ("cm_size", "f", m.geom_size[geoms]),
        ("cm_bound", "f", bounds),
        ("cm_info", "i", info), ("cm_vert", "f", vert), ("cm_graph", "i", graph),
        ("cm_extrema", "i", extrema), ("cm_mapadr", "i", mapadr), ("cm_mapnum", "i", mapnum),
        ("cm_polymap", "i", polymap), ("cm_polyadr", "i", polyadr), ("cm_polynum", "i", polynum),
        ("cm_polyvert", "i", polyvert), ("cm_normal", "f", normal),
    ]


def constants(m, geoms):
    """Float32 packed arrays and ``(symbol, kind, count)`` in compact-geom order."""
    arrays, layout = [], []
    for name, kind, value in _tables(m, list(geoms)):
        value = np.asarray(value, np.float32).ravel()
        if not len(value):
            value = np.zeros(1, np.float32)
        arrays.append(value)
        layout.append((name, kind, len(value)))
    return np.concatenate(arrays), layout


def _align8(n):
    return (n + 7) // 8 * 8


def scratch_bytes(m):
    """One candidate's global/shared scratch, including the analytic replay."""
    iterations = int(m.opt.ccd_iterations)
    polygon, degree = max(16, m.npolygonmax), max(3, m.nmeshdegmax)
    edges = 24 + max(0, iterations - 35)
    # Native double-precision Vertex is 80B, Face is 56B; pointers are 8B.
    epa = sum(map(_align8, (80 * (5 + iterations), 56 * 6 * iterations,
                           8 * 6 * iterations, 4 * edges, 4 * edges)))
    # Replay uses one dual (value, derivative) per scalar, with the native layout.
    multiccd = 2 * _align8(4 * degree) + 3 * _align8(8 * 3 * degree)
    multiccd += 2 * _align8(8 * 3 * polygon) + _align8(8 * 16 * polygon)
    return max(epa, multiccd) + _align8(4 * 4 * 6 * iterations)


def header(m, geoms):
    """Tables, fixed tape ABI, and bounded scratch declarations for this model."""
    geoms = list(geoms)
    native_geoms = [g for g in geoms if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_MESH,
                                                        mujoco.mjtGeom.mjGEOM_BOX)]
    if np.any(m.geom_margin[native_geoms]) or np.any(m.geom_gap[native_geoms]):
        raise ValueError("native contacts require the authored zero geom margins and gaps")
    if any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in geoms):
        if int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD):
            raise ValueError("convex_mesh implements the selected native CCD algorithm")
    _, layout = constants(m, geoms)
    definitions = "\n".join(f'DKC {"int" if kind == "i" else "float"} {name}[{count}];'
                              for name, kind, count in layout)
    return f"""
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#ifndef CM_DEVICE
#ifdef __CUDACC__
#define CM_DEVICE __device__ __forceinline__
#else
#define CM_DEVICE inline
#endif
#endif
#define CM_CCD_ITERS {int(m.opt.ccd_iterations)}
#define CM_CCD_TOL {float(m.opt.ccd_tolerance):.17e}
#define CM_POLYGON_MAX {max(16, m.npolygonmax)}
#define CM_DEGREE_MAX {max(3, m.nmeshdegmax)}
#define CM_MULTI {int(not (int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)))}
#define CM_SCRATCH_BYTES {scratch_bytes(m)}
#define CM_FEATURE_WORDS 12
#define CM_REPLAY_SLOTS {8 if sum(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in geoms) >= 2 else 4}
{definitions}
struct CMFeature {{
  int support1[3], support2[3];
  int nsupp1, nsupp2, slot, count, outward_sign, reserved;
}};
// Box features (reserved=1): support1=(SAT code, normal sign, incident axis or
// edge1 corner bits), support2=(incident sign or edge2 corner bits, clamp bits,
// boundary diagnostics). nsupp1/nsupp2/outward_sign are unused for this tag.
enum {{ CM_BOX_AXIS_TIE=1, CM_BOX_CLIP_BOUNDARY=2, CM_BOX_CLAMP_BOUNDARY=4,
        CM_BOX_SUPPORT_AMBIGUOUS=8, CM_BOX_ZERO_DEPTH=16, CM_BOX_ORIENTATION_BOUNDARY=32,
        CM_BOX_INCIDENT_AXIS_TIE=64, CM_BOX_DEDUPE_BOUNDARY=128 }};
static_assert(sizeof(CMFeature) == 4 * CM_FEATURE_WORDS, "contact feature tape ABI");
struct CMContact {{ float dist, pos[3], frame[9]; CMFeature feature; }};
struct CMManifold {{ int count, status; CMContact contact[8]; }};
// One selected manifold's geometry and twelve pose-tangent derivatives. The
// adjoint reuses it across recorded slots; it is never part of the forward tape.
struct CMLinearization {{
  float value[CM_REPLAY_SLOTS][13], jacobian[CM_REPLAY_SLOTS][12][13];
}};
struct alignas(8) CMScratch {{ unsigned char bytes[CM_SCRATCH_BYTES]; }};
CM_DEVICE void convex_manifold(int, int, const float*, const float*, CMScratch*, CMManifold*);
CM_DEVICE int convex_contact_replay(int, int, const float*, const float*, const CMFeature&,
                                    CMScratch*, CMContact*);
CM_DEVICE int convex_contact_vjp(int, int, const float*, const float*, const CMFeature&,
                                 float, const float*, const float*, CMScratch*, float*);
CM_DEVICE void box_box_manifold(int, int, const float*, const float*, CMScratch*, CMManifold*);
CM_DEVICE int box_box_contact_replay(int, int, const float*, const float*, const CMFeature&,
                                     CMScratch*, CMContact*);
CM_DEVICE int box_box_contact_vjp(int, int, const float*, const float*, const CMFeature&,
                                  float, const float*, const float*, CMScratch*, float*);
CM_DEVICE int convex_manifold_linearize(int, int, const float*, const float*, const CMFeature&,
                                        CMScratch*, CMLinearization*);
CM_DEVICE int box_box_manifold_linearize(int, int, const float*, const float*, const CMFeature&,
                                         CMScratch*, CMLinearization*);
CM_DEVICE void contact_linearization_vjp(const CMLinearization&, int, float, const float*, const float*, float*);
"""


def source(m=None, geoms=None):
    """The portable device implementation (compile after :func:`header`)."""
    return SOURCE


SOURCE = r"""
// Copyright 2021, 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Ported narrowphase only from MuJoCo 3.17.0:
// https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_gjk.c
// https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_convex.c

namespace cm {
enum { mjGEOM_BOX = 6, mjGEOM_MESH = 7 };
template<class T> struct Obj {
  int geom, geom_type, vertindex, meshindex, nsupp;
  T size[4], pos[3], mat[9], margin, suppvert[12];
  int suppind[4];
  struct { struct {
    int nvert, mesh_polynum;
    const float *vert, *polynormal;
    const int *mpolymapadr, *mpolymapnum, *polymap, *polyvertadr, *polyvertnum, *polyvert;
    const int *graph, *extrema;
  } mesh; } data;
};
struct Dual {
  float v, d;
  CM_DEVICE Dual(float value = 0.f, float derivative = 0.f) : v(value), d(derivative) {}
  CM_DEVICE explicit operator bool() const { return v != 0.f; }
};
CM_DEVICE float vadd(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a,b);
#else
  return a+b;
#endif
}
CM_DEVICE float vsub(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(a,b);
#else
  return a-b;
#endif
}
CM_DEVICE float vmul(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a,b);
#else
  return a*b;
#endif
}
CM_DEVICE Dual operator+(Dual a, Dual b) { return {vadd(a.v,b.v), a.d + b.d}; }
CM_DEVICE Dual operator-(Dual a, Dual b) { return {vsub(a.v,b.v), a.d - b.d}; }
CM_DEVICE Dual operator-(Dual a) { return {-a.v, -a.d}; }
CM_DEVICE Dual operator*(Dual a, Dual b) { return {vmul(a.v,b.v), a.d * b.v + a.v * b.d}; }
CM_DEVICE Dual operator/(Dual a, Dual b) {
  float value = a.v / b.v;
  return {value, (a.d - value * b.d) / b.v};
}
CM_DEVICE Dual& operator+=(Dual& a, Dual b) { a = a + b; return a; }
CM_DEVICE Dual& operator-=(Dual& a, Dual b) { a = a - b; return a; }
CM_DEVICE Dual& operator*=(Dual& a, Dual b) { a = a * b; return a; }
CM_DEVICE bool operator<(Dual a, Dual b) { return a.v < b.v; }
CM_DEVICE bool operator>(Dual a, Dual b) { return a.v > b.v; }
CM_DEVICE bool operator<=(Dual a, Dual b) { return a.v <= b.v; }
CM_DEVICE bool operator>=(Dual a, Dual b) { return a.v >= b.v; }
CM_DEVICE bool operator==(Dual a, Dual b) { return a.v == b.v; }
CM_DEVICE bool operator!=(Dual a, Dual b) { return a.v != b.v; }
template<class T> CM_DEVICE T number(float value, float derivative) { return T(value, derivative); }
template<> CM_DEVICE float number<float>(float value, float derivative) { return value; }
template<> CM_DEVICE double number<double>(float value, float derivative) { return value; }
template<class T> CM_DEVICE T absn(T x) { return x < T(0) ? -x : x; }
template<class T> CM_DEVICE T sqrtn(T x);
template<> CM_DEVICE float sqrtn<float>(float x) { return sqrtf(x); }
template<> CM_DEVICE double sqrtn<double>(double x) { return sqrt(x); }
template<> CM_DEVICE Dual sqrtn<Dual>(Dual x) {
  float value = sqrtf(x.v);
  return {value, x.d / (2.f * value)};
}
template<class T> CM_DEVICE T normalize3(T* x) {
  T norm = sqrtn<T>(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  if (norm < T(1e-15f)) { x[0] = 1; x[1] = 0; x[2] = 0; return 0; }
  T scale = T(1) / norm;
  x[0] *= scale; x[1] *= scale; x[2] *= scale;
  return norm;
}
CM_DEVICE size_t align8(size_t n) { return ((n + 7) / 8) * 8; }

CM_DEVICE void quatmat(const float* q, float* r) {
  float q00=vmul(q[0],q[0]), q01=vmul(q[0],q[1]), q02=vmul(q[0],q[2]), q03=vmul(q[0],q[3]);
  float q11=vmul(q[1],q[1]), q12=vmul(q[1],q[2]), q13=vmul(q[1],q[3]);
  float q22=vmul(q[2],q[2]), q23=vmul(q[2],q[3]), q33=vmul(q[3],q[3]);
  r[0]=vsub(vsub(vadd(q00,q11),q22),q33); r[4]=vsub(vadd(vsub(q00,q11),q22),q33);
  r[8]=vadd(vsub(vsub(q00,q11),q22),q33);
  r[1]=vmul(2.f,vsub(q12,q03)); r[2]=vmul(2.f,vadd(q13,q02)); r[3]=vmul(2.f,vadd(q12,q03));
  r[5]=vmul(2.f,vsub(q23,q01)); r[6]=vmul(2.f,vsub(q13,q02)); r[7]=vmul(2.f,vadd(q23,q01));
}
// Bounds enclose the actual packed vertices under the same float rotation as
// native narrowphase. Directed rounding retains touching pairs, including the
// rounding of a transformed vertex's world translation.
CM_DEVICE float add_up(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fadd_ru(a,b);
#else
  double exact=double(a)+double(b); float result=float(exact);
  return double(result)<exact ? nextafterf(result,HUGE_VALF) : result;
#endif
}
CM_DEVICE float mul_up(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fmul_ru(a,b);
#else
  double exact=double(a)*double(b); float result=float(exact);
  return double(result)<exact ? nextafterf(result,HUGE_VALF) : result;
#endif
}
CM_DEVICE float sub_down(float a,float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rd(a,b);
#else
  double exact=double(a)-double(b); float result=float(exact);
  return double(result)>exact ? nextafterf(result,-HUGE_VALF) : result;
#endif
}
CM_DEVICE void world_aabb(int g,const float* x,const float* q,float* bounds) {
  float r[9]; quatmat(q,r);
  for(int k=0;k<3;k++) {
    float extent=mul_up(fabsf(r[3*k]),cm_bound[3*g]);
    extent=add_up(extent,mul_up(fabsf(r[3*k+1]),cm_bound[3*g+1]));
    extent=add_up(extent,mul_up(fabsf(r[3*k+2]),cm_bound[3*g+2]));
    bounds[k]=sub_down(x[k],extent);
    bounds[k+3]=add_up(x[k],extent);
  }
}
CM_DEVICE bool aabb_overlap(const float* a,const float* b) {
  return a[0]<=b[3] && b[0]<=a[3] && a[1]<=b[4] && b[1]<=a[4] && a[2]<=b[5] && b[2]<=a[5];
}
template<class T>
CM_DEVICE void init(Obj<T>* o, int g, const float* gx, const float* gq, int seed = -1) {
  o->geom=g; o->geom_type=cm_type[g]; o->vertindex=o->meshindex=-1; o->nsupp=0; o->margin=0;
  float r[9]; quatmat(gq+4*g,r);
  for(int k=0;k<4;k++) o->size[k]=(k<3 ? cm_size[3*g+k] : 0.f);
  for(int k=0;k<3;k++) o->pos[k]=number<T>(gx[3*g+k],seed==k ? 1.f : 0.f);
  for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
    float dr=0.f;
    if(seed>=3 && seed<6) {
      int axis=seed-3;
      if(i==(axis+1)%3) dr=-r[3*((axis+2)%3)+j];
      if(i==(axis+2)%3) dr= r[3*((axis+1)%3)+j];
    }
    o->mat[3*i+j]=number<T>(r[3*i+j],dr);
  }
  int mesh=cm_mesh[g];
  if(mesh>=0) {
    const int* info=cm_info+6*mesh;
    o->data.mesh.nvert=info[1]; o->data.mesh.mesh_polynum=info[3];
    o->data.mesh.vert=cm_vert+3*info[0];
    o->data.mesh.mpolymapadr=cm_mapadr+info[0]; o->data.mesh.mpolymapnum=cm_mapnum+info[0];
    o->data.mesh.polymap=cm_polymap; o->data.mesh.polyvertadr=cm_polyadr+info[2];
    o->data.mesh.polyvertnum=cm_polynum+info[2]; o->data.mesh.polyvert=cm_polyvert;
    o->data.mesh.polynormal=cm_normal+3*info[2];
    o->data.mesh.graph=cm_graph+info[4]; o->data.mesh.extrema=cm_extrema+info[5];
  }
}
} // namespace cm
namespace cmgeom {
using cm::Obj;
using cm::absn;
using cm::sqrtn;
using cm::normalize3;
using cm::align8;
enum { mjGEOM_BOX=6, mjGEOM_MESH=7 };
#define CM_MINVAL 1e-15f
#define CM_FACE_TOL 0.996f
#define CM_EDGE_TOL 0.0888f
template<class T> CM_DEVICE void add3(T res[3], const T v1[3], const T v2[3]) {
  res[0] = v1[0] + v2[0], res[1] = v1[1] + v2[1], res[2] = v1[2] + v2[2];
}

template<class T> CM_DEVICE void sub3(T res[3], const T v1[3], const T v2[3]) {
  res[0] = v1[0] - v2[0], res[1] = v1[1] - v2[1], res[2] = v1[2] - v2[2];
}

template<class T> CM_DEVICE T dot3(const T v1[3], const T v2[3]) {
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

template<class T> CM_DEVICE T norm3(const T v[3]) {
  return sqrtn<T>(dot3<T>(v, v));
}

template<class T> CM_DEVICE void copy3(T res[3], const T v[3]) {
  res[0] = v[0], res[1] = v[1], res[2] = v[2];
}

template<class T> CM_DEVICE void scl3(T res[3], const T v[3], T s) {
  res[0] = s*v[0], res[1] = s*v[1], res[2] = s*v[2];
}

template<class T> CM_DEVICE void addScl3(T res[3], const T v1[3], const T v2[3], T s) {
  res[0] = v1[0] + s*v2[0], res[1] = v1[1] + s*v2[1], res[2] = v1[2] + s*v2[2];
}

template<class T> CM_DEVICE void cross3(T res[3], const T v1[3], const T v2[3]) {
  res[0] = v1[1]*v2[2] - v1[2]*v2[1];
  res[1] = v1[2]*v2[0] - v1[0]*v2[2];
  res[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

template<class T> CM_DEVICE int projectOriginPlane(T res[3], const T v1[3], const T v2[3],
                              const T v3[3]) {
  T diff21[3], diff31[3], diff32[3], n[3], nv, nn;
  sub3<T>(diff21, v2, v1);
  sub3<T>(diff31, v3, v1);
  sub3<T>(diff32, v3, v2);

  // n = (v1 - v2) x (v3 - v2)
  cross3<T>(n, diff32, diff21);
  nv = dot3<T>(n, v2);
  nn = dot3<T>(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > CM_MINVAL) {
    scl3<T>(res, n, nv / nn);
    return 0;
  }

  // n = (v2 - v1) x (v3 - v1)
  cross3<T>(n, diff21, diff31);
  nv = dot3<T>(n, v1);
  nn = dot3<T>(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > CM_MINVAL) {
    scl3<T>(res, n, nv / nn);
    return 0;
  }

  // n = (v1 - v3) x (v2 - v3)
  cross3<T>(n, diff31, diff32);
  nv = dot3<T>(n, v3);
  nn = dot3<T>(n, n);
  scl3<T>(res, n, nv / nn);
  return 0;
}

template<class T> CM_DEVICE void triAffineCoord(T lambda[3], const T v1[3], const T v2[3],
                           const T v3[3], const T p[3]) {
  // compute minors as in S2D
  T M_14 = v2[1]*v3[2] - v2[2]*v3[1] - v1[1]*v3[2] + v1[2]*v3[1] + v1[1]*v2[2] - v1[2]*v2[1];
  T M_24 = v2[0]*v3[2] - v2[2]*v3[0] - v1[0]*v3[2] + v1[2]*v3[0] + v1[0]*v2[2] - v1[2]*v2[0];
  T M_34 = v2[0]*v3[1] - v2[1]*v3[0] - v1[0]*v3[1] + v1[1]*v3[0] + v1[0]*v2[1] - v1[1]*v2[0];

  // exclude one of the axes with the largest projection of the simplex using the computed minors
  T M_max = 0;
  int x, y;
  T mu1 = absn<T>(M_14), mu2 = absn<T>(M_24), mu3 = absn<T>(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    x = 1;
    y = 2;
  } else if (mu2 >= mu3) {
    M_max = M_24;
    x = 0;
    y = 2;
  } else {
    M_max = M_34;
    x = 0;
    y = 1;
  }

  // C31 corresponds to the signed area of 2-simplex: (v, s2, s3)
  T C31 = p[x]*v2[y] + p[y]*v3[x] + v2[x]*v3[y]
             - p[x]*v3[y] - p[y]*v2[x] - v3[x]*v2[y];

  // C32 corresponds to the signed area of 2-simplex: (v, s1, s3)
  T C32 = p[x]*v3[y] + p[y]*v1[x] + v3[x]*v1[y]
             - p[x]*v1[y] - p[y]*v3[x] - v1[x]*v3[y];

  // C33 corresponds to the signed area of 2-simplex: (v, s1, s2)
  T C33 = p[x]*v1[y] + p[y]*v2[x] + v1[x]*v2[y]
             - p[x]*v2[y] - p[y]*v1[x] - v2[x]*v1[y];

  // compute affine coordinates
  lambda[0] = C31 / M_max;
  lambda[1] = C32 / M_max;
  lambda[2] = C33 / M_max;
}

template<class T> CM_DEVICE T area4(const T* hull, int a, int b, int c, int d) {
  T ca[3], db[3], cross[3];
  sub3<T>(ca, hull + 3*a, hull + 3*c);
  sub3<T>(db, hull + 3*b, hull + 3*d);
  cross3<T>(cross, ca, db);
  return 0.5 * norm3<T>(cross);
}

template<class T> CM_DEVICE void hull4(int res[4], const T* hull, int nhull) {
  int a = 0, b = 1, c = 2, d = 3;
  res[0] = 0, res[1] = 1, res[2] = 2, res[3] = 3;
  T m = area4<T>(hull, a, b, c, d), m_next;
  for (; a < nhull; a++) {
    while (1) {
      int d_next = (d + 1) % nhull;
      m_next = area4<T>(hull, a, b, c, d_next);
      if (m_next <= m) {
        break;
      }
      d = d_next, m = m_next;
      res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      while (1) {
        int c_next = (c + 1) % nhull;
        m_next = area4<T>(hull, a, b, c_next, d);
        if (m_next <= m) {
          break;
        }
        c = c_next, m = m_next;
        res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      }
      while (1) {
        int b_next = (b + 1) % nhull;
        m_next = area4<T>(hull, a, b_next, c, d);
        if (m_next <= m) {
          break;
        }
        b = b_next, m = m_next;
        res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      }
    }
    if (b == a) {
      b = (b + 1) % nhull;
      if (c == b) {
        c = (c + 1) % nhull;
        if (d == c) {
          d = (d + 1) % nhull;
        }
      }
    }
  }
}

template<class T> CM_DEVICE T planeNormal(T res[3], const T v1[3], const T v2[3],
                          const T n[3]) {
  T v3[3], diff1[3], diff2[3];
  add3<T>(v3, v1, n);
  sub3<T>(diff1, v2, v1);
  sub3<T>(diff2, v3, v1);
  cross3<T>(res, diff1, diff2);

  // normalize isn't needed (cancelled out), but done to avoid asymmetric rounding later on
  normalize3<T>(res);
  return dot3<T>(res, v1);
}

template<class T> CM_DEVICE int halfspace(const T a[3], const T n[3], const T p[3]) {
  T diff[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  return dot3<T>(diff, n) > -CM_MINVAL;
}

template<class T> CM_DEVICE T witnessOnFace(T w1[3], T w2[3], const T v[3],
                                   const T* p, const T n[3], const T dir[3]) {
  T d[3] = {v[0] - p[0], v[1] - p[1], v[2] - p[2]};
  addScl3<T>(w1, v, n, -dot3<T>(d, n));
  sub3<T>(d, w1, v);
  T dist = dot3<T>(d, dir);
  addScl3<T>(w2, w1, dir, -dist);
  return dist;
}

template<class T> CM_DEVICE int polygonClip(T* x1, T* x2, T* dist, int nconmax,
                       const T* face1, int nface1,
                       const T* face2, int nface2, const T n[3],
                       const T dir[3], T* buffer, int npolygonmax,
                       T margin) {
  // clipping face needs to be at least a triangle
  if (nface1 < 3) {
    return 0;
  }
  T* polygon = buffer;
  T* clipped = polygon + 6 * npolygonmax;
  T* pn = clipped + 6 * npolygonmax;
  T* pd = pn + 3 * npolygonmax;

  // compute plane normal and distance to plane for each vertex
  for (int i = 0; i < nface1 - 1; i++) {
    pd[i] = planeNormal<T>(&pn[3*i], &face1[3*i], &face1[3*i + 3], n);
  }
  pd[nface1 - 1] = planeNormal<T>(&pn[3*(nface1 - 1)], &face1[3*(nface1 - 1)], &face1[0], n);

  // reserve 2 * max_sides as max sides for a clipped polygon
  int npolygon = nface2, nclipped = 0;

  for (int i = 0; i < nface2; i++) {
    copy3<T>(polygon + 3*i, face2 + 3*i);
  }

  // clip the polygon by one edge e at a time
  for (int e = 0; e < (3 * nface1); e += 3) {
    for (int i = 0; i < npolygon; i++) {
      // get edge PQ of the polygon
      T *P = polygon + 3*i;
      T *Q = (i < npolygon - 1) ? polygon + 3*(i+1) : polygon;
      T PQ[3] = {Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]};

      // determine if P and Q are in the halfspace of the clipping edge
      int inside1 = halfspace<T>(face1 + e, pn + e, P);
      int inside2 = halfspace<T>(face1 + e, pn + e, Q);

      // PQ entirely outside the clipping edge, skip
      if (!inside1 && !inside2) {
        continue;
      }

      // edge PQ is inside the clipping edge, add Q
      if (inside1 && inside2) {
        copy3<T>(clipped + 3*nclipped++, Q);
        continue;
      }

      // add new vertex to clipped polygon where PQ intersects the clipping edge
      T tmp = dot3<T>(pn + e, PQ);
      if (tmp != 0.0) {
        T t = (pd[e/3] - dot3<T>(pn + e, P)) / tmp;
        if (t >= 0.0 && t <= 1.0) {
          addScl3<T>(clipped + 3*nclipped++, P, PQ, t);
        }
      }

      // add Q as PQ is now back inside the clipping edge
      if (inside2) {
        copy3<T>(clipped + 3*nclipped++, Q);
      }
    }

    // swap clipped and polygon
    T* tmp = polygon;
    polygon = clipped;
    clipped = tmp;
    npolygon = nclipped;
    nclipped = 0;
  }

  // prune out vertices with distance from the face exceeding margin
  int m = npolygon;
  npolygon = 0;
  for (int i = 0; i < m; i++) {
    T diff[3];
    sub3<T>(diff, polygon + 3*i, face1);
    if (dot3<T>(diff, n) <= margin) {
      if (npolygon != i) {
        copy3<T>(polygon + 3*npolygon, polygon + 3*i);
      }
      npolygon++;
    }
  }

  if (npolygon < 1) {
    return 0;
  }

  // copy final clipped polygon
  if (nconmax < 5 && npolygon > 4) {
    int idx[4];
    hull4<T>(idx, polygon, npolygon);
    for (int i = 0; i < 4; i++) {
      T d = witnessOnFace<T>(x1 + 3*i, x2 + 3*i, polygon + 3*idx[i], face1, n, dir);
      if (dist) dist[i] = d;
    }
    return 4;
  }

  // if the face is an edge, remove potential duplicates
  if (nface2 == 2 && npolygon > 2) {
    // find the two most distant vertices in the polygon
    int best1 = 0, best2 = 1;
    T d = 0;
    for (int i = 0; i < npolygon; i++) {
      for (int j = i + 1; j < npolygon; j++) {
        T diff[3];
        sub3<T>(diff, polygon + 3*j, polygon + 3*i);
        T d2 = dot3<T>(diff, diff);
        if (d2 > d) {
          d = d2;
          best1 = i;
          best2 = j;
        }
      }
    }
    T d0 = witnessOnFace<T>(x1, x2, polygon + 3*best1, face1, n, dir);
    T d1 = witnessOnFace<T>(x1 + 3, x2 + 3, polygon + 3*best2, face1, n, dir);
    if (dist) {
      dist[0] = d0;
      dist[1] = d1;
    }
    return 2;
  }

  // no pruning needed (cap to max contacts)
  npolygon = (npolygon < nconmax) ? npolygon : nconmax;
  for (int i = 0; i < npolygon; i++) {
    T d = witnessOnFace<T>(x1 + 3*i, x2 + 3*i, polygon + 3*i, face1, n, dir);
    if (dist) dist[i] = d;
  }
  return npolygon;
}

template<class T> CM_DEVICE void globalcoord(T res[3], const T mat[9], const T pos[3],
                              T l1, T l2, T l3) {
  // perform mat * (l1, l2, l3) + pos
  res[0] = mat[0]*l1 + mat[1]*l2 + mat[2]*l3;
  res[1] = mat[3]*l1 + mat[4]*l2 + mat[5]*l3;
  res[2] = mat[6]*l1 + mat[7]*l2 + mat[8]*l3;
  if (pos) {
    res[0] += pos[0];
    res[1] += pos[1];
    res[2] += pos[2];
  }
}

CM_DEVICE int intersect(int res[2], const int* arr1, const int* arr2, int n, int m) {
  int count = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (arr1[i] == arr2[j]) {
        res[count++] = arr1[i];
        if (count == 2) return 2;
      }
    }
  }
  return count;
}

template<class T> CM_DEVICE int meshNormals(T* res, int resind[3], int dim, Obj<T>* obj, const int vi[3]) {
  int v1 = vi[0], v2 = vi[1], v3 = vi[2];
  const int* polymap = obj->data.mesh.polymap;
  const float* polynormal = obj->data.mesh.polynormal;

  if (dim == 3) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    int v2_adr = obj->data.mesh.mpolymapadr[v2];
    int v2_num = obj->data.mesh.mpolymapnum[v2];

    int v3_adr = obj->data.mesh.mpolymapadr[v3];
    int v3_num = obj->data.mesh.mpolymapnum[v3];

    int edgeset[2], faceset[2];
    int n = intersect(edgeset, polymap + v1_adr, polymap + v2_adr, v1_num, v2_num);
    if (n == 0) return 0;
    n = intersect(faceset, edgeset, polymap + v3_adr, n, v3_num);
    if (n == 0) return 0;

    // three vertices on mesh define a unique face
    const float* normal = polynormal + 3*faceset[0];
    globalcoord<T>(res, obj->mat, NULL, normal[0], normal[1], normal[2]);
    resind[0] = faceset[0];
    return 1;
  }

  if (dim == 2) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    int v2_adr = obj->data.mesh.mpolymapadr[v2];
    int v2_num = obj->data.mesh.mpolymapnum[v2];

    // up to two faces as vertices on mesh define an edge
    int edgeset[2];
    int n = intersect(edgeset, polymap + v1_adr, polymap + v2_adr, v1_num, v2_num);
    if (n == 0) return 0;
    for (int i = 0; i < n; i++) {
      const float* normal = polynormal + 3*edgeset[i];
      globalcoord<T>(res + 3*i, obj->mat, NULL, normal[0], normal[1], normal[2]);
      resind[i] = edgeset[i];
    }
    return n;
  }

  if (dim == 1) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    // cap number of possible faces intersecting at a vertex
    for (int i = 0; i < v1_num; i++) {
      int index = polymap[v1_adr + i];
      const float* normal = polynormal + 3*index;
      globalcoord<T>(res + 3*i, obj->mat, NULL, normal[0], normal[1], normal[2]);
      resind[i] = index;
    }
    return v1_num;
  }
  return 0;
}

template<class T> CM_DEVICE int meshEdgeNormals(T* res, T* endverts, int dim, Obj<T>* obj,
                           T v[9], const int vi[3]) {
  T* v1 = v;
  T* v2 = v + 3;
  int v1i = vi[0], v2i = vi[1];
  // mesh data
  const T* mat = obj->mat;
  const T* pos = obj->pos;
  const float* vert = obj->data.mesh.vert;
  const int* polyvert = obj->data.mesh.polyvert;
  globalcoord<T>(v1, mat, pos, vert[3*v1i + 0], vert[3*v1i + 1], vert[3*v1i + 2]);

  // only one edge
  if (dim == 2) {
    globalcoord<T>(v2, mat, pos, vert[3*v2i + 0], vert[3*v2i + 1], vert[3*v2i + 2]);
    copy3<T>(endverts, v2);
    sub3<T>(res, v2, v1);
    normalize3<T>(res);
    return 1;
  }

  if (dim == 1) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1i];
    int v1_num = obj->data.mesh.mpolymapnum[v1i];

    // loop through all faces with vertex v1
    for (int i = 0; i < v1_num; i++) {
      int idx = obj->data.mesh.polymap[v1_adr + i];
      int adr = obj->data.mesh.polyvertadr[idx];
      int nvert = obj->data.mesh.polyvertnum[idx];
      // find previous vertex in polygon to form edge
      for (int j = 0; j < nvert; j++) {
        if (polyvert[adr + j] == v1i) {
          int k = (j == 0) ? nvert - 1 : j - 1;
          const float* vert_k = vert + 3*polyvert[adr + k];
          globalcoord<T>(endverts + 3*i, mat, pos, vert_k[0], vert_k[1], vert_k[2]);
          sub3<T>(res + 3*i, endverts + 3*i, v1);
          normalize3<T>(res + 3*i);
          break;
        }
      }
    }
    return v1_num;
  }
  return 0;
}

template<class T> CM_DEVICE int boxNormals2(T res[9], int resind[3], const T mat[9], const T n[3]) {
  // list of box face normals
  T normals[18] = {1, 0, 0,    -1,  0,  0,
                        0, 1, 0,     0, -1,  0,
                        0, 0, 1,     0,  0, -1};

  // get local coordinates of the normal
  T local_n[3];
  local_n[0] = mat[0]*n[0] + mat[3]*n[1] + mat[6]*n[2];
  local_n[1] = mat[1]*n[0] + mat[4]*n[1] + mat[7]*n[2];
  local_n[2] = mat[2]*n[0] + mat[5]*n[1] + mat[8]*n[2];
  scl3<T>(local_n, local_n, 1/sqrtn<T>(dot3<T>(local_n, local_n)));

  // determine if there is a side close to the normal
  for (int i = 0; i < 6; i++) {
    if (dot3<T>(local_n, normals + 3*i) > CM_FACE_TOL) {
      globalcoord<T>(res, mat, NULL, normals[3*i], normals[3*i + 1], normals[3*i + 2]);
      resind[0] = i;
      return 1;
    }
  }
  return 0;
}

template<class T> CM_DEVICE int boxNormals(T res[9], int resind[3], int dim, Obj<T>* obj,
                      const int vi[3], const T dir[3]) {
  int v1 = vi[0], v2 = vi[1], v3 = vi[2];
  const T* mat = obj->mat;
  if (dim == 3) {
    int c = 0;
    int x = ((v1 & 1) && (v2 & 1) && (v3 & 1)) - (!(v1 & 1) && !(v2 & 1) && !(v3 & 1));
    int y = ((v1 & 2) && (v2 & 2) && (v3 & 2)) - (!(v1 & 2) && !(v2 & 2) && !(v3 & 2));
    int z = ((v1 & 4) && (v2 & 4) && (v3 & 4)) - (!(v1 & 4) && !(v2 & 4) && !(v3 & 4));
    globalcoord<T>(res, mat, NULL, x, y, z);
    int sgn = x + y + z;
    if (x) resind[c++] = 0;
    if (y) resind[c++] = 2;
    if (z) resind[c++] = 4;
    if (sgn == -1) resind[0]++;
    return c == 1 ? 1 : boxNormals2<T>(res, resind, mat, dir);
  }

  if (dim == 2) {
    int c = 0;
    int x = ((v1 & 1) && (v2 & 1)) - (!(v1 & 1) && !(v2 & 1));
    int y = ((v1 & 2) && (v2 & 2)) - (!(v1 & 2) && !(v2 & 2));
    int z = ((v1 & 4) && (v2 & 4)) - (!(v1 & 4) && !(v2 & 4));
    if (x) {
      globalcoord<T>(res, mat, NULL, x, 0, 0);
      resind[c++] = (x > 0) ? 0 : 1;
    }
    if (y) {
      globalcoord<T>(res + 3*c, mat, NULL, 0, y, 0);
      resind[c++] = (y > 0) ? 2 : 3;
    }
    if (z) {
      globalcoord<T>(res + 3, mat, NULL, 0, 0, z);
      resind[c++] = (z > 0) ? 4 : 5;
    }
    return c == 2 ? 2 : boxNormals2<T>(res, resind, mat, dir);
  }

  if (dim == 1) {
    T x = (v1 & 1) ? 1 : -1;
    T y = (v1 & 2) ? 1 : -1;
    T z = (v1 & 4) ? 1 : -1;
    globalcoord<T>(res + 0, mat, NULL, x, 0, 0);
    globalcoord<T>(res + 3, mat, NULL, 0, y, 0);
    globalcoord<T>(res + 6, mat, NULL, 0, 0, z);
    resind[0] = (x > 0) ? 0 : 1;
    resind[1] = (y > 0) ? 2 : 3;
    resind[2] = (z > 0) ? 4 : 5;
    return 3;
  }
  return 0;
}

template<class T> CM_DEVICE int boxEdgeNormals(T res[9], T endverts[9], int dim, Obj<T>* obj,
                          T v[9], const int vi[3]) {
  T* v1 = v;
  T* v2 = v + 3;
  int v1i = vi[0], v2i = vi[1];
  // box data
  const T* mat = obj->mat;
  const T* pos = obj->pos;
  const T* size = obj->size;
  T x = (v1i & 1) ? size[0] : -size[0];
  T y = (v1i & 2) ? size[1] : -size[1];
  T z = (v1i & 4) ? size[2] : -size[2];
  globalcoord<T>(v1, mat, pos, x, y, z);

  if (dim == 2) {
    T x2 = (v2i & 1) ? size[0] : -size[0];
    T y2 = (v2i & 2) ? size[1] : -size[1];
    T z2 = (v2i & 4) ? size[2] : -size[2];
    globalcoord<T>(v2, mat, pos, x2, y2, z2);
    copy3<T>(endverts, v2);
    sub3<T>(res, v2, v1);
    normalize3<T>(res);
    return 1;
  }

  // return 3 adjacent vertices
  if (dim == 1) {
    globalcoord<T>(endverts, mat, pos, -x, y, z);
    sub3<T>(res, endverts, v1);
    normalize3<T>(res);

    globalcoord<T>(endverts + 3, mat, pos, x, -y, z);
    sub3<T>(res + 3, endverts + 3, v1);
    normalize3<T>(res + 3);

    globalcoord<T>(endverts + 6, mat, pos, x, y, -z);
    sub3<T>(res + 6, endverts + 6, v1);
    normalize3<T>(res + 6);
    return 3;
  }
  return 0;
}

template<class T> CM_DEVICE int boxFace(T res[12], Obj<T>* obj, int idx) {
  // box data
  const T* mat = obj->mat;
  const T* pos = obj->pos;
  const T* size = obj->size;

  // compute global coordinates of the box face and face normal
  switch (idx) {
    case 0:  // right
      globalcoord<T>(res + 0, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord<T>(res + 3, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord<T>(res + 6, mat, pos,  size[0], -size[1], -size[2]);
      globalcoord<T>(res + 9, mat, pos,  size[0], -size[1],  size[2]);
      return 4;
     case 1:  // left
      globalcoord<T>(res + 0, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord<T>(res + 3, mat, pos, -size[0],  size[1],  size[2]);
      globalcoord<T>(res + 6, mat, pos, -size[0], -size[1],  size[2]);
      globalcoord<T>(res + 9, mat, pos, -size[0], -size[1], -size[2]);
      return 4;
    case 2:  // top
      globalcoord<T>(res + 0, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord<T>(res + 3, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord<T>(res + 6, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord<T>(res + 9, mat, pos, -size[0],  size[1],  size[2]);
      return 4;
    case 3:  // bottom
      globalcoord<T>(res + 0, mat, pos, -size[0], -size[1],  size[2]);
      globalcoord<T>(res + 3, mat, pos,  size[0], -size[1],  size[2]);
      globalcoord<T>(res + 6, mat, pos,  size[0], -size[1], -size[2]);
      globalcoord<T>(res + 9, mat, pos, -size[0], -size[1], -size[2]);
      return 4;
    case 4:  // front
      globalcoord<T>(res + 0, mat, pos, -size[0],  size[1],  size[2]);
      globalcoord<T>(res + 3, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord<T>(res + 6, mat, pos,  size[0], -size[1],  size[2]);
      globalcoord<T>(res + 9, mat, pos, -size[0], -size[1],  size[2]);
      return 4;
    case 5:  // back
      globalcoord<T>(res + 0, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord<T>(res + 3, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord<T>(res + 6, mat, pos, -size[0], -size[1], -size[2]);
      globalcoord<T>(res + 9, mat, pos,  size[0], -size[1], -size[2]);
      return 4;
  }
  return 0;
}

template<class T> CM_DEVICE int meshFace(T* res, Obj<T>* obj, int idx) {
  const T* mat = obj->mat;
  const T* pos = obj->pos;

  int adr = obj->data.mesh.polyvertadr[idx], j = 0;
  int nvert = obj->data.mesh.polyvertnum[idx];
  const float* vert = obj->data.mesh.vert;
  const int* polyvert = obj->data.mesh.polyvert + adr;
  for (int i = nvert - 1; i >= 0; i--) {
    const float* v = vert + 3*polyvert[i];
    globalcoord<T>(res + 3*j++, mat, pos, v[0], v[1], v[2]);
  }
  return nvert;
}

template<class T> CM_DEVICE int alignedFaces(int res[2], const T* v, int nv,
                               const T* w, int nw, const T dir[3]) {
  int found = 0;
  T best = -CM_FACE_TOL;
  T dir_tol = CM_EDGE_TOL * norm3<T>(dir);
  for (int i = 0; i < nv; i++) {
    if (dot3<T>(v + 3*i, dir) >= -dir_tol) {
      continue;
    }
    for (int j = 0; j < nw; j++) {
      if (dot3<T>(w + 3*j, dir) <= dir_tol) {
        continue;
      }
      T tmp = dot3<T>(v + 3*i, w + 3*j);
      if (tmp < best) {
        best = tmp;
        res[0] = i;
        res[1] = j;
        found = 1;
      }
    }
  }
  return found;
}

template<class T> CM_DEVICE int alignedFaceEdge(int res[2], const T* edge, int nedge,
                                  const T* face, int nface, const T dir[3]) {
  for (int i = 0; i < nface; i++) {
    // ignore faces pointing away from the collision direction (negative dot product)
    if (dot3<T>(face + 3*i, dir) <= CM_MINVAL) {
      continue;
    }

    for (int j = 0; j < nedge; j++) {
      if (absn<T>(dot3<T>(edge + 3*j, face + 3*i)) < CM_EDGE_TOL) {
        res[0] = j;
        res[1] = i;
        return 1;
      }
    }
  }
  return 0;
}

template<class T> CM_DEVICE int simplexDim(int vi[3], T v[9]) {
  if (vi[0] != vi[1]) {
    return (vi[2] == vi[0] || vi[2] == vi[1]) ? 2 : 3;
  }
  if (vi[0] != vi[2]) {
    vi[1] = vi[2];
    copy3<T>(v + 3, v + 6);
    return 2;
  }
  return 1;
}

template<class T> CM_DEVICE void inflate(int nx, T* x1, T* x2, T* dist,
                           T margin1, T margin2, const T normal[3]) {
  if (!margin1 && !margin2) {
    return;
  }
  T n[3];
  if (normal) {
    copy3<T>(n, normal);
  } else {
    sub3<T>(n, x2, x1);
    normalize3<T>(n);
  }
  for (int i = 0; i < nx; i++) {
    if (margin1) addScl3<T>(x1 + 3*i, x1 + 3*i, n, margin1);
    if (margin2) addScl3<T>(x2 + 3*i, x2 + 3*i, n, -margin2);
    if (dist) dist[i] -= (margin1 + margin2);
  }
}

template<class T> CM_DEVICE int mjc_multicontact(Obj<T>* obj1, Obj<T>* obj2, T* x1, T* x2, T* dist,
                     int nconmax, int nmeshdegmax, int npolygonmax, uint8_t* buffer, T* normal_out=nullptr) {
  if (obj1->nsupp < 1 || obj2->nsupp < 1) {
    return 1;
  }
  if (obj1->geom_type == mjGEOM_MESH && !obj1->data.mesh.mesh_polynum) {
    return 1;
  }
  if (obj2->geom_type == mjGEOM_MESH && !obj2->data.mesh.mesh_polynum) {
    return 1;
  }
  if (obj1->geom_type == mjGEOM_BOX || obj2->geom_type == mjGEOM_BOX) {
    nmeshdegmax = nmeshdegmax < 3 ? 3 : nmeshdegmax;
    npolygonmax = npolygonmax < 4 ? 4 : npolygonmax;
  }

  int s1_1 = obj1->nsupp > 1 ? 1 : 0, s1_2 = obj1->nsupp > 2 ? 2 : 0;
  int s2_1 = obj2->nsupp > 1 ? 1 : 0, s2_2 = obj2->nsupp > 2 ? 2 : 0;
  int triface1i[3] = {obj1->suppind[0], obj1->suppind[s1_1], obj1->suppind[s1_2]};
  int triface2i[3] = {obj2->suppind[0], obj2->suppind[s2_1], obj2->suppind[s2_2]};
  T triface1[9], triface2[9];
  copy3<T>(triface1 + 0, obj1->suppvert + 0);
  copy3<T>(triface1 + 3, obj1->suppvert + 3*s1_1);
  copy3<T>(triface1 + 6, obj1->suppvert + 3*s1_2);
  copy3<T>(triface2 + 0, obj2->suppvert + 0);
  copy3<T>(triface2 + 3, obj2->suppvert + 3*s2_1);
  copy3<T>(triface2 + 6, obj2->suppvert + 3*s2_2);

  uint8_t* ptr = buffer;
  int* idx1 = (int*)ptr;           ptr += align8(sizeof(int) * nmeshdegmax);
  int* idx2 = (int*)ptr;           ptr += align8(sizeof(int) * nmeshdegmax);
  T* n1 = (T*)ptr;       ptr += align8(sizeof(T) * 3 * nmeshdegmax);
  T* n2 = (T*)ptr;       ptr += align8(sizeof(T) * 3 * nmeshdegmax);
  T* endverts = (T*)ptr; ptr += align8(sizeof(T) * 3 * nmeshdegmax);
  T* face1 = (T*)ptr;    ptr += align8(sizeof(T) * 3 * npolygonmax);
  T* face2 = (T*)ptr;    ptr += align8(sizeof(T) * 3 * npolygonmax);
  T* polygon = (T*)ptr;  // buffer for polygonClip

  // get dimensions of features of geoms 1 and 2
  int nface1 = simplexDim<T>(triface1i, triface1);
  int nface2 = simplexDim<T>(triface2i, triface2);
  int nnorms1 = 0, nnorms2 = 0;

  T dir[3], dir_neg[3];
  sub3<T>(dir, x2, x1);
  sub3<T>(dir_neg, x1, x2);

  // get all possible face normals for each geom
  if (obj1->geom_type == mjGEOM_BOX) {
    nnorms1 = boxNormals<T>(n1, idx1, nface1, obj1, triface1i, dir_neg);
  } else if (obj1->geom_type == mjGEOM_MESH) {
    nnorms1 = meshNormals<T>(n1, idx1, nface1, obj1, triface1i);
  }
  if (obj2->geom_type == mjGEOM_BOX) {
    nnorms2 = boxNormals<T>(n2, idx2, nface2, obj2, triface2i, dir);
  } else if (obj2->geom_type == mjGEOM_MESH) {
    nnorms2 = meshNormals<T>(n2, idx2, nface2, obj2, triface2i);
  }

  // determine if any two face normals match
  int res[2], edgecon1 = 0, edgecon2 = 0;
  if (!alignedFaces<T>(res, n1, nnorms1, n2, nnorms2, dir)) {
    // check if edge-face collision
    if (nface1 < 3 && nnorms2 && (nface1 <= nface2 || !nnorms1)) {
      nnorms1 = 0;
      if (obj1->geom_type == mjGEOM_BOX) {
        nnorms1 = boxEdgeNormals<T>(n1, endverts, nface1, obj1, triface1, triface1i);
      } else if (obj1->geom_type == mjGEOM_MESH) {
        nnorms1 = meshEdgeNormals<T>(n1, endverts, nface1, obj1, triface1, triface1i);
      }
      if (!alignedFaceEdge<T>(res, n1, nnorms1, n2, nnorms2, dir)) return 1;
      edgecon1 = 1;

    // check if face-edge collision
    } else if (nface2 < 3) {
      nnorms2 = 0;
      if (obj2->geom_type == mjGEOM_BOX) {
        nnorms2 = boxEdgeNormals<T>(n2, endverts, nface2, obj2, triface2, triface2i);
      } else if (obj2->geom_type == mjGEOM_MESH) {
        nnorms2 = meshEdgeNormals<T>(n2, endverts, nface2, obj2, triface2, triface2i);
      }
      if (!alignedFaceEdge<T>(res, n2, nnorms2, n1, nnorms1, dir_neg)) return 1;
      edgecon2 = 1;
    } else {
      // no multi-contact
      return 1;
    }
  }
  int i = res[0], j = res[1];

  // recover geom1 matching edge or face
  if (edgecon1) {
    copy3<T>(face1, triface1);
    copy3<T>(face1 + 3, endverts + 3*i);
    nface1 = 2;
  } else {
    if (obj1->geom_type == mjGEOM_BOX) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = boxFace<T>(face1, obj1, ind);
    } else if (obj1->geom_type == mjGEOM_MESH) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = meshFace<T>(face1, obj1, ind);
    }
  }

  // recover geom2 matching edge or face
  if (edgecon2) {
    copy3<T>(face2, triface2);
    copy3<T>(face2 + 3, endverts + 3*i);
    nface2 = 2;
  } else {
    if (obj2->geom_type == mjGEOM_BOX) {
      nface2 = boxFace<T>(face2, obj2, idx2[j]);
    } else if (obj2->geom_type == mjGEOM_MESH) {
      nface2 = meshFace<T>(face2, obj2, idx2[j]);
    }
  }

  // normal direction for witness recovery and contact inflation
  T wit_dir[3], normal[3];

  // remaining uninflated margins (capsule edges are already inflated via x1/x2)
  T margin1 = (obj1->geom >= 0 && 1) ? 0.5 * obj1->margin : 0;
  T margin2 = (obj2->geom >= 0 && 1) ? 0.5 * obj2->margin : 0;
  T margin = margin1 + margin2;

  int nx;
  if (edgecon1) {
    // face1 is an edge; clip face1 against face2
    scl3<T>(wit_dir, n2 + 3*j, -1.0);
    copy3<T>(normal, wit_dir);
    nx = polygonClip<T>(x2, x1, dist, nconmax, face2, nface2, face1, nface1, n2 + 3*j, wit_dir,
                     polygon, npolygonmax, margin);
  } else if (edgecon2) {
    // face2 is an edge; clip face2 against face1
    scl3<T>(wit_dir, n1 + 3*j, -1.0);
    copy3<T>(normal, n1 + 3*j);
    nx = polygonClip<T>(x1, x2, dist, nconmax, face1, nface1, face2, nface2, n1 + 3*j, wit_dir,
                     polygon, npolygonmax, margin);
  } else {
    // face-face collision
    copy3<T>(wit_dir, n2 + 3*j);
    scl3<T>(normal, n2 + 3*j, -1.0);
    nx = polygonClip<T>(x1, x2, dist, nconmax, face1, nface1, face2, nface2, n1 + 3*i, wit_dir,
                     polygon, npolygonmax, margin);
  }

  if (nx > 0) {
    if (normal_out) copy3<T>(normal_out, normal);
    inflate<T>(nx, x1, x2, dist, margin1, margin2, normal);
    return nx;
  }
  return 1;
}
} // namespace cmgeom

namespace cm {
// Float support sums can tie on distinct hull vertices and select a different
// manifold. Keep the native CCD decisions in the reference model's precision.
typedef double mjtNum;
typedef Obj<mjtNum> mjCCDObj;
enum {
  mjEPA_NOCONTACT=-1, mjEPA_SUCCESS=0, mjEPA_P2_INVALID_FACES,
  mjEPA_P2_NONCONVEX, mjEPA_P2_ORIGIN_ON_FACE, mjEPA_P3_BAD_NORMAL,
  mjEPA_P3_INVALID_V4, mjEPA_P3_INVALID_V5, mjEPA_P3_MISSING_ORIGIN,
  mjEPA_P3_ORIGIN_ON_FACE, mjEPA_P4_MISSING_ORIGIN
};
struct mjCCDConfig {
  int max_iterations; mjtNum tolerance; int max_contacts; mjtNum dist_cutoff;
  void* buffer; int npolygonmax, nmeshdegmax; int* stack;
};
struct mjCCDStatus {
  int separated; mjtNum dist[8], x1[24], x2[24]; int nx;
  int max_iterations; mjtNum tolerance; int max_contacts; mjtNum dist_cutoff;
  int gjk_iterations, epa_iterations, epa_status, fault, final_sign;
};
#define mjMINVAL 1e-15
#define mjMINVAL2 (mjMINVAL * mjMINVAL)
#define mjMAXVAL 1e10
#define mjMAXVAL2 (mjMAXVAL * mjMAXVAL)
#define mjMAX_LIMIT DBL_MAX
#define mjMINDIST2 mjMINVAL2
#define mjMINDIST3 mjMINVAL2
#define mjMINDIST4 mjMINVAL2
#define mjMINEPATOL mjMINVAL
CM_DEVICE void mulMatTVec3(mjtNum* res, const mjtNum* mat, const mjtNum* vec) {
  for(int i=0;i<3;i++) res[i]=mat[i]*vec[0]+mat[3+i]*vec[1]+mat[6+i]*vec[2];
}
CM_DEVICE void localToGlobal(mjtNum* res,const mjtNum* mat,const mjtNum* vec,const mjtNum* pos) {
  for(int i=0;i<3;i++) res[i]=mat[3*i]*vec[0]+mat[3*i+1]*vec[1]+mat[3*i+2]*vec[2]+pos[i];
}
CM_DEVICE void mulMatVec3(mjtNum* res,const mjtNum* mat,const mjtNum* vec) {
  for(int i=0;i<3;i++) res[i]=mat[3*i]*vec[0]+mat[3*i+1]*vec[1]+mat[3*i+2]*vec[2];
}
CM_DEVICE void mjc_boxSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // find support point in local frame
  local_supp[0] = local_dir[0] >= 0 ? size[0] : -size[0];
  local_supp[1] = local_dir[1] >= 0 ? size[1] : -size[1];
  local_supp[2] = local_dir[2] >= 0 ? size[2] : -size[2];

  // mark the index of the corner of the box for fast lookup
  obj->vertindex  = (local_supp[0] > 0) ? 1 : 0;
  obj->vertindex |= (local_supp[1] > 0) ? 2 : 0;
  obj->vertindex |= (local_supp[2] > 0) ? 4 : 0;

  // transform support point to global frame
  localToGlobal(res, mat, local_supp, pos);
}

CM_DEVICE mjtNum dot3f(const mjtNum a[3], const float b[3]) {
  return a[0]*(mjtNum)b[0] + a[1]*(mjtNum)b[1] + a[2]*(mjtNum)b[2];
}

CM_DEVICE void mjc_meshSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const float* verts = obj->data.mesh.vert;
  int nverts = obj->data.mesh.nvert;

  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum max = -FLT_MAX;
  int imax = 0;

  // used cached results from previous search
  if (obj->vertindex >= 0) {
    imax = obj->vertindex;
    max = dot3f(local_dir, verts + 3*imax);
  }

  // search all vertices, find maximum dot product
  for (int i=0; i < nverts; i++) {
    mjtNum vdot = dot3f(local_dir, verts + 3*i);

    // update max
    if (vdot > max) {
      max = vdot;
      imax = i;
    }
  }

  // record vertex index of maximum
  obj->vertindex = imax;

  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];

  // transform result to global frame
  localToGlobal(res, mat, local_dir, pos);
}

CM_DEVICE void mjc_hillclimbSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  int numvert = obj->data.mesh.graph[0];
  const int* vert_edgeadr = obj->data.mesh.graph + 2;
  const int* vert_globalid = obj->data.mesh.graph  + 2 + numvert;
  const int* edge_localid = obj->data.mesh.graph + 2 + 2*numvert;
  const float* verts = obj->data.mesh.vert;
  const mjtNum* pos = obj->pos;
  const mjtNum* mat = obj->mat;

  // rotate dir to geom local frame
  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);

  int prev = -1;
  int imax;

  // map continuous direction to discrete 3x3x3 grid (-1, 0, 1) indices
  int cx = (local_dir[0] > 0.4) - (local_dir[0] < -0.4) + 1;
  int cy = (local_dir[1] > 0.4) - (local_dir[1] < -0.4) + 1;
  int cz = (local_dir[2] > 0.4) - (local_dir[2] < -0.4) + 1;
  int grid_idx = obj->data.mesh.extrema[cx*9 + cy*3 + cz];

  if (obj->meshindex >= 0) {
    // warm start: pick the better of cached vertex vs grid seed
    mjtNum cached_dot = dot3f(local_dir, verts + 3*vert_globalid[obj->meshindex]);
    mjtNum seed_dot = dot3f(local_dir, verts + 3*vert_globalid[grid_idx]);
    imax = (seed_dot > cached_dot) ? grid_idx : obj->meshindex;
  } else {
    // cold start: use grid seed
    imax = grid_idx;
  }

  mjtNum max = dot3f(local_dir, verts + 3*vert_globalid[imax]);

  // hillclimb until no change
  while (imax != prev) {
    prev = imax;
    int subidx;
    for (int i = vert_edgeadr[imax]; (subidx = edge_localid[i]) >= 0; i++) {
      mjtNum vdot = dot3f(local_dir, verts + 3*vert_globalid[subidx]);
      if (vdot > max) {
        max = vdot;
        imax = subidx;  // update maximum vertex index
      }
    }
  }

  // record vertex index of maximum (local id)
  obj->meshindex = imax;

  // get resulting support vertex
  obj->vertindex = imax = vert_globalid[imax];
  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];

  // transform result to global frame
  localToGlobal(res, mat, local_dir, pos);
}
CM_DEVICE void native_support(mjtNum* res,mjCCDObj* o,const mjtNum* dir) {
  if(o->geom_type==mjGEOM_BOX) mjc_boxSupport(res,o,dir);
  else if(o->data.mesh.nvert<10) mjc_meshSupport(res,o,dir);
  else mjc_hillclimbSupport(res,o,dir);
}

// vertex in a polytope
typedef struct {
  mjtNum vert[3];   // v1 - v2; vertex in Minkowski sum making up polytope
  mjtNum vert1[3];  // vertex of polytope in obj1
  mjtNum vert2[3];  // vertex of polytope in obj2
  int index1;       // vertex index in mesh 1
  int index2;       // vertex index in mesh 2
} Vertex;

// subdistance algorithm for GJK that computes the barycentric coordinates of the point in a
// simplex closest to the origin
// implementation adapted from Montanari et al, ToG 2017
CM_DEVICE void subdistance(mjtNum lambda[4], int n, const mjtNum simplex[12]);

// compute the barycentric coordinates of the closest point to the origin in the n-simplex,
// for n = 3, 2, 1 respectively
CM_DEVICE void S3D(mjtNum lambda[4], const mjtNum s1[3], const mjtNum s2[3], const mjtNum s3[3],
                const mjtNum s4[3]);
CM_DEVICE void S2D(mjtNum lambda[3], const mjtNum s1[3], const mjtNum s2[3], const mjtNum s3[3]);
CM_DEVICE void S1D(mjtNum lambda[2], const mjtNum s1[3], const mjtNum s2[3]);

// compute the support point for GJK
CM_DEVICE void gjkSupport(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                       const mjtNum x_k[3], mjtNum x_norm);

// compute the linear combination of up to 4 3D vectors
CM_DEVICE void lincomb(mjtNum res[3], const mjtNum* coef, int n, const mjtNum v1[3],
                           const mjtNum v2[3], const mjtNum v3[3], const mjtNum v4[3]);

// one face in a polytope
typedef struct {
  int verts;         // indices of the three vertices of the face in the polytope
  int adj[3];        // adjacent faces, one for each edge: [v1,v2], [v2,v3], [v3,v1]
  mjtNum v[3];       // projection of the origin on face, can be used as face normal
  mjtNum dist2;      // squared norm of v; negative if deleted
  int index;         // index in map; -1: not in map, -2: deleted from polytope
} Face;

// return int[3] with lower 30 bits split between three int
#define EPA_VERT_EXPAND(n)                     \
  { (n) & 0x3FF,                               \
    (n) >> 10 & 0x3FF,                         \
    (n) >> 20 & 0x3FF }


// polytope used in the Expanding Polytope Algorithm (EPA)
typedef struct {
  Vertex* verts;      // list of vertices that make up the polytope
  int nverts;         // number of vertices
  Face* faces;        // list of faces that make up the polytope
  int nfaces;         // number of faces
  int maxfaces;       // max number of faces that can be stored in polytope
  mjtNum center[3];   // center of the polytope
  Face** map;         // linear map storing faces
  int nmap;           // number of faces in map
  struct Horizon {    // polytope boundary edges that can be seen from w
    int* indices;     // indices of faces on horizon
    int* edges;       // corresponding edge of each face on the horizon
    int nedges;       // number of edges in horizon
    int maxedges;     // max number of edges in horizon
    const mjtNum* w;  // point where horizon is created
    int* stack;       // iterative DFS frames, caller-owned
  } horizon;
} Polytope;

// compute the support point for EPA
CM_DEVICE int epaSupport(Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2,
                      const mjtNum d[3], mjtNum dnorm);

// copy support vertex i from obj1 and obj2 into polytope and return its index
CM_DEVICE int insertVertex(Polytope* pt, const mjCCDObj* obj1, const mjCCDObj* obj2, int i);

// attach a face to the polytope with the given vertex indices; return squared distance to origin
CM_DEVICE mjtNum attachFace(Polytope* pt, int v1, int v2, int v3, int adj1, int adj2, int adj3);

// return 1 if objects are in contact; 0 if not; -1 if inconclusive
// status must have initial tetrahedrons
CM_DEVICE int gjkIntersect(mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);

// create initial polytope for EPA for a 2, 3, or 4-simplex
CM_DEVICE int polytope2(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);
CM_DEVICE int polytope3(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);
CM_DEVICE int polytope4(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);

// return a face of the expanded polytope that best approximates the pentration depth
// witness points are in status->{x1, x2}
CM_DEVICE Face* epa(mjCCDStatus* status, Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2);

// -------------------------------- inlined  3D vector utils --------------------------------------

// res = v1 + v2
CM_DEVICE void add3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  res[0] = v1[0] + v2[0], res[1] = v1[1] + v2[1], res[2] = v1[2] + v2[2];
}

// res = v1 - v2
CM_DEVICE void sub3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  res[0] = v1[0] - v2[0], res[1] = v1[1] - v2[1], res[2] = v1[2] - v2[2];
}

// dot product
CM_DEVICE mjtNum dot3(const mjtNum v1[3], const mjtNum v2[3]) {
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

// norm of v
CM_DEVICE mjtNum norm3(const mjtNum v[3]) {
  return sqrt(dot3(v, v));
}

// res = v
CM_DEVICE void copy3(mjtNum res[3], const mjtNum v[3]) {
  res[0] = v[0], res[1] = v[1], res[2] = v[2];
}

// scalar product: res = s*v
CM_DEVICE void scl3(mjtNum res[3], const mjtNum v[3], mjtNum s) {
  res[0] = s*v[0], res[1] = s*v[1], res[2] = s*v[2];
}

// res = v1 + s*v2
CM_DEVICE void addScl3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3], mjtNum s) {
  res[0] = v1[0] + s*v2[0], res[1] = v1[1] + s*v2[1], res[2] = v1[2] + s*v2[2];
}

// cross product: res = v1 x v2
CM_DEVICE void cross3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  res[0] = v1[1]*v2[2] - v1[2]*v2[1];
  res[1] = v1[2]*v2[0] - v1[0]*v2[2];
  res[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

// return determinant of the 3x3 matrix with columns v1, v2, v3
CM_DEVICE mjtNum det3(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3]) {
  // v1 * (v2 x v3)
  return v1[0]*(v2[1]*v3[2] - v2[2]*v3[1])
       + v1[1]*(v2[2]*v3[0] - v2[0]*v3[2])
       + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0]);
}

// ---------------------------------------- GJK ---------------------------------------------------


// return true if object is a discrete shape or shrunk to a point/line with no margin
CM_DEVICE int discreteGeom(const mjCCDObj* obj) { return 1; }


// GJK algorithm
CM_DEVICE void gjk(mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  int get_dist = status->dist_cutoff > 0;  // need to recover geom distances if not in contact
  int backup_gjk = !get_dist;              // use gjkIntersect if no geom distances needed
  mjtNum simplex[12];                      // Minkowski difference vertices in the simplex
  mjtNum* simplex1 = obj1->suppvert;       // obj1 support vertices in the simplex
  mjtNum* simplex2 = obj2->suppvert;       // obj2 support vertices in the simplex
  int* index1 = obj1->suppind;             // obj1 support vertex indices
  int* index2 = obj2->suppind;             // obj2 support vertex indices
  int n = 0;                               // number of vertices in the simplex
  int k = 0;                               // current iteration
  int kmax = status->max_iterations;       // max number of iterations
  mjtNum* x1_k = status->x1;               // the kth approximation point for obj1
  mjtNum* x2_k = status->x2;               // the kth approximation point for obj2
  mjtNum x_k[3];                           // the kth approximation point in Minkowski difference
  mjtNum lambda[4];                        // barycentric coordinates for x_k
  mjtNum tol2 = status->tolerance * status->tolerance;
  status->separated = 0;
  obj1->nsupp = 0;
  obj2->nsupp = 0;

  // if both geoms are discrete, finite convergence is guaranteed; set tolerance to 0
  int discrete = discreteGeom(obj1) && discreteGeom(obj2);
  mjtNum epsilon = discrete ? 0 : 0.5 * tol2;

  // minimum norm of x_k; resolve distances below tolerance for discrete shapes or distance queries
  mjtNum min_norm = (discrete || get_dist) ? mjMINVAL : status->tolerance;

  // set initial guess
  sub3(x_k, x1_k, x2_k);
  mjtNum x_norm = norm3(x_k), x_norm_prev = 0;  // set to 0 for no-op on first iteration

  for (; k < kmax; k++) {
    // in tolerance for geoms to be in contact, or x_norm has stagnated
    if (x_norm < min_norm || fabs(x_norm_prev - x_norm) < mjMINVAL) {
      break;
    }

    // compute the kth support point
    Vertex v;
    gjkSupport(&v, obj1, obj2, x_k, x_norm);
    mjtNum* s_k = simplex + 3*n;
    copy3(s_k, v.vert);
    copy3(simplex1 + 3*n, v.vert1);
    copy3(simplex2 + 3*n, v.vert2);
    index1[n] = v.index1;
    index2[n] = v.index2;

    // stopping criteria using the Frank-Wolfe duality gap given by
    //  |f(x_k) - f(x_min)|^2 <= < grad f(x_k), (x_k - s_k) >
    mjtNum diff[3] = {x_k[0] - s_k[0], x_k[1] - s_k[1], x_k[2] - s_k[2]};
    if (dot3(x_k, diff) < epsilon) {
      break;
    }

    // the lower bound on distance between the two geoms is (lower / x_norm)
    // if lower > 0, then the geoms are separated
    mjtNum lower = dot3(x_k, s_k);
    if (!get_dist) {
      if (lower > 0) {
        status->separated = 1;
        status->gjk_iterations = k;
        status->nx = 0;
        status->dist[0] = mjMAX_LIMIT;
        return;
      }
    } else if (status->dist_cutoff < mjMAX_LIMIT) {
      if (lower > 0 && lower >= status->dist_cutoff * x_norm) {
        status->separated = 1;
        status->gjk_iterations = k;
        status->nx = 0;
        status->dist[0] = mjMAX_LIMIT;
        return;
      }
    }

    // tetrahedron is generated and only need contact info; fallback to gjkIntersect to
    // determine contact
    if (n == 3 && backup_gjk) {
      status->gjk_iterations = k;
      int ret = gjkIntersect(status, obj1, obj2);
      if (ret != -1) {
        status->nx = 0;
        status->separated = ret == 0;
        status->dist[0] = ret > 0 ? 0 : mjMAX_LIMIT;
        return;
      }
      k = status->gjk_iterations;
      backup_gjk = 0;
    }

    // run the distance subalgorithm to compute the barycentric coordinates
    // of the closest point to the origin in the simplex
    subdistance(lambda, n + 1, simplex);

    // remove vertices from the simplex no longer needed
    n = 0;
    for (int i = 0; i < 4; i++) {
      if (!lambda[i]) continue;
      if (n != i) {
        copy3(simplex + 3*n, simplex + 3*i);
        copy3(simplex1 + 3*n, simplex1 + 3*i);
        copy3(simplex2 + 3*n, simplex2 + 3*i);
        index1[n] = index1[i];
        index2[n] = index2[i];
      }
      lambda[n++] = lambda[i];
    }

    // SHOULD NOT OCCUR
    if (n < 1) {
      status->gjk_iterations = k;
      status->nx = 0;
      status->dist[0] = mjMAX_LIMIT;
      status->separated = 1;
      return;
    }

    // get the next iteration of x_k, save previous x_norm
    lincomb(x_k, lambda, n, simplex + 0, simplex + 3, simplex + 6, simplex + 9);
    x_norm_prev = x_norm;
    x_norm = norm3(x_k);

    // we should have a tetrahedron containing the origin so return early
    if (n == 4) {
      break;
    }
  }

  // compute the approximate witness points
  if (n > 0) {
    lincomb(x1_k, lambda, n, simplex1 + 0, simplex1 + 3, simplex1 + 6, simplex1 + 9);
    lincomb(x2_k, lambda, n, simplex2 + 0, simplex2 + 3, simplex2 + 6, simplex2 + 9);
  }

  // GJK exited early; do a final separation check
  Vertex tmp;
  gjkSupport(&tmp, obj1, obj2, x_k, x_norm);
  if (dot3(x_k, tmp.vert) > 0) {
    status->separated = 1;
  }

  status->nx = 1;
  status->gjk_iterations = k;
  obj1->nsupp = n;
  obj2->nsupp = n;

  // if 3-simplex and not separated, then the origin is contained in the simplex
  status->dist[0] = (n == 4 && !status->separated) ? 0 : x_norm;
}


// compute the support point in obj1 and obj2 for Minkowski difference
CM_DEVICE void support(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                           const mjtNum dir[3], const mjtNum dir_neg[3]) {
  // obj1
  native_support(v->vert1, obj1, dir);
  if (obj1->margin > 0 && obj1->geom >= 0) {
    mjtNum margin = 0.5 * obj1->margin;
    v->vert1[0] += dir[0] * margin;
    v->vert1[1] += dir[1] * margin;
    v->vert1[2] += dir[2] * margin;
  }

  // obj2
  native_support(v->vert2, obj2, dir_neg);
  if (obj2->margin > 0 && obj2->geom >= 0) {
    mjtNum margin = 0.5 * obj2->margin;
    v->vert2[0] += dir_neg[0] * margin;
    v->vert2[1] += dir_neg[1] * margin;
    v->vert2[2] += dir_neg[2] * margin;
  }

  // compute S_{A-B}(dir) = S_A(dir) - S_B(-dir)
  sub3(v->vert, v->vert1, v->vert2);

  // copy vertex indices of discrete geoms
  v->index1 = obj1->vertindex;
  v->index2 = obj2->vertindex;
}


// compute the support points in obj1 and obj2 for the kth approximation point
CM_DEVICE void gjkSupport(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                              const mjtNum x_k[3], mjtNum x_norm) {
  mjtNum dir[3], dir_neg[3];

  // mjc_support requires a normalized direction
  scl3(dir_neg, x_k, 1 / x_norm);
  scl3(dir, dir_neg, -1);
  support(v, obj1, obj2, dir, dir_neg);
}


// compute support points in Minkowski difference, return index of new vertex in polytope
CM_DEVICE int epaSupport(Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2,
                     const mjtNum d[3], mjtNum dnorm) {
  mjtNum dir[3] = {1, 0, 0}, dir_neg[3] = {-1, 0, 0};

  // mjc_support assumes a normalized direction
  if (dnorm > mjMINVAL) {
    dir[0] = d[0] / dnorm;
    dir[1] = d[1] / dnorm;
    dir[2] = d[2] / dnorm;
    scl3(dir_neg, dir, -1);
  }

  int n = pt->nverts++;
  Vertex* v = pt->verts + n;
  support(v, obj1, obj2, dir, dir_neg);
  return n;
}


// compute the support point in the Minkowski difference for gjkIntersect (without normalization)
CM_DEVICE void gjkIntersectSupport(Vertex* v, mjCCDObj* obj1, mjCCDObj* obj2,
                                const mjtNum dir[3]) {
  mjtNum dir_neg[3] = {-dir[0], -dir[1], -dir[2]};
  support(v, obj1, obj2, dir, dir_neg);
}


// compute the signed distance of a face along with the normal
CM_DEVICE mjtNum signedDistance(mjtNum normal[3], const Vertex* v1, const Vertex* v2,
                                    const Vertex* v3) {
  mjtNum diff1[3], diff2[3];
  sub3(diff1, v3->vert, v1->vert);
  sub3(diff2, v2->vert, v1->vert);
  cross3(normal, diff1, diff2);
  mjtNum norm2 = dot3(normal, normal);
  if (norm2 > mjMINVAL2 && norm2 < mjMAXVAL2) {
    scl3(normal, normal, 1 / sqrt(norm2));
    return dot3(normal, v1->vert);
  }
  return mjMAX_LIMIT;  // cannot recover normal (ignore face)
}


// return 1 if objects are in contact; 0 if not; -1 if inconclusive
CM_DEVICE int gjkIntersect(mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  Vertex simplex[4];
  for (int i = 0; i < 4; i++) {
    copy3(simplex[i].vert1, obj1->suppvert + 3*i);
    copy3(simplex[i].vert2, obj2->suppvert + 3*i);
    sub3(simplex[i].vert, simplex[i].vert1, simplex[i].vert2);
    simplex[i].index1 = obj1->suppind[i];
    simplex[i].index2 = obj2->suppind[i];
  }
  int s[4] = {0, 1, 2, 3};

  int k = status->gjk_iterations, kmax = status->max_iterations;
  for (; k < kmax; k++) {
    // compute the signed distance to each face in the simplex along with normals
    mjtNum dist[4], normals[12];
    dist[0] = signedDistance(&normals[0], simplex + s[2], simplex + s[1], simplex + s[3]);
    dist[1] = signedDistance(&normals[3], simplex + s[0], simplex + s[2], simplex + s[3]);
    dist[2] = signedDistance(&normals[6], simplex + s[1], simplex + s[0], simplex + s[3]);
    dist[3] = signedDistance(&normals[9], simplex + s[0], simplex + s[1], simplex + s[2]);

    // if origin is on any affine hull, convergence will fail
    if (!dist[3] || !dist[2] || !dist[1] || !dist[0]) {
      status->gjk_iterations = k;
      return -1;
    }

    // find the face with the smallest distance to the origin
    int i = (dist[0] < dist[1]) ? 0 : 1;
    int j = (dist[2] < dist[3]) ? 2 : 3;
    int index = (dist[i] < dist[j]) ? i : j;

    // origin inside of simplex (run EPA for contact information)
    if (dist[index] > 0) {
      obj1->nsupp = 4;
      obj2->nsupp = 4;
      for (int m = 0; m < 4; m++) {
        copy3(obj1->suppvert + 3*m, simplex[s[m]].vert1);
        copy3(obj2->suppvert + 3*m, simplex[s[m]].vert2);
        obj1->suppind[m] = simplex[s[m]].index1;
        obj2->suppind[m] = simplex[s[m]].index2;
      }
      status->gjk_iterations = k;
      return 1;
    }

    // replace worst vertex (farthest from origin) with new candidate
    gjkIntersectSupport(simplex + s[index], obj1, obj2, normals + 3*index);

    // found origin outside the Minkowski difference (return no collision)
    if (dot3(&normals[3*index], simplex[s[index]].vert) < 0) {
      obj1->nsupp = 0;
      obj2->nsupp = 0;
      status->gjk_iterations = k;
      return 0;
    }

    // swap vertices in the simplex to retain orientation
    i = (index + 1) & 3;
    j = (index + 2) & 3;
    int swap = s[i];
    s[i] = s[j];
    s[j] = swap;
  }
  status->gjk_iterations = k;
  return -1;  // never found origin
}


// compute the linear combination of up to 4 3D vectors
CM_DEVICE void lincomb(mjtNum res[3], const mjtNum* coef, int n, const mjtNum v1[3],
                           const mjtNum v2[3], const mjtNum v3[3], const mjtNum v4[3]) {
  switch (n) {
    case 0:
      res[0] = res[1] = res[2] = 0;
      break;
    case 1:
      res[0] = coef[0]*v1[0];
      res[1] = coef[0]*v1[1];
      res[2] = coef[0]*v1[2];
      break;
    case 2:
      res[0] = coef[0]*v1[0] + coef[1]*v2[0];
      res[1] = coef[0]*v1[1] + coef[1]*v2[1];
      res[2] = coef[0]*v1[2] + coef[1]*v2[2];
      break;
    case 3:
      res[0] = coef[0]*v1[0] + coef[1]*v2[0] + coef[2]*v3[0];
      res[1] = coef[0]*v1[1] + coef[1]*v2[1] + coef[2]*v3[1];
      res[2] = coef[0]*v1[2] + coef[1]*v2[2] + coef[2]*v3[2];
      break;
    case 4:
      res[0] = coef[0]*v1[0] + coef[1]*v2[0] + coef[2]*v3[0] + coef[3]*v4[0];
      res[1] = coef[0]*v1[1] + coef[1]*v2[1] + coef[2]*v3[1] + coef[3]*v4[1];
      res[2] = coef[0]*v1[2] + coef[1]*v2[2] + coef[2]*v3[2] + coef[3]*v4[2];
      break;
  }
}


// res = origin projected onto plane defined by v1, v2, v3
CM_DEVICE int projectOriginPlane(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3],
                              const mjtNum v3[3]) {
  mjtNum diff21[3], diff31[3], diff32[3], n[3], nv, nn;
  sub3(diff21, v2, v1);
  sub3(diff31, v3, v1);
  sub3(diff32, v3, v2);

  // n = (v1 - v2) x (v3 - v2)
  cross3(n, diff32, diff21);
  nv = dot3(n, v2);
  nn = dot3(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > mjMINVAL) {
    scl3(res, n, nv / nn);
    return 0;
  }

  // n = (v2 - v1) x (v3 - v1)
  cross3(n, diff21, diff31);
  nv = dot3(n, v1);
  nn = dot3(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > mjMINVAL) {
    scl3(res, n, nv / nn);
    return 0;
  }

  // n = (v1 - v3) x (v2 - v3)
  cross3(n, diff31, diff32);
  nv = dot3(n, v3);
  nn = dot3(n, n);
  scl3(res, n, nv / nn);
  return 0;
}


// res = origin projected onto line defined by v1, v2
CM_DEVICE void projectOriginLine(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  // res = v2 - <v2, v2 - v1> / <v2 - v1, v2 - v1> * (v2 - v1)
  mjtNum diff[3];
  sub3(diff, v2, v1);
  mjtNum scl = -(dot3(v2, diff) / dot3(diff, diff));
  res[0] = v2[0] + scl*diff[0];
  res[1] = v2[1] + scl*diff[1];
  res[2] = v2[2] + scl*diff[2];
}


// return 1 if both numbers are positive, -1 if both negative and 0 otherwise
CM_DEVICE int sameSign2(mjtNum a, mjtNum b) {
  if (a > 0 && b > 0) return 1;
  if (a < 0 && b < 0) return -1;
  return 0;
}


// subdistance algorithm for GJK that computes the barycentric coordinates of the point in a
// simplex closest to the origin
// implementation adapted from Montanari et al, ToG 2017
CM_DEVICE void subdistance(mjtNum lambda[4], int n, const mjtNum simplex[12]) {
  lambda[0] = lambda[1] = lambda[2] = lambda[3] = 0;
  const mjtNum* s1 = simplex + 0;
  const mjtNum* s2 = simplex + 3;
  const mjtNum* s3 = simplex + 6;
  const mjtNum* s4 = simplex + 9;

  switch (n) {
  case 4:
     S3D(lambda, s1, s2, s3, s4);
     break;
  case 3:
     S2D(lambda, s1, s2, s3);
      break;
  case 2:
     S1D(lambda, s1, s2);
     break;
  default:
    lambda[0] = 1;
    break;
  }
}


CM_DEVICE void S3D(mjtNum lambda[4], const mjtNum s1[3], const mjtNum s2[3],
                const mjtNum s3[3], const mjtNum s4[3]) {
  // the matrix M is given by
  //  [[ s1_x, s2_x, s3_x, s4_x ],
  //   [ s1_y, s2_y, s3_y, s4_y ],
  //   [ s1_z, s2_z, s3_z, s4_z ],
  //   [ 1,    1,    1,    1    ]]
  // we want to solve M*lambda = P, where P = [p_x, p_y, p_z, 1] with [p_x, p_y, p_z] is the
  // origin projected onto the simplex

  // compute cofactors to find det(M)
  mjtNum C41 = -det3(s2, s3, s4);
  mjtNum C42 =  det3(s1, s3, s4);
  mjtNum C43 = -det3(s1, s2, s4);
  mjtNum C44 =  det3(s1, s2, s3);

  // note that m_det = 6*SignVol(simplex) with C4i corresponding to the volume of the 3-simplex
  // with vertices {s1, s2, s3, 0} - si
  mjtNum m_det = C41 + C42 + C43 + C44;

  int comp1 = sameSign2(m_det, C41),
      comp2 = sameSign2(m_det, C42),
      comp3 = sameSign2(m_det, C43),
      comp4 = sameSign2(m_det, C44);

  // if all signs are the same then the origin is inside the simplex
  if (comp1 && comp2 && comp3 && comp4) {
    lambda[0] = C41 / m_det;
    lambda[1] = C42 / m_det;
    lambda[2] = C43 / m_det;
    lambda[3] = C44 / m_det;
    return;
  }

  // find the smallest distance, and use the corresponding barycentric coordinates
  mjtNum dmin = mjMAX_LIMIT;

  if (!comp1) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s2, s3, s4);
    lincomb(x, lambda_2d, 3, s2, s3, s4, NULL);
    mjtNum d = dot3(x, x);
    lambda[0] = 0;
    lambda[1] = lambda_2d[0];
    lambda[2] = lambda_2d[1];
    lambda[3] = lambda_2d[2];
    dmin = d;
  }

  if (!comp2) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s3, s4);
    lincomb(x, lambda_2d, 3, s1, s3, s4, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = 0;
      lambda[2] = lambda_2d[1];
      lambda[3] = lambda_2d[2];
      dmin = d;
    }
  }

  if (!comp3) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s2, s4);
    lincomb(x, lambda_2d, 3, s1, s2, s4, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = lambda_2d[1];
      lambda[2] = 0;
      lambda[3] = lambda_2d[2];
      dmin = d;
    }
  }

  if (!comp4) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s2, s3);
    lincomb(x, lambda_2d, 3, s1, s2, s3, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = lambda_2d[1];
      lambda[2] = lambda_2d[2];
      lambda[3] = 0;
    }
  }
}


CM_DEVICE void S2D(mjtNum lambda[3], const mjtNum s1[3], const mjtNum s2[3], const mjtNum s3[3]) {
  // project origin onto affine hull of the simplex
  mjtNum p_o[3];
  if (projectOriginPlane(p_o, s1, s2, s3)) {
    S1D(lambda, s1, s2);
    lambda[2] = 0;
    return;
  }

  // Below are the minors M_i4 of the matrix M given by
  // [[ s1_x, s2_x, s3_x, s4_x ],
  //  [ s1_y, s2_y, s3_y, s4_y ],
  //  [ s1_z, s2_z, s3_z, s4_z ],
  //  [ 1,    1,    1,    1    ]]
  mjtNum M_14 = s2[1]*s3[2] - s2[2]*s3[1] - s1[1]*s3[2] + s1[2]*s3[1] + s1[1]*s2[2] - s1[2]*s2[1];
  mjtNum M_24 = s2[0]*s3[2] - s2[2]*s3[0] - s1[0]*s3[2] + s1[2]*s3[0] + s1[0]*s2[2] - s1[2]*s2[0];
  mjtNum M_34 = s2[0]*s3[1] - s2[1]*s3[0] - s1[0]*s3[1] + s1[1]*s3[0] + s1[0]*s2[1] - s1[1]*s2[0];

  // exclude the axis with the largest projection of the simplex using the computed minors
  mjtNum M_max = 0;
  mjtNum s1_2D[2], s2_2D[2], s3_2D[2], p_o_2D[2];
  mjtNum mu1 = fabs(M_14), mu2 = fabs(M_24), mu3 = fabs(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    s1_2D[0] = s1[1];
    s1_2D[1] = s1[2];

    s2_2D[0] = s2[1];
    s2_2D[1] = s2[2];

    s3_2D[0] = s3[1];
    s3_2D[1] = s3[2];

    p_o_2D[0] = p_o[1];
    p_o_2D[1] = p_o[2];
  } else if (mu2 >= mu3) {
    M_max = M_24;
    s1_2D[0] = s1[0];
    s1_2D[1] = s1[2];

    s2_2D[0] = s2[0];
    s2_2D[1] = s2[2];

    s3_2D[0] = s3[0];
    s3_2D[1] = s3[2];

    p_o_2D[0] = p_o[0];
    p_o_2D[1] = p_o[2];
  } else {
    M_max = M_34;
    s1_2D[0] = s1[0];
    s1_2D[1] = s1[1];

    s2_2D[0] = s2[0];
    s2_2D[1] = s2[1];

    s3_2D[0] = s3[0];
    s3_2D[1] = s3[1];

    p_o_2D[0] = p_o[0];
    p_o_2D[1] = p_o[1];
  }

  // compute the cofactors C3i of the following matrix:
  // [[ s1_2D[0] - p_o_2D[0], s2_2D[0] - p_o_2D[0], s3_2D[0] - p_o_2D[0] ],
  //  [ s1_2D[1] - p_o_2D[1], s2_2D[1] - p_o_2D[1], s3_2D[1] - p_o_2D[1] ],
  //  [ 1,                    1,                    1                    ]]

  // C31 corresponds to the signed area of 2-simplex: (p_o_2D, s2_2D, s3_2D)
  mjtNum C31 = p_o_2D[0]*s2_2D[1] + p_o_2D[1]*s3_2D[0] + s2_2D[0]*s3_2D[1]
             - p_o_2D[0]*s3_2D[1] - p_o_2D[1]*s2_2D[0] - s3_2D[0]*s2_2D[1];

  // C32 corresponds to the signed area of 2-simplex: (_po_2D, s1_2D, s3_2D)
  mjtNum C32 = p_o_2D[0]*s3_2D[1] + p_o_2D[1]*s1_2D[0] + s3_2D[0]*s1_2D[1]
             - p_o_2D[0]*s1_2D[1] - p_o_2D[1]*s3_2D[0] - s1_2D[0]*s3_2D[1];

  // C33 corresponds to the signed area of 2-simplex: (p_o_2D, s1_2D, s2_2D)
  mjtNum C33 = p_o_2D[0]*s1_2D[1] + p_o_2D[1]*s2_2D[0] + s1_2D[0]*s2_2D[1]
             - p_o_2D[0]*s2_2D[1] - p_o_2D[1]*s1_2D[0] - s2_2D[0]*s1_2D[1];

  int comp1 = sameSign2(M_max, C31),
      comp2 = sameSign2(M_max, C32),
      comp3 = sameSign2(M_max, C33);

  // all the same sign, p_o is inside the 2-simplex
  if (comp1 && comp2 && comp3) {
    lambda[0] = C31 / M_max;
    lambda[1] = C32 / M_max;
    lambda[2] = C33 / M_max;
    return;
  }

  // find the smallest distance, and use the corresponding barycentric coordinates
  mjtNum dmin = mjMAX_LIMIT;

  if (!comp1) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s2, s3);
    lincomb(x, lambda_1d, 2, s2, s3, NULL, NULL);
    mjtNum d = dot3(x, x);
    lambda[0] = 0;
    lambda[1] = lambda_1d[0];
    lambda[2] = lambda_1d[1];
    dmin = d;
  }

  if (!comp2) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s1, s3);
    lincomb(x, lambda_1d, 2, s1, s3, NULL, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_1d[0];
      lambda[1] = 0;
      lambda[2] = lambda_1d[1];
      dmin = d;
    }
  }

  if (!comp3) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s1, s2);
    lincomb(x, lambda_1d, 2, s1, s2, NULL, NULL);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_1d[0];
      lambda[1] = lambda_1d[1];
      lambda[2] = 0;
    }
  }
}


CM_DEVICE void S1D(mjtNum lambda[2], const mjtNum s1[3], const mjtNum s2[3]) {
  // find projection of origin onto the 1-simplex:
  mjtNum p_o[3];
  projectOriginLine(p_o, s1, s2);

  // find the axis with the largest projection "shadow" of the simplex
  mjtNum mu = s1[0] - s2[0];
  mjtNum mu_max = mu;
  int index = 0;

  mu = s1[1] - s2[1];
  if (fabs(mu) >= fabs(mu_max)) {
    mu_max = mu;
    index = 1;
  }

  mu = s1[2] - s2[2];
  if (fabs(mu) >= fabs(mu_max)) {
    mu_max = mu;
    index = 2;
  }

  mjtNum C1 = p_o[index] - s2[index];
  mjtNum C2 = s1[index] - p_o[index];

  // determine if projection of origin lies inside 1-simplex
  int same = sameSign2(mu_max, C1) && sameSign2(mu_max, C2);
  lambda[0] = same ? C1 / mu_max : 0;
  lambda[1] = same ? C2 / mu_max : 1;
}


// ---------------------------------------- EPA ---------------------------------------------------

// replace a 3-simplex with one of its faces
CM_DEVICE void replaceSimplex3(Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2,
                                   int v1, int v2, int v3) {
  int v[3] = {v1, v2, v3};
  obj1->nsupp = 3;
  obj2->nsupp = 3;
  for (int i = 0; i < 3; i++) {
    copy3(obj1->suppvert + 3*i, pt->verts[v[i]].vert1);
    copy3(obj2->suppvert + 3*i, pt->verts[v[i]].vert2);
    obj1->suppind[i] = pt->verts[v[i]].index1;
    obj2->suppind[i] = pt->verts[v[i]].index2;
  }

  // reset polytope
  pt->nfaces = 0;
  pt->nverts = 0;
  pt->nmap = 0;
}


// return 1 if the origin and p3 are on the same side of the plane defined by p0, p1, p2
CM_DEVICE int sameSide(const mjtNum p0[3], const mjtNum p1[3],
                    const mjtNum p2[3], const mjtNum p3[3]) {
    mjtNum diff1[3], diff2[3], diff3[3], diff4[3], n[3];
    sub3(diff1, p1, p0);
    sub3(diff2, p2, p0);
    cross3(n, diff1, diff2);

    sub3(diff3, p3, p0);
    mjtNum dot1 = dot3(n, diff3);

    scl3(diff4, p0, -1);
    mjtNum dot2 = dot3(n, diff4);
    if (dot1 > 0 && dot2 > 0) return 1;
    if (dot1 < 0 && dot2 < 0) return 1;
    return 0;
}


// return 1 if the origin is contained in the tetrahedron, 0 otherwise
CM_DEVICE int testTetra(const mjtNum p0[3], const mjtNum p1[3],
                     const mjtNum p2[3], const mjtNum p3[3]) {
  return sameSide(p0, p1, p2, p3)
      && sameSide(p1, p2, p3, p0)
      && sameSide(p2, p3, p0, p1)
      && sameSide(p3, p0, p1, p2);
}


// matrix for 120 degrees rotation around given axis
CM_DEVICE void rotmat(mjtNum R[9], const mjtNum axis[3]) {
  mjtNum n = norm3(axis);
  mjtNum u1 = axis[0] / n, u2 = axis[1] / n, u3 = axis[2] / n;
  const mjtNum sin = 0.86602540378;  // sin(120 deg)
  const mjtNum cos = -0.5;           // cos(120 deg)
  R[0] = cos + u1*u1*(1 - cos);
  R[1] = u1*u2*(1 - cos) - u3*sin;
  R[2] = u1*u3*(1 - cos) + u2*sin;
  R[3] = u2*u1*(1 - cos) + u3*sin;
  R[4] = cos + u2*u2*(1 - cos);
  R[5] = u2*u3*(1 - cos) - u1*sin;
  R[6] = u1*u3*(1 - cos) - u2*sin;
  R[7] = u2*u3*(1 - cos) + u1*sin;
  R[8] = cos + u3*u3*(1 - cos);
}


// return nonzero if the ray v1v2 intersects the triangle v3v4v5
CM_DEVICE int rayTriangle(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3],
                              const mjtNum v4[3], const mjtNum v5[3]) {
  mjtNum diff12[3], diff13[3], diff14[3], diff15[3];
  sub3(diff12, v2, v1);
  sub3(diff13, v3, v1);
  sub3(diff14, v4, v1);
  sub3(diff15, v5, v1);

  mjtNum vol1 = det3(diff13, diff14, diff12);
  mjtNum vol2 = det3(diff14, diff15, diff12);
  mjtNum vol3 = det3(diff15, diff13, diff12);

  if (vol1 >= 0 && vol2 >= 0 && vol3 >= 0) return 1;
  if (vol1 <= 0 && vol2 <= 0 && vol3 <= 0) return -1;
  return 0;
}


// create a polytope from a 1-simplex (returns 0 on success)
CM_DEVICE int polytope2(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  int v1i = insertVertex(pt, obj1, obj2, 0);
  int v2i = insertVertex(pt, obj1, obj2, 1);
  mjtNum *v1 = pt->verts[v1i].vert, *v2 = pt->verts[v2i].vert;

  // set the polytope center
  add3(pt->center, v1, v2);
  scl3(pt->center, pt->center, 0.5);

  mjtNum diff[3];
  sub3(diff, v2, v1);

  // find component with smallest magnitude (so cross product is largest)
  mjtNum value = mjMAX_LIMIT;
  int index = 0;
  for (int i = 0; i < 3; i++) {
    if (fabs(diff[i]) < value) {
      value = fabs(diff[i]);
      index = i;
    }
  }

  // cross product with best coordinate axis
  mjtNum e[3] = {0, 0, 0};
  e[index] = 1;
  mjtNum d1[3], d2[3], d3[3];
  cross3(d1, e, diff);

  // rotate around the line segment to get three more points spaced 120 degrees apart
  mjtNum R[9];
  rotmat(R, diff);

  mulMatVec3(d2, R, d1);
  mulMatVec3(d3, R, d2);

  // save vertices and get indices for each one
  int v3i = epaSupport(pt, obj1, obj2, d1, norm3(d1));
  int v4i = epaSupport(pt, obj1, obj2, d2, norm3(d2));
  int v5i = epaSupport(pt, obj1, obj2, d3, norm3(d3));

  mjtNum* v3 = pt->verts[v3i].vert;
  mjtNum* v4 = pt->verts[v4i].vert;
  mjtNum* v5 = pt->verts[v5i].vert;

  // build hexahedron
  if (attachFace(pt, v1i, v3i, v4i, 1, 3, 2) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v1i, v3i, v4i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1i, v5i, v3i, 2, 4, 0) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v1i, v5i, v3i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1i, v4i, v5i, 0, 5, 1) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v1i, v4i, v5i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v2i, v4i, v3i, 5, 0, 4) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v2i, v4i, v3i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v2i, v3i, v5i, 3, 1, 5) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v2i, v3i, v5i);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v2i, v5i, v4i, 4, 2, 3) < mjMINDIST2) {
    replaceSimplex3(pt, obj1, obj2, v2i, v5i, v4i);
    return polytope3(pt, status, obj1, obj2);
  }

  // check hexahedron is convex
  if (!rayTriangle(v1, v2, v3, v4, v5)) {
    return mjEPA_P2_NONCONVEX;
  }

  // populate face map
  for (int i = 0; i < 6; i++) {
    pt->map[i] = pt->faces + i;
    pt->faces[i].index = i;
  }
  pt->nmap = 6;
  return 0;
}


// compute the affine coordinates of p on the triangle v1v2v3
CM_DEVICE void triAffineCoord(mjtNum lambda[3], const mjtNum v1[3], const mjtNum v2[3],
                           const mjtNum v3[3], const mjtNum p[3]) {
  // compute minors as in S2D
  mjtNum M_14 = v2[1]*v3[2] - v2[2]*v3[1] - v1[1]*v3[2] + v1[2]*v3[1] + v1[1]*v2[2] - v1[2]*v2[1];
  mjtNum M_24 = v2[0]*v3[2] - v2[2]*v3[0] - v1[0]*v3[2] + v1[2]*v3[0] + v1[0]*v2[2] - v1[2]*v2[0];
  mjtNum M_34 = v2[0]*v3[1] - v2[1]*v3[0] - v1[0]*v3[1] + v1[1]*v3[0] + v1[0]*v2[1] - v1[1]*v2[0];

  // exclude one of the axes with the largest projection of the simplex using the computed minors
  mjtNum M_max = 0;
  int x, y;
  mjtNum mu1 = fabs(M_14), mu2 = fabs(M_24), mu3 = fabs(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    x = 1;
    y = 2;
  } else if (mu2 >= mu3) {
    M_max = M_24;
    x = 0;
    y = 2;
  } else {
    M_max = M_34;
    x = 0;
    y = 1;
  }

  // C31 corresponds to the signed area of 2-simplex: (v, s2, s3)
  mjtNum C31 = p[x]*v2[y] + p[y]*v3[x] + v2[x]*v3[y]
             - p[x]*v3[y] - p[y]*v2[x] - v3[x]*v2[y];

  // C32 corresponds to the signed area of 2-simplex: (v, s1, s3)
  mjtNum C32 = p[x]*v3[y] + p[y]*v1[x] + v3[x]*v1[y]
             - p[x]*v1[y] - p[y]*v3[x] - v1[x]*v3[y];

  // C33 corresponds to the signed area of 2-simplex: (v, s1, s2)
  mjtNum C33 = p[x]*v1[y] + p[y]*v2[x] + v1[x]*v2[y]
             - p[x]*v2[y] - p[y]*v1[x] - v2[x]*v1[y];

  // compute affine coordinates
  lambda[0] = C31 / M_max;
  lambda[1] = C32 / M_max;
  lambda[2] = C33 / M_max;
}


// return true if point p and triangle v1v2v3 intersect
CM_DEVICE int triPointIntersect(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3],
                             const mjtNum p[3]) {
  mjtNum lambda[3];
  triAffineCoord(lambda, v1, v2, v3, p);
  if (lambda[0] < 0 || lambda[1] < 0 || lambda[2] < 0) {
    return 0;
  }
  mjtNum pr[3], diff[3];
  pr[0] = v1[0]*lambda[0] + v2[0]*lambda[1] + v3[0]*lambda[2];
  pr[1] = v1[1]*lambda[0] + v2[1]*lambda[1] + v3[1]*lambda[2];
  pr[2] = v1[2]*lambda[0] + v2[2]*lambda[1] + v3[2]*lambda[2];
  sub3(diff, pr, p);
  return norm3(diff) < mjMINVAL;
}


// create a polytope from a 2-simplex (returns 0 on success)
CM_DEVICE int polytope3(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  // save vertices and get indices for each one
  int v1i = insertVertex(pt, obj1, obj2, 0);
  int v2i = insertVertex(pt, obj1, obj2, 1);
  int v3i = insertVertex(pt, obj1, obj2, 2);
  const mjtNum *v1 = pt->verts[v1i].vert,
               *v2 = pt->verts[v2i].vert,
               *v3 = pt->verts[v3i].vert;

  // set the polytope center
  add3(pt->center, v1, v2);
  add3(pt->center, pt->center, v3);
  scl3(pt->center, pt->center, 1.0 / 3.0);

  // get normals in both directions
  mjtNum diff1[3], diff2[3], n[3], n_neg[3];
  sub3(diff1, v2, v1);
  sub3(diff2, v3, v1);
  cross3(n, diff1, diff2);
  mjtNum n_norm = norm3(n);
  if (n_norm < mjMINVAL) {
    return mjEPA_P3_BAD_NORMAL;
  }

  // negative of triangle normal n
  scl3(n_neg, n, -1);

  int v5i = epaSupport(pt, obj1, obj2, n_neg, n_norm);
  int v4i = epaSupport(pt, obj1, obj2, n, n_norm);
  mjtNum* v4 = pt->verts[v4i].vert;
  mjtNum* v5 = pt->verts[v5i].vert;

  // check that v4 is not contained in the 2-simplex
  if (triPointIntersect(v1, v2, v3, v4)) {
    return mjEPA_P3_INVALID_V4;
  }

  // check that v5 is not contained in the 2-simplex
  if (triPointIntersect(v1, v2, v3, v5)) {
    return mjEPA_P3_INVALID_V5;
  }

  // check that v4 and v5 are not coplanar with the 2-simplex
  mjtNum d4[3], d5[3];
  sub3(d4, v4, v1);
  sub3(d5, v5, v1);
  if (dot3(n, d4) < mjMINEPATOL*n_norm || dot3(n_neg, d5) < mjMINEPATOL*n_norm) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }

  // if origin does not lie on simplex then we need to check that the hexahedron contains the
  // origin
  //
  // TODO(kylebayes): It's possible for GJK to return a 2-simplex with the origin not contained in
  // it but within tolerance from it. In that case the hexahedron could possibly be constructed
  // that doesn't contain the origin, but nonetheless there is penetration depth.
  if (status->dist[0] > 10*mjMINVAL && !testTetra(v1, v2, v3, v4) && !testTetra(v1, v2, v3, v5)) {
    return mjEPA_P3_MISSING_ORIGIN;
  }

  // create hexahedron for EPA
  if (attachFace(pt, v4i, v1i, v2i, 1, 3, 2) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v4i, v3i, v1i, 2, 4, 0) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v4i, v2i, v3i, 0, 5, 1) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v2i, v1i, 5, 0, 4) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v1i, v3i, 3, 1, 5) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v3i, v2i, 4, 2, 3) < mjMINDIST3) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }

  // populate face map
  for (int i = 0; i < 6; i++) {
    pt->map[i] = pt->faces + i;
    pt->faces[i].index = i;
  }
  pt->nmap = 6;
  return 0;
}


// create a polytope from a 3-simplex (returns 0 on success)
CM_DEVICE int polytope4(Polytope* pt, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  int v1 = insertVertex(pt, obj1, obj2, 0);
  int v2 = insertVertex(pt, obj1, obj2, 1);
  int v3 = insertVertex(pt, obj1, obj2, 2);
  int v4 = insertVertex(pt, obj1, obj2, 3);

  // set the polytope center
  add3(pt->center, pt->verts[v1].vert, pt->verts[v2].vert);
  add3(pt->center, pt->center, pt->verts[v3].vert);
  add3(pt->center, pt->center, pt->verts[v4].vert);
  scl3(pt->center, pt->center, 0.25);

  // if the origin is on a face, replace the 3-simplex with a 2-simplex
  if (attachFace(pt, v1, v2, v3, 1, 3, 2) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v1, v2, v3);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1, v4, v2, 2, 3, 0) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v1, v4, v2);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v1, v3, v4, 0, 3, 1) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v1, v3, v4);
    return polytope3(pt, status, obj1, obj2);
  }
  if (attachFace(pt, v4, v3, v2, 2, 0, 1) < mjMINDIST4) {
    replaceSimplex3(pt, obj1, obj2, v4, v3, v2);
    return polytope3(pt, status, obj1, obj2);
  }

  // numerically verify that the origin lies in the tetrahedron interior
  if (!testTetra(pt->verts[v1].vert, pt->verts[v2].vert, pt->verts[v3].vert, pt->verts[v4].vert)) {
    return mjEPA_P4_MISSING_ORIGIN;
  }

  // populate face map
  for (int i = 0; i < 4; i++) {
    pt->map[i] = pt->faces + i;
    pt->faces[i].index = i;
  }
  pt->nmap = 4;
  return 0;
}


// copy support vertex i from obj1 and obj2 into polytope and return its index
CM_DEVICE int insertVertex(Polytope* pt, const mjCCDObj* obj1, const mjCCDObj* obj2, int i) {
  int n = pt->nverts++;
  Vertex* v = &pt->verts[n];
  copy3(v->vert1, obj1->suppvert + 3*i);
  copy3(v->vert2, obj2->suppvert + 3*i);
  sub3(v->vert, v->vert1, v->vert2);
  v->index1 = obj1->suppind[i];
  v->index2 = obj2->suppind[i];
  return n;
}


// delete face from map (return non-zero on error)
CM_DEVICE void deleteFace(Polytope* pt, Face* face) {
  if (face->index >= 0) {
    pt->map[face->index] = pt->map[--pt->nmap];
    pt->map[face->index]->index = face->index;
  }
  face->index = -2;  // mark face as deleted from map and polytope
}


// return max number of faces that can be stored in polytope
CM_DEVICE int maxFaces(Polytope* pt) {
  return pt->maxfaces - pt->nfaces;
}


// attach a face to the polytope with the given vertex indices; return squared distance to origin
CM_DEVICE mjtNum attachFace(Polytope* pt, int v1, int v2, int v3,
                                int adj1, int adj2, int adj3) {
  Face* face = &pt->faces[pt->nfaces++];
  face->verts = v1 + (v2 << 10) + (v3 << 20);

  // adjacent faces
  face->adj[0] = adj1;
  face->adj[1] = adj2;
  face->adj[2] = adj3;

  // compute witness point v
  int ret = projectOriginPlane(face->v, pt->verts[v3].vert, pt->verts[v2].vert, pt->verts[v1].vert);
  if (ret) {
    return 0;
  }

  // ensure projection points outward from the polytope
  mjtNum outward[3];
  sub3(outward, pt->verts[v1].vert, pt->center);
  if (dot3(face->v, outward) < 0) {
    scl3(face->v, face->v, -1);
  }

  face->dist2 = dot3(face->v, face->v);
  face->index = -1;

  return face->dist2;
}


// add an edge to the horizon; return number of edges added
CM_DEVICE int addEdge(Polytope* pt, int index, int edge) {
  // always increment nedges so we can produce a useful warning if threshold is exceeded
  int i = pt->horizon.nedges++;
  if (i < pt->horizon.maxedges) {
    pt->horizon.edges[i] = edge;
    pt->horizon.indices[i] = index;
    return 1;
  }
  return 0;
}


// get edge index where vertex lies
CM_DEVICE int getEdge(Face* face, int vertex) {
  int verts[3] = EPA_VERT_EXPAND(face->verts);
  if (verts[0] == vertex) return 0;
  if (verts[1] == vertex) return 1;
  return 2;
}


// recursive call to build horizon; return 1 if face is visible from w otherwise 0
CM_DEVICE int horizonRec(Polytope* pt, Face* face, int e) {
  if (dot3(face->v, pt->horizon.w) - face->dist2 <= mjMINVAL) return 0;
  int* stack=pt->horizon.stack;
  int depth=0;
  stack[0]=(int)(face-pt->faces); stack[1]=e; stack[2]=1; stack[3]=1;
  deleteFace(pt,face);
  while(depth>=0) {
    int* f=stack+4*depth;
    if(f[2]==3) {
      int ret=f[3];
      if(--depth<0) return ret;
      if(ret<0) stack[4*depth+3]=-1;
      continue;
    }
    face=pt->faces+f[0];
    int i=(f[1]+f[2]++)%3;
    Face* adjacent=pt->faces+face->adj[i];
    if(adjacent->index<=-2) continue;
    int vertices[3]=EPA_VERT_EXPAND(face->verts);
    int edge=getEdge(adjacent,vertices[(i+1)%3]);
    if(dot3(adjacent->v,pt->horizon.w)-adjacent->dist2>mjMINVAL) {
      ++depth;
      int* child=stack+4*depth;
      child[0]=face->adj[i]; child[1]=edge; child[2]=1; child[3]=1;
      deleteFace(pt,adjacent);
    } else if(!addEdge(pt,face->adj[i],edge)) {
      f[3]=-1;
    }
  }
  return 1;
}


// create horizon given the face as starting point; return 0 on success
CM_DEVICE int horizon(Polytope* pt, Face* face) {
  deleteFace(pt, face);
  int verts[3] = EPA_VERT_EXPAND(face->verts);
  int ret = 0;
  for (int i = 0; i < 3; i++) {
    Face* adjFace = &pt->faces[face->adj[i]];
    if (adjFace->index > -2) {
      int adjEdge = getEdge(adjFace, verts[(i + 1) % 3]);
      int r = horizonRec(pt, adjFace, adjEdge);
      if (r < 0 || (!r && !addEdge(pt, face->adj[i], adjEdge))) {
        ret = -1;
      }
    }
  }
  return ret;
}


// recover witness points from EPA polytope, return signed distance between witness points
CM_DEVICE mjtNum epaWitness(const Polytope* pt, const Face* face, mjtNum x1[3], mjtNum x2[3]) {
  // compute affine coordinates for witness points on plane defined by face
  mjtNum lambda[3];
  int verts[3] = EPA_VERT_EXPAND(face->verts);
  Vertex* v1 = pt->verts + verts[0];
  Vertex* v2 = pt->verts + verts[1];
  Vertex* v3 = pt->verts + verts[2];
  triAffineCoord(lambda, v1->vert, v2->vert, v3->vert, face->v);

  // witness point on geom 1
  lincomb(x1, lambda, 3, v1->vert1, v2->vert1, v3->vert1, NULL);

  // witness point on geom 2
  lincomb(x2, lambda, 3, v1->vert2, v2->vert2, v3->vert2, NULL);

  return -sqrt(face->dist2);
}


// return a face of the expanded polytope that best approximates the pentration depth
// witness points are in status->{x1, x2}
CM_DEVICE Face* epa(mjCCDStatus* status, Polytope* pt, mjCCDObj* obj1, mjCCDObj* obj2) {
  mjtNum upper = mjMAX_LIMIT, upper2 = mjMAX_LIMIT, lower2;
  Face* face = NULL, *pface = NULL;  // face closest to origin
  int discrete = discreteGeom(obj1) && discreteGeom(obj2);

  // discrete geoms return in a finite number of iterations, a non-zero tolerance avoids
  // absurdly small lower and upper bounds
  mjtNum tolerance = discrete ? mjMINEPATOL : status->tolerance;

  int k, kmax = status->max_iterations < 1000 ? status->max_iterations : 1000;
  for (k = 0; k < kmax; k++) {
    pface = face;

    // find the face closest to the origin (lower bound for penetration depth)
    lower2 = mjMAX_LIMIT;
    for (int i = 0; i < pt->nmap; i++) {
      if (pt->map[i]->dist2 < lower2) {
        face = pt->map[i];
        lower2 = face->dist2;
      }
    }

    // face not valid, return previous face
    if (lower2 > upper2 || !face) {
      face = pface;
      break;
    }

    // check if lower bound is 0
    if (lower2 <= 0) {
      status->fault |= 2;
      break;
    }

    // compute support point w from the closest face's normal
    mjtNum lower = sqrt(lower2);
    int wi = epaSupport(pt, obj1, obj2, face->v, lower);
    const Vertex* w = pt->verts + wi;
    mjtNum upper_k = dot3(face->v, w->vert) / lower;  // upper bound for kth iteration
    if (upper_k < upper) {
      upper = upper_k;
      upper2 = upper * upper;
    }
    if (upper - lower < tolerance) {
      // terminate without contact when upper < lower on first iteration
      if (k == 0 && upper < lower - 1e-10) {
        face = NULL;
        status->epa_status = mjEPA_NOCONTACT;
      }
      break;
    }

    // check if vertex w is a repeated support point
    if (discrete) {
      int i = 0, nverts = pt->nverts - 1;
      for (; i < nverts; i++) {
        if (w->index1 == pt->verts[i].index1 && w->index2 == pt->verts[i].index2) {
          break;
        }
      }
      if (i != nverts) {
        break;
      }
    }

    pt->horizon.w = w->vert;
    if (horizon(pt, face) < 0) {
      status->fault |= 4;
      face = NULL;
      break;
    }

    // unrecoverable numerical issue; at least one face was deleted so nedges is 3 or more
    if (pt->horizon.nedges < 3) {
      face = NULL;
      break;
    }

    // insert w as new vertex and attach faces along the horizon
    int nfaces = pt->nfaces, nedges = pt->horizon.nedges;

    // check if there's enough memory to store new faces
    if (nedges > maxFaces(pt)) {
      status->fault |= 8;
      break;
    }

    // attach first face
    int hznIndex = pt->horizon.indices[0], hznEdge = pt->horizon.edges[0];
    Face* hznFace = &pt->faces[hznIndex];
    int hznVerts[3] = EPA_VERT_EXPAND(hznFace->verts);
    int v1 = hznVerts[hznEdge],
        v2 = hznVerts[(hznEdge + 1) % 3];
    hznFace->adj[hznEdge] = nfaces;
    mjtNum dist2 = attachFace(pt, wi, v2, v1, nfaces + nedges - 1, hznIndex, nfaces + 1);

    // unrecoverable numerical issue
    if (dist2 == 0) {
      face = NULL;
      break;
    }

    // store face in map
    if (dist2 >= lower2 && dist2 <= upper2) {
      int i = pt->nmap++;
      pt->map[i] = &pt->faces[pt->nfaces - 1];
      pt->map[i]->index = i;
    }

    // attach remaining faces
    for (int i = 1; i < nedges; i++) {
      int cur = nfaces + i;                  // index of attached face
      int next = nfaces + (i + 1) % nedges;  // index of next face

      hznIndex = pt->horizon.indices[i], hznEdge = pt->horizon.edges[i];
      hznFace = &pt->faces[hznIndex];
      int hznVerts2[3] = EPA_VERT_EXPAND(hznFace->verts);
      v1 = hznVerts2[hznEdge];
      v2 = hznVerts2[(hznEdge + 1) % 3];
      hznFace->adj[hznEdge] = cur;
      dist2 = attachFace(pt, wi, v2, v1, cur - 1, hznIndex, next);

      // unrecoverable numerical issue
      if (dist2 == 0) {
        face = NULL;
        break;
      }

      // store face in map
      if (dist2 >= lower2 && dist2 <= upper2) {
        int idx = pt->nmap++;
        pt->map[idx] = &pt->faces[pt->nfaces - 1];
        pt->map[idx]->index = idx;
      }
    }
    pt->horizon.nedges = 0;  // clear horizon

    // no face candidates left
    if (!pt->nmap || !face) {
      break;
    }
  }

  status->epa_iterations = k;
  if (face) {
    status->dist[0] = epaWitness(pt, face, status->x1, status->x2);
    int fi[3]=EPA_VERT_EXPAND(face->verts);
    mjtNum raw_projection[3];
    projectOriginPlane(raw_projection,pt->verts[fi[2]].vert,pt->verts[fi[1]].vert,pt->verts[fi[0]].vert);
    status->final_sign=dot3(raw_projection,face->v)<0 ? -1 : 1;
    status->nx = 1;
    int verts[3] = EPA_VERT_EXPAND(face->verts);
    obj1->nsupp = 3;
    obj2->nsupp = 3;
    for (int i = 0; i < 3; i++) {
      copy3(obj1->suppvert + 3*i, pt->verts[verts[i]].vert1);
      copy3(obj2->suppvert + 3*i, pt->verts[verts[i]].vert2);
      obj1->suppind[i] = pt->verts[verts[i]].index1;
      obj2->suppind[i] = pt->verts[verts[i]].index2;
    }
  } else {
    status->nx = 0;
    status->dist[0] = 0;
  }
  return face;
}



CM_DEVICE mjtNum mjc_ccd(const mjCCDConfig* config, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2) {
  // setup
  copy3(status->x1, obj1->pos);
  copy3(status->x2, obj2->pos);
  status->gjk_iterations = 0;
  status->epa_iterations = 0;
  status->epa_status = mjEPA_NOCONTACT;
  status->tolerance = config->tolerance;
  status->max_iterations = config->max_iterations;
  int maxcon = sizeof(status->dist) / sizeof(status->dist[0]);
  status->max_contacts = config->max_contacts < maxcon ? config->max_contacts : maxcon;
  status->dist_cutoff = config->dist_cutoff;

  gjk(status, obj1, obj2);

  // penetration recovery for contacts not needed
  if (!config->max_contacts) {
    return status->dist[0];
  }

  if (status->dist[0] <= config->tolerance && obj1->nsupp > 1
      && config->buffer && !status->separated) {
    mjtNum gjk_dist = status->dist[0];
    status->dist[0] = 0;  // assume touching
    Polytope pt;
    pt.nfaces = pt.nmap = pt.nverts = pt.horizon.nedges = 0;
    pt.horizon.stack=config->stack;

    // allocate memory for polytope
    int N = config->max_iterations;
    pt.maxfaces = 6 * N;
    pt.horizon.maxedges = 24 + (N > 35 ? N - 35 : 0);
    uint8_t* buffer = (uint8_t*)config->buffer;
    pt.verts = (Vertex*)buffer;
    buffer += align8(sizeof(Vertex) * (5 + N));
    pt.faces = (Face*)buffer;
    buffer += align8(sizeof(Face) * (6 * N));
    pt.map = (Face**)buffer;
    buffer += align8(sizeof(Face*) * (6 * N));
    pt.horizon.indices = (int*)buffer;
    buffer += align8(sizeof(int) * pt.horizon.maxedges);
    pt.horizon.edges = (int*)buffer;

    int ret;
    if (obj1->nsupp == 2) {
      ret = polytope2(&pt, status, obj1, obj2);
    } else if (obj1->nsupp == 3) {
      ret = polytope3(&pt, status, obj1, obj2);
    } else {
      ret = polytope4(&pt, status, obj1, obj2);
    }
    status->epa_status = ret;

    // simplex not on boundary (objects are penetrating)
    if (!ret) {
      Face* face = epa(status, &pt, obj1, obj2);

      // get multiple contacts if requested
      if (config->max_contacts > 1 && face) {
        status->nx = cmgeom::mjc_multicontact<mjtNum>(obj1, obj2, status->x1, status->x2, status->dist,
                                      status->max_contacts, config->nmeshdegmax,
                                      config->npolygonmax, (uint8_t*)config->buffer);
      }

      // if EPA found no contact during a distance query, restore GJK distance and witness count
      if (!face && config->dist_cutoff > 0 && status->epa_status == mjEPA_NOCONTACT) {
        status->nx = 1;
        status->dist[0] = gjk_dist;
      }
    }
  }
  mjtNum min_dist = status->dist[0];
  for (int i = 1; i < status->nx; i++) {
    if (status->dist[i] < min_dist) {
      min_dist = status->dist[i];
    }
  }
  return min_dist;
}
} // namespace cm
#undef mjMINVAL
#undef mjMINVAL2
#undef mjMAXVAL
#undef mjMAXVAL2
#undef mjMAX_LIMIT
#undef mjMINDIST2
#undef mjMINDIST3
#undef mjMINDIST4
#undef mjMINEPATOL
#undef EPA_VERT_EXPAND

namespace cmgeom {
template<class T> CM_DEVICE void frame(const T* normal,T* result) {
  T n[3]={normal[0],normal[1],normal[2]}, b[3]={0,0,0}, c[3];
  normalize3<T>(n);
  if(n[1]<T(0.5f) && n[1]>T(-0.5f)) b[1]=1; else b[2]=1;
  T dot=dot3<T>(n,b);
  for(int i=0;i<3;i++) b[i]-=n[i]*dot;
  normalize3<T>(b);
  cross3<T>(c,n,b);
  for(int i=0;i<3;i++){result[i]=n[i];result[3+i]=b[i];result[6+i]=c[i];}
}
}
namespace cm {
struct Replay {
  Dual dist[8], x1[24], x2[24], normal[3];
  int count;
};
CM_DEVICE void vertex(Obj<Dual>* o,int index,Dual* result) {
  if(o->geom_type==mjGEOM_BOX) {
    cmgeom::globalcoord<Dual>(result,o->mat,o->pos,
      (index&1)?o->size[0]:-o->size[0],
      (index&2)?o->size[1]:-o->size[1],
      (index&4)?o->size[2]:-o->size[2]);
  } else {
    const float* v=o->data.mesh.vert+3*index;
    cmgeom::globalcoord<Dual>(result,o->mat,o->pos,v[0],v[1],v[2]);
  }
}
// The active support IDs are discrete. The selected face, all witnesses, clipping
// intersections, normals and tangent frame below retain their state derivatives.
CM_DEVICE int replay(int g1,int g2,const float* gx,const float* gq,const CMFeature& feature,
                     int seed,CMScratch* scratch,Replay* output) {
  Obj<Dual> a,b;
  init<Dual>(&a,g1,gx,gq,seed<6 ? seed : -1);
  init<Dual>(&b,g2,gx,gq,seed>=6 ? seed-6 : -1);
  a.nsupp=b.nsupp=3;
  Dual minkowski[9];
  for(int i=0;i<3;i++) {
    a.suppind[i]=feature.support1[i]; b.suppind[i]=feature.support2[i];
    vertex(&a,a.suppind[i],a.suppvert+3*i); vertex(&b,b.suppind[i],b.suppvert+3*i);
    cmgeom::sub3<Dual>(minkowski+3*i,a.suppvert+3*i,b.suppvert+3*i);
  }
  Dual projection[3],lambda[3];
  if(cmgeom::projectOriginPlane<Dual>(projection,minkowski+6,minkowski+3,minkowski)) return 16;
  cmgeom::scl3<Dual>(projection,projection,feature.outward_sign);
  cmgeom::triAffineCoord<Dual>(lambda,minkowski,minkowski+3,minkowski+6,projection);
  for(int k=0;k<3;k++) {
    output->x1[k]=lambda[0]*a.suppvert[k]+lambda[1]*a.suppvert[3+k]+lambda[2]*a.suppvert[6+k];
    output->x2[k]=lambda[0]*b.suppvert[k]+lambda[1]*b.suppvert[3+k]+lambda[2]*b.suppvert[6+k];
  }
  cmgeom::copy3<Dual>(output->normal,projection);
  output->dist[0]=-cmgeom::norm3<Dual>(projection);
  output->count=1;
  if(CM_MULTI) {
    output->count=cmgeom::mjc_multicontact<Dual>(&a,&b,output->x1,output->x2,output->dist,
      4,CM_DEGREE_MAX,CM_POLYGON_MAX,scratch->bytes,output->normal);
  }
  return 0;
}
CM_DEVICE void contact_geometry(const Replay& result,int slot,Dual* dist,Dual* pos,Dual* frame) {
  *dist=result.dist[slot];
  Dual normal[3];
  for(int k=0;k<3;k++) {
    pos[k]=(result.x1[3*slot+k]+result.x2[3*slot+k])*Dual(0.5f);
    // The same selected face direction avoids subtracting nearly coincident
    // world witnesses, which corrupts shallow-contact normals and their VJP.
    normal[k]=result.normal[k];
  }
  cmgeom::frame<Dual>(normal,frame);
}
} // namespace cm

CM_DEVICE void cm_feature_pack(float* words,const CMFeature& feature) {
  const unsigned char* source=(const unsigned char*)&feature;
  unsigned char* target=(unsigned char*)words;
  for(int i=0;i<(int)sizeof(CMFeature);i++) target[i]=source[i];
}
CM_DEVICE CMFeature cm_feature_unpack(const float* words) {
  CMFeature feature;
  const unsigned char* source=(const unsigned char*)words;
  unsigned char* target=(unsigned char*)&feature;
  for(int i=0;i<(int)sizeof(CMFeature);i++) target[i]=source[i];
  return feature;
}

// Invoke on lane zero of one candidate warp. The host launches a bounded
// candidate batch and reuses its scratch pool between batches.
CM_DEVICE void convex_manifold(int g1,int g2,const float* gx,const float* gq,
                               CMScratch* scratch,CMManifold* manifold) {
  manifold->count=0; manifold->status=0;
  if((cm_type[g1]!=6 && cm_type[g1]!=7)||(cm_type[g2]!=6 && cm_type[g2]!=7)
      || (cm_type[g1]!=7 && cm_type[g2]!=7)) {
    manifold->status=1; return;
  }
  // Native dispatch orders primitive types, then source geom IDs. Keep its
  // tie-breaking even when a caller requests the opposite orientation.
  bool flip=cm_type[g1]>cm_type[g2] || (cm_type[g1]==cm_type[g2] && g1>g2);
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  cm::mjCCDObj a,b;
  cm::init<cm::mjtNum>(&a,g1,gx,gq); cm::init<cm::mjtNum>(&b,g2,gx,gq);
  cm::mjCCDConfig config;
  config.max_iterations=CM_CCD_ITERS; config.tolerance=CM_CCD_TOL;
  config.max_contacts=1; config.dist_cutoff=0;
  config.npolygonmax=CM_POLYGON_MAX; config.nmeshdegmax=CM_DEGREE_MAX;
  config.buffer=scratch->bytes;
  config.stack=(int*)(scratch->bytes+CM_SCRATCH_BYTES-cm::align8(4*4*6*CM_CCD_ITERS));
  cm::mjCCDStatus status={};
  cm::mjtNum distance=cm::mjc_ccd(&config,&status,&a,&b);
  manifold->status=status.fault;
  if(distance>=0 || status.nx==0) return;
  if(a.nsupp!=3 || b.nsupp!=3) { manifold->status|=16; return; }
  CMFeature feature={};
  for(int i=0;i<3;i++){feature.support1[i]=a.suppind[i]; feature.support2[i]=b.suppind[i];}
  feature.nsupp1=feature.nsupp2=3; feature.outward_sign=status.final_sign;
  cm::Replay replay;
  int error=cm::replay(g1,g2,gx,gq,feature,-1,scratch,&replay);
  if(error){manifold->status|=error;return;}
  feature.count=replay.count;
  manifold->count=replay.count;
  for(int slot=0;slot<replay.count;slot++) {
    cm::Dual d,p[3],frame[9];
    cm::contact_geometry(replay,slot,&d,p,frame);
    CMContact* c=manifold->contact+slot;
    c->dist=d.v;
    for(int k=0;k<3;k++) c->pos[k]=p[k].v;
    for(int k=0;k<9;k++) c->frame[k]=frame[k].v;
    if(flip) for(int k=0;k<3;k++){c->frame[k]=-c->frame[k];c->frame[6+k]=-c->frame[6+k];}
    feature.slot=slot; c->feature=feature;
  }
}


// Copyright 2016 Svetoslav Kolev. Apache License 2.0, retained notice above.
// Specialized narrowphase only, ported from:
// https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_box.c
namespace cmbb {
using T=cm::Dual;
struct Contact { T dist,pos[3],normal[3]; };
struct Scratch {
  T polygon[2][12][3],accepted[12][3],clip_distance[12];
  Contact contact[8];
};
static_assert(sizeof(Scratch)<=CM_SCRATCH_BYTES,"box geometry fits caller scratch");
constexpr float SEPEPS=1e-6f, PAREPS=1e-7f, SGNEPS=1e-5f, DUPEPS=1e-10f, EDGEBIAS=1e-6f;
constexpr int MAXVERT=12;
CM_DEVICE void copy3(T* out,const T* in){for(int i=0;i<3;i++)out[i]=in[i];}
CM_DEVICE void add3(T* out,const T* a,const T* b){for(int i=0;i<3;i++)out[i]=a[i]+b[i];}
CM_DEVICE void sub3(T* out,const T* a,const T* b){for(int i=0;i<3;i++)out[i]=a[i]-b[i];}
CM_DEVICE void addTo3(T* out,const T* in){for(int i=0;i<3;i++)out[i]+=in[i];}
CM_DEVICE void addToScl3(T* out,const T* in,T scale){for(int i=0;i<3;i++)out[i]+=in[i]*scale;}
CM_DEVICE T dot3(const T* a,const T* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
CM_DEVICE void mulMatVec3(T* out,const T* m,const T* v){
  for(int i=0;i<3;i++)out[i]=m[3*i]*v[0]+m[3*i+1]*v[1]+m[3*i+2]*v[2];
}
CM_DEVICE void mulMatTVec3(T* out,const T* m,const T* v){
  for(int i=0;i<3;i++)out[i]=m[i]*v[0]+m[3+i]*v[1]+m[6+i]*v[2];
}
CM_DEVICE void mulMatTMat3(T* out,const T* a,const T* b){
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)
    out[3*i+j]=a[i]*b[j]+a[3+i]*b[3+j]+a[6+i]*b[6+j];
}
CM_DEVICE void copy(T* out,const T* in,int n){for(int i=0;i<n;i++)out[i]=in[i];}
CM_DEVICE void transpose(T* out,const T* in,int nr,int nc){
  for(int i=0;i<nr;i++)for(int j=0;j<nc;j++)out[j*nr+i]=in[i*nc+j];
}
CM_DEVICE void zero3(T* out){out[0]=out[1]=out[2]=0;}
CM_DEVICE int clamp_choice(T x,T low,T high,CMFeature* active){
  if(x==low || x==high)active->support2[2]|=CM_BOX_CLAMP_BOUNDARY;
  return x<low ? 1 : (x>high ? 2 : 0);
}
CM_DEVICE T clamp_selected(T x,T low,T high,int choice){
  return choice==1 ? low : (choice==2 ? high : x);
}

CM_DEVICE int clipHalfPlane(int nin, T (**cur)[3], T (**spare)[3],
                         int coord, T sign, T limit, T* d, CMFeature* active) {
  T (*in)[3] = *cur;
  int all_inside = 1;
  for (int k = 0; k < nin; k++) {
    d[k] = sign*in[k][coord] - limit;
    all_inside &= d[k] <= 0;
    if(d[k]==0)active->support2[2]|=CM_BOX_CLIP_BOUNDARY;
  }
  if (all_inside) {
    return nin;
  }

  T (*out)[3] = *spare;
  int nout = 0;
  for (int k = 0; k < nin; k++) {
    const T* p = in[k];
    int k1 = k + 1 == nin ? 0 : k + 1;
    T dp = d[k], dq = d[k1];

    // emit p if inside
    if (dp <= 0 && nout < MAXVERT) {
      copy3(out[nout++], p);
    }

    // emit intersection if the edge strictly crosses the plane
    if (((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) && nout < MAXVERT) {
      const T* q = in[k1];
      T t = dp / (dp - dq);
      out[nout][0] = p[0] + t*(q[0] - p[0]);
      out[nout][1] = p[1] + t*(q[1] - p[1]);
      out[nout][2] = p[2] + t*(q[2] - p[2]);
      nout++;
    }
  }
  *cur = out;
  *spare = in;
  return nout;
}

CM_DEVICE int geometry(const T* pos1,const T* mat1,const T* size1,
                       const T* pos2,const T* mat2,const T* size2,
                       const CMFeature* saved,CMFeature* active,Scratch* scratch) {
  const T margin=0;
  Contact* con=scratch->contact;
  // rot: box2 axes in box1 frame (columns); pos21: box2 center in box1 frame;
  // pos12: box1 center in box2 frame
  T rot[9], rotabs[9], pos21[3], pos12[3], tmp[3];
  sub3(tmp, pos2, pos1);
  mulMatTVec3(pos21, mat1, tmp);
  sub3(tmp, pos1, pos2);
  mulMatTVec3(pos12, mat2, tmp);
  mulMatTMat3(rot, mat1, mat2);
  for (int i = 0; i < 9; i++) {
    rotabs[i] = cm::absn<T>(rot[i]);
  }

  //------------------------------ stage 1: separating-axis test

  // the separation tests decide contact against no contact, so they carry rounding slack:
  // without it a box pair that genuinely overlaps by less than the rounding error of its
  // own support evaluation is reported as separated, and the boxes pass through each other
  T septol = margin + SEPEPS*(size1[0] + size1[1] + size1[2] +
                                            size2[0] + size2[1] + size2[2]);

  // best separation so far (most positive; negative = penetration), and the winning axis:
  // code 0..2 face of box1, 3..5 face of box2, >= 6 edge pair (i, j) as 6 + 3*i + j
  T sep_best = -1e10f;
  T sep_face = -1e10f;
  int code = -1;
  int axis_tie=0, face_tie=0;

  if(saved) {
    code=saved->support1[0];
  } else {
  // face axes of box1: candidate normal is axis i of box1
  for (int i = 0; i < 3; i++) {
    T radius2 = rotabs[3*i+0]*size2[0] + rotabs[3*i+1]*size2[1] + rotabs[3*i+2]*size2[2];
    T sep = cm::absn<T>(pos21[i]) - size1[i] - radius2;
    if (sep > septol) {
      return 0;
    }
    if (sep > sep_best) {
      sep_best = sep;
      code = i;
      axis_tie=0;
    } else if(sep==sep_best) {
      axis_tie=1;
    }
  }

  // face axes of box2: candidate normal is axis j of box2
  for (int j = 0; j < 3; j++) {
    T radius1 = rotabs[0+j]*size1[0] + rotabs[3+j]*size1[1] + rotabs[6+j]*size1[2];
    T sep = cm::absn<T>(pos12[j]) - size2[j] - radius1;
    if (sep > septol) {
      return 0;
    }
    if (sep > sep_best) {
      sep_best = sep;
      code = 3 + j;
      axis_tie=0;
    } else if(sep==sep_best) {
      axis_tie=1;
    }
  }
  sep_face = sep_best;
  face_tie=axis_tie;
  int code_face = code;

  // edge-cross axes: candidate direction is axis i of box1 crossed with axis j of box2
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // cross product of e_i with column j of rot, in box1 frame; component i is zero
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      T ax1 = -rot[3*i2+j];
      T ax2 = rot[3*i1+j];

      // the cross product of two unit vectors has norm sin(angle); for nearly parallel
      // edges the components above are pure cancellation noise and the direction is
      // meaningless, so require sin(angle) well above rounding; the skipped axes are
      // covered by the face normals, which the cross product converges to as the angle
      // vanishes
      T norm2 = ax1*ax1 + ax2*ax2;
      if (norm2 < PAREPS) {
        continue;
      }
      T inv = 1/cm::sqrtn<T>(norm2);
      ax1 *= inv;
      ax2 *= inv;

      // support radius of box1: component i of axis is zero by construction
      T radius1 = size1[i1]*cm::absn<T>(ax1) + size1[i2]*cm::absn<T>(ax2);

      // support radius of box2: transform axis to box2 frame; component j is zero there,
      // and only components i1, i2 of the axis are nonzero here
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      T a2_1 = ax1*rot[3*i1+j1] + ax2*rot[3*i2+j1];
      T a2_2 = ax1*rot[3*i1+j2] + ax2*rot[3*i2+j2];
      T radius2 = size2[j1]*cm::absn<T>(a2_1) + size2[j2]*cm::absn<T>(a2_2);

      T sep = cm::absn<T>(ax1*pos21[i1] + ax2*pos21[i2]) - radius1 - radius2;
      if (sep > septol) {
        return 0;
      }

      // an edge axis must beat the best face axis by a bias-scaled amount: on exact ties
      // the face manifold (multiple points) is strictly better for the solver
      if (sep - EDGEBIAS*cm::absn<T>(sep) > sep_best && sep > sep_face) {
        sep_best = sep;
        code = 6 + 3*i + j;
        axis_tie=0;
      }
    }
  }

  if (code < 0) {
    return 0;  // cannot happen: some face axis always sets code
  }

  // a winning edge axis nearly parallel to the best face axis (within ~8 degrees)
  // duplicates it: the face manifold covers the same contact with multiple points, and
  // resting stacks flip between the two codes by rounding noise if the near-tie is
  // allowed to alternate. The face is substituted unless the edge is better by five
  // percent of the face depth (ODE's classic fudge): resting-stack energy degrades
  // continuously as this margin shrinks, while the depth cost of the substitution is
  // bounded by the same five percent. Substituting after the search, rather than
  // filtering during it, prevents a worse non-aliasing edge from stealing the contact
  // that the substitution meant to give to the face.
  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    T axis[3];
    axis[i] = 0;
    axis[i1] = -rot[3*i2+j];
    axis[i2] = rot[3*i1+j];
    cm::normalize3<T>(axis);
    T face_dot;
    if (code_face < 3) {
      face_dot = cm::absn<T>(axis[code_face]);
    } else {
      int f = code_face - 3;
      face_dot = cm::absn<T>(axis[0]*rot[0+f] + axis[1]*rot[3+f] + axis[2]*rot[6+f]);
    }
    if (face_dot > 0.99 && sep_best < sep_face + 0.05*cm::absn<T>(sep_face) + 1e-15f) {
      code = code_face;
      sep_best = sep_face;
      axis_tie=face_tie;
    }
  }

    active->support1[0]=code;
    if(axis_tie)active->support2[2]|=CM_BOX_AXIS_TIE;
  }

  //------------------------------ stage 2a: edge-edge contact

  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    int j1 = (j + 1) % 3, j2 = (j + 2) % 3;

    // unit separating axis in box1 frame, oriented from box1 toward box2
    T axis[3];
    axis[i] = 0;
    axis[i1] = -rot[3*i2+j];
    axis[i2] = rot[3*i1+j];
    cm::normalize3<T>(axis);
    T orientation=dot3(axis,pos21);
    if(orientation==0)active->support2[2]|=CM_BOX_ORIENTATION_BOUNDARY;
    int normal_sign=saved ? saved->support1[1] : (orientation<0 ? -1 : 1);
    active->support1[1]=normal_sign;
    if (normal_sign<0) {
      axis[0] = -axis[0];
      axis[1] = -axis[1];
      axis[2] = -axis[2];
    }

    // supporting edges: the box1 edge runs along e_i at a corner selected by the axis
    // signs in (i1, i2); the box2 edge runs along column j at a corner selected by the
    // signs of the axis in box2 coordinates. A near-zero component makes the sign choice
    // meaningless -- both edges support the axis -- and rounding can pick the wrong one,
    // producing witness points on the wrong side of the box. Enumerate both signs for any
    // ambiguous component (at most one per box) and keep the closest witness pair.
    T a2[3] = {
      axis[0]*rot[0+0] + axis[1]*rot[3+0] + axis[2]*rot[6+0],
      axis[0]*rot[0+1] + axis[1]*rot[3+1] + axis[2]*rot[6+1],
      axis[0]*rot[0+2] + axis[1]*rot[3+2] + axis[2]*rot[6+2],
    };
    const T ambig = SGNEPS;
    int amb1 = -1, amb2 = -1;
    if (cm::absn<T>(axis[i1]) < ambig) amb1 = i1;
    else if (cm::absn<T>(axis[i2]) < ambig) amb1 = i2;
    if (cm::absn<T>(a2[j1]) < ambig) amb2 = j1;
    else if (cm::absn<T>(a2[j2]) < ambig) amb2 = j2;

    if(amb1>=0 || amb2>=0)active->support2[2]|=CM_BOX_SUPPORT_AMBIGUOUS;
    T d2[3] = {rot[0+j], rot[3+j], rot[6+j]};
    T b = d2[i];  // d1 . d2, with d1 = e_i
    T denom = 1 - b*b;

    T w1[3], w2[3];
    T best_d2 = 1e10f;
    for (int v1 = 0; v1 < (saved ? 1 : (amb1 >= 0 ? 2 : 1)); v1++) {
      for (int v2 = 0; v2 < (saved ? 1 : (amb2 >= 0 ? 2 : 1)); v2++) {
        // corner of the box1 edge: support along +axis, ambiguous component flipped by v1
        T c1[3];
        c1[i] = 0;
        c1[i1] = axis[i1] >= 0 ? size1[i1] : -size1[i1];
        c1[i2] = axis[i2] >= 0 ? size1[i2] : -size1[i2];
        if (amb1 >= 0 && v1) c1[amb1] = -c1[amb1];
        if(saved) {
          c1[i1]=(saved->support1[2]&(1<<i1)) ? size1[i1] : -size1[i1];
          c1[i2]=(saved->support1[2]&(1<<i2)) ? size1[i2] : -size1[i2];
        }

        // corner of the box2 edge: support along -axis in box2 coordinates
        T cc[3];
        cc[j] = 0;
        cc[j1] = a2[j1] >= 0 ? -size2[j1] : size2[j1];
        cc[j2] = a2[j2] >= 0 ? -size2[j2] : size2[j2];
        if (amb2 >= 0 && v2) cc[amb2] = -cc[amb2];
        if(saved) {
          cc[j1]=(saved->support2[0]&(1<<j1)) ? size2[j1] : -size2[j1];
          cc[j2]=(saved->support2[0]&(1<<j2)) ? size2[j2] : -size2[j2];
        }
        T c2[3];
        mulMatVec3(c2, rot, cc);
        addTo3(c2, pos21);

        // closest points between the two edge segments (directions are unit vectors)
        T e[3];
        sub3(e, c2, c1);
        T d1e = e[i];  // d1 . e
        T d2e = dot3(d2, e);
        T s = denom < 1e-15f ? 0 : (d1e - b*d2e) / denom;

        // clamp into the segments, letting each clamp re-solve the other parameter
        int choices=0;
        int choice=saved ? (saved->support2[1]&3) : clamp_choice(s,-size1[i],size1[i],active);
        choices|=choice; s=clamp_selected(s,-size1[i],size1[i],choice);
        T raw_t=b*s-d2e;
        choice=saved ? ((saved->support2[1]>>2)&3) : clamp_choice(raw_t,-size2[j],size2[j],active);
        choices|=choice<<2; T t=clamp_selected(raw_t,-size2[j],size2[j],choice);
        T raw_s=d1e+b*t;
        choice=saved ? ((saved->support2[1]>>4)&3) : clamp_choice(raw_s,-size1[i],size1[i],active);
        choices|=choice<<4; s=clamp_selected(raw_s,-size1[i],size1[i],choice);

        T p1[3], p2[3], gap[3];
        copy3(p1, c1);
        p1[i] += s;
        copy3(p2, c2);
        addToScl3(p2, d2, t);
        sub3(gap, p2, p1);
        T gap2 = dot3(gap, gap);
        if (gap2 < best_d2) {
          best_d2 = gap2;
          active->support1[2]=(c1[i1]>=0 ? (1<<i1) : 0) | (c1[i2]>=0 ? (1<<i2) : 0);
          active->support2[0]=(cc[j1]>=0 ? (1<<j1) : 0) | (cc[j2]>=0 ? (1<<j2) : 0);
          active->support2[1]=choices;
          copy3(w1, p1);
          copy3(w2, p2);
        }
      }
    }

    // signed distance along the axis
    T gap[3];
    sub3(gap, w2, w1);
    T dist = dot3(gap, axis);
    if (dist > septol) {
      return 0;
    }

    // contact at the midpoint of the witness pair: for penetrating edges this is inside both
    // boxes; in the margin band it is midway between the two surfaces
    T mid[3] = {0.5*(w1[0] + w2[0]), 0.5*(w1[1] + w2[1]), 0.5*(w1[2] + w2[2])};

    con[0].dist = dist;
    mulMatVec3(tmp, mat1, mid);
    add3(con[0].pos, tmp, pos1);
    mulMatVec3(con[0].normal, mat1, axis);

    return 1;
  }

  //------------------------------ stage 2b: face contact

  // reference box: owner of the winning face; incident box: the other one
  int ref1 = code < 3;              // is box1 the reference?
  int a = ref1 ? code : code - 3;   // face axis of the reference box
  const T* sizeref = ref1 ? size1 : size2;
  const T* sizeinc = ref1 ? size2 : size1;
  const T* posref = ref1 ? pos1 : pos2;
  const T* matref = ref1 ? mat1 : mat2;
  const T* posoi = ref1 ? pos21 : pos12;  // incident center in reference frame

  // incident box axes in reference frame: rot maps box2 to box1, transpose maps box1 to box2;
  // rinc(r, c) = component r of incident axis c, in reference frame
  T rinc[9];
  if (ref1) {
    copy(rinc, rot, 9);
  } else {
    transpose(rinc, rot, 3, 3);
  }

  // face direction: +1 if the incident box lies along +a, else -1
  if(posoi[a]==0)active->support2[2]|=CM_BOX_ORIENTATION_BOUNDARY;
  T sgn=saved ? saved->support1[1] : (posoi[a]>=0 ? 1 : -1);
  active->support1[1]=sgn.v>0 ? 1 : -1;

  // incident face: the face of the incident box most opposed to the reference face normal
  int binc = 0;
  for (int k = 1; k < 3; k++) {
    if (cm::absn<T>(rinc[3*a+k]) > cm::absn<T>(rinc[3*a+binc])) {
      binc = k;
    }
  }
  if(saved)binc=saved->support1[2];
  active->support1[2]=binc;
  for(int k=0;k<3;k++)if(k!=binc && cm::absn<T>(rinc[3*a+k])==cm::absn<T>(rinc[3*a+binc]))
    active->support2[2]|=CM_BOX_INCIDENT_AXIS_TIE;
  T tinc=saved ? saved->support2[0] : (sgn*rinc[3*a+binc]>0 ? -1 : 1);
  active->support2[0]=tinc.v>0 ? 1 : -1;

  // corners of the incident face in reference frame, cyclic winding; the in-plane
  // coordinates are (x, y) = the two non-a reference axes, z is the signed distance
  // above the reference face plane (negative = inside the reference box)
  int ax = (a + 1) % 3, ay = (a + 2) % 3;
  int bu = (binc + 1) % 3, bv = (binc + 2) % 3;
  T (*poly)[MAXVERT][3]=scratch->polygon;

  // face center and in-face half-edge offsets, in the projected (x, y, z) coordinates
  T cx[3], du[3], dv[3];
  for (int r = 0; r < 3; r++) {
    int c = r == 0 ? ax : (r == 1 ? ay : a);
    cx[r] = posoi[c] + tinc*sizeinc[binc]*rinc[3*c+binc];
    du[r] = sizeinc[bu]*rinc[3*c+bu];
    dv[r] = sizeinc[bv]*rinc[3*c+bv];
  }
  cx[2] = sgn*cx[2] - sizeref[a];
  du[2] *= sgn;
  dv[2] *= sgn;
  const int corner_sign[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
  for (int k = 0; k < 4; k++) {
    T su = corner_sign[k][0], sv = corner_sign[k][1];
    poly[0][k][0] = cx[0] + su*du[0] + sv*dv[0];
    poly[0][k][1] = cx[1] + su*du[1] + sv*dv[1];
    poly[0][k][2] = cx[2] + su*du[2] + sv*dv[2];
  }

  // clip against the four side planes of the reference face; the buffers swap only on
  // passes that actually clip
  int nvert = 4;
  T (*cur)[3] = poly[0];
  T (*spare)[3] = poly[1];
  nvert = clipHalfPlane(nvert, &cur, &spare, 0, 1, sizeref[ax],scratch->clip_distance,active);
  nvert = clipHalfPlane(nvert, &cur, &spare, 0, -1, sizeref[ax],scratch->clip_distance,active);
  nvert = clipHalfPlane(nvert, &cur, &spare, 1, 1, sizeref[ay],scratch->clip_distance,active);
  nvert = clipHalfPlane(nvert, &cur, &spare, 1, -1, sizeref[ay],scratch->clip_distance,active);

  // accept vertices within the margin band, dropping near-duplicates produced by clipping
  // at polygon corners; duplicate radius is relative to the reference face scale
  T (*accepted)[3]=scratch->accepted;
  int naccept = 0;
  T dupe2 = DUPEPS*(sizeref[ax]*sizeref[ax] + sizeref[ay]*sizeref[ay]);
  for (int k = 0; k < nvert; k++) {
    if(cur[k][2]==margin)active->support2[2]|=CM_BOX_ZERO_DEPTH;
    if (cur[k][2] > margin) {
      continue;
    }
    int dupe = 0;
    for (int q = 0; q < naccept; q++) {
      T dx = accepted[q][0] - cur[k][0];
      T dy = accepted[q][1] - cur[k][1];
      if(dx*dx+dy*dy==dupe2)active->support2[2]|=CM_BOX_DEDUPE_BOUNDARY;
      if (dx*dx + dy*dy < dupe2) {
        dupe = 1;
        break;
      }
    }
    if (!dupe) {
      copy3(accepted[naccept++], cur[k]);
    }
  }
  if(naccept>8)return -32;
  if (naccept == 0) {
    return 0;
  }

  // world normal points from geom1 to geom2: along +sgn*a of the reference frame when box1
  // is the reference, opposite when box2 is
  T normal[3];
  T nsign = ref1 ? sgn : -sgn;
  normal[0] = nsign*matref[3*0+a];
  normal[1] = nsign*matref[3*1+a];
  normal[2] = nsign*matref[3*2+a];

  for (int k = 0; k < naccept; k++) {
    const T* v = accepted[k];

    // contact position: on the clipped incident polygon in (x, y), midway between the
    // reference face plane and the incident surface along the face axis
    T posc[3];
    posc[ax] = v[0];
    posc[ay] = v[1];
    posc[a] = sgn*(sizeref[a] + 0.5*v[2]);

    con[k].dist = v[2];
    mulMatVec3(tmp, matref, posc);
    add3(con[k].pos, tmp, posref);
    copy3(con[k].normal, normal);

  }
  return naccept;
}
} // namespace cmbb

namespace cmbb {
CM_DEVICE int evaluate(int g1,int g2,const float* gx,const float* gq,const CMFeature* saved,
                       int seed,CMScratch* scratch,CMFeature* active) {
  cm::Obj<T> a,b;
  cm::init<T>(&a,g1,gx,gq,seed<6 ? seed : -1);
  cm::init<T>(&b,g2,gx,gq,seed>=6 ? seed-6 : -1);
  *active=saved ? *saved : CMFeature{};
  active->reserved=1;
  return geometry(a.pos,a.mat,a.size,b.pos,b.mat,b.size,saved,active,(Scratch*)scratch->bytes);
}
CM_DEVICE void output(const Contact& c,bool flip,const CMFeature& feature,CMContact* out) {
  T frame[9];
  cmgeom::frame<T>(c.normal,frame);
  out->dist=c.dist.v;
  for(int k=0;k<3;k++)out->pos[k]=c.pos[k].v;
  for(int k=0;k<9;k++)out->frame[k]=frame[k].v;
  if(flip)for(int k=0;k<3;k++){out->frame[k]=-out->frame[k];out->frame[6+k]=-out->frame[6+k];}
  out->feature=feature;
}
CM_DEVICE bool valid_feature(const CMFeature& f) {
  return f.reserved==1 && f.support1[0]>=0 && f.support1[0]<15 &&
         f.count>0 && f.count<=8 && f.slot>=0 && f.slot<f.count;
}
} // namespace cmbb

CM_DEVICE void box_box_manifold(int g1,int g2,const float* gx,const float* gq,
                                CMScratch* scratch,CMManifold* manifold) {
  manifold->count=manifold->status=0;
  if(cm_type[g1]!=6 || cm_type[g2]!=6){manifold->status=1;return;}
  bool flip=g1>g2;
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  CMFeature feature;
  int count=cmbb::evaluate(g1,g2,gx,gq,nullptr,-1,scratch,&feature);
  if(count<0){manifold->status=-count;return;}
  feature.count=count;
  manifold->count=count;
  const cmbb::Contact* contacts=((cmbb::Scratch*)scratch->bytes)->contact;
  for(int slot=0;slot<count;slot++){
    feature.slot=slot;
    cmbb::output(contacts[slot],flip,feature,manifold->contact+slot);
  }
}

CM_DEVICE int box_box_contact_replay(int g1,int g2,const float* gx,const float* gq,
                                     const CMFeature& feature,CMScratch* scratch,CMContact* output) {
  if(!cmbb::valid_feature(feature))return 16;
  bool flip=g1>g2;
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  CMFeature active;
  int count=cmbb::evaluate(g1,g2,gx,gq,&feature,-1,scratch,&active);
  if(count!=feature.count)return 16;
  const cmbb::Contact* contacts=((cmbb::Scratch*)scratch->bytes)->contact;
  cmbb::output(contacts[feature.slot],flip,feature,output);
  return 0;
}

CM_DEVICE int box_box_contact_vjp(int g1,int g2,const float* gx,const float* gq,
                                  const CMFeature& feature,float dbar,const float* pbar,
                                  const float* frbar,CMScratch* scratch,float* gb12) {
  for(int k=0;k<12;k++)gb12[k]=0.f;
  if(!cmbb::valid_feature(feature))return 16;
  bool flip=g1>g2;
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  for(int seed=0;seed<12;seed++) {
    int canonical_seed=flip ? (seed<6 ? seed+6 : seed-6) : seed;
    CMFeature active;
    int count=cmbb::evaluate(g1,g2,gx,gq,&feature,canonical_seed,scratch,&active);
    if(count!=feature.count)return 16;
    const cmbb::Contact& c=((cmbb::Scratch*)scratch->bytes)->contact[feature.slot];
    cmbb::T frame[9];cmgeom::frame<cmbb::T>(c.normal,frame);
    if(flip)for(int k=0;k<3;k++){frame[k]=-frame[k];frame[6+k]=-frame[6+k];}
    float g=dbar*c.dist.d;
    for(int k=0;k<3;k++)g+=pbar[k]*c.pos[k].d;
    for(int k=0;k<9;k++)g+=frbar[k]*frame[k].d;
    gb12[seed]=g;
  }
  return 0;
}


// Returns a geometry VJP in [translation, world rotation tangent] order for
// each geom. No finite differences, frozen contact normals or straight-through
// updates enter this derivative. At a feature change the local derivative is
// the selected branch's derivative, as with the core primitive collider.
CM_DEVICE int convex_contact_replay(int g1,int g2,const float* gx,const float* gq,
                                   const CMFeature& feature,CMScratch* scratch,CMContact* output) {
  if(feature.nsupp1!=3 || feature.nsupp2!=3 || feature.slot<0 || feature.slot>=feature.count) return 16;
  bool flip=cm_type[g1]>cm_type[g2] || (cm_type[g1]==cm_type[g2] && g1>g2);
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  cm::Replay replay;
  int error=cm::replay(g1,g2,gx,gq,feature,-1,scratch,&replay);
  if(error || replay.count!=feature.count) return 16;
  cm::Dual d,p[3],frame[9];
  cm::contact_geometry(replay,feature.slot,&d,p,frame);
  output->dist=d.v;
  for(int k=0;k<3;k++) output->pos[k]=p[k].v;
  for(int k=0;k<9;k++) output->frame[k]=frame[k].v;
  if(flip) for(int k=0;k<3;k++){output->frame[k]=-output->frame[k];output->frame[6+k]=-output->frame[6+k];}
  output->feature=feature;
  return 0;
}

CM_DEVICE int convex_contact_vjp(int g1,int g2,const float* gx,const float* gq,
                                const CMFeature& feature,float dbar,const float* pbar,
                                const float* frbar,CMScratch* scratch,float* gb12) {
  for(int i=0;i<12;i++) gb12[i]=0.f;
  if(feature.nsupp1!=3 || feature.nsupp2!=3 || feature.slot<0 || feature.slot>=feature.count) return 16;
  bool flip=cm_type[g1]>cm_type[g2] || (cm_type[g1]==cm_type[g2] && g1>g2);
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  for(int seed=0;seed<12;seed++) {
    cm::Replay replay;
    int canonical_seed=flip ? (seed<6 ? seed+6 : seed-6) : seed;
    int error=cm::replay(g1,g2,gx,gq,feature,canonical_seed,scratch,&replay);
    if(error || replay.count!=feature.count) return 16;
    cm::Dual d,p[3],frame[9];
    cm::contact_geometry(replay,feature.slot,&d,p,frame);
    if(flip) for(int k=0;k<3;k++){frame[k]=-frame[k];frame[6+k]=-frame[6+k];}
    float g=dbar*d.d;
    for(int k=0;k<3;k++) g+=pbar[k]*p[k].d;
    for(int k=0;k<9;k++) g+=frbar[k]*frame[k].d;
    gb12[seed]=g;
  }
  return 0;
}

// Seeded replay always reconstructs every slot of the selected manifold. Keep
// those results once per pair instead of rebuilding the same clipping polygon
// twelve times for each contact. Each slot's contraction below retains the
// single-contact VJP's operation order.
CM_DEVICE void cm_linearization_store(CMLinearization* result,int slot,int seed,
                                      cm::Dual dist,const cm::Dual* pos,const cm::Dual* frame) {
  if(seed==0) {
    result->value[slot][0]=dist.v;
    for(int k=0;k<3;k++)result->value[slot][1+k]=pos[k].v;
    for(int k=0;k<9;k++)result->value[slot][4+k]=frame[k].v;
  }
  float* derivative=result->jacobian[slot][seed];
  derivative[0]=dist.d;
  for(int k=0;k<3;k++)derivative[1+k]=pos[k].d;
  for(int k=0;k<9;k++)derivative[4+k]=frame[k].d;
}

CM_DEVICE int convex_manifold_linearize(int g1,int g2,const float* gx,const float* gq,
                                        const CMFeature& feature,CMScratch* scratch,CMLinearization* result) {
  if(feature.nsupp1!=3 || feature.nsupp2!=3 || feature.slot<0 || feature.slot>=feature.count ||
     feature.count>CM_REPLAY_SLOTS)return 16;
  bool flip=cm_type[g1]>cm_type[g2] || (cm_type[g1]==cm_type[g2] && g1>g2);
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  for(int seed=0;seed<12;seed++) {
    cm::Replay replay;
    int canonical_seed=flip ? (seed<6 ? seed+6 : seed-6) : seed;
    int error=cm::replay(g1,g2,gx,gq,feature,canonical_seed,scratch,&replay);
    if(error || replay.count!=feature.count)return 16;
    for(int slot=0;slot<replay.count;slot++) {
      cm::Dual d,p[3],frame[9];
      cm::contact_geometry(replay,slot,&d,p,frame);
      if(flip)for(int k=0;k<3;k++){frame[k]=-frame[k];frame[6+k]=-frame[6+k];}
      cm_linearization_store(result,slot,seed,d,p,frame);
    }
  }
  return 0;
}

CM_DEVICE int box_box_manifold_linearize(int g1,int g2,const float* gx,const float* gq,
                                         const CMFeature& feature,CMScratch* scratch,CMLinearization* result) {
  if(!cmbb::valid_feature(feature) || feature.count>CM_REPLAY_SLOTS)return 16;
  bool flip=g1>g2;
  if(flip){int tmp=g1;g1=g2;g2=tmp;}
  for(int seed=0;seed<12;seed++) {
    int canonical_seed=flip ? (seed<6 ? seed+6 : seed-6) : seed;
    CMFeature active;
    int count=cmbb::evaluate(g1,g2,gx,gq,&feature,canonical_seed,scratch,&active);
    if(count!=feature.count)return 16;
    const cmbb::Contact* contacts=((cmbb::Scratch*)scratch->bytes)->contact;
    for(int slot=0;slot<count;slot++) {
      cmbb::T frame[9];cmgeom::frame<cmbb::T>(contacts[slot].normal,frame);
      if(flip)for(int k=0;k<3;k++){frame[k]=-frame[k];frame[6+k]=-frame[6+k];}
      cm_linearization_store(result,slot,seed,contacts[slot].dist,contacts[slot].pos,frame);
    }
  }
  return 0;
}

CM_DEVICE void contact_linearization_vjp(const CMLinearization& result,int slot,float dbar,
                                         const float* pbar,const float* frbar,float* gb12) {
  for(int seed=0;seed<12;seed++) {
    const float* derivative=result.jacobian[slot][seed];
    float g=dbar*derivative[0];
    for(int k=0;k<3;k++)g+=pbar[k]*derivative[1+k];
    for(int k=0;k<9;k++)g+=frbar[k]*derivative[4+k];
    gb12[seed]=g;
  }
}



"""
