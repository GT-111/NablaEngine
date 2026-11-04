"""CPU oracle and analytic geometry checks; no CUDA or physics time advance.

PYTHONPATH=src python scripts/convex_mesh_check.py
The generated narrowphase is compiled without linking MuJoCo. MuJoCo 3.17 is
the independent collision oracle. A temporary directory owns all build output.
"""

import ctypes
import pathlib
import subprocess
import tempfile
import unittest

import mujoco
import numpy as np

from nablaengine.g1_engine import convex_mesh


VERTICES = "-.1 -.12 -.08 .1 -.12 -.08 -.1 .12 -.08 .1 .12 -.08 -.1 -.12 .08 .1 -.12 .08 -.1 .12 .08 .1 .12 .08"


def pair_model(second="mesh", *, multiccd=True):
    flag = "enable" if multiccd else "disable"
    geom = 'type="mesh" mesh="hull"' if second == "mesh" else 'type="box" size=".1 .12 .08"'
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option ccd_iterations="35" ccd_tolerance="1e-6">'
        f'<flag nativeccd="enable" multiccd="{flag}"/></option><asset>'
        f'<mesh name="hull" vertex="{VERTICES}"/></asset><worldbody>'
        '<body name="a"><freejoint/><geom name="ga" type="mesh" mesh="hull"/></body>'
        f'<body name="b" pos=".17 .013 .009"><freejoint/><geom name="gb" {geom}/></body>'
        '</worldbody></mujoco>')


def many_sided_model(pyramid=False):
    n = 23
    angles = 2 * np.pi * (np.arange(n) + .07 * np.sin(np.arange(n))) / n
    if pyramid:
        vertices = [(.1 * np.cos(t), .12 * np.sin(t), -.08) for t in angles] + [(.013, .009, .15)]
        pose = 'pos=".013 .009 -.14" quat="0 1 0 0"'
    else:
        vertices = [(.1 * np.cos(t), .12 * np.sin(t), z) for z in (-.08, .08) for t in angles]
        pose = 'pos=".013 .009 .14" euler="0 0 11"'
    vertex_text = " ".join(str(float(k)) for vertex in vertices for k in vertex)
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option ccd_iterations="35" ccd_tolerance="1e-6"/><asset>'
        f'<mesh name="hull" vertex="{vertex_text}"/></asset><worldbody>'
        '<body><freejoint/><geom type="mesh" mesh="hull"/></body>'
        f'<body {pose}><freejoint/><geom type="mesh" mesh="hull"/></body>'
        '</worldbody></mujoco>')


def multiple_mesh_model():
    vertices = np.asarray([float(x) for x in VERTICES.split()]).reshape(-1, 3)
    scaled = " ".join(str(x) for x in (vertices * (.8, .9, 1.1)).ravel())
    return mujoco.MjModel.from_xml_string(
        '<mujoco><asset>'
        f'<mesh name="unused" vertex="{VERTICES}"/>'
        f'<mesh name="a" vertex="{VERTICES}"/><mesh name="b" vertex="{scaled}"/>'
        '</asset><worldbody><body><freejoint/><geom type="mesh" mesh="a"/></body>'
        '<body pos=".15 .013 .009"><freejoint/><geom type="mesh" mesh="b"/></body>'
        '</worldbody></mujoco>')


def box_model(*, octagon=False, multiccd=True):
    size = '.1 .1 .03' if octagon else '.1 .12 .08'
    pose = 'pos=".003 .007 .045" euler=".02 .03 .4"' if octagon else 'pos=".17 .013 .009"'
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option>'
        f'<flag multiccd="{"enable" if multiccd else "disable"}"/></option><worldbody>'
        f'<body><freejoint/><geom type="box" size="{size}"/></body>'
        f'<body {pose}><freejoint/><geom type="box" size="{size}"/></body>'
        '</worldbody></mujoco>')


def poses(m, qpos=None):
    d = mujoco.MjData(m)
    if qpos is not None:
        d.qpos[:] = qpos
    mujoco.mj_fwdPosition(m, d)
    q = np.empty((m.ngeom, 4))
    for g in range(m.ngeom):
        mujoco.mju_mat2Quat(q[g], d.geom_xmat[g])
    return d, np.asarray(d.geom_xpos, np.float32).copy(), np.asarray(q, np.float32)


class NativeGeometry:
    """One compiled model and its direct C ABI, independent of MuJoCo at run time."""

    def __init__(self, m, geoms=None):
        self.directory = tempfile.TemporaryDirectory(prefix="nabla-convex-cpu-")
        geoms = list(range(m.ngeom)) if geoms is None else list(geoms)
        packed, layout = convex_mesh.constants(m, geoms)
        load = []
        for name, kind, size in layout:
            load.append(f"for(int i=0;i<{size};i++) {name}[i]=({ 'int' if kind == 'i' else 'float' })*p++;")
        wrapper = r'''
extern "C" void cm_load(const float* p) { LOAD }
extern "C" int cm_support_gap() {
  const float vertices[6]={1.f,1.f,0.f,1.f,1.00000011920928955078125f,0.f};
  const int graph[10]={2,4,0,2,0,1,1,-1,0,-1}, extrema[27]={};
  cm::mjCCDObj object={};
  object.geom_type=7; object.vertindex=object.meshindex=-1;
  object.mat[0]=object.mat[4]=object.mat[8]=1;
  object.data.mesh.vert=vertices; object.data.mesh.nvert=2;
  object.data.mesh.graph=graph; object.data.mesh.extrema=extrema;
  const cm::mjtNum direction[3]={1,.001,0}; cm::mjtNum result[3];
  cm::mjc_meshSupport(result,&object,direction);
  int ordered=object.vertindex==1;
  object.vertindex=object.meshindex=-1;
  cm::mjc_hillclimbSupport(result,&object,direction);
  return ordered | ((object.vertindex==1)<<1);
}
struct GuardedScratch { uint64_t before; CMScratch scratch; uint64_t after; };
extern "C" int cm_forward(int g1,int g2,const float* x,const float* q,float* out,int* feature,int* error) {
  GuardedScratch guard; guard.before=0x18273645; guard.after=0x54637281;
  CMManifold m;
  if(cm_type[g1]==6 && cm_type[g2]==6)box_box_manifold(g1,g2,x,q,&guard.scratch,&m);
  else convex_manifold(g1,g2,x,q,&guard.scratch,&m);
  *error=m.status;
  for(int i=0;i<m.count;i++) {
    out[13*i]=m.contact[i].dist;
    for(int k=0;k<3;k++) out[13*i+1+k]=m.contact[i].pos[k];
    for(int k=0;k<9;k++) out[13*i+4+k]=m.contact[i].frame[k];
    float packed[CM_FEATURE_WORDS]; cm_feature_pack(packed,m.contact[i].feature);
    CMFeature unpacked=cm_feature_unpack(packed);
    const unsigned char* p=(const unsigned char*)&unpacked;
    unsigned char* f=(unsigned char*)(feature+12*i);
    for(int k=0;k<(int)sizeof(CMFeature);k++) f[k]=p[k];
  }
  if(guard.before!=0x18273645 || guard.after!=0x54637281) *error|=32;
  return m.count;
}
extern "C" int cm_reverse(int g1,int g2,const float* x,const float* q,const int* feature,
                           const float* cotangent,float* gradient) {
  GuardedScratch guard; guard.before=0x18273645; guard.after=0x54637281;
  CMFeature f; const unsigned char* p=(const unsigned char*)feature; unsigned char* b=(unsigned char*)&f;
  for(int i=0;i<(int)sizeof(f);i++) b[i]=p[i];
  int error=f.reserved==1
      ? box_box_contact_vjp(g1,g2,x,q,f,cotangent[0],cotangent+1,cotangent+4,&guard.scratch,gradient)
      : convex_contact_vjp(g1,g2,x,q,f,cotangent[0],cotangent+1,cotangent+4,&guard.scratch,gradient);
  if(guard.before!=0x18273645 || guard.after!=0x54637281) error|=32;
  return error;
}
extern "C" int cm_replay(int g1,int g2,const float* x,const float* q,const int* feature,float* out) {
  GuardedScratch guard; guard.before=0x18273645; guard.after=0x54637281;
  CMFeature f; const unsigned char* p=(const unsigned char*)feature; unsigned char* b=(unsigned char*)&f;
  for(int i=0;i<(int)sizeof(f);i++) b[i]=p[i];
  CMContact c;
  int error=f.reserved==1 ? box_box_contact_replay(g1,g2,x,q,f,&guard.scratch,&c)
                        : convex_contact_replay(g1,g2,x,q,f,&guard.scratch,&c);
  out[0]=c.dist;
  for(int i=0;i<3;i++) out[1+i]=c.pos[i];
  for(int i=0;i<9;i++) out[4+i]=c.frame[i];
  if(guard.before!=0x18273645 || guard.after!=0x54637281) error|=32;
  return error;
}
extern "C" int cm_linearize(int g1,int g2,const float* x,const float* q,const int* feature,
                             const float* cotangent,float* value,float* gradient) {
  GuardedScratch guard; guard.before=0x18273645; guard.after=0x54637281;
  struct { uint64_t before; CMLinearization data; uint64_t after; } cache;
  cache.before=0x18273645; cache.after=0x54637281;
  CMFeature f; const unsigned char* p=(const unsigned char*)feature; unsigned char* b=(unsigned char*)&f;
  for(int i=0;i<(int)sizeof(f);i++) b[i]=p[i];
  int error=f.reserved==1 ? box_box_manifold_linearize(g1,g2,x,q,f,&guard.scratch,&cache.data)
                        : convex_manifold_linearize(g1,g2,x,q,f,&guard.scratch,&cache.data);
  if(error)return error;
  for(int slot=0;slot<f.count;slot++) {
    for(int k=0;k<13;k++)value[13*slot+k]=cache.data.value[slot][k];
    contact_linearization_vjp(cache.data,slot,cotangent[13*slot],cotangent+13*slot+1,
                              cotangent+13*slot+4,gradient+12*slot);
  }
  if(guard.before!=0x18273645 || guard.after!=0x54637281 ||
     cache.before!=0x18273645 || cache.after!=0x54637281)error|=32;
  return error;
}
'''.replace("LOAD", "\n".join(load))
        path = pathlib.Path(self.directory.name)
        (path / "geometry.cc").write_text("#define DKC\n" + convex_mesh.header(m, geoms)
                                          + convex_mesh.source() + wrapper)
        subprocess.run(["c++", "-shared", "-fPIC", "-std=c++17", "-O2", "-ffp-contract=off",
                        str(path / "geometry.cc"), "-o", str(path / "geometry.so")], check=True,
                       capture_output=True, text=True)
        self.library = ctypes.CDLL(str(path / "geometry.so"))
        fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
        self.library.cm_load.argtypes = [fp]
        self.library.cm_forward.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp, fp, ip, ip]
        self.library.cm_reverse.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp, ip, fp, fp]
        self.library.cm_replay.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp, ip, fp]
        self.library.cm_linearize.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp, ip, fp, fp, fp]
        self.library.cm_load(packed.ctypes.data_as(fp))

    @staticmethod
    def _float(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _int(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def forward(self, x, q, pair=(0, 1)):
        x, q = np.ascontiguousarray(x, np.float32), np.ascontiguousarray(q, np.float32)
        result, feature, error = np.empty((8, 13), np.float32), np.empty((8, 12), np.int32), np.zeros(1, np.int32)
        n = self.library.cm_forward(*pair, self._float(x), self._float(q), self._float(result),
                                    self._int(feature), self._int(error))
        if error[0]:
            raise AssertionError(f"native narrowphase status={error[0]}")
        return result[:n].copy(), feature[:n].copy()

    def reverse(self, x, q, feature, cotangent, pair=(0, 1)):
        x, q, cotangent = (np.ascontiguousarray(a, np.float32) for a in (x, q, cotangent))
        feature, gradient = np.ascontiguousarray(feature, np.int32), np.zeros(12, np.float32)
        error = self.library.cm_reverse(*pair, self._float(x), self._float(q), self._int(feature),
                                        self._float(cotangent), self._float(gradient))
        if error:
            raise AssertionError(f"analytic replay status={error}")
        return gradient

    def replay(self, x, q, feature, pair=(0, 1)):
        x, q = (np.ascontiguousarray(a, np.float32) for a in (x, q))
        feature, result = np.ascontiguousarray(feature, np.int32), np.empty(13, np.float32)
        error = self.library.cm_replay(*pair, self._float(x), self._float(q), self._int(feature), self._float(result))
        if error:
            raise AssertionError(f"feature replay status={error}")
        return result

    def linearize(self, x, q, feature, cotangent, pair=(0, 1)):
        x, q, cotangent = (np.ascontiguousarray(a, np.float32) for a in (x, q, cotangent))
        feature = np.ascontiguousarray(feature, np.int32)
        value, gradient = np.empty((8, 13), np.float32), np.empty((8, 12), np.float32)
        error = self.library.cm_linearize(*pair, self._float(x), self._float(q), self._int(feature),
            self._float(cotangent), self._float(value), self._float(gradient))
        if error:
            raise AssertionError(f"manifold linearization status={error}")
        return value[:feature[9]], gradient[:feature[9]]


def perturb(x, q, index, amount):
    x, q = x.astype(np.float64).copy(), q.astype(np.float64).copy()
    geom, dof = divmod(index, 6)
    if dof < 3:
        x[geom, dof] += amount
    else:
        delta = np.zeros(4)
        delta[0], delta[dof - 2] = np.cos(amount / 2), np.sin(amount / 2)
        changed = np.empty(4)
        mujoco.mju_mulQuat(changed, delta, q[geom])
        q[geom] = changed
    return np.asarray(x, np.float32), np.asarray(q, np.float32)


class ConvexMeshCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if mujoco.__version__ != "3.17.0":
            raise unittest.SkipTest("independent native CCD oracle is pinned to MuJoCo 3.17.0")
        cls.mesh = pair_model()
        cls.box = pair_model("box")
        cls.single = pair_model(multiccd=False)
        cls.prism = many_sided_model()
        cls.pyramid = many_sided_model(pyramid=True)
        cls.multiple = multiple_mesh_model()
        cls.boxes = box_model()
        cls.octagon = box_model(octagon=True)
        cls.octagon_no_multi = box_model(octagon=True, multiccd=False)
        cls.engines = {id(m): NativeGeometry(m) for m in
                       (cls.mesh, cls.box, cls.single, cls.prism, cls.pyramid, cls.multiple,
                        cls.boxes, cls.octagon, cls.octagon_no_multi)}

    @classmethod
    def tearDownClass(cls):
        for engine in cls.engines.values():
            engine.directory.cleanup()

    def assert_oracle(self, model, qpos=None):
        data, x, q = poses(model, qpos)
        actual, _ = self.engines[id(model)].forward(x, q)
        expected = np.asarray([[c.dist, *c.pos, *c.frame] for c in data.contact]).reshape(-1, 13)
        for row, contact in zip(expected, data.contact):
            if contact.geom[0] != 0:
                row[4:7] *= -1
                row[10:13] *= -1
        self.assertEqual(len(actual), len(expected))
        # Contact order can differ at equivalent hull ties; compare nearest points.
        remaining = list(range(len(expected)))
        for contact in actual:
            nearest = min(remaining, key=lambda i: np.linalg.norm(contact[1:4] - expected[i, 1:4]))
            np.testing.assert_allclose(contact, expected[nearest], atol=3e-6, rtol=3e-5)
            remaining.remove(nearest)
        self.assertEqual(data.time, 0)
        return actual, x, q

    def test_native_mesh_mesh_and_box_mesh_manifolds(self):
        for model in (self.mesh, self.box):
            with self.subTest(second=int(model.geom_type[1])):
                actual, x, q = self.assert_oracle(model)
                self.assertEqual(len(actual), 4)
                reverse, _ = self.engines[id(model)].forward(x, q, pair=(1, 0))
                self.assertEqual(len(reverse), len(actual))
                np.testing.assert_allclose(sorted(reverse[:, 0]), sorted(actual[:, 0]), atol=3e-6)

    def test_native_support_preserves_gaps_below_float_dot_product_precision(self):
        # Adjacent float32 hull vertices can have distinct support values whose
        # dot products round equal in float32, selecting a different EPA face.
        self.assertEqual(self.engines[id(self.mesh)].library.cm_support_gap(), 3)

    def test_touching_separated_rotated_and_deep_penetration(self):
        for offset, angle in ((2 * float(np.float32(.1)), 0), (0.23, .13), (.085, .21), (.174, .18)):
            qpos = self.mesh.qpos0.copy()
            qpos[7:10] = (offset, .013, .009)
            qpos[10:14] = (np.cos(angle / 2), 0, 0, np.sin(angle / 2))
            with self.subTest(offset=offset, angle=angle):
                self.assert_oracle(self.mesh, qpos)

    def test_multiccd_off_keeps_native_single_witness(self):
        actual, _, _ = self.assert_oracle(self.single)
        self.assertEqual(len(actual), 1)

    def test_high_polygon_count_and_vertex_degree_prune_to_native_four_points(self):
        self.assertEqual(self.prism.npolygonmax, 23)
        self.assertEqual(self.pyramid.nmeshdegmax, 23)
        for model in (self.prism, self.pyramid):
            actual, _, _ = self.assert_oracle(model)
            self.assertEqual(len(actual), 4)

    def test_multiple_hulls_rebase_global_mesh_maps_and_skip_unused_mesh(self):
        self.assertEqual(self.multiple.nmesh, 3)
        np.testing.assert_array_equal(self.multiple.geom_dataid, [1, 2])
        actual, _, _ = self.assert_oracle(self.multiple)
        self.assertEqual(len(actual), 4)

    def test_edge_edge_and_edge_face_replay_preserves_normal_derivatives(self):
        axis = np.array([.3, .5, .8])
        axis /= np.linalg.norm(axis)
        engine = self.engines[id(self.mesh)]
        bar = np.random.default_rng(31).normal(size=13).astype(np.float32)
        for angle, count in ((.1, 2), (.25, 1)):
            qpos = self.mesh.qpos0.copy()
            qpos[10], qpos[11:14] = np.cos(angle / 2), axis * np.sin(angle / 2)
            contacts, x, q = self.assert_oracle(self.mesh, qpos)
            self.assertEqual(len(contacts), count)
            _, features = engine.forward(x, q)
            self.assertEqual(len(set(features[0, :3])), 2)
            self.assertEqual(len(set(features[0, 3:6])), 2)
            for slot in range(count):
                analytic = engine.reverse(x, q, features[slot], bar)
                finite = np.zeros(12)
                for dof in range(12):
                    plus, _ = engine.forward(*perturb(x, q, dof, 2e-4))
                    minus, _ = engine.forward(*perturb(x, q, dof, -2e-4))
                    self.assertEqual((len(plus), len(minus)), (count, count))
                    finite[dof] = np.dot(bar, plus[slot] - minus[slot]) / 4e-4
                np.testing.assert_allclose(analytic, finite, atol=2e-3, rtol=6e-3)

    def test_analytic_vjp_translation_rotation_normal_and_tangent_frame(self):
        rng = np.random.default_rng(4)
        for model in (self.mesh, self.box, self.single):
            qpos = model.qpos0.copy()
            angle = .067
            qpos[10:14] = (np.cos(angle / 2), .2 * np.sin(angle / 2),
                           .3 * np.sin(angle / 2), np.sqrt(.87) * np.sin(angle / 2))
            _, x, q = poses(model, qpos)
            engine = self.engines[id(model)]
            contacts, features = engine.forward(x, q)
            self.assertGreater(len(contacts), 0)
            for slot in range(len(contacts)):
                bar = rng.normal(size=13).astype(np.float32)
                gradient = engine.reverse(x, q, features[slot], bar)
                finite = np.zeros(12)
                step = 2e-4
                for dof in range(12):
                    plus, _ = engine.forward(*perturb(x, q, dof, step))
                    minus, _ = engine.forward(*perturb(x, q, dof, -step))
                    self.assertEqual(len(plus), len(contacts))
                    self.assertEqual(len(minus), len(contacts))
                    # Slots follow native clipping order on this open branch.
                    finite[dof] = np.dot(bar, (plus[slot] - minus[slot]) / (2 * step))
                np.testing.assert_allclose(gradient, finite, atol=2e-3, rtol=6e-3)
                np.testing.assert_allclose(gradient[:3] + gradient[6:9], bar[1:4], atol=3e-5)
            frame_only = np.zeros(13, np.float32)
            frame_only[4:] = rng.normal(size=9)
            frame_gradient = engine.reverse(x, q, features[0], frame_only)
            self.assertGreater(np.linalg.norm(frame_gradient[[3, 4, 5, 9, 10, 11]]), .1)

    def test_generator_preserves_flags_and_raw_feature_abi(self):
        header = convex_mesh.header(self.mesh, [0, 1])
        self.assertIn("#define CM_MULTI 1", header)
        self.assertIn("#define CM_FEATURE_WORDS 12", header)
        self.assertIn("#define CM_MULTI 0", convex_mesh.header(self.single, [0, 1]))
        self.mesh.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        try:
            with self.assertRaisesRegex(ValueError, "native CCD"):
                convex_mesh.header(self.mesh, [0, 1])
        finally:
            self.mesh.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)

    def test_public_feature_replay_matches_forward_without_gjk(self):
        for model in (self.mesh, self.box, self.single, self.prism, self.pyramid, self.multiple,
                      self.boxes, self.octagon, self.octagon_no_multi):
            _, x, q = poses(model)
            engine = self.engines[id(model)]
            for pair in ((0, 1), (1, 0)):
                contacts, features = engine.forward(x, q, pair=pair)
                for contact, feature in zip(contacts, features):
                    np.testing.assert_array_equal(engine.replay(x, q, feature, pair=pair), contact)

    def test_manifold_linearization_reuses_all_slots_without_changing_geometry_or_vjp(self):
        rng = np.random.default_rng(421)
        for model in (self.mesh, self.box, self.single, self.prism, self.pyramid,
                      self.multiple, self.boxes, self.octagon, self.octagon_no_multi):
            engine = self.engines[id(model)]
            _, x, q = poses(model)
            for pair in ((0, 1), (1, 0)):
                with self.subTest(model=model, pair=pair):
                    contacts, features = engine.forward(x, q, pair=pair)
                    self.assertGreater(len(contacts), 0)
                    bar = rng.normal(size=contacts.shape).astype(np.float32)
                    bar[::2] = 0.  # inactive slots retain their exact zero VJP
                    values, gradients = engine.linearize(x, q, features[-1], bar, pair=pair)
                    np.testing.assert_array_equal(values.view(np.uint32), contacts.view(np.uint32))
                    for slot, feature in enumerate(features):
                        expected = engine.reverse(x, q, feature, bar[slot], pair=pair)
                        np.testing.assert_array_equal(gradients[slot].view(np.uint32), expected.view(np.uint32))
                    invalid = features[-1].copy()
                    invalid[9] += 1
                    with self.assertRaisesRegex(AssertionError, "linearization status=16"):
                        engine.linearize(x, q, invalid, bar, pair=pair)

    def test_shallow_mesh_face_frame_preserves_translation_independence(self):
        engine = self.engines[id(self.mesh)]
        _, deep_x, deep_q = poses(self.mesh)
        _, features = engine.forward(deep_x, deep_q)
        qpos = self.mesh.qpos0.copy()
        rotation = np.array([.99991, .008, .004, .009])
        rotation /= np.linalg.norm(rotation)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, rotation)
        origin = np.array([.01, -.04, -.15])
        qpos[:3], qpos[3:7] = origin, rotation
        qpos[7:10] = origin + matrix.reshape(3, 3) @ np.array([.19995, .013, .009])
        qpos[10:14] = rotation
        native, x, q = poses(self.mesh, qpos)
        self.assertEqual(len(features), native.ncon)
        # Replay the same face at 50 micrometres of penetration. Its normal is
        # independent of translation, even when world witnesses almost coincide.
        bar = np.array([0, 0, 0, 0, .2, -.4, .8, .3, .6, -.2, -.1, .5, .7], np.float32)
        for feature in features:
            contact = engine.replay(x, q, feature)
            expected = min(native.contact, key=lambda c: np.linalg.norm(c.pos - contact[1:4]))
            np.testing.assert_allclose(contact[4:], expected.frame, atol=2e-6, rtol=0)
            gradient = engine.reverse(x, q, feature, bar)
            np.testing.assert_allclose(gradient[[0, 1, 2, 6, 7, 8]], 0, atol=1e-6, rtol=0)
            self.assertGreater(np.linalg.norm(gradient[[3, 4, 5, 9, 10, 11]]), .1)

    def test_native_box_box_keeps_all_eight_contacts_independent_of_multiccd(self):
        for model in (self.octagon, self.octagon_no_multi):
            contacts, _, _ = self.assert_oracle(model)
            self.assertEqual(len(contacts), 8)
        for offset in (.2, .24, .07):
            qpos = self.boxes.qpos0.copy()
            qpos[7] = offset
            self.assert_oracle(self.boxes, qpos)

    def test_native_box_box_face_and_edge_geometry_vjp(self):
        axis = np.array([.3, .5, .8])
        axis /= np.linalg.norm(axis)
        cases = [(self.octagon, self.octagon.qpos0.copy())]
        for angle in (.05, .25, .5, .8):
            qpos = self.boxes.qpos0.copy()
            qpos[10], qpos[11:14] = np.cos(angle / 2), axis * np.sin(angle / 2)
            if angle == .5:
                qpos[7] = .15  # keep all selected vertices inside through the FD interval
            cases.append((self.boxes, qpos))
        rng = np.random.default_rng(7)
        for model, qpos in cases:
            contacts, x, q = self.assert_oracle(model, qpos)
            engine = self.engines[id(model)]
            _, features = engine.forward(x, q)
            self.assertTrue(np.all(features[:, 11] == 1))
            self.assertTrue(np.all(features[:, 5] == 0))  # open branches for two-sided FD
            if model is self.octagon:
                self.assertEqual(len(contacts), 8)
            for pair in ((0, 1), (1, 0)):
                oriented, tape = engine.forward(x, q, pair=pair)
                for contact, feature in zip(oriented, tape):
                    np.testing.assert_array_equal(engine.replay(x, q, feature, pair=pair), contact)
                    bar = rng.normal(size=13).astype(np.float32)
                    analytic = engine.reverse(x, q, feature, bar, pair=pair)
                    finite = np.zeros(12)
                    for dof in range(12):
                        # gb is in the requested pair's geom order.
                        world_dof = dof if pair == (0, 1) else (dof + 6) % 12
                        plus, _ = engine.forward(*perturb(x, q, world_dof, 2e-4), pair=pair)
                        minus, _ = engine.forward(*perturb(x, q, world_dof, -2e-4), pair=pair)
                        self.assertEqual((len(plus), len(minus)), (len(contacts), len(contacts)),
                                         (qpos.tolist(), int(feature[8]), dof))
                        finite[dof] = np.dot(bar, plus[feature[8]] - minus[feature[8]]) / 4e-4
                    np.testing.assert_allclose(analytic, finite, atol=2e-3, rtol=6e-3)
                    np.testing.assert_allclose(analytic[:3] + analytic[6:9], bar[1:4], atol=3e-5)

    def test_box_branch_boundaries_are_marked_without_inventing_a_smooth_derivative(self):
        _, x, q = poses(self.boxes)
        contacts, feature = self.engines[id(self.boxes)].forward(x, q)
        self.assertEqual(len(contacts), 4)
        self.assertTrue(np.all(feature[:, 5] & 1))  # exact tie between parallel reference faces
        qpos = self.boxes.qpos0.copy()
        qpos[7:10] = (.2, 0, 0)
        _, x, q = poses(self.boxes, qpos)
        contacts, feature = self.engines[id(self.boxes)].forward(x, q)
        self.assertEqual(len(contacts), 4)
        self.assertTrue(np.all(feature[:, 5] & 2))  # vertices on clipping planes
        self.assertTrue(np.all(feature[:, 5] & 16))  # exact zero-depth, excluded by the solver


if __name__ == "__main__":
    unittest.main()
