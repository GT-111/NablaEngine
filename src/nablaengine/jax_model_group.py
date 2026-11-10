"""Concurrent native model steps on XLA's stream, with one tape per model.

The small dispatcher calls the already compiled model libraries. It changes
stream scheduling, not their forward or adjoint CUDA programs.
"""

import ctypes
import functools

import numpy as np

from nablaengine import engines, jax_ffi as F


GROUP_UNIT = r"""
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
using Forward = int (*)(cudaStream_t,int,const float*,const float*,const float*,const float*,const float*,
                       const float*,int,int,float*,float*,float*,int*,float*,float*,void*,const float*,const float*);
using Adjoint = int (*)(cudaStream_t,int,int,const float*,const float*,const float*,const float*,const float*,
                       float*,float*,float*,int*,const float*,float,const float*,const float*,float*);
using Work = long long (*)(int);
using Kin = int (*)(cudaStream_t,int,const float*,const float*,const float*,float*,float*,float*,float*,float*);
using KinVjp = int (*)(cudaStream_t,int,const float*,const float*,const float*,const float*,const float*,
                      const float*,const float*,const float*,const float*,const float*,float*,float*);
struct Model {
  int nq,nv,nu,com,maxcon,rec,flags,refact,nonconv,singular,replay,iterations,nbody;
  Forward forward;
  Adjoint adjoint;
  Work work;
  Kin kin;
  KinVjp kin_vjp;
  cudaStream_t stream;
  cudaEvent_t done;
  unsigned long long* counters;
};
struct Group {
  int device;
  std::mutex mutex;
  cudaEvent_t ready;
  std::vector<Model> models;
};
static std::mutex registry_mutex;
static std::vector<std::unique_ptr<Group>> groups;
static constexpr int NDIAG = 18;
static ffi::Error error(cudaError_t rc, const char* where) {
  return rc == cudaSuccess ? ffi::Error::Success()
    : ffi::Error::Internal(std::string("nablaengine.model_group: ") + where + ": " + cudaGetErrorString(rc));
}
static ffi::Error bad(const char* message) { return ffi::Error::InvalidArgument(message); }

// Called at the setup boundary, outside XLA/capture. Models and their raw
// function addresses remain immutable for the lifetime of every executable.
extern "C" int dk_group_create(int count, const long long* metadata, const uintptr_t* functions, int64_t* id) {
  auto group = std::make_unique<Group>();
  int rc = (int)cudaGetDevice(&group->device);
  if (rc) return rc;
  if ((rc = (int)cudaEventCreateWithFlags(&group->ready, cudaEventDisableTiming))) return rc;
  for (int i = 0; i < count; ++i) {
    const long long* p = metadata + 13*i;
    Model m{(int)p[0],(int)p[1],(int)p[2],(int)p[3],(int)p[4],(int)p[5],(int)p[6],
            (int)p[7],(int)p[8],(int)p[9],(int)p[10],(int)p[11],(int)p[12],
            reinterpret_cast<Forward>(functions[5*i]), reinterpret_cast<Adjoint>(functions[5*i+1]),
            reinterpret_cast<Work>(functions[5*i+2]), reinterpret_cast<Kin>(functions[5*i+3]),
            reinterpret_cast<KinVjp>(functions[5*i+4]), nullptr, nullptr, nullptr};
    if ((rc = (int)cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking))) return rc;
    if ((rc = (int)cudaEventCreateWithFlags(&m.done, cudaEventDisableTiming))) return rc;
    if ((rc = (int)cudaMalloc(&m.counters, NDIAG*sizeof(unsigned long long)))) return rc;
    if ((rc = (int)cudaMemset(m.counters, 0, NDIAG*sizeof(unsigned long long)))) return rc;
    group->models.push_back(m);
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  *id = (int64_t)groups.size();
  groups.push_back(std::move(group));
  return 0;
}
static Group* lookup(int64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  return id >= 0 && id < (int64_t)groups.size() ? groups[id].get() : nullptr;
}
__device__ void add(unsigned long long* counters, int index, unsigned value) {
  __syncwarp();
  for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  if ((threadIdx.x & 31) == 0 && value) atomicAdd(counters + index, (unsigned long long)value);
}
__global__ void forward_diagnostics(Model m, int worlds, int substeps, const int* info, const float* tape) {
  unsigned unsupported=0,limit=0,refact=0,nonconv=0,singular=0,contacts=0,rows=0;
  for (int w=blockIdx.x*blockDim.x+threadIdx.x; w<worlds; w+=gridDim.x*blockDim.x) {
    unsigned flag = (unsigned)info[4*w+3];
    unsupported += flag != 0; contacts += (flag & 4u) != 0; rows += (flag & 16u) != 0;
    limit += info[4*w] >= m.iterations;
    if (tape) for (int step=0; step<substeps; ++step) {
      unsigned bits = __float_as_uint(tape[((size_t)w*substeps+step)*m.rec+m.flags]);
      refact += (bits & m.refact) != 0; nonconv += (bits & m.nonconv) != 0; singular += (bits & m.singular) != 0;
    }
  }
  add(m.counters,2,unsupported); add(m.counters,3,limit); add(m.counters,5,refact);
  add(m.counters,6,nonconv); add(m.counters,7,singular); add(m.counters,16,contacts); add(m.counters,17,rows);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(m.counters,1ull); atomicAdd(m.counters+1,(unsigned long long)worlds);
    if (tape) atomicAdd(m.counters+4,(unsigned long long)worlds*substeps);
  }
}
__device__ bool finite(const float* p, int size) {
  bool ok=true; for (int i=0;i<size;++i) ok &= isfinite(p[i]); return ok;
}
__global__ void backward_diagnostics(Model m, int worlds, const float* q, const float* v, const float* u,
                                     const float* cq, const float* cv, int* info) {
  unsigned rows=0,rq=0,rv=0,ru=0,rc=0,replay=0;
  for (int w=blockIdx.x*blockDim.x+threadIdx.x;w<worlds;w+=gridDim.x*blockDim.x) {
    unsigned bits = (finite(q+(size_t)w*m.nq,m.nq)?0u:1u) | (finite(v+(size_t)w*m.nv,m.nv)?0u:2u)
                  | (finite(u+(size_t)w*m.nu,m.nu)?0u:4u)
                  | (finite(cq+(size_t)w*m.nq,m.nq)&&finite(cv+(size_t)w*m.nv,m.nv)?0u:8u);
    int flags=info[w]; info[w]=flags|(int)(bits<<16);
    rows += (bits&7u)!=0; rq += (bits&1u)!=0; rv += (bits&2u)!=0; ru += (bits&4u)!=0; rc += (bits&8u)!=0;
    replay += (flags&m.replay)!=0;
  }
  add(m.counters,10,rows); add(m.counters,11,rq); add(m.counters,12,rv); add(m.counters,13,ru);
  add(m.counters,14,rc); add(m.counters,15,replay);
  if (blockIdx.x==0&&threadIdx.x==0) { atomicAdd(m.counters+8,1ull); atomicAdd(m.counters+9,(unsigned long long)worlds); }
}
static unsigned grid(int worlds) { return (unsigned)std::min(160,std::max(1,(worlds+255)/256)); }
static bool shape(ffi::AnyBuffer b, ffi::DataType type, std::initializer_list<int64_t> dims) {
  if (b.element_type()!=type || b.dimensions().size()!=dims.size()) return false;
  size_t index=0; for (int64_t dim:dims) if (b.dimensions()[index++]!=dim) return false;
  return true;
}
static ffi::Error dispatch(cudaStream_t main, ffi::ScratchAllocator scratch, ffi::RemainingArgs args,
                           ffi::RemainingRets rets, int64_t group_id, int32_t substeps, int32_t backward,
                           int32_t taped, int32_t kinematics, float clamp_soft) {
  Group* group=lookup(group_id);
  if (!group || substeps<1) return bad("invalid model group/substeps");
  std::lock_guard<std::mutex> lock(group->mutex);
  int device=-1; cudaGetDevice(&device);
  if (device!=group->device) return bad("model group belongs to another CUDA device");
  const size_t kin_offset=taped?6:5, count=group->models.size();
  const size_t stride=backward?5:(kin_offset+(kinematics?5:0)), input_stride=backward&&kinematics?15:8;
  if (args.size()!=count*input_stride || rets.size()!=count*stride) return bad("model group operand/result count");
  std::vector<ffi::AnyBuffer> inputs, outputs;
  for (size_t i=0;i<args.size();++i) {
    auto b=args.get<ffi::AnyBuffer>(i); if (!b) return b.error(); inputs.push_back(*b);
  }
  for (size_t i=0;i<rets.size();++i) {
    auto b=rets.get<ffi::AnyBuffer>(i); if (!b) return b.error(); outputs.push_back(**b);
  }
  std::vector<int> worlds(count);
  std::vector<void*> work(count,nullptr);
  // Decode/allocate all operands before forking the stream, so an argument or
  // allocation error cannot leave a worker using already-released scratch.
  for (size_t i=0;i<count;++i) {
    const Model& m=group->models[i]; auto* a=inputs.data()+input_stride*i; auto* o=outputs.data()+stride*i;
    if (a[0].dimensions().size()!=2 || a[0].dimensions()[0]>INT_MAX) return bad("model group batch rank/size");
    int B=worlds[i]=(int)a[0].dimensions()[0];
    std::vector<std::vector<int64_t>> expected = backward
      ? std::vector<std::vector<int64_t>>{{B,(int64_t)substeps*m.rec},{B},{B,m.com},{B,m.nv},{B},{B,m.nu},{B,m.nq},{B,m.nv}}
      : std::vector<std::vector<int64_t>>{{B,m.nq},{B,m.nv},{B,m.nv},{B,m.nu},{B},{B,m.com},{B,m.nv},{B}};
    if(backward&&kinematics) expected.insert(expected.end(),{{B,m.nq},{B,m.nv},{B,m.nq},
        {B,m.nbody,3},{B,m.nbody,4},{B,m.nbody,6},{B,m.nbody,3}});
    for (size_t j=0;j<input_stride;++j) {
      if (a[j].element_type()!=ffi::DataType::F32 || a[j].dimensions().size()!=expected[j].size()) return bad("model group input dtype/rank");
      for (size_t k=0;k<expected[j].size();++k) if(a[j].dimensions()[k]!=expected[j][k]) return bad("model group input shape");
    }
    if (!shape(o[0],ffi::DataType::F32,{B,m.nq}) || !shape(o[1],ffi::DataType::F32,{B,m.nv})
        || !shape(o[2],ffi::DataType::F32,{B,backward?m.nu:m.nv})
        || !(backward?shape(o[3],ffi::DataType::S32,{B}):shape(o[3],ffi::DataType::S32,{B,4}))
        || !shape(o[4],ffi::DataType::F32,{B,backward?m.nv:m.maxcon*8})
        || (!backward&&taped&&!shape(o[5],ffi::DataType::F32,{B,(int64_t)substeps*m.rec}))) return bad("model group result shape");
    if(!backward&&kinematics) {
      auto* k=o+kin_offset;
      if(!shape(k[0],ffi::DataType::F32,{B,m.nq}) || !shape(k[1],ffi::DataType::F32,{B,m.nbody,3})
          || !shape(k[2],ffi::DataType::F32,{B,m.nbody,4}) || !shape(k[3],ffi::DataType::F32,{B,m.nbody,6})
          || !shape(k[4],ffi::DataType::F32,{B,m.nbody,3})) return bad("model group kinematics result shape");
    }
    if (!backward&&B) if (long long bytes=m.work(B)) {
      auto p=scratch.Allocate((size_t)bytes,16);
      if(!p) return ffi::Error::Internal("model group forward scratch allocation failed");
      work[i]=*p;
    }
    if(backward&&kinematics&&B) {
      auto p=scratch.Allocate((size_t)B*(m.nq+m.nv)*sizeof(float),16);
      if(!p) return ffi::Error::Internal("model group kinematics cotangent allocation failed");
      work[i]=*p;
    }
  }
  cudaError_t rc=cudaEventRecord(group->ready,main);
  if(rc!=cudaSuccess) return error(rc,"record inputs ready");
  int first_error=0;
  for (size_t i=0;i<count;++i) {
    Model& m=group->models[i]; int B=worlds[i]; if(!B) continue;
    auto* a=inputs.data()+input_stride*i; auto* o=outputs.data()+stride*i;
    int code=(int)cudaStreamWaitEvent(m.stream,group->ready,0);
    if (!code&&!backward) {
      float* tape=taped?o[5].typed_data<float>():nullptr;
      code=m.forward(m.stream,B,a[0].typed_data<float>(),a[1].typed_data<float>(),a[2].typed_data<float>(),
          a[3].typed_data<float>(),a[4].typed_data<float>(),a[5].typed_data<float>(),m.iterations,substeps,
          o[0].typed_data<float>(),o[1].typed_data<float>(),o[2].typed_data<float>(),o[3].typed_data<int32_t>(),
          o[4].typed_data<float>(),tape,work[i],a[6].typed_data<float>(),a[7].typed_data<float>());
      if(!code&&kinematics) {
        auto* k=o+kin_offset;
        code=m.kin(m.stream,B,o[0].typed_data<float>(),o[1].typed_data<float>(),a[5].typed_data<float>(),
            k[0].typed_data<float>(),k[1].typed_data<float>(),k[2].typed_data<float>(),k[3].typed_data<float>(),k[4].typed_data<float>());
      }
      if(!code) { forward_diagnostics<<<grid(B),256,0,m.stream>>>(m,B,substeps,o[3].typed_data<int32_t>(),tape); code=(int)cudaGetLastError(); }
    } else if (!code) {
      const float *cq=a[6].typed_data<float>(), *cv=a[7].typed_data<float>();
      if(kinematics) {
        float *sq=static_cast<float*>(work[i]), *sv=sq+(size_t)B*m.nq;
        code=m.kin_vjp(m.stream,B,a[8].typed_data<float>(),a[9].typed_data<float>(),a[2].typed_data<float>(),
            a[10].typed_data<float>(),a[11].typed_data<float>(),a[12].typed_data<float>(),a[13].typed_data<float>(),
            a[14].typed_data<float>(),cq,cv,sq,sv);
        cq=sq; cv=sv;
      }
      if(!code) code=m.adjoint(m.stream,B,substeps,a[0].typed_data<float>(),a[1].typed_data<float>(),a[2].typed_data<float>(),
          cq,cv,o[0].typed_data<float>(),o[1].typed_data<float>(),
          o[2].typed_data<float>(),o[3].typed_data<int32_t>(),a[5].typed_data<float>(),clamp_soft,
          a[3].typed_data<float>(),a[4].typed_data<float>(),o[4].typed_data<float>());
      if(!code) { backward_diagnostics<<<grid(B),256,0,m.stream>>>(m,B,o[0].typed_data<float>(),o[1].typed_data<float>(),
          o[2].typed_data<float>(),cq,cv,o[3].typed_data<int32_t>()); code=(int)cudaGetLastError(); }
    }
    if(code&&!first_error) first_error=code;
    code=(int)cudaEventRecord(m.done,m.stream); if(code&&!first_error) first_error=code;
    code=(int)cudaStreamWaitEvent(main,m.done,0); if(code&&!first_error) first_error=code;
  }
  return error((cudaError_t)first_error,"model launch/join");
}
__global__ void read_counters(const unsigned long long* source, unsigned* out, int reset) {
  for(int k=threadIdx.x;k<NDIAG;k+=blockDim.x) {
    auto* counter=const_cast<unsigned long long*>(source)+k;
    unsigned long long value=reset?atomicExch(counter,0ull):*counter;
    out[2*k]=(unsigned)value; out[2*k+1]=(unsigned)(value>>32);
  }
}
static ffi::Error diagnostics(cudaStream_t main, ffi::RemainingArgs, int64_t id, int32_t reset,
                              ffi::Result<ffi::Buffer<ffi::U32>> output) {
  Group* group=lookup(id); if(!group) return bad("invalid model group");
  std::lock_guard<std::mutex> lock(group->mutex);
  if(output->dimensions().size()!=3 || output->dimensions()[0]!=(int64_t)group->models.size()
      || output->dimensions()[1]!=NDIAG || output->dimensions()[2]!=2) return bad("model group diagnostics shape");
  for(size_t i=0;i<group->models.size();++i)
    read_counters<<<1,32,0,main>>>(group->models[i].counters,output->typed_data()+i*NDIAG*2,reset);
  return error(cudaGetLastError(),"diagnostics read");
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkModelGroup,dispatch,ffi::Ffi::Bind()
  .Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>().RemainingArgs().RemainingRets()
  .Attr<int64_t>("group_id").Attr<int32_t>("substeps").Attr<int32_t>("backward").Attr<int32_t>("taped")
  .Attr<int32_t>("kinematics").Attr<float>("clamp_soft"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(DkModelGroupDiag,diagnostics,ffi::Ffi::Bind()
  .Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().Attr<int64_t>("group_id").Attr<int32_t>("reset")
  .Ret<ffi::Buffer<ffi::U32>>());
"""


@functools.lru_cache(maxsize=None)
def _library(arch):
    import jax

    path, built, key = F._build({"dk_ffi.cu": GROUP_UNIT}, {"dk_ffi.cu": []}, arch,
                               jax.ffi.include_dir(), np.zeros(1, np.float32), prefix="libdk_model_group")
    lib = ctypes.cdll.LoadLibrary(str(path))
    targets = {}
    for kind, symbol in (("step", "DkModelGroup"), ("diag", "DkModelGroupDiag")):
        targets[kind] = f"dk_model_group_{kind}_{key[:8]}"
        jax.ffi.register_ffi_target(targets[kind], jax.ffi.pycapsule(getattr(lib, symbol)), platform="CUDA")
    lib.dk_group_create.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    return lib, targets, dict(path=str(path), built=built, key=key)


class ModelGroup:
    """An immutable set of native models, independently dispatched per step.

    Each input item is the context's eight float32 operands. Empty plant batches
    stay empty. Each model keeps its own tape and receives exactly its own VJP.
    """

    def __init__(self, contexts, iterations, *, clamp_kv=F.CLAMP_KV, normal=F.NORMAL,
                 clamp_soft=None, grad_decay=F.GRAD_DECAY):
        import jax
        import jax.numpy as jnp

        self.contexts = tuple(contexts)
        self.iterations = tuple(int(value) for value in iterations)
        self.grad_decay = tuple(float(np.float32(value)) for value in grad_decay) if isinstance(grad_decay, (tuple, list)) else (float(np.float32(grad_decay)),) * len(self.contexts)
        if not self.contexts or len(self.iterations) != len(self.contexts) or len(self.grad_decay) != len(self.contexts):
            raise ValueError("Each model needs its own iteration/gradient-decay setting")
        if (int(clamp_kv), int(normal)) not in F.VARIANTS:
            raise ValueError("Unsupported native adjoint variant")
        if min(self.iterations) < 0 or not np.isfinite(self.grad_decay).all():
            raise ValueError("Invalid native iteration/gradient-decay setting")
        self.clamp_soft = F._soft(clamp_kv, clamp_soft)
        arch = self.contexts[0].info["arch"]
        if any(context.info["arch"] != arch for context in self.contexts):
            raise ValueError("A model group must use one CUDA architecture")
        # The existing library owns constant uploads and adjoint launch setup.
        # A real setup FK initializes both once, before worker-stream execution.
        for context in self.contexts:
            model, info = context._state["m"], context.info
            jax.block_until_ready(context.kinematics(jnp.asarray(model.qpos0[None], jnp.float32),
                jnp.zeros((1, info["nv"]), jnp.float32), jnp.zeros((1, info["com_width"]), jnp.float32)))
        self._lib, self._targets, self.info = _library(arch)
        metadata, functions = [], []
        for context, count in zip(self.contexts, self.iterations, strict=True):
            info, state = context.info, context._state
            modules = engines.modules(state["m"])
            metadata.append((info["nq"], info["nv"], info["nu"], info["com_width"], info["maxcon"], info["rec_words"],
                modules.tape.layout(state["m"])["flags"][0], modules.tape.FLAG_REFACT, modules.tape.FLAG_NONCONV,
                modules.tape.FLAG_SINGULAR, modules.adjoint.ADJ_REPLAY_MISMATCH, count, info["nbody"]))
            functions.extend(ctypes.cast(getattr(state["lib"], symbol), ctypes.c_void_p).value for symbol in (
                info["forward_unit"] + "_forward", F._tag((int(clamp_kv), int(normal))) + "_adjoint",
                info["forward_unit"] + "_work_bytes", info["forward_unit"] + "_kinematics",
                info["forward_unit"] + "_kinematics_vjp"))
        metadata, functions = np.asarray(metadata, np.int64), np.asarray(functions, np.uintp)
        group_id = ctypes.c_int64()
        code = self._lib.dk_group_create(len(self.contexts), metadata.ctypes.data, functions.ctypes.data, ctypes.byref(group_id))
        if code:
            raise RuntimeError(f"CUDA model group setup failed ({code})")
        self.group_id = group_id.value

    def _forward(self, inputs, substeps, tape, kinematics):
        import jax

        if len(inputs) != len(self.contexts):
            raise ValueError("The model group needs one operand tuple per model")
        outputs = []
        for context, values in zip(self.contexts, inputs, strict=True):
            B, info = values[0].shape[0], context.info
            if len(values) != 8:
                raise ValueError("A native model step needs eight operands")
            shapes = ((B, info["nq"]), (B, info["nv"]), (B, info["nv"]), (B, info["nu"]), (B,),
                      (B, info["com_width"]), (B, info["nv"]), (B,))
            for name, value, shape in zip(("qpos", "qvel", "warm", "ctrl", "friction", "com", "qfrc", "kp"), values, shapes, strict=True):
                F._check(name, value, shape)
            outputs.extend(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
                ((B, info["nq"]), np.float32), ((B, info["nv"]), np.float32), ((B, info["nv"]), np.float32),
                ((B, 4), np.int32), ((B, info["maxcon"] * 8), np.float32)))
            if tape:
                outputs.append(jax.ShapeDtypeStruct((B, substeps * info["rec_words"]), np.float32))
            if kinematics:
                outputs.extend(F._kin_structs(F._kin_lib(context), B))
        result = jax.ffi.ffi_call(self._targets["step"], outputs)(*(value for item in inputs for value in item),
            group_id=np.int64(self.group_id), substeps=np.int32(substeps), backward=np.int32(0),
            taped=np.int32(tape), kinematics=np.int32(kinematics), clamp_soft=np.float32(self.clamp_soft or 0.0))
        offset = 6 if tape else 5
        stride = offset + (5 if kinematics else 0)
        return tuple(tuple(result[i:i + 4]) + tuple(result[i + offset:i + stride])
                     + ((result[i + 5],) if tape else ()) for i in range(0, len(result), stride))

    def _backward(self, residual, cotangent, substeps, kinematics):
        import jax
        import jax.numpy as jnp

        inputs, outputs = [], []
        for context, (tape, friction, com, force, kp, ctrl, state), cot in zip(self.contexts, residual, cotangent, strict=True):
            info, B = context.info, tape.shape[0]
            if ctrl is None:
                ctrl = jnp.zeros((B, info["nu"]), np.float32)
            inputs.extend((tape, friction, com, force, kp, ctrl, cot[0], cot[1]))
            if kinematics:
                inputs.extend((*state, *cot[4:]))
            outputs.extend(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
                ((B, info["nq"]), np.float32), ((B, info["nv"]), np.float32), ((B, info["nu"]), np.float32),
                ((B,), np.int32), ((B, info["nv"]), np.float32)))
        result = jax.ffi.ffi_call(self._targets["step"], outputs)(*inputs, group_id=np.int64(self.group_id),
            substeps=np.int32(substeps), backward=np.int32(1), taped=np.int32(0), kinematics=np.int32(kinematics),
            clamp_soft=np.float32(self.clamp_soft or 0.0))
        gradients = []
        for index, ((_tape, friction, com, _force, kp, _ctrl, _state), decay) in enumerate(zip(residual, self.grad_decay, strict=True)):
            dq, dv, du, _info, df = result[index * 5:index * 5 + 5]
            if decay != 1.0:
                dq, dv = dq * np.float32(decay), dv * np.float32(decay)
            gradients.append((dq, dv, jnp.zeros_like(dv), du, jnp.zeros_like(friction),
                              jnp.zeros_like(com), df, jnp.zeros_like(kp)))
        return tuple(gradients)

    @functools.lru_cache(maxsize=None)
    def make_step(self, substeps=1, kinematics=False):
        import jax
        from jax.ad_checkpoint import checkpoint_name

        @jax.custom_vjp
        def step(inputs):
            return self._forward(inputs, substeps, False, kinematics)

        def forward(inputs):
            result = self._forward(inputs, substeps, True, kinematics)
            outputs = tuple(tuple(checkpoint_name(value, F.OUT_NAME) for value in item[:-1]) for item in result)
            residual = tuple((checkpoint_name(item[-1], F.TAPE_NAME), values[4], values[5], values[6], values[7],
                              values[3] if self.clamp_soft is not None else None, out[:2] if kinematics else ())
                             for item, values, out in zip(result, inputs, outputs, strict=True))
            return outputs, residual

        def backward(residual, cotangent):
            return (self._backward(residual, cotangent, substeps, kinematics),)

        step.defvjp(forward, backward)
        return step

    def step(self, inputs, *, substeps=1, kinematics=False):
        return self.make_step(int(substeps), bool(kinematics))(tuple(tuple(values) for values in inputs))

    @functools.lru_cache(maxsize=None)
    def _diagnostics_reader(self, reset):
        import jax

        return jax.jit(lambda: jax.ffi.ffi_call(self._targets["diag"],
            jax.ShapeDtypeStruct((len(self.contexts), len(F.DIAG), 2), np.uint32), has_side_effect=True)(
                group_id=np.int64(self.group_id), reset=np.int32(reset)))

    def read_diagnostics(self, reset=False):
        return tuple(F.decode_diagnostics(item) for item in self._diagnostics_reader(bool(reset))())
