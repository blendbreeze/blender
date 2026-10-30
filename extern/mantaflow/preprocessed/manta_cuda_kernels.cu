// SPDX-License-Identifier: Apache-2.0
// Experimental CUDA kernels for select mantaflow operations (flood fill,
// levelset extrapolation, MAC velocity extrapolation, semi-Lagrangian scalar
// advection, pressure CG solve, particle levelsets). Ported 1:1 from the CPU
// kernels in levelset.cpp / fastmarch.cpp / advection.cpp / pressure.cpp /
// flip.cpp - see PROJECT_NOTES.md.
//
// Enabled at runtime via MANTA_CUDA=1. Grids are staged through persistent
// pinned host buffers with asynchronous copies; each operation queues its
// kernels back-to-back and synchronizes once at the end (the flood fill uses
// direction-alternating in-place sweeps instead of a per-layer BFS, so it
// needs no intermediate device-host round trips). All kernels are
// deterministic: the flood fill computes an order-independent reachable set,
// and the extrapolation passes only write cells whose values are never read
// within the same pass.
//
// After the first CUDA error the path latches off for the session with a
// single log line and all callers fall back to the CPU implementations.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "manta_cuda.h"
#include "mcubes.h"

namespace Manta {
namespace manta_cuda {

// Latched off after the first CUDA error so one failure cannot spam the log
// or poison further calls; the sim silently continues on CPU.
static bool s_latched = false;

static bool initEnvDevice()
{
  const char *env = getenv("MANTA_CUDA");
  if (!env || !atoi(env)) {
    return false;
  }
  int count = 0;
  cudaError_t e1 = cudaGetDeviceCount(&count);
  if (e1 != cudaSuccess || count <= 0) {
    fprintf(stderr,
            "MANTA_CUDA: no usable CUDA device (%s), staying on CPU\n",
            cudaGetErrorString(e1));
    return false;
  }
  cudaError_t e2 = cudaSetDevice(0);
  cudaError_t e3 = (e2 == cudaSuccess) ? cudaFree(0) : cudaErrorUnknown;
  if (e2 != cudaSuccess || e3 != cudaSuccess) {
    fprintf(stderr,
            "MANTA_CUDA: device init failed (%s), staying on CPU\n",
            cudaGetErrorString(e2 != cudaSuccess ? e2 : e3));
    return false;
  }
  fprintf(stderr, "MANTA_CUDA: enabled\n");
  return true;
}

static void cudaFail(const char *where, cudaError_t rc)
{
  if (!s_latched) {
    s_latched = true;
    fprintf(stderr,
            "MANTA_CUDA: error in %s: %s - CUDA disabled for this session "
            "(continuing on CPU)\n",
            where,
            cudaGetErrorString(rc));
  }
}

bool enabled()
{
  static bool s_env = initEnvDevice();
  return s_env && !s_latched;
}

// Check the return code of a CUDA call, latch off and bail out on error.
#define CUDA_RC(call, where) \
  do { \
    cudaError_t _rc = (call); \
    if (_rc != cudaSuccess) { \
      cudaFail(where, _rc); \
      return false; \
    } \
  } while (0)

// Persistent device buffers plus pinned host staging (grows to the largest
// request, never shrinks). Plugin calls are serialized by the Python GIL
// (verified: no Py_BEGIN_ALLOW_THREADS in the mantaflow wrapper layer), so no
// locking. `phi` is sized in floats and holds a scalar grid (n) or a MAC grid
// (3 floats per cell) depending on the operation.
struct DevBuffers {
  float *phi = nullptr;    // device: scalar or MAC data
  float *phi2 = nullptr;   // device: second scalar buffer (advection source)
  int *ibuf = nullptr;     // device: visited / tmp (also reused as float dst)
  int *flagsBuf = nullptr; // device: flag grid copy
  float *hPhi = nullptr;   // pinned host staging for phi
  int *hFlags = nullptr;   // pinned host staging for flags / scalar bytes
  long phiFloats = 0;
  long intCells = 0;

  bool ensure(long needFloats, long needInts)
  {
    if (needFloats <= phiFloats && needInts <= intCells) {
      return true;
    }
    freeAll();
    bool ok = cudaMalloc(&phi, needFloats * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&phi2, needFloats * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&ibuf, needInts * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&flagsBuf, needInts * sizeof(int)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hPhi, needFloats * sizeof(float), cudaHostAllocDefault) ==
             cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hFlags, needInts * sizeof(int), cudaHostAllocDefault) ==
             cudaSuccess;
    if (!ok) {
      cudaFail("device buffer alloc", cudaGetLastError());
      freeAll();
      return false;
    }
    phiFloats = needFloats;
    intCells = needInts;
    return true;
  }
  void freeAll()
  {
    cudaFree(phi);
    cudaFree(phi2);
    cudaFree(ibuf);
    cudaFree(flagsBuf);
    cudaFreeHost(hPhi);
    cudaFreeHost(hFlags);
    phi = nullptr;
    phi2 = nullptr;
    ibuf = nullptr;
    flagsBuf = nullptr;
    hPhi = nullptr;
    hFlags = nullptr;
    phiFloats = 0;
    intCells = 0;
  }
};

static DevBuffers g_dev;

// ---------------------------------------------------------------------------
// device helpers

__device__ __forceinline__ bool signOkFill(float v, bool outside)
{
  return outside ? (v >= 0.0f) : (v < 0.0f);
}

__device__ __forceinline__ void coords(
    long t, int sx, long strideZ, int *i, int *j, int *k)
{
  int kk = (int)(t / strideZ);
  long r = t - (long)kk * strideZ;
  int jj = (int)(r / sx);
  *k = kk;
  *j = jj;
  *i = (int)(r - (long)jj * sx);
}

__device__ __forceinline__ bool inInnerBox(int i, int j, int k, int sx, int sy, int sz, int bw)
{
  return i >= bw && j >= bw && k >= bw && i < sx - bw && j < sy - bw && k < sz - bw;
}

// ---------------------------------------------------------------------------
// flood fill: seed the wall ring, then propagate with direction-alternating
// in-place sweeps. The reachable set is identical to the CPU stack-based
// flood fill (order-independent reachability), so the result matches.

__global__ void ffInit(const float *__restrict__ phi,
                       int *__restrict__ visited,
                       int sx,
                       int sy,
                       int sz,
                       long strideZ,
                       int bw,
                       bool outside)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    bool seed = false;
    if (inInnerBox(i, j, k, sx, sy, sz, bw) && signOkFill(phi[t], outside)) {
      const bool wall = (i == bw || i == sx - 1 - bw) || (j == bw || j == sy - 1 - bw) ||
                        (sz > 1 && (k == bw || k == sz - 1 - bw));
      seed = wall;
    }
    visited[t] = seed ? 1 : 0;
  }
}

// One Gauss-Seidel-style propagation sweep; `reverse` flips the traversal
// direction so information spreads from both ends between iterations.
__global__ void ffSweep(const float *__restrict__ phi,
                        int *__restrict__ visited,
                        int sx,
                        int sy,
                        int sz,
                        long strideZ,
                        int bw,
                        bool outside,
                        int reverse)
{
  long n = (long)sx * sy * sz;
  for (long u = (long)blockIdx.x * blockDim.x + threadIdx.x; u < n;
       u += (long)gridDim.x * blockDim.x) {
    const long t = reverse ? (n - 1 - u) : u;
    if (visited[t]) {
      continue;
    }
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, bw) || !signOkFill(phi[t], outside)) {
      continue;
    }
    bool hit = visited[t - 1] || visited[t + 1] || visited[t - sx] || visited[t + sx];
    if (!hit && sz > 1) {
      hit = visited[t - strideZ] || visited[t + strideZ];
    }
    if (hit) {
      visited[t] = 1;
    }
  }
}

// Same as KnFillApply (interior of the inner box only): overwrite unvisited
// cells with matching sign.
__global__ void applyFill(float *__restrict__ phi,
                          const int *__restrict__ visited,
                          int sx,
                          int sy,
                          int sz,
                          long strideZ,
                          int bw,
                          float value,
                          bool outside)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, bw)) {
      continue;
    }
    if (visited[t]) {
      continue;
    }
    if (outside && phi[t] < 0.0f) {
      continue;
    }
    if (!outside && phi[t] >= 0.0f) {
      continue;
    }
    phi[t] = value;
  }
}

// ---------------------------------------------------------------------------
// extrapolateLsSimple

__global__ void lsMarkSkip(const float *__restrict__ phi,
                           int *__restrict__ tmp,
                           int sx,
                           int sy,
                           int sz,
                           long strideZ,
                           bool inside)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    if ((!inside && phi[t] < 0.0f) || (inside && phi[t] > 0.0f)) {
      tmp[t] = 1;
    }
  }
}

__global__ void lsFirstLayer(int *__restrict__ tmp, int sx, int sy, int sz, long strideZ, int dim)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (tmp[t]) {
      continue;
    }
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    bool hit = false;
    if (tmp[t - 1] == 1 || tmp[t + 1] == 1 || tmp[t - sx] == 1 || tmp[t + sx] == 1) {
      hit = true;
    }
    if (!hit && dim == 3 && (tmp[t - strideZ] == 1 || tmp[t + strideZ] == 1)) {
      hit = true;
    }
    if (hit) {
      tmp[t] = 2;
    }
  }
}

// One dilation step d: tmp==0 cells next to tmp==d take the average of those
// neighbors plus direction, and become layer d+1.
__global__ void lsDilate(float *__restrict__ phi,
                         int *__restrict__ tmp,
                         int sx,
                         int sy,
                         int sz,
                         long strideZ,
                         int dim,
                         int d,
                         float direction)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (tmp[t]) {
      continue;
    }
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    int nbs = 0;
    float avg = 0.0f;
    long nb[6] = {t - 1, t + 1, t - sx, t + sx, t - strideZ, t + strideZ};
    int cnt = (dim == 3) ? 6 : 4;
    for (int q = 0; q < cnt; q++) {
      if (tmp[nb[q]] == d) {
        avg += phi[nb[q]];
        nbs++;
      }
    }
    if (nbs > 0) {
      tmp[t] = d + 1;
      phi[t] = avg / (float)nbs + direction;
    }
  }
}

__global__ void lsRemaining(float *__restrict__ phi,
                            const int *__restrict__ tmp,
                            int sx,
                            int sy,
                            int sz,
                            long strideZ,
                            float value)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (tmp[t]) {
      continue;
    }
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    phi[t] = value;
  }
}

// ---------------------------------------------------------------------------
// extrapolateMACSimple

// Mark faces whose component `axis` is initialized (fluid, honoring intoObs).
__global__ void macMark(const int *__restrict__ flags,
                        int *__restrict__ tmp,
                        int sx,
                        int sy,
                        int sz,
                        long strideZ,
                        int axis,
                        bool intoObs)
{
  long n = (long)sx * sy * sz;
  long axisOff = (axis == 0) ? 1 : (axis == 1) ? sx : strideZ;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (t < axisOff) {
      continue;
    }
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    const int f = flags[t];
    const int fm = flags[t - axisOff];
    const bool fluid = (f & 1) || (fm & 1);
    bool mark = fluid;
    if (intoObs) {
      mark = fluid && !(f & 2) && !(fm & 2);
    }
    if (mark) {
      tmp[t] = 1;
    }
  }
}

// Dilation step d for MAC component c (vel is 3 floats per cell).
__global__ void macDilate(float *__restrict__ vel,
                          int *__restrict__ tmp,
                          int sx,
                          int sy,
                          int sz,
                          long strideZ,
                          int dim,
                          int d,
                          int c)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (tmp[t]) {
      continue;
    }
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    int nbs = 0;
    float avg = 0.0f;
    long nb[6] = {t - 1, t + 1, t - sx, t + sx, t - strideZ, t + strideZ};
    int cnt = (dim == 3) ? 6 : 4;
    for (int q = 0; q < cnt; q++) {
      if (tmp[nb[q]] == d) {
        avg += vel[3 * nb[q] + c];
        nbs++;
      }
    }
    if (nbs > 0) {
      tmp[t] = d + 1;
      vel[3 * t + c] = avg / (float)nbs;
    }
  }
}

// ---------------------------------------------------------------------------
// scalar semi-Lagrange advection (order 1, linear, first-order trace)

// Exact device port of mantaflow's BUILD_INDEX + interpol (linear, clamped).
__device__ __forceinline__ float interpLinear(const float *__restrict__ data,
                                              int sx,
                                              int sy,
                                              int sz,
                                              long strideZ,
                                              float px0,
                                              float py0,
                                              float pz0)
{
  float px = px0 - 0.5f, py = py0 - 0.5f, pz = pz0 - 0.5f;
  int xi = (int)px, yi = (int)py, zi = (int)pz;
  float s1 = px - (float)xi, s0 = 1.f - s1;
  float t1 = py - (float)yi, t0 = 1.f - t1;
  float f1 = pz - (float)zi, f0 = 1.f - f1;
  if (px < 0.f) {
    xi = 0;
    s0 = 1.f;
    s1 = 0.f;
  }
  if (py < 0.f) {
    yi = 0;
    t0 = 1.f;
    t1 = 0.f;
  }
  if (pz < 0.f) {
    zi = 0;
    f0 = 1.f;
    f1 = 0.f;
  }
  if (xi >= sx - 1) {
    xi = sx - 2;
    s0 = 0.f;
    s1 = 1.f;
  }
  if (yi >= sy - 1) {
    yi = sy - 2;
    t0 = 0.f;
    t1 = 1.f;
  }
  if (sz > 1 && zi >= sz - 1) {
    zi = sz - 2;
    f0 = 0.f;
    f1 = 1.f;
  }
  const long X = 1, Y = sx, Z = strideZ;
  long idx = xi + Y * yi + Z * zi;
  return ((data[idx] * t0 + data[idx + Y] * t1) * s0 +
          (data[idx + X] * t0 + data[idx + X + Y] * t1) * s1) *
             f0 +
         ((data[idx + Z] * t0 + data[idx + Y + Z] * t1) * s0 +
          (data[idx + X + Z] * t0 + data[idx + X + Y + Z] * t1) * s1) *
             f1;
}

__global__ void slScalar(const float *__restrict__ vel,
                         const float *__restrict__ src,
                         float *__restrict__ dst,
                         int sx,
                         int sy,
                         int sz,
                         long strideZ,
                         float dt)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    // MACGrid::getCentered: average of the two adjacent faces per axis.
    const float vx = 0.5f * (vel[3 * t] + vel[3 * (t + 1)]);
    const float vy = 0.5f * (vel[3 * t + 1] + vel[3 * (t + sx) + 1]);
    const float vz = (sz > 1) ? 0.5f * (vel[3 * t + 2] + vel[3 * (t + strideZ) + 2]) : 0.f;
    const float x = ((float)i + 0.5f) - vx * dt;
    const float y = ((float)j + 0.5f) - vy * dt;
    const float z = ((float)k + 0.5f) - vz * dt;
    dst[t] = interpLinear(src, sx, sy, sz, strideZ, x, y, z);
  }
}

// ---------------------------------------------------------------------------
// pressure solve (unpreconditioned CG), 1:1 with MakeLaplaceMatrix /
// ApplyMatrix / GridCg(PcNone) / correctVelocity in the CPU code

// Persistent buffers for the pressure solve (grow-only).
struct PressBufs {
  int *flags = nullptr;   // n
  float *rhs = nullptr;   // n
  float *frac = nullptr;  // 3n (obstacle fractions, optional input)
  float *p = nullptr;     // n pressure
  float *res = nullptr;
  float *search = nullptr;
  float *tmp = nullptr;
  float *A0 = nullptr;
  float *Ai = nullptr;
  float *Aj = nullptr;
  float *Ak = nullptr;
  double *dotOut = nullptr;
  unsigned long long *maxOut = nullptr;
  float *hRhs = nullptr;
  float *hP = nullptr;
  float *hFrac = nullptr;
  int *hFlags = nullptr;
  long cap = 0;

  bool ensure(long cells)
  {
    if (cells <= cap) {
      return true;
    }
    freeAll();
    bool ok = cudaMalloc(&flags, cells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&rhs, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&frac, 3 * cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&p, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&res, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&search, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&tmp, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&A0, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&Ai, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&Aj, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&Ak, cells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&dotOut, sizeof(double)) == cudaSuccess;
    ok = ok && cudaMalloc(&maxOut, sizeof(unsigned long long)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hRhs, cells * sizeof(float), cudaHostAllocDefault) ==
             cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hP, cells * sizeof(float), cudaHostAllocDefault) ==
             cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hFrac, 3 * cells * sizeof(float), cudaHostAllocDefault) ==
             cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hFlags, cells * sizeof(int), cudaHostAllocDefault) ==
             cudaSuccess;
    if (!ok) {
      cudaFail("pressure buffer alloc", cudaGetLastError());
      freeAll();
      return false;
    }
    cap = cells;
    return true;
  }
  void freeAll()
  {
    cudaFree(flags);
    cudaFree(rhs);
    cudaFree(frac);
    cudaFree(p);
    cudaFree(res);
    cudaFree(search);
    cudaFree(tmp);
    cudaFree(A0);
    cudaFree(Ai);
    cudaFree(Aj);
    cudaFree(Ak);
    cudaFree(dotOut);
    cudaFree(maxOut);
    cudaFreeHost(hRhs);
    cudaFreeHost(hP);
    cudaFreeHost(hFrac);
    cudaFreeHost(hFlags);
    flags = nullptr;
    rhs = frac = p = res = search = tmp = nullptr;
    A0 = Ai = Aj = Ak = nullptr;
    dotOut = nullptr;
    maxOut = nullptr;
    hRhs = hP = hFrac = nullptr;
    hFlags = nullptr;
    cap = 0;
  }
};

static PressBufs g_pc;

// Same as MakeLaplaceMatrix (interior only; A-grids must be zeroed first).
__global__ void cgMakeMatrix(const int *__restrict__ flags,
                             float *__restrict__ A0,
                             float *__restrict__ Ai,
                             float *__restrict__ Aj,
                             float *__restrict__ Ak,
                             const float *__restrict__ frac, /* 3 per cell or null */
                             int sx,
                             int sy,
                             int sz,
                             long strideZ)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    if (!(flags[t] & 1)) {  // fluid
      continue;
    }
    if (frac == nullptr) {
      if (!(flags[t - 1] & 2))
        A0[t] += 1.f;
      if (!(flags[t + 1] & 2))
        A0[t] += 1.f;
      if (!(flags[t - sx] & 2))
        A0[t] += 1.f;
      if (!(flags[t + sx] & 2))
        A0[t] += 1.f;
      if (!(flags[t - strideZ] & 2))
        A0[t] += 1.f;
      if (!(flags[t + strideZ] & 2))
        A0[t] += 1.f;
      if (flags[t + 1] & 1)
        Ai[t] = -1.f;
      if (flags[t + sx] & 1)
        Aj[t] = -1.f;
      if (flags[t + strideZ] & 1)
        Ak[t] = -1.f;
    }
    else {
      A0[t] += frac[3 * t] + frac[3 * (t + 1)] + frac[3 * t + 1] + frac[3 * (t + sx) + 1] +
               frac[3 * t + 2] + frac[3 * (t + strideZ) + 2];
      if (flags[t + 1] & 1)
        Ai[t] = -frac[3 * (t + 1)];
      if (flags[t + sx] & 1)
        Aj[t] = -frac[3 * (t + sx) + 1];
      if (flags[t + strideZ] & 1)
        Ak[t] = -frac[3 * (t + strideZ) + 2];
    }
  }
}

// Same as ApplyMatrix (whole grid; non-fluid cells copy src through).
__global__ void cgApplyMat(const int *__restrict__ flags,
                           float *__restrict__ dst,
                           const float *__restrict__ src,
                           const float *__restrict__ A0,
                           const float *__restrict__ Ai,
                           const float *__restrict__ Aj,
                           const float *__restrict__ Ak,
                           int sx,
                           int sy,
                           int sz,
                           long strideZ)
{
  long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (!(flags[t] & 1)) {
      dst[t] = src[t];
      continue;
    }
    // Neighbor reads are in bounds for all fluid cells of the interior box;
    // clamp at the outer border (walls, never fluid in practice).
    long xm = (t % sx == 0) ? t : t - 1;
    long ym = ((t / sx) % sy == 0) ? t : t - sx;
    long zm = (t / strideZ == 0) ? t : t - strideZ;
    long xp = (t % sx == sx - 1) ? t : t + 1;
    long yp = ((t / sx) % sy == sy - 1) ? t : t + sx;
    long zp = (t / strideZ == sz - 1) ? t : t + strideZ;
    dst[t] = src[t] * A0[t] + src[xm] * Ai[xm] + src[xp] * Ai[t] + src[ym] * Aj[ym] +
             src[yp] * Aj[t] + src[zm] * Ak[zm] + src[zp] * Ak[t];
  }
}

__global__ void cgCopy(float *__restrict__ dst, const float *__restrict__ src, long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    dst[t] = src[t];
  }
}

// p += alpha * search; res -= alpha * tmp(Ap); tmp = res   (one fused pass)
__global__ void cgUpdate(float *__restrict__ p,
                         float *__restrict__ res,
                         float *__restrict__ tmp,
                         const float *__restrict__ search,
                         float alpha,
                         long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    p[t] += alpha * search[t];
    res[t] -= alpha * tmp[t];
    tmp[t] = res[t];
  }
}

// search = tmp + beta * search
__global__ void cgSearchUpdate(float *__restrict__ search,
                               const float *__restrict__ tmp,
                               float beta,
                               long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    search[t] = tmp[t] + beta * search[t];
  }
}

// Double-precision dot product (block reduce + atomicAdd), and optionally the
// max-abs norm of r in the same pass (bit trick: fabs bits are monotonic).
__global__ void cgDotMax(const float *__restrict__ a,
                         const float *__restrict__ b,
                         const float *__restrict__ r,
                         long n,
                         double *dotOut,
                         unsigned long long *maxOut)
{
  __shared__ double sh[256];
  __shared__ unsigned int shm[256];
  double s = 0.0;
  unsigned int m = 0u;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    s += (double)a[t] * (double)b[t];
    if (r != nullptr) {
      unsigned int bits = __float_as_uint(fabsf(r[t]));
      if (bits > m) {
        m = bits;
      }
    }
  }
  sh[threadIdx.x] = s;
  shm[threadIdx.x] = m;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      sh[threadIdx.x] += sh[threadIdx.x + off];
      if (shm[threadIdx.x + off] > shm[threadIdx.x]) {
        shm[threadIdx.x] = shm[threadIdx.x + off];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(dotOut, sh[0]);
    if (r != nullptr) {
      atomicMax(maxOut, (unsigned long long)shm[0]);
    }
  }
}

// ---------------------------------------------------------------------------
// particle levelsets (stage 4a): counting-sorted particle index plus the
// averaged levelset / correction / smoothing passes of
// improvedParticleLevelset and unionParticleLevelset. Ported 1:1 from
// plugin/flip.cpp; the counting sort replaces the serial count / prefix-sum /
// scatter passes of gridParticleIndex. Within-bin particle order depends on
// atomic scheduling, so the accumulators can differ from the CPU run in the
// last float ULP (visually equivalent, same policy as the CG solve port).

// Counting sort pass 1: per-cell counts of active, in-bounds particles (the
// same filter as gridParticleIndex; PDELETE = 1<<10).
__global__ void plsCount(const float4 *__restrict__ parts,
                         long nParts,
                         int *__restrict__ counts,
                         int sx,
                         int sy,
                         int sz,
                         long strideZ)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < nParts;
       t += (long)gridDim.x * blockDim.x) {
    const float4 p = parts[t];
    if (__float_as_int(p.w) & (1 << 10)) {
      continue;
    }
    const int i = (int)floorf(p.x);
    const int j = (int)floorf(p.y);
    const int k = (int)floorf(p.z);
    if (i < 0 || j < 0 || k < 0 || i >= sx || j >= sy || k >= sz) {
      continue;
    }
    atomicAdd(&counts[i + (long)sx * j + strideZ * k], 1);
  }
}

// Exclusive scan over the cell counts, three steps: block-local inclusive
// scan, serial scan of the block totals, then combine.
__global__ void plsScanBlocks(const int *__restrict__ counts,
                              int *__restrict__ incl,
                              int *__restrict__ blockTotals,
                              long n)
{
  __shared__ int sh[256];
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  sh[threadIdx.x] = (t < n) ? counts[t] : 0;
  __syncthreads();
  for (int off = 1; off < 256; off <<= 1) {
    const int v = (threadIdx.x >= off) ? sh[threadIdx.x - off] : 0;
    __syncthreads();
    sh[threadIdx.x] += v;
    __syncthreads();
  }
  if (t < n) {
    incl[t] = sh[threadIdx.x];
  }
  if (threadIdx.x == blockDim.x - 1) {
    blockTotals[blockIdx.x] = sh[threadIdx.x];
  }
}

// Single-thread; also stores the grand total (active particle count) as the
// final bin end in cellStart[n].
__global__ void plsScanOffsets(int *__restrict__ blockTotals,
                               int numBlocks,
                               int *__restrict__ cellStart,
                               long n)
{
  int run = 0;
  for (int b = 0; b < numBlocks; b++) {
    const int v = blockTotals[b];
    blockTotals[b] = run;
    run += v;
  }
  cellStart[n] = run;
}

__global__ void plsScanApply(const int *__restrict__ counts,
                             const int *__restrict__ incl,
                             const int *__restrict__ blockTotals,
                             int *__restrict__ cellStart,
                             long n)
{
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) {
    cellStart[t] = blockTotals[blockIdx.x] + incl[t] - counts[t];
  }
}

// Counting sort pass 2: scatter particle indices into their cell bins.
__global__ void plsScatter(const float4 *__restrict__ parts,
                           long nParts,
                           const int *__restrict__ cellStart,
                           int *__restrict__ cursor,
                           int *__restrict__ sorted,
                           int sx,
                           int sy,
                           int sz,
                           long strideZ)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < nParts;
       t += (long)gridDim.x * blockDim.x) {
    const float4 p = parts[t];
    if (__float_as_int(p.w) & (1 << 10)) {
      continue;
    }
    const int i = (int)floorf(p.x);
    const int j = (int)floorf(p.y);
    const int k = (int)floorf(p.z);
    if (i < 0 || j < 0 || k < 0 || i >= sx || j >= sy || k >= sz) {
      continue;
    }
    const long c = i + (long)sx * j + strideZ * k;
    const long slot = (long)cellStart[c] + atomicAdd(&cursor[c], 1);
    sorted[slot] = (int)t;
  }
}

// Same as ComputeAveragedLevelsetWeight (full grid): phi = |gridPos - pAcc| -
// rAcc with w = max(0, 1 - d^2/(4r^2)) weights; pAcc/rAcc saved for the
// correction pass (zeroed first, like the freshly constructed CPU grids).
__global__ void plsAveraged(const float4 *__restrict__ parts,
                            const int *__restrict__ sorted,
                            const int *__restrict__ cellStart,
                            float *__restrict__ phi,
                            float3 *__restrict__ pAcc,
                            float *__restrict__ rAcc,
                            int sx,
                            int sy,
                            int sz,
                            long strideZ,
                            int r,
                            float radius,
                            float sradiusInv)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    const float gx = i + 0.5f, gy = j + 0.5f, gz = k + 0.5f;
    float phiv = radius;
    float wacc = 0.f;
    float3 pacc = make_float3(0.f, 0.f, 0.f);
    float racc = 0.f;

    const int z1 = min(k + r, sz - 1);
    const int y1 = min(j + r, sy - 1);
    const int x1 = min(i + r, sx - 1);
    for (int zj = max(0, k - r); zj <= z1; zj++) {
      for (int yj = max(0, j - r); yj <= y1; yj++) {
        for (int xj = max(0, i - r); xj <= x1; xj++) {
          const long c = xj + (long)sx * yj + strideZ * zj;
          const int p1 = cellStart[c + 1];
          for (int p = cellStart[c]; p < p1; p++) {
            const float4 pp = parts[sorted[p]];
            const float dx = gx - pp.x, dy = gy - pp.y, dz = gz - pp.z;
            const float s = (dx * dx + dy * dy + dz * dz) * sradiusInv;
            const float w = fmaxf(0.f, 1.f - s);
            wacc += w;
            racc += radius * w;
            pacc.x += pp.x * w;
            pacc.y += pp.y * w;
            pacc.z += pp.z * w;
          }
        }
      }
    }
    if (wacc > 1e-6f) {  // VECTOR_EPSILON
      racc /= wacc;
      pacc.x /= wacc;
      pacc.y /= wacc;
      pacc.z /= wacc;
      const float ex = gx - pacc.x, ey = gy - pacc.y, ez = gz - pacc.z;
      phiv = sqrtf(ex * ex + ey * ey + ez * ez) - racc;
      pAcc[t] = pacc;
      rAcc[t] = racc;
    }
    phi[t] = phiv;
  }
}

// Cardano eigenvalues, ported from Matrix3x3<Real>::eigenvalues()
// (helper/util/matrixbase.h); returns the largest of the three. The
// intermediate expressions use double math wherever the original has a double
// literal (3.0, /2.0, pow(x, 1.0/3.0), ...) and narrow to float on store -
// the discriminant h sits near zero for typical Jacobians, so deviating
// precision flips the h>/<0 branch and changes the eigenvalues completely.
__device__ __forceinline__ float plsMaxEigen(float v00,
                                             float v01,
                                             float v02,
                                             float v10,
                                             float v11,
                                             float v12,
                                             float v20,
                                             float v21,
                                             float v22)
{
  const float b = -v00 - v11 - v22;
  const float c = v00 * (v11 + v22) + v11 * v22 - v12 * v21 - v01 * v10 - v02 * v20;
  float d = -v00 * (v11 * v22 - v12 * v21) - v20 * (v01 * v12 - v11 * v02) -
            v10 * (v02 * v21 - v22 * v01);
  const float f = (float)((3.0 * (double)c - (double)(b * b)) / 3.0);
  const float g = (float)((((2.0 * (double)b) * b * b - (9.0 * (double)b) * c) +
                          27.0 * (double)d) /
                         27.0);
  const float h = (float)((double)(g * g) / 4.0 + (double)(f * f * f) / 27.0);

  float e0, e1, e2;
  float sign;
  if (h > 0.0f) {
    float r = (float)(-(double)g / 2.0 + (double)sqrtf(h));
    if (r < 0.0f) {
      r = -r;
      sign = -1.0f;
    }
    else {
      sign = 1.0f;
    }
    const float s = (float)((double)sign * pow((double)r, 1.0 / 3.0));
    float t = (float)(-(double)g / 2.0 - (double)sqrtf(h));
    if (t < 0.0f) {
      t = -t;
      sign = -1.0f;
    }
    else {
      sign = 1.0f;
    }
    const float u = (float)((double)sign * pow((double)t, 1.0 / 3.0));
    e0 = (float)((double)(s + u) - (double)b / 3.0);
    e1 = e2 = 0;
  }
  else if (h == 0.0f) {
    if (d < 0.0f) {
      d = -d;
      sign = -1.0f;
    }
    else {
      sign = 1.0f;
    }
    e0 = (float)(-1.0 * (double)sign * pow((double)d, 1.0 / 3.0));
    e1 = e2 = 0;
  }
  else {
    const float i = (float)sqrt((double)(g * g) / 4.0 - (double)h);
    const float j = (float)pow((double)i, 1.0 / 3.0);
    const float k = (float)acos(-(double)g / (2.0 * (double)i));
    const float l = -j;
    const float m = (float)cos((double)k / 3.0);
    const float n = (float)(sqrt(3.0) * sin((double)k / 3.0));
    const float p = (float)(-(double)b / 3.0);
    e0 = (float)(2.0 * (double)j * m + (double)p);
    e1 = l * (m + n) + p;
    e2 = l * (m - n) + p;
  }
  return fmaxf(e0, fmaxf(e1, e2));
}

// Same as correctLevelset (bnd 1): Jacobian of pAcc via central differences,
// cubic correction from the largest eigenvalue, clamp outside values.
__global__ void plsCorrect(float *__restrict__ phi,
                           const float3 *__restrict__ pAcc,
                           const float *__restrict__ rAcc,
                           int sx,
                           int sy,
                           int sz,
                           long strideZ,
                           float radius,
                           float t_low,
                           float t_high)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    if (rAcc[t] <= 1e-6f) {  // VECTOR_EPSILON
      continue;
    }
    const float maxEV = plsMaxEigen(
        0.5f * (pAcc[t + 1].x - pAcc[t - 1].x),
        0.5f * (pAcc[t + sx].x - pAcc[t - sx].x),
        0.5f * (pAcc[t + strideZ].x - pAcc[t - strideZ].x),
        0.5f * (pAcc[t + 1].y - pAcc[t - 1].y),
        0.5f * (pAcc[t + sx].y - pAcc[t - sx].y),
        0.5f * (pAcc[t + strideZ].y - pAcc[t - strideZ].y),
        0.5f * (pAcc[t + 1].z - pAcc[t - 1].z),
        0.5f * (pAcc[t + sx].z - pAcc[t - sx].z),
        0.5f * (pAcc[t + strideZ].z - pAcc[t - strideZ].z));

    float correction = 1.0f;
    if (maxEV >= t_low) {
      const float tt = (t_high - maxEV) / (t_high - t_low);
      correction = tt * tt * tt - 3 * tt * tt + 3 * tt;
    }
    correction = fminf(fmaxf(correction, 0.f), 1.f);

    const float gx = i + 0.5f, gy = j + 0.5f, gz = k + 0.5f;
    const float ex = gx - pAcc[t].x, ey = gy - pAcc[t].y, ez = gz - pAcc[t].z;
    const float correctedPhi = fabsf(sqrtf(ex * ex + ey * ey + ez * ez)) - rAcc[t] * correction;
    phi[t] = (correctedPhi > radius) ? radius : correctedPhi;
  }
}

// Same as knSmoothGrid (bnd 1): tmp = (center + axis neighbors) * factor.
__global__ void plsSmooth(const float *__restrict__ me,
                          float *__restrict__ tmp,
                          int sx,
                          int sy,
                          int sz,
                          long strideZ,
                          float factor)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    float val = me[t] + me[t + 1] + me[t - 1] + me[t + sx] + me[t - sx];
    if (sz > 1) {
      val += me[t + strideZ] + me[t - strideZ];
    }
    tmp[t] = val * factor;
  }
}

// Same as knSmoothGridNeg (bnd 1): tmp = smoothed < tmp ? smoothed : me. The
// tmp read matches the CPU call sequence - fresh grid (zero) on the first
// pass of an iteration, previous-swap content (old phi) on the second.
__global__ void plsSmoothNeg(const float *__restrict__ me,
                             float *__restrict__ tmp,
                             int sx,
                             int sy,
                             int sz,
                             long strideZ,
                             float factor)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    float val = me[t] + me[t + 1] + me[t - 1] + me[t + sx] + me[t - sx];
    if (sz > 1) {
      val += me[t + strideZ] + me[t - strideZ];
    }
    val *= factor;
    tmp[t] = (val < tmp[t]) ? val : me[t];
  }
}

// Same as knSetBoundary for setBound(0.5, 0): overwrite the outer ring.
__global__ void plsSetBound(float *__restrict__ phi, int sx, int sy, int sz)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, (long)sx * sy, &i, &j, &k);
    const bool bnd = (i <= 0 || i >= sx - 1 || j <= 0 || j >= sy - 1 ||
                      (sz > 1 && (k <= 0 || k >= sz - 1)));
    if (bnd) {
      phi[t] = 0.5f;
    }
  }
}

// Same as ComputeUnionLevelsetPindex (full grid): phi = min over nearby
// particles of |gridPos - pos| - radius.
__global__ void plsUnion(const float4 *__restrict__ parts,
                         const int *__restrict__ sorted,
                         const int *__restrict__ cellStart,
                         float *__restrict__ phi,
                         int sx,
                         int sy,
                         int sz,
                         long strideZ,
                         int r,
                         float radius)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    const float gx = i + 0.5f, gy = j + 0.5f, gz = k + 0.5f;
    float phiv = radius;

    const int z1 = min(k + r, sz - 1);
    const int y1 = min(j + r, sy - 1);
    const int x1 = min(i + r, sx - 1);
    for (int zj = max(0, k - r); zj <= z1; zj++) {
      for (int yj = max(0, j - r); yj <= y1; yj++) {
        for (int xj = max(0, i - r); xj <= x1; xj++) {
          const long c = xj + (long)sx * yj + strideZ * zj;
          const int p1 = cellStart[c + 1];
          for (int p = cellStart[c]; p < p1; p++) {
            const float4 pp = parts[sorted[p]];
            const float dx = gx - pp.x, dy = gy - pp.y, dz = gz - pp.z;
            phiv = fminf(phiv, sqrtf(dx * dx + dy * dy + dz * dz) - radius);
          }
        }
      }
    }
    phi[t] = phiv;
  }
}

// Persistent buffers for the particle levelset ops (same grow-only scheme as
// DevBuffers / PressBufs). `parts` holds the raw BasicParticleData array
// (pos.xyz + flag bits, 16 bytes) reinterpreted as float4.
struct PlsBufs {
  float4 *parts = nullptr;
  int *sorted = nullptr;     // particle indices, counting-sorted by cell
  int *cellStart = nullptr;  // n + 1 per-cell bin starts
  int *scanA = nullptr;      // counts, then scatter cursor
  int *scanB = nullptr;      // block-local inclusive scan
  int *blockTotals = nullptr;
  float *phi = nullptr;
  float *phiT = nullptr;
  float3 *pAcc = nullptr;
  float *rAcc = nullptr;
  float4 *hParts = nullptr;  // pinned staging
  float *hPhi = nullptr;
  float *pvel = nullptr;   // device: per-particle velocity (3 floats each)
  float *hPvel = nullptr;  // pinned staging
  long pvelFloats = 0;
  long cells = 0;
  long partsCap = 0;
  int scanBlocksCap = 0;

  bool ensurePvel(long floats)
  {
    if (floats <= pvelFloats) {
      return true;
    }
    cudaFree(pvel);
    cudaFreeHost(hPvel);
    pvel = nullptr;
    hPvel = nullptr;
    const long fc = (floats > 0) ? floats : 1;
    bool ok = cudaMalloc(&pvel, fc * sizeof(float)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hPvel, fc * sizeof(float), cudaHostAllocDefault) ==
               cudaSuccess;
    if (!ok) {
      cudaFail("pls pvel alloc", cudaGetLastError());
      cudaFree(pvel);
      cudaFreeHost(hPvel);
      pvel = nullptr;
      hPvel = nullptr;
      pvelFloats = 0;
      return false;
    }
    pvelFloats = fc;
    return true;
  }

  bool ensure(long nCells, long nParts)
  {
    const long nb = (nCells + 255) / 256;
    if (nCells <= cells && nParts <= partsCap && nb <= (long)scanBlocksCap) {
      return true;
    }
    freeAll();
    const long pc = (nParts > 0) ? nParts : 1;
    bool ok = cudaMalloc(&parts, pc * sizeof(float4)) == cudaSuccess;
    ok = ok && cudaMalloc(&sorted, pc * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&cellStart, (nCells + 1) * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&scanA, nCells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&scanB, nCells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&blockTotals, nb * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&phi, nCells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&phiT, nCells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&pAcc, nCells * sizeof(float3)) == cudaSuccess;
    ok = ok && cudaMalloc(&rAcc, nCells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hParts, pc * sizeof(float4), cudaHostAllocDefault) ==
             cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&hPhi, nCells * sizeof(float), cudaHostAllocDefault) ==
             cudaSuccess;
    if (!ok) {
      cudaFail("pls buffer alloc", cudaGetLastError());
      freeAll();
      return false;
    }
    cells = nCells;
    partsCap = pc;
    scanBlocksCap = (int)nb;
    return true;
  }
  void freeAll()
  {
    cudaFree(parts);
    cudaFree(sorted);
    cudaFree(cellStart);
    cudaFree(scanA);
    cudaFree(scanB);
    cudaFree(blockTotals);
    cudaFree(phi);
    cudaFree(phiT);
    cudaFree(pAcc);
    cudaFree(rAcc);
    cudaFreeHost(hParts);
    cudaFreeHost(hPhi);
    cudaFree(pvel);
    cudaFreeHost(hPvel);
    parts = nullptr;
    sorted = nullptr;
    cellStart = nullptr;
    scanA = scanB = blockTotals = nullptr;
    phi = phiT = nullptr;
    pAcc = nullptr;
    rAcc = nullptr;
    hParts = nullptr;
    hPhi = nullptr;
    pvel = nullptr;
    hPvel = nullptr;
    pvelFloats = 0;
    cells = 0;
    partsCap = 0;
    scanBlocksCap = 0;
  }
};

static PlsBufs g_pls;

// ---------------------------------------------------------------------------
// marching cubes (stage 4b): GPU port of LevelsetGrid::createMesh
// (levelset.cpp). Edge ids are (axis * n + cellFlat); the 12 edges of a cube
// map onto shared ids so the CPU's edgeVX/VY/VZ dedup grids become plain
// flags. Vertices are emitted by the same cube that the serial CPU loop
// would have used first (smallest flat cell index that references the edge),
// with that cube's exact corner ordering from mcEdges - so interpolated
// positions are bit-identical to the CPU run; only the vertex array ORDER
// differs. Normals use sqrtf where the CPU uses std::hypot (last-ULP
// cosmetic difference).

// Marching cubes LUTs on the device (raw byte copies of the mcubes.h data).
__constant__ short cMcEdgeTable[256];
__constant__ short cMcTriTable[256][16];
__constant__ int cMcEdges[24];
__constant__ int cMcCubieX[8];
__constant__ int cMcCubieY[8];
__constant__ int cMcCubieZ[8];
static bool s_mcLutsCopied = false;

static bool mcInitLuts()
{
  if (s_mcLutsCopied) {
    return true;
  }
  if (cudaMemcpyToSymbol(cMcEdgeTable, mcEdgeTable, sizeof(mcEdgeTable)) != cudaSuccess ||
      cudaMemcpyToSymbol(cMcTriTable, mcTriTable, sizeof(mcTriTable)) != cudaSuccess ||
      cudaMemcpyToSymbol(cMcEdges, mcEdges, sizeof(mcEdges)) != cudaSuccess ||
      cudaMemcpyToSymbol(cMcCubieX, cubieOffsetX, sizeof(cubieOffsetX)) != cudaSuccess ||
      cudaMemcpyToSymbol(cMcCubieY, cubieOffsetY, sizeof(cubieOffsetY)) != cudaSuccess ||
      cudaMemcpyToSymbol(cMcCubieZ, cubieOffsetZ, sizeof(cubieOffsetZ)) != cudaSuccess) {
    return false;
  }
  s_mcLutsCopied = true;
  return true;
}

// Cube classification: value[l] = -phi(corner l); skip on invalid cells
// (phi <= -1000), bit l set when -phi < 1e-4 (same expressions as the CPU).
__device__ __forceinline__ int mcCubeIdx(const float *__restrict__ phi,
                                         long t,
                                         int sx,
                                         long strideZ,
                                         bool *skip)
{
  const float v[8] = {-phi[t],
                      -phi[t + 1],
                      -phi[t + 1 + sx],
                      -phi[t + sx],
                      -phi[t + strideZ],
                      -phi[t + 1 + strideZ],
                      -phi[t + 1 + sx + strideZ],
                      -phi[t + sx + strideZ]};
  bool sk = false;
  int ci = 0;
#pragma unroll
  for (int l = 0; l < 8; l++) {
    if (-v[l] <= -1000.0f) {  // FastMarch::InvalidTime()
      sk = true;
    }
    if (v[l] < 1e-4f) {  // isoValue
      ci |= 1 << l;
    }
  }
  *skip = sk;
  return ci;
}

// Global edge id of edge number e of the cube at flat index t.
__device__ __forceinline__ long mcEdgeId(int e, long t, int sx, long strideZ, long n)
{
  const int axis = (e == 1 || e == 3 || e == 5 || e == 7) ? 1 : ((e >= 8) ? 2 : 0);
  long flat = t;
  switch (e) {
    case 1:
      flat = t + 1;
      break;
    case 2:
      flat = t + sx;
      break;
    case 5:
      flat = t + 1 + strideZ;
      break;
    case 4:
      flat = t + strideZ;
      break;
    case 6:
      flat = t + sx + strideZ;
      break;
    case 7:
      flat = t + strideZ;
      break;
    case 9:
      flat = t + 1;
      break;
    case 10:
      flat = t + 1 + sx;
      break;
    case 11:
      flat = t + sx;
      break;
    default:
      break;
  }
  return axis * n + flat;
}

// getGradient (grid.h): per-axis clamped central differences; note the z
// component clamps k only after x/y are read, exactly like the CPU version.
__device__ __forceinline__ float3 mcGradient(const float *__restrict__ phi,
                                             int i,
                                             int j,
                                             int k,
                                             int sx,
                                             int sy,
                                             int sz,
                                             long strideZ)
{
  if (i > sx - 2)
    i = sx - 2;
  if (j > sy - 2)
    j = sy - 2;
  if (i < 1)
    i = 1;
  if (j < 1)
    j = 1;
  const long c = i + (long)sx * j + strideZ * k;
  float3 g = make_float3(phi[c + 1] - phi[c - 1], phi[c + sx] - phi[c - sx], 0.f);
  if (k > sz - 2)
    k = sz - 2;
  if (k < 1)
    k = 1;
  const long cz = i + (long)sx * j + strideZ * k;
  g.z = phi[cz + strideZ] - phi[cz - strideZ];
  return g;
}

// getNormalized (vectorbase.h); length via sqrtf instead of std::hypot.
__device__ __forceinline__ float3 mcNormalize(float x, float y, float z)
{
  const float l = sqrtf(x * x + y * y + z * z);
  if (fabsf(l - 1.0f) < 1e-6f) {  // VECTOR_EPSILON
    return make_float3(x, y, z);
  }
  if (l > 1e-6f) {
    const float fac = (float)(1. / (double)l);
    return make_float3(x * fac, y * fac, z * fac);
  }
  return make_float3(0.f, 0.f, 0.f);
}

// Pass 1 (per cube): mark referenced edges and count triangles.
__global__ void mcMark(const float *__restrict__ phi,
                       int *__restrict__ edgeFlag,  // 3n, zeroed
                       int *__restrict__ cubeTris,  // n, zeroed
                       int *__restrict__ triCounter,
                       int sx,
                       int sy,
                       int sz,
                       long strideZ,
                       long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (i > sx - 2 || j > sy - 2 || k > sz - 2) {
      continue;
    }
    bool skip;
    const int ci = mcCubeIdx(phi, t, sx, strideZ, &skip);
    const short et = (skip || cMcEdgeTable[ci] == 0) ? 0 : cMcEdgeTable[ci];
    if (et == 0) {
      continue;
    }
    for (int e = 0; e < 12; e++) {
      if (et & (1 << e)) {
        atomicMax(&edgeFlag[mcEdgeId(e, t, sx, strideZ, n)], 1);
      }
    }
    int q = 0;
    while (cMcTriTable[ci][q] != -1) {
      q++;
    }
    cubeTris[t] = q / 3;
    if (q > 0) {
      atomicAdd(triCounter, q / 3);
    }
  }
}

// Pass 2 (per edge): interpolate the vertex exactly as the CPU loop would -
// by the first cube (smallest flat index) referencing this edge, using that
// cube's corner pair from mcEdges. Also turns the flag into vertIdx+1.
__global__ void mcVertices(const float *__restrict__ phi,
                           int *__restrict__ edgeFlag,  // 3n flags -> vertIdx+1
                           const int *__restrict__ edgeVert,  // 3n exclusive scan
                           float *__restrict__ verts,
                           float *__restrict__ normals,
                           int sx,
                           int sy,
                           int sz,
                           long strideZ,
                           long n)
{
  const long n3 = 3 * n;
  for (long eid = (long)blockIdx.x * blockDim.x + threadIdx.x; eid < n3;
       eid += (long)gridDim.x * blockDim.x) {
    if (!edgeFlag[eid]) {
      continue;
    }
    const long v = edgeVert[eid];
    const int axis = (int)(eid / n);
    const long t = eid - (long)axis * n;
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);

    // owner candidates (di, dj, dk, edgeNumber) in increasing flat order
    const int candX[4][4] = {{0, -1, -1, 6}, {0, 0, -1, 4}, {0, -1, 0, 2}, {0, 0, 0, 0}};
    const int candY[4][4] = {{-1, 0, -1, 5}, {0, 0, -1, 7}, {-1, 0, 0, 1}, {0, 0, 0, 3}};
    const int candZ[4][4] = {{-1, -1, 0, 10}, {0, -1, 0, 11}, {-1, 0, 0, 9}, {0, 0, 0, 8}};
    const int(*cand)[4] = (axis == 0) ? candX : ((axis == 1) ? candY : candZ);

    for (int c = 0; c < 4; c++) {
      const int ci_ = i + cand[c][0], cj = j + cand[c][1], ck = k + cand[c][2];
      if (ci_ < 0 || cj < 0 || ck < 0 || ci_ > sx - 2 || cj > sy - 2 || ck > sz - 2) {
        continue;
      }
      const long ct = t + cand[c][0] + (long)sx * cand[c][1] + strideZ * cand[c][2];
      const int en = cand[c][3];
      bool skip;
      const int cidx = mcCubeIdx(phi, ct, sx, strideZ, &skip);
      if (skip || !(cMcEdgeTable[cidx] & (1 << en))) {
        continue;
      }
      // this cube creates the vertex - replicate its arithmetic exactly
      const int q1 = cMcEdges[2 * en], q2 = cMcEdges[2 * en + 1];
      const long c1 = ct + cMcCubieX[q1] + (long)sx * cMcCubieY[q1] + strideZ * cMcCubieZ[q1];
      const long c2 = ct + cMcCubieX[q2] + (long)sx * cMcCubieY[q2] + strideZ * cMcCubieZ[q2];
      const float valp1 = -phi[c1], valp2 = -phi[c2];
      const float mu = (1e-4f - valp1) / (valp2 - valp1);
      const float p1x = (float)(ci_ + cMcCubieX[q1]), p1y = (float)(cj + cMcCubieY[q1]),
                  p1z = (float)(ck + cMcCubieZ[q1]);
      const float p2x = (float)(ci_ + cMcCubieX[q2]), p2y = (float)(cj + cMcCubieY[q2]),
                  p2z = (float)(ck + cMcCubieZ[q2]);
      verts[3 * v] = p1x + (p2x - p1x) * mu + 0.5f;
      verts[3 * v + 1] = p1y + (p2y - p1y) * mu + 0.5f;
      verts[3 * v + 2] = p1z + (p2z - p1z) * mu + 0.5f;

      const float3 g1 = mcGradient(
          phi, ci_ + cMcCubieX[q1], cj + cMcCubieY[q1], ck + cMcCubieZ[q1], sx, sy, sz, strideZ);
      const float3 g2 = mcGradient(
          phi, ci_ + cMcCubieX[q2], cj + cMcCubieY[q2], ck + cMcCubieZ[q2], sx, sy, sz, strideZ);
      const float3 nn = mcNormalize(g1.x * (1.0f - mu) + g2.x * mu,
                                    g1.y * (1.0f - mu) + g2.y * mu,
                                    g1.z * (1.0f - mu) + g2.z * mu);
      normals[3 * v] = nn.x;
      normals[3 * v + 1] = nn.y;
      normals[3 * v + 2] = nn.z;

      edgeFlag[eid] = (int)v + 1;
      break;  // only the first owner creates it, as on the CPU
    }
  }
}

// Pass 3 (per cube): emit the LUT triangles via the edge -> vertex map.
__global__ void mcTriangles(const float *__restrict__ phi,
                            const int *__restrict__ edgeFlag,  // holds vertIdx+1
                            const int *__restrict__ cubeTriStart,
                            int *__restrict__ tris,
                            int sx,
                            int sy,
                            int sz,
                            long strideZ,
                            long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (i > sx - 2 || j > sy - 2 || k > sz - 2) {
      continue;
    }
    bool skip;
    const int ci = mcCubeIdx(phi, t, sx, strideZ, &skip);
    if (skip || cMcEdgeTable[ci] == 0) {
      continue;
    }
    const long base = cubeTriStart[t];
    int q = 0;
    for (; cMcTriTable[ci][q] != -1; q++) {
      const long eid = mcEdgeId(cMcTriTable[ci][q], t, sx, strideZ, n);
      tris[3 * base + q] = edgeFlag[eid] - 1;
    }
  }
}

// Persistent buffers for marching cubes (grow-only, same scheme as the
// others; vertex/triangle caps grow after the counts are known each frame).
struct McBufs {
  float *phi = nullptr;        // n
  int *edgeFlag = nullptr;     // 3n flags -> vertIdx+1
  int *edgeVert = nullptr;     // 3n+1 exclusive scan (+ total at [3n])
  int *scanB = nullptr;        // 3n block-local inclusive scan
  int *blockTotA = nullptr;    // blocksFor(3n)
  int *cubeTris = nullptr;     // n per-cube triangle counts
  int *cubeTriStart = nullptr;  // n+1 exclusive scan (+ total at [n])
  int *scanB2 = nullptr;       // n
  int *blockTotB = nullptr;    // blocksFor(n)
  int *triCounter = nullptr;   // 1
  float *verts = nullptr;      // 3 * capV
  float *normals = nullptr;    // 3 * capV
  int *tris = nullptr;         // 3 * capT
  long cells = 0;
  long capV = 0;
  long capT = 0;

  bool ensure(long nCells)
  {
    if (nCells <= cells) {
      return true;
    }
    freeAll();
    const long nbA = (3 * nCells + 255) / 256;
    const long nbB = (nCells + 255) / 256;
    bool ok = cudaMalloc(&phi, nCells * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&edgeFlag, 3 * nCells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&edgeVert, (3 * nCells + 1) * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&scanB, 3 * nCells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&blockTotA, nbA * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&cubeTris, nCells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&cubeTriStart, (nCells + 1) * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&scanB2, nCells * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&blockTotB, nbB * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&triCounter, sizeof(int)) == cudaSuccess;
    if (!ok) {
      cudaFail("mc buffer alloc", cudaGetLastError());
      freeAll();
      return false;
    }
    cells = nCells;
    return true;
  }
  bool ensureVT(long V, long T)
  {
    if (V <= capV && T <= capT) {
      return true;
    }
    const long nv = (V > capV) ? V : capV;
    const long nt = (T > capT) ? T : capT;
    cudaFree(verts);
    cudaFree(normals);
    cudaFree(tris);
    verts = nullptr;
    normals = nullptr;
    tris = nullptr;
    bool ok = (nv > 0) ? cudaMalloc(&verts, 3 * nv * sizeof(float)) == cudaSuccess : true;
    ok = ok && ((nv > 0) ? cudaMalloc(&normals, 3 * nv * sizeof(float)) == cudaSuccess : true);
    ok = ok && ((nt > 0) ? cudaMalloc(&tris, 3 * nt * sizeof(int)) == cudaSuccess : true);
    if (!ok) {
      cudaFail("mc vt buffer alloc", cudaGetLastError());
      cudaFree(verts);
      cudaFree(normals);
      cudaFree(tris);
      verts = nullptr;
      normals = nullptr;
      tris = nullptr;
      capV = 0;
      capT = 0;
      return false;
    }
    capV = nv;
    capT = nt;
    return true;
  }
  void freeAll()
  {
    cudaFree(phi);
    cudaFree(edgeFlag);
    cudaFree(edgeVert);
    cudaFree(scanB);
    cudaFree(blockTotA);
    cudaFree(cubeTris);
    cudaFree(cubeTriStart);
    cudaFree(scanB2);
    cudaFree(blockTotB);
    cudaFree(triCounter);
    cudaFree(verts);
    cudaFree(normals);
    cudaFree(tris);
    phi = nullptr;
    edgeFlag = nullptr;
    edgeVert = nullptr;
    scanB = nullptr;
    blockTotA = nullptr;
    cubeTris = nullptr;
    cubeTriStart = nullptr;
    scanB2 = nullptr;
    blockTotB = nullptr;
    triCounter = nullptr;
    verts = nullptr;
    normals = nullptr;
    tris = nullptr;
    cells = 0;
    capV = 0;
    capT = 0;
  }
};

static McBufs g_mc;

// ---------------------------------------------------------------------------
// particle transfer + extrapolation ops (stage 4c): mapPartsToMAC (the
// particle->grid scatter, serial on CPU), flipVelocityUpdate (per-particle
// gather), extrapolateMACFromWeight and extrapolateVec3Simple. Transcribed
// from plugin/flip.cpp + fastmarch.cpp + helper/util/interpol.h
// (BUILD_INDEX / interpolMAC / setInterpolMAC, including the pos-0.5 shift
// and border clamping). The scatter uses atomicAdd, so weight/velocity sums
// differ from the serial CPU run in the last float ULP (documented policy).

// interpolMAC (staggered trilinear with per-axis shifted base cells).
__device__ __forceinline__ float3 macInterp(const float *__restrict__ data,
                                            int sx,
                                            int sy,
                                            int sz,
                                            long strideZ,
                                            float x0,
                                            float y0,
                                            float z0)
{
  const float px = x0 - 0.5f, py = y0 - 0.5f, pz = z0 - 0.5f;
  int xi = (int)px, yi = (int)py, zi = (int)pz;
  float s1 = px - (float)xi, s0 = 1.f - s1;
  float t1 = py - (float)yi, t0 = 1.f - t1;
  float f1 = pz - (float)zi, f0 = 1.f - f1;
  if (px < 0.f) {
    xi = 0;
    s0 = 1.f;
    s1 = 0.f;
  }
  if (py < 0.f) {
    yi = 0;
    t0 = 1.f;
    t1 = 0.f;
  }
  if (pz < 0.f) {
    zi = 0;
    f0 = 1.f;
    f1 = 0.f;
  }
  if (xi >= sx - 1) {
    xi = sx - 2;
    s0 = 0.f;
    s1 = 1.f;
  }
  if (yi >= sy - 1) {
    yi = sy - 2;
    t0 = 0.f;
    t1 = 1.f;
  }
  if (sz > 1 && zi >= sz - 1) {
    zi = sz - 2;
    f0 = 0.f;
    f1 = 1.f;
  }
  int sxi = (int)x0, syi = (int)y0, szi = (int)z0;
  float ss1 = x0 - (float)sxi, ss0 = 1.f - ss1;
  float st1 = y0 - (float)syi, st0 = 1.f - st1;
  float sf1 = z0 - (float)szi, sf0 = 1.f - sf1;
  if (x0 < 0.f) {
    sxi = 0;
    ss0 = 1.f;
    ss1 = 0.f;
  }
  if (y0 < 0.f) {
    syi = 0;
    st0 = 1.f;
    st1 = 0.f;
  }
  if (z0 < 0.f) {
    szi = 0;
    sf0 = 1.f;
    sf1 = 0.f;
  }
  if (sxi >= sx - 1) {
    sxi = sx - 2;
    ss0 = 0.f;
    ss1 = 1.f;
  }
  if (syi >= sy - 1) {
    syi = sy - 2;
    st0 = 0.f;
    st1 = 1.f;
  }
  if (sz > 1 && szi >= sz - 1) {
    szi = sz - 2;
    sf0 = 0.f;
    sf1 = 1.f;
  }

  const int X = 1, Y = sx;
  const long Z = strideZ;
  float3 ret = make_float3(0.f, 0.f, 0.f);
  {  // X component: shifted x base
    const long idx = ((long)zi * sy + yi) * sx + sxi;
    ret.x = f0 * ((data[3 * idx] * t0 + data[3 * (idx + Y)] * t1) * ss0 +
                  (data[3 * (idx + X)] * t0 + data[3 * (idx + X + Y)] * t1) * ss1) +
            f1 * ((data[3 * (idx + Z)] * t0 + data[3 * (idx + Y + Z)] * t1) * ss0 +
                  (data[3 * (idx + X + Z)] * t0 + data[3 * (idx + X + Y + Z)] * t1) * ss1);
  }
  {  // Y component: shifted y base
    const long idx = ((long)zi * sy + syi) * sx + xi;
    ret.y = f0 * ((data[3 * idx + 1] * st0 + data[3 * (idx + Y) + 1] * st1) * s0 +
                  (data[3 * (idx + X) + 1] * st0 + data[3 * (idx + X + Y) + 1] * st1) * s1) +
            f1 * ((data[3 * (idx + Z) + 1] * st0 + data[3 * (idx + Y + Z) + 1] * st1) * s0 +
                  (data[3 * (idx + X + Z) + 1] * st0 + data[3 * (idx + X + Y + Z) + 1] * st1) *
                      s1);
  }
  {  // Z component: shifted z base
    const long idx = ((long)szi * sy + yi) * sx + xi;
    ret.z = sf0 * ((data[3 * idx + 2] * t0 + data[3 * (idx + Y) + 2] * t1) * s0 +
                   (data[3 * (idx + X) + 2] * t0 + data[3 * (idx + X + Y) + 2] * t1) * s1) +
            sf1 * ((data[3 * (idx + Z) + 2] * t0 + data[3 * (idx + Y + Z) + 2] * t1) * s0 +
                   (data[3 * (idx + X + Z) + 2] * t0 + data[3 * (idx + X + Y + Z) + 2] * t1) *
                       s1);
  }
  return ret;
}

// setInterpolMAC as atomic scatter: velocity weight into both vel and wgt
// (3 floats per cell each), same 8 corners per component as the CPU version.
__device__ __forceinline__ void macScatterAtomic(const float4 *__restrict__ parts,
                                                 const float *__restrict__ pvel,
                                                 float *__restrict__ vel,
                                                 float *__restrict__ wgt,
                                                 int sx,
                                                 int sy,
                                                 int sz,
                                                 long strideZ,
                                                 long idx)
{
  const float4 p = parts[idx];
  if (__float_as_int(p.w) & (1 << 10)) {  // PDELETE
    return;
  }
  const float x0 = p.x, y0 = p.y, z0 = p.z;
  const float px = x0 - 0.5f, py = y0 - 0.5f, pz = z0 - 0.5f;
  int xi = (int)px, yi = (int)py, zi = (int)pz;
  float s1 = px - (float)xi, s0 = 1.f - s1;
  float t1 = py - (float)yi, t0 = 1.f - t1;
  float f1 = pz - (float)zi, f0 = 1.f - f1;
  if (px < 0.f) {
    xi = 0;
    s0 = 1.f;
    s1 = 0.f;
  }
  if (py < 0.f) {
    yi = 0;
    t0 = 1.f;
    t1 = 0.f;
  }
  if (pz < 0.f) {
    zi = 0;
    f0 = 1.f;
    f1 = 0.f;
  }
  if (xi >= sx - 1) {
    xi = sx - 2;
    s0 = 0.f;
    s1 = 1.f;
  }
  if (yi >= sy - 1) {
    yi = sy - 2;
    t0 = 0.f;
    t1 = 1.f;
  }
  if (sz > 1 && zi >= sz - 1) {
    zi = sz - 2;
    f0 = 0.f;
    f1 = 1.f;
  }
  int sxi = (int)x0, syi = (int)y0, szi = (int)z0;
  float ss1 = x0 - (float)sxi, ss0 = 1.f - ss1;
  float st1 = y0 - (float)syi, st0 = 1.f - st1;
  float sf1 = z0 - (float)szi, sf0 = 1.f - sf1;
  if (x0 < 0.f) {
    sxi = 0;
    ss0 = 1.f;
    ss1 = 0.f;
  }
  if (y0 < 0.f) {
    syi = 0;
    st0 = 1.f;
    st1 = 0.f;
  }
  if (z0 < 0.f) {
    szi = 0;
    sf0 = 1.f;
    sf1 = 0.f;
  }
  if (sxi >= sx - 1) {
    sxi = sx - 2;
    ss0 = 0.f;
    ss1 = 1.f;
  }
  if (syi >= sy - 1) {
    syi = sy - 2;
    st0 = 0.f;
    st1 = 1.f;
  }
  if (sz > 1 && szi >= sz - 1) {
    szi = sz - 2;
    sf0 = 0.f;
    sf1 = 1.f;
  }

  const int X = 1, Y = sx;
  const long Z = strideZ;
  const float vx = pvel[3 * idx], vy = pvel[3 * idx + 1], vz = pvel[3 * idx + 2];
  {  // X
    const long b = ((long)zi * sy + yi) * sx + sxi;
    const float s0f0 = ss0 * f0, s1f0 = ss1 * f0, s0f1 = ss0 * f1, s1f1 = ss1 * f1;
    const float w0 = t0 * s0f0, wx = t0 * s1f0, wy = t1 * s0f0, wxy = t1 * s1f0;
    const float wz = t0 * s0f1, wxz = t0 * s1f1, wyz = t1 * s0f1, wxyz = t1 * s1f1;
    atomicAdd(&wgt[3 * b], w0);
    atomicAdd(&wgt[3 * (b + X)], wx);
    atomicAdd(&wgt[3 * (b + Y)], wy);
    atomicAdd(&wgt[3 * (b + X + Y)], wxy);
    atomicAdd(&wgt[3 * (b + Z)], wz);
    atomicAdd(&wgt[3 * (b + X + Z)], wxz);
    atomicAdd(&wgt[3 * (b + Y + Z)], wyz);
    atomicAdd(&wgt[3 * (b + X + Y + Z)], wxyz);
    atomicAdd(&vel[3 * b], w0 * vx);
    atomicAdd(&vel[3 * (b + X)], wx * vx);
    atomicAdd(&vel[3 * (b + Y)], wy * vx);
    atomicAdd(&vel[3 * (b + X + Y)], wxy * vx);
    atomicAdd(&vel[3 * (b + Z)], wz * vx);
    atomicAdd(&vel[3 * (b + X + Z)], wxz * vx);
    atomicAdd(&vel[3 * (b + Y + Z)], wyz * vx);
    atomicAdd(&vel[3 * (b + X + Y + Z)], wxyz * vx);
  }
  {  // Y
    const long b = ((long)zi * sy + syi) * sx + xi;
    const float s0f0 = s0 * f0, s1f0 = s1 * f0, s0f1 = s0 * f1, s1f1 = s1 * f1;
    const float w0 = st0 * s0f0, wx = st0 * s1f0, wy = st1 * s0f0, wxy = st1 * s1f0;
    const float wz = st0 * s0f1, wxz = st0 * s1f1, wyz = st1 * s0f1, wxyz = st1 * s1f1;
    atomicAdd(&wgt[3 * b + 1], w0);
    atomicAdd(&wgt[3 * (b + X) + 1], wx);
    atomicAdd(&wgt[3 * (b + Y) + 1], wy);
    atomicAdd(&wgt[3 * (b + X + Y) + 1], wxy);
    atomicAdd(&wgt[3 * (b + Z) + 1], wz);
    atomicAdd(&wgt[3 * (b + X + Z) + 1], wxz);
    atomicAdd(&wgt[3 * (b + Y + Z) + 1], wyz);
    atomicAdd(&wgt[3 * (b + X + Y + Z) + 1], wxyz);
    atomicAdd(&vel[3 * b + 1], w0 * vy);
    atomicAdd(&vel[3 * (b + X) + 1], wx * vy);
    atomicAdd(&vel[3 * (b + Y) + 1], wy * vy);
    atomicAdd(&vel[3 * (b + X + Y) + 1], wxy * vy);
    atomicAdd(&vel[3 * (b + Z) + 1], wz * vy);
    atomicAdd(&vel[3 * (b + X + Z) + 1], wxz * vy);
    atomicAdd(&vel[3 * (b + Y + Z) + 1], wyz * vy);
    atomicAdd(&vel[3 * (b + X + Y + Z) + 1], wxyz * vy);
  }
  {  // Z
    const long b = ((long)szi * sy + yi) * sx + xi;
    const float s0f0 = s0 * sf0, s1f0 = s1 * sf0, s0f1 = s0 * sf1, s1f1 = s1 * sf1;
    const float w0 = t0 * s0f0, wx = t0 * s1f0, wy = t1 * s0f0, wxy = t1 * s1f0;
    const float wz = t0 * s0f1, wxz = t0 * s1f1, wyz = t1 * s0f1, wxyz = t1 * s1f1;
    atomicAdd(&wgt[3 * b + 2], w0);
    atomicAdd(&wgt[3 * (b + X) + 2], wx);
    atomicAdd(&wgt[3 * (b + Y) + 2], wy);
    atomicAdd(&wgt[3 * (b + X + Y) + 2], wxy);
    atomicAdd(&wgt[3 * (b + Z) + 2], wz);
    atomicAdd(&wgt[3 * (b + X + Z) + 2], wxz);
    atomicAdd(&wgt[3 * (b + Y + Z) + 2], wyz);
    atomicAdd(&wgt[3 * (b + X + Y + Z) + 2], wxyz);
    atomicAdd(&vel[3 * b + 2], w0 * vz);
    atomicAdd(&vel[3 * (b + X) + 2], wx * vz);
    atomicAdd(&vel[3 * (b + Y) + 2], wy * vz);
    atomicAdd(&vel[3 * (b + X + Y) + 2], wxy * vz);
    atomicAdd(&vel[3 * (b + Z) + 2], wz * vz);
    atomicAdd(&vel[3 * (b + X + Z) + 2], wxz * vz);
    atomicAdd(&vel[3 * (b + Y + Z) + 2], wyz * vz);
    atomicAdd(&vel[3 * (b + X + Y + Z) + 2], wxyz * vz);
  }
}

// knMapLinearVec3ToMACGrid equivalent: per-particle scatter (atomicAdd).
__global__ void pmScatter(const float4 *__restrict__ parts,
                          const float *__restrict__ pvel,
                          float *__restrict__ vel,
                          float *__restrict__ wgt,
                          long nParts,
                          int sx,
                          int sy,
                          int sz,
                          long strideZ)
{
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < nParts;
       idx += (long)gridDim.x * blockDim.x) {
    macScatterAtomic(parts, pvel, vel, wgt, sx, sy, sz, strideZ, idx);
  }
}

// weight stomp (Vec3, < 1e-6 -> 0) + safeDivide (b ? a/b : a), per component.
__global__ void pmNormalize(float *__restrict__ vel, float *__restrict__ wgt, long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
#pragma unroll
    for (int c = 0; c < 3; c++) {
      float w = wgt[3 * t + c];
      if (w < 1e-6f) {
        w = 0.f;
        wgt[3 * t + c] = 0.f;
      }
      if (w != 0.f) {
        vel[3 * t + c] = vel[3 * t + c] / w;
      }
    }
  }
}

// knMapLinearMACGridToVec3_FLIP equivalent: pvel = flip*(pvel+delta)+(1-flip)*v
__global__ void fuUpdate(const float4 *__restrict__ parts,
                         float *__restrict__ pvel,
                         const float *__restrict__ vel,
                         const float *__restrict__ velOld,
                         long nParts,
                         int sx,
                         int sy,
                         int sz,
                         long strideZ,
                         float flipRatio)
{
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < nParts;
       idx += (long)gridDim.x * blockDim.x) {
    if (__float_as_int(parts[idx].w) & (1 << 10)) {  // PDELETE
      continue;
    }
    const float3 v = macInterp(vel, sx, sy, sz, strideZ, parts[idx].x, parts[idx].y, parts[idx].z);
    const float3 ov =
        macInterp(velOld, sx, sy, sz, strideZ, parts[idx].x, parts[idx].y, parts[idx].z);
    const float dx = v.x - ov.x, dy = v.y - ov.y, dz = v.z - ov.z;
    pvel[3 * idx] = flipRatio * (pvel[3 * idx] + dx) + (1.0f - flipRatio) * v.x;
    pvel[3 * idx + 1] = flipRatio * (pvel[3 * idx + 1] + dy) + (1.0f - flipRatio) * v.y;
    pvel[3 * idx + 2] = flipRatio * (pvel[3 * idx + 2] + dz) + (1.0f - flipRatio) * v.z;
  }
}

// extrapolateMACFromWeight: init pass (weight > 0 -> 1) per component.
__global__ void mfwInit(float *__restrict__ wgt, int sx, int sy, int sz, long strideZ, int c)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    if (wgt[3 * t + c] > 0.f) {
      wgt[3 * t + c] = 1.f;
    }
  }
}

// extrapolateMACFromWeight dilation: average over 6 neighbors with weight==d.
__global__ void mfwDilate(float *__restrict__ vel,
                          float *__restrict__ wgt,
                          int sx,
                          int sy,
                          int sz,
                          long strideZ,
                          int c,
                          int d)
{
  const long n = (long)sx * sy * sz;
  const bool is3D = (sz > 1);
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    if (wgt[3 * t + c] != 0.f) {
      continue;
    }
    const long nb[6] = {t + 1, t - 1, t + sx, t - sx, t + strideZ, t - strideZ};
    float avgVel = 0.f;
    int nbs = 0;
    const int cnt = is3D ? 6 : 4;
    for (int q = 0; q < cnt; q++) {
      if (wgt[3 * nb[q] + c] == (float)d) {
        avgVel += vel[3 * nb[q] + c];
        nbs++;
      }
    }
    if (nbs > 0) {
      wgt[3 * t + c] = (float)(d + 1);
      vel[3 * t + c] = avgVel / (float)nbs;
    }
  }
}

// extrapolateVec3Simple: same mark/first-layer as the scalar variant (lsMarkSkip/
// lsFirstLayer reuse), Vec3 copy-dilation (no direction offset) and zero fill.
__global__ void lsDilateVec3(float *__restrict__ vel,
                             int *__restrict__ tmp,
                             int sx,
                             int sy,
                             int sz,
                             long strideZ,
                             int dim,
                             int d)
{
  const long n = (long)sx * sy * sz;
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    int i, j, k;
    coords(t, sx, strideZ, &i, &j, &k);
    if (!inInnerBox(i, j, k, sx, sy, sz, 1)) {
      continue;
    }
    if (tmp[t] != 0) {
      continue;
    }
    const long nb[6] = {t + 1, t - 1, t + sx, t - sx, t + strideZ, t - strideZ};
    const int cnt = (dim == 3) ? 6 : 4;
    float ax = 0.f, ay = 0.f, az = 0.f;
    int nbs = 0;
    for (int q = 0; q < cnt; q++) {
      if (tmp[nb[q]] == d) {
        ax += vel[3 * nb[q]];
        ay += vel[3 * nb[q] + 1];
        az += vel[3 * nb[q] + 2];
        nbs++;
      }
    }
    if (nbs > 0) {
      tmp[t] = d + 1;
      vel[3 * t] = ax / (float)nbs;
      vel[3 * t + 1] = ay / (float)nbs;
      vel[3 * t + 2] = az / (float)nbs;
    }
  }
}

__global__ void lsRemainingVec3(float *__restrict__ vel,
                                const int *__restrict__ tmp,
                                long n)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
    if (tmp[t] == 0) {
      vel[3 * t] = 0.f;
      vel[3 * t + 1] = 0.f;
      vel[3 * t + 2] = 0.f;
    }
  }
}

// ---------------------------------------------------------------------------
// host entry points (declared in Manta namespace, see manta_cuda.h)

static int blocksFor(long n)
{
  return (int)((n + 255) / 256);
}

}  // namespace manta_cuda

bool cudaFloodFill(Grid<Real> &phi, Real value, bool outside, int boundaryWidth)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = phi.getSizeX(), sy = phi.getSizeY(), sz = phi.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX) {
    return false;
  }
  const long bytes = n * sizeof(float);

  if (!g_dev.ensure(n, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &phi(0), bytes);
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, bytes, cudaMemcpyHostToDevice),
          "floodFill: copy in");
  CUDA_RC(cudaMemsetAsync(buf.ibuf, 0, n * sizeof(int)), "floodFill: memset visited");

  ffInit<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, boundaryWidth, outside);
  // sx+sy+sz alternating sweeps is a safe upper bound on the propagation
  // depth; all launches are queued asynchronously with no round trips.
  const int sweeps = sx + sy + sz;
  for (int s = 0; s < sweeps; s++) {
    ffSweep<<<blocksFor(n), 256>>>(
        buf.phi, buf.ibuf, sx, sy, sz, strideZ, boundaryWidth, outside, s & 1);
  }
  applyFill<<<blocksFor(n), 256>>>(
      buf.phi, buf.ibuf, sx, sy, sz, strideZ, boundaryWidth, (float)value, outside);
  CUDA_RC(cudaMemcpyAsync(buf.hPhi, buf.phi, bytes, cudaMemcpyDeviceToHost),
          "floodFill: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "floodFill: sync");
  memcpy(&phi(0), buf.hPhi, bytes);
  return true;
}

bool cudaExtrapolateLsSimple(Grid<Real> &phi, int distance, bool inside)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = phi.getSizeX(), sy = phi.getSizeY(), sz = phi.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  const int dim = (sz > 1) ? 3 : 2;
  if (n <= 0 || n >= (long)INT_MAX) {
    return false;
  }
  const long bytes = n * sizeof(float);

  if (!g_dev.ensure(n, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &phi(0), bytes);
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, bytes, cudaMemcpyHostToDevice),
          "extrapolateLs: copy in");
  CUDA_RC(cudaMemsetAsync(buf.ibuf, 0, n * sizeof(int)), "extrapolateLs: memset tmp");

  const float direction = inside ? -1.0f : 1.0f;
  lsMarkSkip<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, inside);
  lsFirstLayer<<<blocksFor(n), 256>>>(buf.ibuf, sx, sy, sz, strideZ, dim);
  for (int d = 2; d < 1 + distance; ++d) {
    lsDilate<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, dim, d, direction);
  }
  lsRemaining<<<blocksFor(n), 256>>>(
      buf.phi, buf.ibuf, sx, sy, sz, strideZ, direction * (distance + 2));

  CUDA_RC(cudaMemcpyAsync(buf.hPhi, buf.phi, bytes, cudaMemcpyDeviceToHost),
          "extrapolateLs: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "extrapolateLs: sync");
  memcpy(&phi(0), buf.hPhi, bytes);
  return true;
}

bool cudaExtrapolateMACSimple(FlagGrid &flags, MACGrid &vel, int distance, bool intoObs, int dim)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = flags.getSizeX(), sy = flags.getSizeY(), sz = flags.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  // A MAC grid stores one Vec3 (3 floats) per cell.
  if (n <= 0 || n >= (long)INT_MAX || sizeof(Vec3) != 3 * sizeof(Real)) {
    return false;
  }
  const long velFloats = 3 * n;
  const long velBytes = velFloats * sizeof(float);

  if (!g_dev.ensure(velFloats, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &vel(0), velBytes);
  memcpy(buf.hFlags, &flags(0), n * sizeof(int));
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, velBytes, cudaMemcpyHostToDevice),
          "extrapolateMAC: copy in");
  CUDA_RC(cudaMemcpyAsync(buf.flagsBuf, buf.hFlags, n * sizeof(int), cudaMemcpyHostToDevice),
          "extrapolateMAC: flags copy");

  for (int c = 0; c < dim; ++c) {
    CUDA_RC(cudaMemsetAsync(buf.ibuf, 0, n * sizeof(int)), "extrapolateMAC: memset tmp");
    macMark<<<blocksFor(n), 256>>>(buf.flagsBuf, buf.ibuf, sx, sy, sz, strideZ, c, intoObs);
    cudaError_t lrc = cudaGetLastError();
    if (lrc != cudaSuccess) {
      cudaFail("extrapolateMAC: macMark launch", lrc);
      return false;
    }
    for (int d = 1; d < 1 + distance; ++d) {
      macDilate<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, dim, d, c);
      lrc = cudaGetLastError();
      if (lrc != cudaSuccess) {
        cudaFail("extrapolateMAC: macDilate launch", lrc);
        return false;
      }
    }
  }

  CUDA_RC(cudaMemcpyAsync(buf.hPhi, buf.phi, velBytes, cudaMemcpyDeviceToHost),
          "extrapolateMAC: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "extrapolateMAC: sync");
  memcpy(&vel(0), buf.hPhi, velBytes);
  return true;
}

bool cudaAdvectSemiLagrangeScalar(const FlagGrid &flags, MACGrid &vel, Grid<Real> &grid, Real dt)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = flags.getSizeX(), sy = flags.getSizeY(), sz = flags.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX) {
    return false;
  }
  const long velFloats = 3 * n;

  if (!g_dev.ensure(velFloats, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &vel(0), velFloats * sizeof(float));
  memcpy(buf.hFlags, &grid(0), n * sizeof(float));  // scalar src via flags staging
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, velFloats * sizeof(float), cudaMemcpyHostToDevice),
          "advectScalar: vel copy in");
  CUDA_RC(cudaMemcpyAsync(buf.phi2, buf.hFlags, n * sizeof(float), cudaMemcpyHostToDevice),
          "advectScalar: src copy in");

  slScalar<<<blocksFor(n), 256>>>(
      buf.phi, buf.phi2, (float *)buf.ibuf, sx, sy, sz, strideZ, (float)dt);
  cudaError_t lrc = cudaGetLastError();
  if (lrc != cudaSuccess) {
    cudaFail("advectScalar: launch", lrc);
    return false;
  }

  CUDA_RC(cudaMemcpyAsync(buf.hFlags, buf.ibuf, n * sizeof(float), cudaMemcpyDeviceToHost),
          "advectScalar: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "advectScalar: sync");
  memcpy(&grid(0), buf.hFlags, n * sizeof(float));
  return true;
}

bool cudaSolvePressureCg(Grid<Real> &rhs,
                         Grid<Real> &pressure,
                         FlagGrid &flags,
                         Real cgAccuracy,
                         MACGrid *fractions,
                         Real cgMaxIterFac)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = flags.getSizeX(), sy = flags.getSizeY(), sz = flags.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2) {
    return false;
  }
  const long bytes = n * sizeof(float);

  if (!g_pc.ensure(n)) {
    return false;
  }
  PressBufs &buf = g_pc;
  memcpy(buf.hRhs, &rhs(0), bytes);
  memcpy(buf.hFlags, &flags(0), bytes);
  const bool haveFrac = (fractions != nullptr);
  if (haveFrac) {
    memcpy(buf.hFrac, &(*fractions)(0), 3 * bytes);
  }
  CUDA_RC(cudaMemcpyAsync(buf.rhs, buf.hRhs, bytes, cudaMemcpyHostToDevice),
          "pressure: rhs copy in");
  CUDA_RC(cudaMemcpyAsync(buf.flags, buf.hFlags, bytes, cudaMemcpyHostToDevice),
          "pressure: flags copy in");
  if (haveFrac) {
    CUDA_RC(cudaMemcpyAsync(buf.frac, buf.hFrac, 3 * bytes, cudaMemcpyHostToDevice),
            "pressure: fractions copy in");
  }
  CUDA_RC(cudaMemsetAsync(buf.p, 0, bytes), "pressure: memset p");
  CUDA_RC(cudaMemsetAsync(buf.A0, 0, bytes), "pressure: memset A0");
  CUDA_RC(cudaMemsetAsync(buf.Ai, 0, bytes), "pressure: memset Ai");
  CUDA_RC(cudaMemsetAsync(buf.Aj, 0, bytes), "pressure: memset Aj");
  CUDA_RC(cudaMemsetAsync(buf.Ak, 0, bytes), "pressure: memset Ak");

  cgMakeMatrix<<<blocksFor(n), 256>>>(
      buf.flags, buf.A0, buf.Ai, buf.Aj, buf.Ak, haveFrac ? buf.frac : nullptr, sx, sy, sz,
      strideZ);
  cudaError_t lrc = cudaGetLastError();
  if (lrc != cudaSuccess) {
    cudaFail("pressure: matrix launch", lrc);
    return false;
  }

  const int gridDim = (sx > sy && sx > sz) ? sx : ((sy > sz) ? sy : sz);
  // Unpreconditioned CG needs more iterations than the CPU MIC variant;
  // each iteration is far cheaper on the GPU.
  const int maxIter = 2 * (int)(cgMaxIterFac * (Real)gridDim);

  // doInit: res = rhs, search = res (PcNone), sigma = dot(res, rhs)
  cgCopy<<<blocksFor(n), 256>>>(buf.res, buf.rhs, n);
  cgCopy<<<blocksFor(n), 256>>>(buf.search, buf.rhs, n);
  double sigma = 0.0;
  CUDA_RC(cudaMemsetAsync(buf.dotOut, 0, sizeof(double)), "pressure: memset dot");
  cgDotMax<<<blocksFor(n), 256>>>(buf.res, buf.rhs, nullptr, n, buf.dotOut, nullptr);
  CUDA_RC(cudaStreamSynchronize(0), "pressure: init sync");
  CUDA_RC(cudaMemcpy(&sigma, buf.dotOut, sizeof(double), cudaMemcpyDeviceToHost),
          "pressure: init dot read");

  for (int iter = 0; iter < maxIter && sigma > 0.0; iter++) {
    cgApplyMat<<<blocksFor(n), 256>>>(
        buf.flags, buf.tmp, buf.search, buf.A0, buf.Ai, buf.Aj, buf.Ak, sx, sy, sz, strideZ);
    CUDA_RC(cudaMemsetAsync(buf.dotOut, 0, sizeof(double)), "pressure: memset dot");
    cgDotMax<<<blocksFor(n), 256>>>(buf.tmp, buf.search, nullptr, n, buf.dotOut, nullptr);
    CUDA_RC(cudaStreamSynchronize(0), "pressure: iter sync 1");
    double dp = 0.0;
    CUDA_RC(cudaMemcpy(&dp, buf.dotOut, sizeof(double), cudaMemcpyDeviceToHost),
            "pressure: dot read");
    float alpha = 0.f;
    if (fabs(dp) > 0.0) {
      alpha = (float)(sigma / dp);
    }
    cgUpdate<<<blocksFor(n), 256>>>(buf.p, buf.res, buf.tmp, buf.search, alpha, n);
    CUDA_RC(cudaMemsetAsync(buf.dotOut, 0, sizeof(double)), "pressure: memset dot2");
    CUDA_RC(cudaMemsetAsync(buf.maxOut, 0, sizeof(unsigned long long)), "pressure: memset max");
    cgDotMax<<<blocksFor(n), 256>>>(buf.tmp, buf.res, buf.res, n, buf.dotOut, buf.maxOut);
    CUDA_RC(cudaStreamSynchronize(0), "pressure: iter sync 2");
    double sigmaNew = 0.0;
    unsigned long long maxBits = 0;
    CUDA_RC(cudaMemcpy(&sigmaNew, buf.dotOut, sizeof(double), cudaMemcpyDeviceToHost),
            "pressure: dot2 read");
    CUDA_RC(cudaMemcpy(&maxBits, buf.maxOut, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
            "pressure: max read");
    float resNorm;
    memcpy(&resNorm, &maxBits, 4);  // low 32 bits hold the float bits
    if (resNorm < (float)cgAccuracy) {
      break;
    }
    float beta = 0.f;
    if (fabs(sigma) > 0.0) {
      beta = (float)(sigmaNew / sigma);
    }
    cgSearchUpdate<<<blocksFor(n), 256>>>(buf.search, buf.tmp, beta, n);
    sigma = sigmaNew;
  }

  CUDA_RC(cudaMemcpyAsync(buf.hP, buf.p, bytes, cudaMemcpyDeviceToHost), "pressure: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "pressure: final sync");
  memcpy(&pressure(0), buf.hP, bytes);
  return true;
}

// Builds the counting-sorted particle index on the device (counts, exclusive
// scan, scatter) into the persistent PlsBufs. Shared by both levelset ops.
// `parts` is the raw BasicParticleData array (pos.xyz + flag bits) as float4.
static bool plsBuildIndex(manta_cuda::PlsBufs &b,
                          const float4 *parts,
                          long nParts,
                          int sx,
                          int sy,
                          int sz,
                          long strideZ,
                          long n)
{
  using namespace manta_cuda;
  if (nParts > 0) {
    memcpy(b.hParts, parts, nParts * sizeof(float4));
    CUDA_RC(cudaMemcpyAsync(b.parts, b.hParts, nParts * sizeof(float4), cudaMemcpyHostToDevice),
            "pls: parts copy in");
  }
  CUDA_RC(cudaMemsetAsync(b.scanA, 0, n * sizeof(int)), "pls: memset counts");
  if (nParts > 0) {
    plsCount<<<blocksFor(nParts), 256>>>(b.parts, nParts, b.scanA, sx, sy, sz, strideZ);
  }
  const int nb = blocksFor(n);
  plsScanBlocks<<<nb, 256>>>(b.scanA, b.scanB, b.blockTotals, n);
  plsScanOffsets<<<1, 1>>>(b.blockTotals, nb, b.cellStart, n);
  plsScanApply<<<nb, 256>>>(b.scanA, b.scanB, b.blockTotals, b.cellStart, n);
  CUDA_RC(cudaMemsetAsync(b.scanA, 0, n * sizeof(int)), "pls: memset cursor");
  if (nParts > 0) {
    plsScatter<<<blocksFor(nParts), 256>>>(
        b.parts, nParts, b.cellStart, b.scanA, b.sorted, sx, sy, sz, strideZ);
  }
  return true;
}

bool cudaImprovedParticleLevelset(const void *partsData,
                                  long nParts,
                                  Grid<Real> &phi,
                                  Real radiusFactor,
                                  int smoothen,
                                  int smoothenNeg,
                                  Real t_low,
                                  Real t_high)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const float4 *parts = (const float4 *)partsData;
  const int sx = phi.getSizeX(), sy = phi.getSizeY(), sz = phi.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2 || nParts < 0 || nParts >= (long)INT_MAX) {
    return false;
  }
  if (nParts > 0 && parts == nullptr) {
    return false;
  }
  const long bytes = n * sizeof(float);

  // Same math chain as calculateRadiusFactor + the CPU call sites (double
  // intermediates, narrowed to float once).
  const float radius = (float)(0.5 * (sqrt(3.0) * ((double)radiusFactor + .01)));
  const int r = (int)radius + 1;
  const float sradiusInv = (float)(1. / (4. * (double)radius * (double)radius));
  const float factor = (float)(1. / 7.);  // 3D

  if (!g_pls.ensure(n, nParts)) {
    return false;
  }
  PlsBufs &b = g_pls;
  if (!plsBuildIndex(b, parts, nParts, sx, sy, sz, strideZ, n)) {
    return false;
  }

  CUDA_RC(cudaMemsetAsync(b.pAcc, 0, n * sizeof(float3)), "pls: memset pAcc");
  CUDA_RC(cudaMemsetAsync(b.rAcc, 0, bytes), "pls: memset rAcc");
  plsAveraged<<<blocksFor(n), 256>>>(
      b.parts, b.sorted, b.cellStart, b.phi, b.pAcc, b.rAcc, sx, sy, sz, strideZ, r, radius,
      sradiusInv);
  plsCorrect<<<blocksFor(n), 256>>>(
      b.phi, b.pAcc, b.rAcc, sx, sy, sz, strideZ, radius, (float)t_low, (float)t_high);

  float *cur = b.phi;
  float *tmp = b.phiT;
  const int iters = (smoothen > smoothenNeg) ? smoothen : smoothenNeg;
  for (int it = 0; it < iters; ++it) {
    // The CPU loop constructs a fresh (zeroed) tmp grid per iteration; the
    // neg pass then reads whatever the preceding swap left in tmp.
    CUDA_RC(cudaMemsetAsync(tmp, 0, bytes), "pls: memset tmp");
    if (it < smoothen) {
      plsSmooth<<<blocksFor(n), 256>>>(cur, tmp, sx, sy, sz, strideZ, factor);
      float *s = cur;
      cur = tmp;
      tmp = s;
    }
    if (it < smoothenNeg) {
      plsSmoothNeg<<<blocksFor(n), 256>>>(cur, tmp, sx, sy, sz, strideZ, factor);
      float *s = cur;
      cur = tmp;
      tmp = s;
    }
  }
  plsSetBound<<<blocksFor(n), 256>>>(cur, sx, sy, sz);

  CUDA_RC(cudaGetLastError(), "pls: launch");
  CUDA_RC(cudaMemcpyAsync(b.hPhi, cur, bytes, cudaMemcpyDeviceToHost), "pls: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "pls: sync");
  memcpy(&phi(0), b.hPhi, bytes);
  return true;
}

bool cudaUnionParticleLevelset(const void *partsData, long nParts, Grid<Real> &phi, Real radiusFactor)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const float4 *parts = (const float4 *)partsData;
  const int sx = phi.getSizeX(), sy = phi.getSizeY(), sz = phi.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2 || nParts < 0 || nParts >= (long)INT_MAX) {
    return false;
  }
  if (nParts > 0 && parts == nullptr) {
    return false;
  }
  const long bytes = n * sizeof(float);

  const float radius = (float)(0.5 * (sqrt(3.0) * ((double)radiusFactor + .01)));
  const int r = (int)radius + 1;

  if (!g_pls.ensure(n, nParts)) {
    return false;
  }
  PlsBufs &b = g_pls;
  if (!plsBuildIndex(b, parts, nParts, sx, sy, sz, strideZ, n)) {
    return false;
  }

  plsUnion<<<blocksFor(n), 256>>>(
      b.parts, b.sorted, b.cellStart, b.phi, sx, sy, sz, strideZ, r, radius);
  plsSetBound<<<blocksFor(n), 256>>>(b.phi, sx, sy, sz);

  CUDA_RC(cudaGetLastError(), "plsUnion: launch");
  CUDA_RC(cudaMemcpyAsync(b.hPhi, b.phi, bytes, cudaMemcpyDeviceToHost), "plsUnion: copy out");
  CUDA_RC(cudaStreamSynchronize(0), "plsUnion: sync");
  memcpy(&phi(0), b.hPhi, bytes);
  return true;
}

bool cudaMarchingCubes(Grid<Real> &phi,
                       std::vector<float> &verts,
                       std::vector<float> &normals,
                       std::vector<int> &tris)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = phi.getSizeX(), sy = phi.getSizeY(), sz = phi.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2 || 3 * n >= (long)INT_MAX) {
    return false;
  }
  if (!mcInitLuts()) {
    cudaFail("mc: LUT copy", cudaGetLastError());
    return false;
  }
  const long bytes = n * sizeof(float);

  if (!g_mc.ensure(n)) {
    return false;
  }
  McBufs &b = g_mc;
  CUDA_RC(cudaMemcpyAsync(b.phi, &phi(0), bytes, cudaMemcpyHostToDevice), "mc: phi copy in");
  // &phi(0) is regular pageable memory; wait for the upload before kernels
  CUDA_RC(cudaStreamSynchronize(0), "mc: phi upload sync");
  CUDA_RC(cudaMemsetAsync(b.edgeFlag, 0, 3 * n * sizeof(int)), "mc: memset edges");
  CUDA_RC(cudaMemsetAsync(b.cubeTris, 0, bytes), "mc: memset cubeTris");
  CUDA_RC(cudaMemsetAsync(b.triCounter, 0, sizeof(int)), "mc: memset tricount");

  mcMark<<<blocksFor(n), 256>>>(
      b.phi, b.edgeFlag, b.cubeTris, b.triCounter, sx, sy, sz, strideZ, n);
  CUDA_RC(cudaGetLastError(), "mc: mark launch");

  // compact edges (reuse the particle-index scan kernels) and cube->tri slots
  const long n3 = 3 * n;
  const int nbA = blocksFor(n3);
  plsScanBlocks<<<nbA, 256>>>(b.edgeFlag, b.scanB, b.blockTotA, n3);
  plsScanOffsets<<<1, 1>>>(b.blockTotA, nbA, b.edgeVert, n3);
  plsScanApply<<<nbA, 256>>>(b.edgeFlag, b.scanB, b.blockTotA, b.edgeVert, n3);
  const int nbB = blocksFor(n);
  plsScanBlocks<<<nbB, 256>>>(b.cubeTris, b.scanB2, b.blockTotB, n);
  plsScanOffsets<<<1, 1>>>(b.blockTotB, nbB, b.cubeTriStart, n);
  plsScanApply<<<nbB, 256>>>(b.cubeTris, b.scanB2, b.blockTotB, b.cubeTriStart, n);

  int V = 0, T = 0;
  CUDA_RC(cudaStreamSynchronize(0), "mc: scan sync");
  CUDA_RC(cudaMemcpy(&V, &b.edgeVert[n3], sizeof(int), cudaMemcpyDeviceToHost), "mc: read V");
  CUDA_RC(cudaMemcpy(&T, b.triCounter, sizeof(int), cudaMemcpyDeviceToHost), "mc: read T");

  if (!g_mc.ensureVT(V, T)) {
    return false;
  }
  if (V > 0) {
    mcVertices<<<blocksFor(n3), 256>>>(
        b.phi, b.edgeFlag, b.edgeVert, b.verts, b.normals, sx, sy, sz, strideZ, n);
  }
  if (T > 0) {
    mcTriangles<<<blocksFor(n), 256>>>(
        b.phi, b.edgeFlag, b.cubeTriStart, b.tris, sx, sy, sz, strideZ, n);
  }

  CUDA_RC(cudaGetLastError(), "mc: launch");
  verts.resize(3 * (size_t)V);
  normals.resize(3 * (size_t)V);
  tris.resize(3 * (size_t)T);
  if (V > 0) {
    CUDA_RC(cudaMemcpy(verts.data(), b.verts, 3 * V * sizeof(float), cudaMemcpyDeviceToHost),
            "mc: verts out");
    CUDA_RC(cudaMemcpy(normals.data(), b.normals, 3 * V * sizeof(float), cudaMemcpyDeviceToHost),
            "mc: normals out");
  }
  if (T > 0) {
    CUDA_RC(cudaMemcpy(tris.data(), b.tris, 3 * T * sizeof(int), cudaMemcpyDeviceToHost),
            "mc: tris out");
  }
  CUDA_RC(cudaStreamSynchronize(0), "mc: final sync");
  return true;
}

bool cudaMapPartsToMAC(const void *partsData,
                       long nParts,
                       const float *pvelData,
                       MACGrid &vel,
                       Grid<Vec3> &weight)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = vel.getSizeX(), sy = vel.getSizeY(), sz = vel.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2 || nParts < 0 || nParts >= (long)INT_MAX) {
    return false;
  }
  if (nParts > 0 && (partsData == nullptr || pvelData == nullptr)) {
    return false;
  }
  const long velFloats = 3 * n;
  const long velBytes = velFloats * sizeof(float);
  if (!g_dev.ensure(velFloats, velFloats)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  CUDA_RC(cudaMemsetAsync(buf.phi, 0, velBytes), "mapParts: memset vel");
  CUDA_RC(cudaMemsetAsync(buf.phi2, 0, velBytes), "mapParts: memset weight");
  if (nParts > 0) {
    if (!g_pls.ensure(n, nParts) || !g_pls.ensurePvel(3 * nParts)) {
      return false;
    }
    memcpy(g_pls.hParts, partsData, nParts * sizeof(float4));
    memcpy(g_pls.hPvel, pvelData, 3 * nParts * sizeof(float));
    CUDA_RC(cudaMemcpyAsync(
                g_pls.parts, g_pls.hParts, nParts * sizeof(float4), cudaMemcpyHostToDevice),
            "mapParts: parts copy");
    CUDA_RC(cudaMemcpyAsync(g_pls.pvel, g_pls.hPvel, 3 * nParts * sizeof(float),
                            cudaMemcpyHostToDevice),
            "mapParts: pvel copy");
    pmScatter<<<blocksFor(nParts), 256>>>(
        g_pls.parts, g_pls.pvel, buf.phi, buf.phi2, nParts, sx, sy, sz, strideZ);
    CUDA_RC(cudaGetLastError(), "mapParts: scatter launch");
  }
  pmNormalize<<<blocksFor(n), 256>>>(buf.phi, buf.phi2, n);
  CUDA_RC(cudaMemcpyAsync(buf.hPhi, buf.phi, velBytes, cudaMemcpyDeviceToHost),
          "mapParts: vel copy out");
  CUDA_RC(cudaMemcpyAsync((float *)buf.hFlags, buf.phi2, velBytes, cudaMemcpyDeviceToHost),
          "mapParts: weight copy out");
  CUDA_RC(cudaStreamSynchronize(0), "mapParts: sync");
  memcpy(&vel(0), buf.hPhi, velBytes);
  memcpy(&weight(0), buf.hFlags, velBytes);
  return true;
}

bool cudaFlipVelocityUpdate(const void *partsData,
                            long nParts,
                            float *pvelData,
                            MACGrid &vel,
                            MACGrid &velOld,
                            Real flipRatio)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = vel.getSizeX(), sy = vel.getSizeY(), sz = vel.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2 || nParts <= 0 || nParts >= (long)INT_MAX) {
    return false;
  }
  if (partsData == nullptr || pvelData == nullptr) {
    return false;
  }
  const long velFloats = 3 * n;
  const long velBytes = velFloats * sizeof(float);
  if (!g_dev.ensure(velFloats, velFloats)) {
    return false;
  }
  if (!g_pls.ensure(n, nParts) || !g_pls.ensurePvel(3 * nParts)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &vel(0), velBytes);
  memcpy((float *)buf.hFlags, &velOld(0), velBytes);
  memcpy(g_pls.hParts, partsData, nParts * sizeof(float4));
  memcpy(g_pls.hPvel, pvelData, 3 * nParts * sizeof(float));
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, velBytes, cudaMemcpyHostToDevice),
          "flipUpdate: vel copy in");
  CUDA_RC(cudaMemcpyAsync(buf.phi2, (float *)buf.hFlags, velBytes, cudaMemcpyHostToDevice),
          "flipUpdate: velOld copy in");
  CUDA_RC(cudaMemcpyAsync(
              g_pls.parts, g_pls.hParts, nParts * sizeof(float4), cudaMemcpyHostToDevice),
          "flipUpdate: parts copy");
  CUDA_RC(cudaMemcpyAsync(g_pls.pvel, g_pls.hPvel, 3 * nParts * sizeof(float),
                          cudaMemcpyHostToDevice),
          "flipUpdate: pvel copy");
  fuUpdate<<<blocksFor(nParts), 256>>>(
      g_pls.parts, g_pls.pvel, buf.phi, buf.phi2, nParts, sx, sy, sz, strideZ, (float)flipRatio);
  CUDA_RC(cudaGetLastError(), "flipUpdate: launch");
  CUDA_RC(cudaMemcpyAsync(g_pls.hPvel, g_pls.pvel, 3 * nParts * sizeof(float),
                          cudaMemcpyDeviceToHost),
          "flipUpdate: pvel copy out");
  CUDA_RC(cudaStreamSynchronize(0), "flipUpdate: sync");
  memcpy(pvelData, g_pls.hPvel, 3 * nParts * sizeof(float));
  return true;
}

bool cudaExtrapolateMACFromWeight(MACGrid &vel, Grid<Vec3> &weight, int distance)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = vel.getSizeX(), sy = vel.getSizeY(), sz = vel.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2) {
    return false;
  }
  const long velFloats = 3 * n;
  const long velBytes = velFloats * sizeof(float);
  if (!g_dev.ensure(velFloats, velFloats)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &vel(0), velBytes);
  memcpy((float *)buf.hFlags, &weight(0), velBytes);
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, velBytes, cudaMemcpyHostToDevice),
          "mfw: vel copy in");
  CUDA_RC(cudaMemcpyAsync(buf.phi2, (float *)buf.hFlags, velBytes, cudaMemcpyHostToDevice),
          "mfw: weight copy in");

  const int dim = (sz > 1) ? 3 : 2;
  for (int c = 0; c < dim; ++c) {
    mfwInit<<<blocksFor(n), 256>>>(buf.phi2, sx, sy, sz, strideZ, c);
    for (int d = 1; d < 1 + distance; ++d) {
      mfwDilate<<<blocksFor(n), 256>>>(buf.phi, buf.phi2, sx, sy, sz, strideZ, c, d);
    }
  }
  CUDA_RC(cudaGetLastError(), "mfw: launch");
  CUDA_RC(cudaMemcpyAsync(buf.hPhi, buf.phi, velBytes, cudaMemcpyDeviceToHost),
          "mfw: vel copy out");
  CUDA_RC(cudaMemcpyAsync((float *)buf.hFlags, buf.phi2, velBytes, cudaMemcpyDeviceToHost),
          "mfw: weight copy out");
  CUDA_RC(cudaStreamSynchronize(0), "mfw: sync");
  memcpy(&vel(0), buf.hPhi, velBytes);
  memcpy(&weight(0), buf.hFlags, velBytes);
  return true;
}

bool cudaExtrapolateVec3Simple(Grid<Vec3> &vel, Grid<Real> &phi, int distance, bool inside)
{
  using namespace manta_cuda;
  if (!enabled()) {
    return false;
  }
  const int sx = vel.getSizeX(), sy = vel.getSizeY(), sz = vel.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX || sz < 2) {
    return false;
  }
  const long velFloats = 3 * n;
  const long velBytes = velFloats * sizeof(float);
  if (!g_dev.ensure(velFloats, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  memcpy(buf.hPhi, &vel(0), velBytes);
  memcpy(buf.hFlags, &phi(0), n * sizeof(float));  // scalar phi via flags staging
  CUDA_RC(cudaMemcpyAsync(buf.phi, buf.hPhi, velBytes, cudaMemcpyHostToDevice),
          "extrapVec3: vel copy in");
  CUDA_RC(cudaMemcpyAsync(buf.flagsBuf, buf.hFlags, n * sizeof(float), cudaMemcpyHostToDevice),
          "extrapVec3: phi copy in");
  CUDA_RC(cudaMemsetAsync(buf.ibuf, 0, n * sizeof(int)), "extrapVec3: memset tmp");

  const int dim = (sz > 1) ? 3 : 2;
  lsMarkSkip<<<blocksFor(n), 256>>>((const float *)buf.flagsBuf, buf.ibuf, sx, sy, sz, strideZ, inside);
  lsFirstLayer<<<blocksFor(n), 256>>>(buf.ibuf, sx, sy, sz, strideZ, dim);
  for (int d = 2; d < 1 + distance; ++d) {
    lsDilateVec3<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, dim, d);
  }
  lsRemainingVec3<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, n);

  CUDA_RC(cudaGetLastError(), "extrapVec3: launch");
  CUDA_RC(cudaMemcpyAsync(buf.hPhi, buf.phi, velBytes, cudaMemcpyDeviceToHost),
          "extrapVec3: vel copy out");
  CUDA_RC(cudaStreamSynchronize(0), "extrapVec3: sync");
  memcpy(&vel(0), buf.hPhi, velBytes);
  return true;
}

}  // namespace Manta
