// SPDX-License-Identifier: Apache-2.0
// Experimental CUDA kernels for select mantaflow operations (flood fill,
// levelset extrapolation, MAC velocity extrapolation). Ported 1:1 from the
// CPU kernels in levelset.cpp / fastmarch.cpp - see PROJECT_NOTES.md.
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

#include <cuda_runtime.h>

#include "manta_cuda.h"

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

}  // namespace Manta
