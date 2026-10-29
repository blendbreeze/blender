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
  int *ibuf = nullptr;     // device: visited / tmp
  int *flagsBuf = nullptr; // device: flag grid copy
  float *hPhi = nullptr;   // pinned host staging for phi
  int *hFlags = nullptr;   // pinned host staging for flags
  long phiFloats = 0;
  long intCells = 0;

  bool ensure(long needFloats, long needInts)
  {
    if (needFloats <= phiFloats && needInts <= intCells) {
      return true;
    }
    freeAll();
    bool ok = cudaMalloc(&phi, needFloats * sizeof(float)) == cudaSuccess;
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
    cudaFree(ibuf);
    cudaFree(flagsBuf);
    cudaFreeHost(hPhi);
    cudaFreeHost(hFlags);
    phi = nullptr;
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

}  // namespace Manta
