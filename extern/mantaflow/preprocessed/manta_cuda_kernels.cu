// SPDX-License-Identifier: Apache-2.0
// Experimental CUDA kernels for select mantaflow operations (flood fill,
// levelset extrapolation, MAC velocity extrapolation). Ported 1:1 from the
// CPU kernels in levelset.cpp / fastmarch.cpp - see PROJECT_NOTES.md.
//
// Enabled at runtime via MANTA_CUDA=1. All data is staged with cudaMemcpy per
// call (grids are flat contiguous arrays). The kernels are deterministic:
// flood fill is order independent, the extrapolation passes only write cells
// whose values are never read within the same pass.
//
// After the first CUDA error the path latches off for the session with a
// single log line and all callers fall back to the CPU implementations.

#include <climits>
#include <cstdio>
#include <cstdlib>

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

// Run a CUDA call, latch off and bail out on error. The reported error is
// the actual return code of this call (sticky async kernel errors surface
// here at the next synchronizing call).
#define CUDA_SYNC_CHECK(call, where) \
  do { \
    cudaError_t _rc = (call); \
    if (_rc != cudaSuccess) { \
      cudaFail(where, _rc); \
      return false; \
    } \
  } while (0)

// Persistent device buffers (grow to the largest request, never shrink).
// Plugin calls are serialized by the Python GIL (verified: no
// Py_BEGIN_ALLOW_THREADS in the mantaflow wrapper layer), so no locking.
// NOTE: `phi` is sized in floats and holds a scalar grid (n) or a MAC grid
// (3 floats per cell) depending on the operation.
struct DevBuffers {
  float *phi = nullptr;
  int *ibuf = nullptr;
  int *flagsBuf = nullptr;
  int *front[2] = {nullptr, nullptr};
  int *fcount = nullptr;
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
    ok = ok && cudaMalloc(&front[0], needInts * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&front[1], needInts * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&fcount, sizeof(int)) == cudaSuccess;
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
    cudaFree(front[0]);
    cudaFree(front[1]);
    cudaFree(fcount);
    phi = nullptr;
    ibuf = nullptr;
    flagsBuf = nullptr;
    front[0] = front[1] = nullptr;
    fcount = nullptr;
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

// ---------------------------------------------------------------------------
// flood fill

// Seed: wall ring cells of the inner [bw, size-bw) box with matching sign.
__global__ void seedWalls(const float *__restrict__ phi,
                          int *__restrict__ visited,
                          int *__restrict__ front,
                          int *__restrict__ fcount,
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
    if (i < bw || j < bw || k < bw || i >= sx - bw || j >= sy - bw || k >= sz - bw) {
      continue;
    }
    bool wall = (i == bw || i == sx - 1 - bw) || (j == bw || j == sy - 1 - bw) ||
                (sz > 1 && (k == bw || k == sz - 1 - bw));
    if (!wall || !signOkFill(phi[t], outside)) {
      continue;
    }
    if (atomicCAS(&visited[t], 0, 1) == 0) {
      long p = atomicAdd(fcount, 1);
      if (p < n) {
        front[p] = (int)t;
      }
    }
  }
}

// Expand one BFS frontier layer.
__global__ void expand(const float *__restrict__ phi,
                       int *__restrict__ visited,
                       const int *__restrict__ front,
                       int fsize,
                       int *__restrict__ nfront,
                       int *__restrict__ ncount,
                       int sx,
                       int sy,
                       int sz,
                       long strideZ,
                       int bw,
                       bool outside)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= fsize) {
    return;
  }
  long c = front[t];
  int i, j, k;
  coords(c, sx, strideZ, &i, &j, &k);
  long n = (long)sx * sy * sz;

  auto tryNb = [&](long idx, bool ok) {
    if (ok && !visited[idx] && signOkFill(phi[idx], outside)) {
      if (atomicCAS(&visited[idx], 0, 1) == 0) {
        long p = atomicAdd(ncount, 1);
        if (p < n) {
          nfront[p] = (int)idx;
        }
      }
    }
  };
  tryNb(c - 1, i > bw);
  tryNb(c + 1, i < sx - 1 - bw);
  tryNb(c - sx, j > bw);
  tryNb(c + sx, j < sy - 1 - bw);
  if (sz > 1) {
    tryNb(c - strideZ, k > bw);
    tryNb(c + strideZ, k < sz - 1 - bw);
  }
}

// Same as KnFillApply: overwrite unvisited cells with matching sign.
__global__ void applyFill(float *__restrict__ phi,
                          const int *__restrict__ visited,
                          long n,
                          float value,
                          bool outside)
{
  for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
       t += (long)gridDim.x * blockDim.x) {
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
    if (i < 1 || j < 1 || k < 1 || i >= sx - 1 || j >= sy - 1 || k >= sz - 1) {
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
    if (i < 1 || j < 1 || k < 1 || i >= sx - 1 || j >= sy - 1 || k >= sz - 1) {
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
    if (i < 1 || j < 1 || k < 1 || i >= sx - 1 || j >= sy - 1 || k >= sz - 1) {
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
    if (i < 1 || j < 1 || k < 1 || i >= sx - 1 || j >= sy - 1 || k >= sz - 1) {
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
    if (i < 1 || j < 1 || k < 1 || i >= sx - 1 || j >= sy - 1 || k >= sz - 1) {
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
    if (i < 1 || j < 1 || k < 1 || i >= sx - 1 || j >= sy - 1 || k >= sz - 1) {
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

  if (!g_dev.ensure(n, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  CUDA_SYNC_CHECK(cudaMemcpy(buf.phi, &phi(0), n * sizeof(float), cudaMemcpyHostToDevice),
                  "floodFill: copy in");
  CUDA_SYNC_CHECK(cudaMemset(buf.ibuf, 0, n * sizeof(int)), "floodFill: memset visited");
  CUDA_SYNC_CHECK(cudaMemset(buf.fcount, 0, sizeof(int)), "floodFill: memset count");

  seedWalls<<<blocksFor(n), 256>>>(
      buf.phi, buf.ibuf, buf.front[0], buf.fcount, sx, sy, sz, strideZ, boundaryWidth, outside);

  bool exhausted = false;
  int cur = 0;
  for (long iter = 0; iter < n + 8; iter++) {
    int hCount = 0;
    CUDA_SYNC_CHECK(cudaMemcpy(&hCount, buf.fcount, sizeof(int), cudaMemcpyDeviceToHost),
                    "floodFill: count readback");
    if (hCount == 0) {
      break;
    }
    if (hCount > (int)n || iter == n + 7) {
      exhausted = true;  // safety net, should not happen
      break;
    }
    cudaMemset(buf.fcount, 0, sizeof(int));
    expand<<<blocksFor(hCount), 256>>>(buf.phi,
                                       buf.ibuf,
                                       buf.front[cur],
                                       hCount,
                                       buf.front[cur ^ 1],
                                       buf.fcount,
                                       sx,
                                       sy,
                                       sz,
                                       strideZ,
                                       boundaryWidth,
                                       outside);
    cur ^= 1;
  }
  if (exhausted) {
    cudaFail("floodFill: BFS exhausted (bug)", cudaErrorUnknown);
    return false;
  }

  applyFill<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, n, (float)value, outside);
  CUDA_SYNC_CHECK(cudaMemcpy(&phi(0), buf.phi, n * sizeof(float), cudaMemcpyDeviceToHost),
                  "floodFill: copy out");
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

  if (!g_dev.ensure(n, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  CUDA_SYNC_CHECK(cudaMemcpy(buf.phi, &phi(0), n * sizeof(float), cudaMemcpyHostToDevice),
                  "extrapolateLs: copy in");
  CUDA_SYNC_CHECK(cudaMemset(buf.ibuf, 0, n * sizeof(int)), "extrapolateLs: memset tmp");

  const float direction = inside ? -1.0f : 1.0f;
  lsMarkSkip<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, inside);
  lsFirstLayer<<<blocksFor(n), 256>>>(buf.ibuf, sx, sy, sz, strideZ, dim);
  for (int d = 2; d < 1 + distance; ++d) {
    lsDilate<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, dim, d, direction);
  }
  lsRemaining<<<blocksFor(n), 256>>>(
      buf.phi, buf.ibuf, sx, sy, sz, strideZ, direction * (distance + 2));

  CUDA_SYNC_CHECK(cudaMemcpy(&phi(0), buf.phi, n * sizeof(float), cudaMemcpyDeviceToHost),
                  "extrapolateLs: copy out");
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

  if (!g_dev.ensure(velFloats, n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  CUDA_SYNC_CHECK(cudaMemcpy(buf.phi, &vel(0), velFloats * sizeof(float), cudaMemcpyHostToDevice),
                  "extrapolateMAC: copy in");
  CUDA_SYNC_CHECK(cudaMemcpy(buf.flagsBuf, &flags(0), n * sizeof(int), cudaMemcpyHostToDevice),
                  "extrapolateMAC: flags copy");

  for (int c = 0; c < dim; ++c) {
    CUDA_SYNC_CHECK(cudaMemset(buf.ibuf, 0, n * sizeof(int)), "extrapolateMAC: memset tmp");
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

  CUDA_SYNC_CHECK(cudaMemcpy(&vel(0), buf.phi, velFloats * sizeof(float), cudaMemcpyDeviceToHost),
                  "extrapolateMAC: copy out");
  return true;
}

}  // namespace Manta
