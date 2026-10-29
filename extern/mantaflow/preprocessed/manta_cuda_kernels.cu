// SPDX-License-Identifier: Apache-2.0
// Experimental CUDA kernels for select mantaflow operations (flood fill,
// levelset extrapolation, MAC velocity extrapolation). Ported 1:1 from the
// CPU kernels in levelset.cpp / fastmarch.cpp - see PROJECT_NOTES.md.
//
// Enabled at runtime via MANTA_CUDA=1. Data is staged with cudaMemcpy per
// call (grids are flat contiguous arrays). All kernels are deterministic:
// flood fill is order independent, the extrapolation passes only write cells
// whose values are never read within the same pass.

#include <climits>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "manta_cuda.h"

namespace Manta {
namespace manta_cuda {

static bool enabledCached()
{
  static int mode = -1;
  if (mode < 0) {
    mode = 0;
    const char *env = getenv("MANTA_CUDA");
    if (env && atoi(env)) {
      int count = 0;
      cudaError_t e1 = cudaGetDeviceCount(&count);
      cudaError_t e2 = (e1 == cudaSuccess && count > 0) ? cudaSetDevice(0) : cudaErrorUnknown;
      cudaError_t e3 = (e2 == cudaSuccess) ? cudaFree(0) : cudaErrorUnknown;
      if (e1 == cudaSuccess && count > 0 && e2 == cudaSuccess && e3 == cudaSuccess) {
        mode = 1;
      }
      else {
        fprintf(stderr,
                "MANTA_CUDA: disabled (count=%d e1=%d e2=%d e3=%d: %s)\n",
                count,
                (int)e1,
                (int)e2,
                (int)e3,
                cudaGetErrorString(e1 != cudaSuccess ? e1 : (e2 != cudaSuccess ? e2 : e3)));
      }
    }
  }
  return mode == 1;
}

bool enabled()
{
  return enabledCached();
}

#define CUDA_CHECK(expr) ((cudaGetLastError() == cudaSuccess && (expr) == cudaSuccess))

// Persistent device buffer cache (grows to the largest grid seen, never
// shrinks). Plugin calls happen from the single Python solver thread, so no
// locking is needed. This avoids cudaMalloc/cudaFree per call, which costs
// more than the kernels themselves on small grids.
struct DevBuffers {
  float *phi = nullptr;   // levelset / MAC component scratch
  int *ibuf = nullptr;    // visited / tmp
  int *front[2] = {nullptr, nullptr};
  int *fcount = nullptr;
  long cap = 0;

  bool ensure(long cells)
  {
    if (cells <= cap) {
      return true;
    }
    free();
    long n = (cap > 0) ? cap : cells;
    while (n < cells) {
      n *= 2;
    }
    bool ok = cudaMalloc(&phi, n * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc(&ibuf, n * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&front[0], n * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&front[1], n * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&fcount, sizeof(int)) == cudaSuccess;
    if (!ok) {
      free();
      return false;
    }
    cap = n;
    return true;
  }
  void free()
  {
    cudaFree(phi);
    cudaFree(ibuf);
    cudaFree(front[0]);
    cudaFree(front[1]);
    cudaFree(fcount);
    phi = nullptr;
    ibuf = nullptr;
    front[0] = front[1] = nullptr;
    fcount = nullptr;
    cap = 0;
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

// TEMP debug: trace first CUDA-path decisions
static int g_dbgCount = 0;
#define MDBG(...) \
  do { \
    if (g_dbgCount < 20) { \
      fprintf(stderr, "MANTA_CUDA: " __VA_ARGS__); \
      g_dbgCount++; \
    } \
  } while (0)

}  // namespace manta_cuda

bool cudaFloodFill(Grid<Real> &phi, Real value, bool outside, int boundaryWidth)
{
  using namespace manta_cuda;
  if (!enabled()) {
    MDBG("floodFill: not enabled\n");
    return false;
  }
  const int sx = phi.getSizeX(), sy = phi.getSizeY(), sz = phi.getSizeZ();
  const long n = (long)sx * sy * sz;
  const long strideZ = (long)sx * sy;
  if (n <= 0 || n >= (long)INT_MAX) {
    MDBG("floodFill: size out of range (%ld)\n", n);
    return false;
  }
  if (sizeof(Vec3) != 3 * sizeof(Real)) {
    MDBG("floodFill: vec layout mismatch\n");
    return false;
  }

  if (!g_dev.ensure(n)) {
    MDBG("floodFill: buffer alloc failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  DevBuffers &buf = g_dev;
  bool ok = CUDA_CHECK(cudaMemcpy(buf.phi, &phi(0), n * sizeof(float), cudaMemcpyHostToDevice));
  ok = ok && CUDA_CHECK(cudaMemset(buf.ibuf, 0, n * sizeof(int)));
  ok = ok && CUDA_CHECK(cudaMemset(buf.fcount, 0, sizeof(int)));
  if (!ok) {
    MDBG("floodFill: initial transfer failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  MDBG("floodFill: entering GPU path n=%ld\n", n);

  seedWalls<<<blocksFor(n), 256>>>(
      buf.phi, buf.ibuf, buf.front[0], buf.fcount, sx, sy, sz, strideZ, boundaryWidth, outside);

  bool exhausted = false;
  int cur = 0;
  for (long iter = 0; iter < n + 8; iter++) {
    int hCount = 0;
    if (cudaMemcpy(&hCount, buf.fcount, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
      MDBG("floodFill: counter readback failed\n");
      return false;
    }
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
    MDBG("floodFill: BFS exhausted (bug)\n");
    return false;
  }

  applyFill<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, n, (float)value, outside);
  ok = CUDA_CHECK(cudaMemcpy(&phi(0), buf.phi, n * sizeof(float), cudaMemcpyDeviceToHost));
  if (!ok) {
    MDBG("floodFill: final readback failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  MDBG("floodFill: GPU path ok\n");
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

  if (!g_dev.ensure(n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  bool ok = CUDA_CHECK(cudaMemcpy(buf.phi, &phi(0), n * sizeof(float), cudaMemcpyHostToDevice));
  ok = ok && CUDA_CHECK(cudaMemset(buf.ibuf, 0, n * sizeof(int)));
  if (!ok) {
    return false;
  }

  const float direction = inside ? -1.0f : 1.0f;
  lsMarkSkip<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, inside);
  lsFirstLayer<<<blocksFor(n), 256>>>(buf.ibuf, sx, sy, sz, strideZ, dim);
  for (int d = 2; d < 1 + distance; ++d) {
    lsDilate<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, dim, d, direction);
  }
  lsRemaining<<<blocksFor(n), 256>>>(
      buf.phi, buf.ibuf, sx, sy, sz, strideZ, direction * (distance + 2));

  ok = CUDA_CHECK(cudaMemcpy(&phi(0), buf.phi, n * sizeof(float), cudaMemcpyDeviceToHost));
  return ok;
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
  if (n <= 0 || n >= (long)INT_MAX || sizeof(Vec3) != 3 * sizeof(Real)) {
    return false;
  }
  const long velBytes = 3 * n * sizeof(float);

  if (!g_dev.ensure(n)) {
    return false;
  }
  DevBuffers &buf = g_dev;
  bool ok = CUDA_CHECK(cudaMemcpy(buf.phi, &vel(0), velBytes, cudaMemcpyHostToDevice));
  if (!ok) {
    return false;
  }

  for (int c = 0; c < dim; ++c) {
    if (!CUDA_CHECK(cudaMemset(buf.ibuf, 0, n * sizeof(int)))) {
      return false;
    }
    macMark<<<blocksFor(n), 256>>>(
        (const int *)&flags(0), buf.ibuf, sx, sy, sz, strideZ, c, intoObs);
    for (int d = 1; d < 1 + distance; ++d) {
      macDilate<<<blocksFor(n), 256>>>(buf.phi, buf.ibuf, sx, sy, sz, strideZ, dim, d, c);
    }
  }

  ok = CUDA_CHECK(cudaMemcpy(&vel(0), buf.phi, velBytes, cudaMemcpyDeviceToHost));
  return ok;
}

}  // namespace Manta
