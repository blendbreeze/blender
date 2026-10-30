// SPDX-License-Identifier: Apache-2.0
// Experimental CUDA acceleration for select mantaflow kernels.
// Enabled at runtime with MANTA_CUDA=1 in the environment; every entry point
// returns false when the CUDA path is unavailable so callers fall back to the
// CPU implementation.

#ifndef _MANTA_CUDA_H
#define _MANTA_CUDA_H

#include <vector>

#include "grid.h"

namespace Manta {

namespace manta_cuda {
bool enabled();
}

/* LevelsetGrid::floodFill - fill enclosed holes of a levelset. */
bool cudaFloodFill(Grid<Real> &phi, Real value, bool outside, int boundaryWidth);

/* extrapolateLsSimple - reinitialize a narrow band around a levelset. */
bool cudaExtrapolateLsSimple(Grid<Real> &phi, int distance, bool inside);

/* extrapolateMACSimple - all axis passes of the mark + dilation loops. */
bool cudaExtrapolateMACSimple(FlagGrid &flags, MACGrid &vel, int distance, bool intoObs, int dim);

/* advectSemiLagrange for a scalar grid, order 1 / linear / first-order trace. */
bool cudaAdvectSemiLagrangeScalar(const FlagGrid &flags, MACGrid &vel, Grid<Real> &grid, Real dt);

/* Pressure solve: unpreconditioned CG on the GPU (same system / accuracy as
 * the CPU solve). RHS build and velocity correction stay on the CPU. */
bool cudaSolvePressureCg(Grid<Real> &rhs,
                         Grid<Real> &pressure,
                         FlagGrid &flags,
                         Real cgAccuracy,
                         MACGrid *fractions,
                         Real cgMaxIterFac);

/* improvedParticleLevelset / unionParticleLevelset: counting-sorted particle
 * index plus the particle levelset on the GPU. `partsData` is the raw
 * BasicParticleData array (pos.xyz + flag, 16 bytes per particle); callers
 * must route ptype/exclude variants to the CPU path. */
bool cudaImprovedParticleLevelset(const void *partsData,
                                  long nParts,
                                  Grid<Real> &phi,
                                  Real radiusFactor,
                                  int smoothen,
                                  int smoothenNeg,
                                  Real t_low,
                                  Real t_high);
bool cudaUnionParticleLevelset(const void *partsData,
                               long nParts,
                               Grid<Real> &phi,
                               Real radiusFactor);

/* LevelsetGrid::createMesh - marching cubes on the GPU (same LUTs and
 * classification; interpolated vertex positions bit-identical, vertex array
 * order differs). Outputs interleaved xyz positions / normals (3*V floats)
 * and triangle indices (3*T ints); the caller builds the Mesh. */
bool cudaMarchingCubes(Grid<Real> &phi,
                       std::vector<float> &verts,
                       std::vector<float> &normals,
                       std::vector<int> &tris);

/* mapPartsToMAC - particle->MAC velocity scatter (serial on CPU) plus the
 * weight stomp / safeDivide normalization. `partsData` is the raw
 * BasicParticleData array, `pvelData` the contiguous Vec3 particle velocity
 * array (3 floats per particle). Both vel and weight are written back;
 * callers must route ptype/exclude variants to the CPU path and do the
 * velOld copy themselves. */
bool cudaMapPartsToMAC(const void *partsData,
                       long nParts,
                       const float *pvelData,
                       MACGrid &vel,
                       Grid<Vec3> &weight);

/* flipVelocityUpdate - per-particle FLIP blend; pvelData is updated in place. */
bool cudaFlipVelocityUpdate(const void *partsData,
                            long nParts,
                            float *pvelData,
                            MACGrid &vel,
                            MACGrid &velOld,
                            Real flipRatio);

/* extrapolateMACFromWeight - weight-driven MAC extrapolation (destroys the
 * weight values, like the CPU version). */
bool cudaExtrapolateMACFromWeight(MACGrid &vel, Grid<Vec3> &weight, int distance);

/* extrapolateVec3Simple - vec3 variant of extrapolateLsSimple. */
bool cudaExtrapolateVec3Simple(Grid<Vec3> &vel, Grid<Real> &phi, int distance, bool inside);

}  // namespace Manta

#endif /* _MANTA_CUDA_H */
