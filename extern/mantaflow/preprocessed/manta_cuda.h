// SPDX-License-Identifier: Apache-2.0
// Experimental CUDA acceleration for select mantaflow kernels.
// Enabled at runtime with MANTA_CUDA=1 in the environment; every entry point
// returns false when the CUDA path is unavailable so callers fall back to the
// CPU implementation.

#ifndef _MANTA_CUDA_H
#define _MANTA_CUDA_H

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

}  // namespace Manta

#endif /* _MANTA_CUDA_H */
