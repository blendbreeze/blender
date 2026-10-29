// SPDX-License-Identifier: Apache-2.0
// CPU fallback stubs used when building without WITH_MANTA_CUDA.

#include "manta_cuda.h"

namespace Manta {

namespace manta_cuda {
bool enabled()
{
  return false;
}
}  // namespace manta_cuda

bool cudaFloodFill(Grid<Real> & /*phi*/, Real /*value*/, bool /*outside*/, int /*boundaryWidth*/)
{
  return false;
}

bool cudaExtrapolateLsSimple(Grid<Real> & /*phi*/, int /*distance*/, bool /*inside*/)
{
  return false;
}

bool cudaExtrapolateMACSimple(FlagGrid & /*flags*/,
                              MACGrid & /*vel*/,
                              int /*distance*/,
                              bool /*intoObs*/,
                              int /*dim*/)
{
  return false;
}

}  // namespace Manta
