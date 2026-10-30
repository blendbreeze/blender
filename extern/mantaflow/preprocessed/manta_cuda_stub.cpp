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

bool cudaAdvectSemiLagrangeScalar(const FlagGrid & /*flags*/,
                                  MACGrid & /*vel*/,
                                  Grid<Real> & /*grid*/,
                                  Real /*dt*/)
{
  return false;
}

bool cudaSolvePressureCg(Grid<Real> & /*rhs*/,
                         Grid<Real> & /*pressure*/,
                         FlagGrid & /*flags*/,
                         Real /*cgAccuracy*/,
                         MACGrid * /*fractions*/,
                         Real /*cgMaxIterFac*/)
{
  return false;
}

bool cudaImprovedParticleLevelset(const void * /*partsData*/,
                                  long /*nParts*/,
                                  Grid<Real> & /*phi*/,
                                  Real /*radiusFactor*/,
                                  int /*smoothen*/,
                                  int /*smoothenNeg*/,
                                  Real /*t_low*/,
                                  Real /*t_high*/)
{
  return false;
}

bool cudaUnionParticleLevelset(const void * /*partsData*/,
                               long /*nParts*/,
                               Grid<Real> & /*phi*/,
                               Real /*radiusFactor*/)
{
  return false;
}

bool cudaMarchingCubes(Grid<Real> & /*phi*/,
                       std::vector<float> & /*verts*/,
                       std::vector<float> & /*normals*/,
                       std::vector<int> & /*tris*/)
{
  return false;
}

bool cudaMapPartsToMAC(const void * /*partsData*/,
                       long /*nParts*/,
                       const float * /*pvelData*/,
                       MACGrid & /*vel*/,
                       Grid<Vec3> & /*weight*/)
{
  return false;
}

bool cudaFlipVelocityUpdate(const void * /*partsData*/,
                            long /*nParts*/,
                            float * /*pvelData*/,
                            MACGrid & /*vel*/,
                            MACGrid & /*velOld*/,
                            Real /*flipRatio*/)
{
  return false;
}

bool cudaExtrapolateMACFromWeight(MACGrid & /*vel*/, Grid<Vec3> & /*weight*/, int /*distance*/)
{
  return false;
}

bool cudaExtrapolateVec3Simple(Grid<Vec3> & /*vel*/,
                               Grid<Real> & /*phi*/,
                               int /*distance*/,
                               bool /*inside*/)
{
  return false;
}

}  // namespace Manta
