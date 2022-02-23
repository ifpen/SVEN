from numba import jit, njit, prange
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

@jit(nopython=True, fastmath=True)
def biotSavartFilaments(evaluationPoint, leftNodes, rightNodes, circulations, deltaFlts):
    inducedVelocity = np.zeros(3)
    #
    curr_px = evaluationPoint[0];
    curr_py = evaluationPoint[1];
    curr_pz = evaluationPoint[2];
    #
    uix = 0;
    uiy = 0;
    uiz = 0;
    #
    numFilaments = len(circulations)
    # Loop over all source filaments
    for k in range(numFilaments):
        #
        curr_fp1x = leftNodes[k][0]
        curr_fp1y = leftNodes[k][1]
        curr_fp1z = leftNodes[k][2]
        curr_fp2x = rightNodes[k][0]
        curr_fp2y = rightNodes[k][1]
        curr_fp2z = rightNodes[k][2]
        curr_fstr = circulations[k]
        curr_flen = np.linalg.norm(rightNodes[k] - leftNodes[k])
        #
        pxx1 = curr_px - curr_fp1x;
        pyy1 = curr_py - curr_fp1y;
        pzz1 = curr_pz - curr_fp1z;
        #
        pxx2 = curr_px - curr_fp2x;
        pyy2 = curr_py - curr_fp2y;
        pzz2 = curr_pz - curr_fp2z;
        #
        r1 = np.sqrt((pxx1 * pxx1 + pyy1 * pyy1 + pzz1 * pzz1));
        r2 = np.sqrt((pxx2 * pxx2 + pyy2 * pyy2 + pzz2 * pzz2));
        #
        r1dr2 = pxx1 * pxx2 + pyy1 * pyy2 + pzz1 * pzz2;
        r1tr2 = r1 * r2;
        #
        fd = curr_flen * deltaFlts;
        den = (r1tr2 * (r1tr2 + r1dr2) + fd * fd);
        #
        ubar = curr_fstr / (4. * np.pi) * (r1 + r2) / den;
        #
        uix += ubar * (pyy1 * pzz2 - pzz1 * pyy2);
        uiy += ubar * (pzz1 * pxx2 - pxx1 * pzz2);
        uiz += ubar * (pxx1 * pyy2 - pyy1 * pxx2);
        #
    inducedVelocity[0] = uix;
    inducedVelocity[1] = uiy;
    inducedVelocity[2] = uiz;

    return inducedVelocity

#@jit(nopython=True, fastmath=True)
def biotSavartFilaments_v2(evaluationPoints, leftNodes, rightNodes, circulations, deltaFlts):
    #
    inducedVelocities = np.zeros((len(evaluationPoints),3))
    for i in range(len(evaluationPoints)):
        #
        inducedVelocity = np.zeros(3)
        #
        # numFilaments = len(circulations)

        # curr_flen = np.linalg.norm(rightNodes - leftNodes)

        p1 = evaluationPoints[i] - leftNodes
        p2 = evaluationPoints[i] - rightNodes

        ubar = circulations / (4. * np.pi)  # * (r1 + r2) / den;
        cross = np.cross(p1,p2, axisa=1, axisb=1)
        inducedVelocity[0] = np.sum(ubar * cross[:,0])
        inducedVelocity[1] = np.sum(ubar * cross[:,1])
        inducedVelocity[2] = np.sum(ubar * cross[:,2])

        inducedVelocities[i,:] = inducedVelocity

    return inducedVelocities

@njit(fastmath=True) #parallel=True)
def biotSavartFilaments_v3(evaluationPoints, leftNodes, rightNodes, circulations):
    #
    inducedVelocities = np.zeros((len(evaluationPoints), 3))
    inducedVelocity = np.zeros(3)
    #
    cross = np.zeros((len(circulations), 3))
    lengths = np.zeros(len(circulations))

    for k in range(len(circulations)):
        lengths[k] = np.linalg.norm(rightNodes[k] - leftNodes[k])

    r1 = np.zeros(len(circulations))
    r2 = np.zeros(len(circulations))
    r1dr2 = np.zeros(len(circulations))
    #
    for i in range(evaluationPoints.shape[0]):
        #
        p1 = evaluationPoints[i] - leftNodes
        p2 = evaluationPoints[i] - rightNodes
        #
        # Numba does not support aaxis and baxis
        for k in range(len(circulations)):
            cross[k,:] = np.cross(p1[k],p2[k])
            r1[k] = np.sqrt(np.sum(p1[k]*p1[k]))
            r2[k] = np.sqrt(np.sum(p2[k]*p2[k]))
            r1dr2[k] = np.sum(p1[k]*p2[k])

        r1tr2 = r1*r2
        fd = lengths*lengths*lengths*lengths
        den = (r1tr2 * (r1tr2 + r1dr2) + fd );
        #
        ubar = circulations / (4. * np.pi) * (r1 + r2) / den;

        inducedVelocity[0] = np.sum(ubar * cross[:,0])
        inducedVelocity[1] = np.sum(ubar * cross[:,1])
        inducedVelocity[2] = np.sum(ubar * cross[:,2])

        inducedVelocities[i,:] = inducedVelocity

    return inducedVelocities

@jit(nopython=True, fastmath=True)
def biotSavartFilaments_v4(evaluationPoints, leftNodes, rightNodes, circulations, deltaFlts):
    #
    inducedVelocities = np.zeros((len(evaluationPoints),3))
    #
    fd = np.zeros(len(circulations))
    for k in range(len(circulations)):
        fd[k] = np.linalg.norm(rightNodes[k] - leftNodes[k])
    fd = fd * deltaFlts
    #
    for i in range(len(evaluationPoints)):
        #
        evaluationPoint = evaluationPoints[i]
        curr_px = evaluationPoint[0];
        curr_py = evaluationPoint[1];
        curr_pz = evaluationPoint[2];
        #
        numFilaments = len(circulations)

        # Loop over all source filaments
        for k in range(numFilaments):
            #
            curr_fp1x = leftNodes[k][0]
            curr_fp1y = leftNodes[k][1]
            curr_fp1z = leftNodes[k][2]
            curr_fp2x = rightNodes[k][0]
            curr_fp2y = rightNodes[k][1]
            curr_fp2z = rightNodes[k][2]
            #
            pxx1 = curr_px - curr_fp1x;
            pyy1 = curr_py - curr_fp1y;
            pzz1 = curr_pz - curr_fp1z;
            #
            pxx2 = curr_px - curr_fp2x;
            pyy2 = curr_py - curr_fp2y;
            pzz2 = curr_pz - curr_fp2z;
            #
            r1 = np.sqrt((pxx1 * pxx1 + pyy1 * pyy1 + pzz1 * pzz1));
            r2 = np.sqrt((pxx2 * pxx2 + pyy2 * pyy2 + pzz2 * pzz2));
            #
            r1dr2 = pxx1 * pxx2 + pyy1 * pyy2 + pzz1 * pzz2;
            r1tr2 = r1 * r2;
            #
            den = (r1tr2 * (r1tr2 + r1dr2) + fd[k] * fd[k]);
            #
            ubar = circulations[k] * (r1 + r2) / den;
            #
            inducedVelocities[i,0] += ubar * (pyy1 * pzz2 - pzz1 * pyy2);
            inducedVelocities[i,1] += ubar * (pzz1 * pxx2 - pxx1 * pzz2);
            inducedVelocities[i,2] += ubar * (pxx1 * pyy2 - pyy1 * pxx2);
            #
    return inducedVelocities / (4. * np.pi)

@jit(nopython=True, fastmath=True)
def biotSavartFilaments_2D(evaluationPoints, leftNodes, rightNodes, circulations, deltaFlts):
    #
    inducedVelocities = np.zeros(np.shape(evaluationPoints))
    #
    shape = np.shape(evaluationPoints)
    for i in range(shape[0]):
        for j in range(shape[1]):
            #
            evaluationPoint = evaluationPoints[i,j]
            curr_px = evaluationPoint[0];
            curr_py = evaluationPoint[1];
            curr_pz = evaluationPoint[2];
            #
            # Loop over all source filaments
            shapeWake = np.shape(rightNodes)
            for k in range(shapeWake[0]):
                for l in range(shapeWake[1]):
                    #
                    curr_fp1x = leftNodes[k,l][0]
                    curr_fp1y = leftNodes[k,l][1]
                    curr_fp1z = leftNodes[k,l][2]
                    curr_fp2x = rightNodes[k,l][0]
                    curr_fp2y = rightNodes[k,l][1]
                    curr_fp2z = rightNodes[k,l][2]
                    #
                    pxx1 = curr_px - curr_fp1x;
                    pyy1 = curr_py - curr_fp1y;
                    pzz1 = curr_pz - curr_fp1z;
                    #
                    pxx2 = curr_px - curr_fp2x;
                    pyy2 = curr_py - curr_fp2y;
                    pzz2 = curr_pz - curr_fp2z;
                    #
                    r1 = np.sqrt((pxx1 * pxx1 + pyy1 * pyy1 + pzz1 * pzz1));
                    r2 = np.sqrt((pxx2 * pxx2 + pyy2 * pyy2 + pzz2 * pzz2));
                    #
                    r1dr2 = pxx1 * pxx2 + pyy1 * pyy2 + pzz1 * pzz2;
                    r1tr2 = r1 * r2;
                    #
                    fd = deltaFlts * np.linalg.norm(rightNodes[k,l] - leftNodes[k,l])
                    den = (r1tr2 * (r1tr2 + r1dr2) + fd * fd);
                    #
                    # if(np.abs(den) < 1e-3):
                    #     print('den: ', den, r1tr2, leftNodes[k,l], rightNodes[k,l])
                    ubar = circulations[k,l] * (r1 + r2) / den;
                    #
                    inducedVelocities[i,j,0] += ubar * (pyy1 * pzz2 - pzz1 * pyy2);
                    inducedVelocities[i,j,1] += ubar * (pzz1 * pxx2 - pxx1 * pzz2);
                    inducedVelocities[i,j,2] += ubar * (pxx1 * pyy2 - pyy1 * pxx2);
                        #
    return inducedVelocities / (4. * np.pi)

# @jit(nopython=True, fastmath=True)
# def biotSavartFilaments(evaluationPoint, leftNodes, rightNodes, circulations, deltaFlts):
#     inducedVelocity = np.zeros(3)
#     #
#     curr_px = evaluationPoint[0];
#     curr_py = evaluationPoint[1];
#     curr_pz = evaluationPoint[2];
#     #
#     uix = 0;
#     uiy = 0;
#     uiz = 0;
#     #
#     numFilaments = len(circulations)
#     # Loop over all source filaments
#     for k in range(numFilaments):
#         #
#         curr_fp1x = leftNodes[k][0]
#         curr_fp1y = leftNodes[k][1]
#         curr_fp1z = leftNodes[k][2]
#         curr_fp2x = rightNodes[k][0]
#         curr_fp2y = rightNodes[k][1]
#         curr_fp2z = rightNodes[k][2]
#         curr_fstr = circulations[k]
#         curr_flen = np.linalg.norm(rightNodes[k] - leftNodes[k])
#         #
#         pxx1 = curr_px - curr_fp1x;
#         pyy1 = curr_py - curr_fp1y;
#         pzz1 = curr_pz - curr_fp1z;
#         #
#         pxx2 = curr_px - curr_fp2x;
#         pyy2 = curr_py - curr_fp2y;
#         pzz2 = curr_pz - curr_fp2z;
#         #
#         r1 = np.sqrt((pxx1 * pxx1 + pyy1 * pyy1 + pzz1 * pzz1));
#         r2 = np.sqrt((pxx2 * pxx2 + pyy2 * pyy2 + pzz2 * pzz2));
#         #
#         r1dr2 = pxx1 * pxx2 + pyy1 * pyy2 + pzz1 * pzz2;
#         r1tr2 = r1 * r2;
#         #
#         fd = curr_flen * deltaFlts;
#         den = (r1tr2 * (r1tr2 + r1dr2) + fd * fd);
#         #
#         ubar = curr_fstr / (4. * np.pi) * (r1 + r2) / den;
#         #
#         uix += ubar * (pyy1 * pzz2 - pzz1 * pyy2);
#         uiy += ubar * (pzz1 * pxx2 - pxx1 * pzz2);
#         uiz += ubar * (pxx1 * pyy2 - pyy1 * pxx2);
#         #
#         inducedVelocity[0] = uix;
#         inducedVelocity[1] = uiy;
#         inducedVelocity[2] = uiz;
#
#     return inducedVelocity


modPtcles = SourceModule("""
__global__ void particlesOnBladesKernel(float *destUx, float *destUy, float *destUz, float *bladeNodeX, float *bladeNodeY, 
                           float *bladeNodeZ, float *positionX, float *positionY, 
                           float *positionZ, float *vorticityX, float *vorticityY, float *vorticityZ, 
                           float *particleRadius, float deltaPtcles, int numBladePoints, int numParticles)
{
  //Get thread's global index
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  //const int idx = threadIdx.x;
  //float PI = 3.14159265358979323846;
  //float   fPI  = 4. * PI;
  //int numParticles = inputParticles[0];
  // Check point index
  if(idx<numParticles){
  //if(idx<numBladePoints){
    float curr_px  , curr_py  , curr_pz;
    float curr_xp_x, curr_xp_y, curr_xp_z;
    float curr_vp_x, curr_vp_y, curr_vp_z;
    float radius;
    // global  mem fetches coalesced acces?
    curr_xp_x = positionX[idx];
    curr_xp_y = positionY[idx];
    curr_xp_z = positionZ[idx];
    curr_vp_x = vorticityX[idx];
    curr_vp_y = vorticityY[idx];
    curr_vp_z = vorticityZ[idx];
    radius    = particleRadius[idx];
    // Loop over all source particles
    for(int k=0; k < numBladePoints; k++)
    {
        curr_px = bladeNodeX[k];
        curr_py = bladeNodeY[k];
        curr_pz = bladeNodeZ[k];
        float norm			= 0.;
        float x_min_xp_x	= curr_px - curr_xp_x;
        float x_min_xp_y	= curr_py - curr_xp_y;
        float x_min_xp_z	= curr_pz - curr_xp_z;
        norm       	= sqrt(x_min_xp_x*x_min_xp_x + x_min_xp_y*x_min_xp_y + x_min_xp_z*x_min_xp_z);
        // Epsilon for particles regularization
        //float delta = deltaPtcles; //1e-4;
        float epsilon = radius * deltaPtcles;
        // Rosenhead regularisation - G. Pinon thesis
        float d = norm*norm + epsilon*epsilon;
        float cst  = 1. / (d*sqrt(d));
        float numer_x		= cst * x_min_xp_x;
        float numer_y		= cst * x_min_xp_y;
        float numer_z		= cst * x_min_xp_z;
        float cross_x		= numer_y*curr_vp_z - numer_z*curr_vp_y;
        float cross_y		= numer_z*curr_vp_x - numer_x*curr_vp_z;
        float cross_z		= numer_x*curr_vp_y - numer_y*curr_vp_x;
        //destUx[k] -= cross_x / fPI;
        //destUy[k] -= cross_y / fPI;
        //destUz[k] -= cross_z / fPI;
        atomicAdd(&destUx[k], -cross_x );
        atomicAdd(&destUy[k], -cross_y );
        atomicAdd(&destUz[k], -cross_z ); 
    }
  }
}
""")

modFlts = SourceModule("""
__global__ void bladeOnParticlesKernel(float *destUx, float *destUy, float *destUz, float *ptclePosX, float *ptclePosY, 
                           float *ptclePosZ, float *fltsLeftX, float *fltsLeftY, float *fltsLeftZ,
                           float *fltsRightX, float *fltsRightY, float *fltsRightZ, float *fltsCirculations, 
                           int numParticles, int numFilaments, float deltaFlts)
{
  //Get thread's global index
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  //const int idx = threadIdx.x;
  float PI = 3.14159265358979323846;
  float   fPI  = 4. * PI;
  // Check point index
  if(idx<numParticles){
    float curr_px  , curr_py  , curr_pz;
    //
    curr_px = ptclePosX[idx];
    curr_py = ptclePosY[idx];
    curr_pz = ptclePosZ[idx];
    //
    //Loop over all source filaments
    float uix = 0;
    float uiy = 0;
    float uiz = 0;
    // Loop over all source particles
    for(int k=0; k < numFilaments; k++)
    {
        float  pxx1  = curr_px  - fltsLeftX[k];
        float  pyy1  = curr_py  - fltsLeftY[k];
        float  pzz1  = curr_pz  - fltsLeftZ[k];
        //
        float  pxx2  = curr_px - fltsRightX[k];
        float  pyy2  = curr_py - fltsRightY[k];
        float  pzz2  = curr_pz - fltsRightZ[k];
        float  fstr  = fltsCirculations[k];
        float  flen  = sqrt((fltsRightX[k]-fltsLeftX[k])*(fltsRightX[k]-fltsLeftX[k])+
                          (fltsRightY[k]-fltsLeftY[k])*(fltsRightY[k]-fltsLeftY[k])+
                          (fltsRightZ[k]-fltsLeftZ[k])*(fltsRightZ[k]-fltsLeftZ[k])
                          );
        //
        float    r1  = sqrt((pxx1*pxx1 + pyy1*pyy1 + pzz1*pzz1));
        float    r2  = sqrt((pxx2*pxx2 + pyy2*pyy2 + pzz2*pzz2));
        //
        float r1dr2  = pxx1*pxx2 + pyy1*pyy2 + pzz1*pzz2;
        float r1tr2  = r1*r2;
        //
        //float delta = 0.5;
        float   fd = flen*deltaFlts;
        float   den  = r1tr2*(r1tr2 + r1dr2) + fd*fd;
        //
        float ubar = 0.;
        if(abs(den) > 1e-8){
            ubar  = fstr / (4. * PI)*(r1 + r2) / den;
        }
        destUx[idx] += ubar * (pyy1*pzz2-pzz1*pyy2);
        destUy[idx] += ubar * (pzz1*pxx2-pxx1*pzz2);
        destUz[idx] += ubar * (pxx1*pyy2-pyy1*pxx2);
    }
  }
}
""")

mod = SourceModule("""
__global__ void particlesOnParticlesKernel(float *destUx, float *destUy, float *destUz, float *positionX, float *positionY, 
                           float *positionZ, float *vorticityX, float *vorticityY, float *vorticityZ, 
                           float *particleRadius, float deltaPtcles, int numParticles)
{
  //Get thread's global index
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Check point index
  if(idx<numParticles){

    float curr_px  , curr_py  , curr_pz;
    float curr_xp_x, curr_xp_y, curr_xp_z;
    float curr_vp_x, curr_vp_y, curr_vp_z;
    float radius;
    // global  mem fetches coalesced acces?
    curr_px = positionX[idx];
    curr_py = positionY[idx];
    curr_pz = positionZ[idx];
    // Loop over all source particles
    for(int k=0; k < numParticles; k++)
    {
          //global  mem fetches coalesced acces?
          curr_xp_x = positionX[k];
          curr_xp_y = positionY[k];
          curr_xp_z = positionZ[k];

          curr_vp_x = vorticityX[k];
          curr_vp_y = vorticityY[k];
          curr_vp_z = vorticityZ[k];

          radius    = particleRadius[k];

          float norm			= 0.;

          float x_min_xp_x	= curr_px - curr_xp_x;
          float x_min_xp_y	= curr_py - curr_xp_y;
          float x_min_xp_z	= curr_pz - curr_xp_z;

          norm       	= sqrt(x_min_xp_x*x_min_xp_x + x_min_xp_y*x_min_xp_y + x_min_xp_z*x_min_xp_z);

          // Epsilon for particles regularization
          float epsilon = radius * deltaPtcles;
          // Rosenhead regularisation - G. Pinon thesis
          float d = norm*norm + epsilon*epsilon;
          float cst  = 1. / (d*sqrt(d));

          //float numer_x		= cst * x_min_xp_x;
          //float numer_y		= cst * x_min_xp_y;
          //float numer_z		= cst * x_min_xp_z;

          //float cross_x		= x_min_xp_y*curr_vp_z - x_min_xp_z*curr_vp_y;
          //float cross_y		= x_min_xp_z*curr_vp_x - x_min_xp_x*curr_vp_z;
          //float cross_z		= x_min_xp_x*curr_vp_y - x_min_xp_y*curr_vp_x;

          destUx[idx] -= cst * (x_min_xp_y*curr_vp_z - x_min_xp_z*curr_vp_y);
          destUy[idx] -= cst * (x_min_xp_z*curr_vp_x - x_min_xp_x*curr_vp_z);
          destUz[idx] -= cst * (x_min_xp_x*curr_vp_y - x_min_xp_y*curr_vp_x);
      //}
    }
  }
}
""")
