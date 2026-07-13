/*             Parallel Sparse BLAS   GPU plugin  */
/*   (C) Copyright 2013 */

/*                      Salvatore Filippone */
/*                      Alessandro Fanfarillo */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */
/*   1. Redistributions of source code must retain the above copyright */
/*      notice, this list of conditions and the following disclaimer. */
/*   2. Redistributions in binary form must reproduce the above copyright */
/*      notice, this list of conditions, and the following disclaimer in the */
/*      documentation and/or other materials provided with the distribution. */
/*   3. The name of the PSBLAS group or the names of its contributors may */
/*      not be used to endorse or promote products derived from this */
/*      software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED */
/* TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR */
/* PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE PSBLAS GROUP OR ITS CONTRIBUTORS */
/* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR */
/* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF */
/* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS */
/* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN */
/* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) */
/* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */
 

#include <stdio.h>
#include <complex.h>
//#include "utils.h"
//#include "common.h"
#include "dvectordev.h"

int registerMappedDouble(void  *buff, void **d_p, int n, double dummy)
{
  return registerMappedMemory(buff, d_p, ((size_t) n) * sizeof(double));
}

int checkMultiVecDeviceDouble(void* deviceMultiVec)
{
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceMultiVec;
  fprintf(stderr, "checkMultiVecDeviceInt Size: %d  Pointer %p\n", devVec->size_, devVec->v_);
  return SPGPU_SUCCESS;
}

int writeMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  int i = writeRemoteBuffer((void*) hostVec, (void*) devVec->v_, ((size_t) devVec->pitch_) * devVec->count_ * sizeof(double));
  if(i != 0)
    fprintf(stderr, "From routine : %s : %d \n", "FallocMultiVecDevice", i);

  return i;
}

int writeMultiVecDeviceDoubleR2(void* deviceVec, double* hostVec, int ld)
{
  int i = writeMultiVecDeviceDouble(deviceVec, (void *) hostVec);
  if(i != 0)
    fprintf(stderr, "From routine : %s : %d \n", "writeMultiVecDeviceDoubleR2", i);

  return i;
}

int readMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  int i = readRemoteBuffer((void*) hostVec, (void*) devVec->v_, ((size_t) devVec->pitch_) * devVec->count_ * sizeof(double));
  if(i != 0)
    fprintf(stderr, "From routine : %s : %d \n", "readMultiVecDeviceDouble", i);

  return i;
}

int readMultiVecDeviceDoubleR2(void* deviceVec, double* hostVec, int ld)
{
  int i = readMultiVecDeviceDouble(deviceVec, hostVec);
  if(i != 0)
    fprintf(stderr, "From routine : %s : %d \n", "readMultiVecDeviceDoubleR2", i);

  return i;
}

int setscalMultiVecDeviceDouble(double val, int first, int last, int indexBase, void* devMultiVecX) 
{
  spgpuHandle_t handle = psb_cudaGetHandle();
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  spgpuDsetscal(handle, first, last, indexBase, val, (double*) devVecX->v_);
  
  return SPGPU_SUCCESS;
}

int geinsMultiVecDeviceDouble(int n, void* devMultiVecIrl, void* devMultiVecVal, int dupl, int indexBase, void* devMultiVecX)
{
  double beta;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecIrl = (struct MultiVectDevice *) devMultiVecIrl;
  struct MultiVectDevice *devVecVal = (struct MultiVectDevice *) devMultiVecVal;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitch = devVecIrl->pitch_;

  if((n > devVecIrl->size_) || (n > devVecVal->size_)) 
    return SPGPU_UNSUPPORTED;

  //fprintf(stderr, "geins: %d %d  %p %p %p\n", dupl, n, devVecIrl->v_, devVecVal->v_, devVecX->v_);

  if(dupl == INS_OVERWRITE) 
    beta = 0.0;
  else if(dupl == INS_ADD) 
    beta = 1.0;
  else
    beta = 0.0;

  spgpuDscat(handle, (double*) devVecX->v_, n, (double*) devVecVal->v_, (int*) devVecIrl->v_, indexBase, beta);
  
  return SPGPU_SUCCESS;
}

int igathMultiVecDeviceDouble(void* deviceVec, int vectorId, int n, int first, void* indexes, int hfirst, void* host_values, int indexBase)
{
  int* idx = (int*) indexes;
  double* hv = (double*) host_values; 
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle = psb_cudaGetHandle();
  
  hv  = &(hv[hfirst - indexBase]);
  idx = &(idx[first - indexBase]);
  spgpuDgath(handle, hv, n, idx, indexBase, (double*) devVec->v_ + (vectorId * devVec->pitch_));
  return SPGPU_SUCCESS;
}

int igathMultiVecDeviceDoubleVecIdx(void* deviceVec, int vectorId, int n, int first, void* deviceIdx, int hfirst,  void* host_values, int indexBase)
{
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  return igathMultiVecDeviceDouble(deviceVec, vectorId, n, first, (void*) devIdx->v_, hfirst, host_values, indexBase);
}

int iscatMultiVecDeviceDouble(void* deviceVec, int vectorId, int n, int first, void *indexes, int hfirst, void* host_values, int indexBase, double beta)
{
  int* idx = (int*) indexes;
  double* hv  = (double*) host_values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle = psb_cudaGetHandle();

  idx = &(idx[first - indexBase]);
  hv  = &(hv[hfirst - indexBase]);
  spgpuDscat(handle, (double*) devVec->v_, n, hv, idx, indexBase, beta);
  return SPGPU_SUCCESS;
}

int iscatMultiVecDeviceDoubleVecIdx(void* deviceVec, int vectorId, int n, int first, void *deviceIdx, int hfirst, void* host_values, int indexBase, double beta)
{
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  return  iscatMultiVecDeviceDouble(deviceVec, vectorId, n, first, (void*) devIdx->v_, hfirst, host_values, indexBase, beta);
}

//AXPY-like operations
int axpbyMultiVecDeviceDouble1(int n, double alpha, void* devMultiVecX, double beta, void* devMultiVecY)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  int pitchY = devVecY->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;

  if(devVecX->count_ != devVecY->count_)        //case v_full
    for(int j = 0; j < devVecY->count_; ++j)
      spgpuDaxpby(handle, (double*) devVecY->v_ + (pitchY * j), n, beta, (double*) devVecY->v_ + (pitchY * j), alpha, (double*) devVecX->v_);
  else                                          //case m_full or cudaVect
    for(int j = 0; j < devVecY->count_; ++j)
      spgpuDaxpby(handle, (double*) devVecY->v_ + (pitchY * j), n, beta, (double*) devVecY->v_ + (pitchY * j), alpha, (double*) devVecX->v_ + (pitchX * j));
  
  return SPGPU_SUCCESS;
}

int axpbyMultiVecDeviceDouble2(int n, double alpha, void* devMultiVecX, int idxX, double beta, void* devMultiVecY) 
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  spgpuDaxpby(handle, (double*) devVecY->v_, n, beta, (double*) devVecY->v_, alpha, (double*) devVecX->v_ + (pitchX * idxX));

  return SPGPU_SUCCESS;
}

int axpbyMultiVecDeviceDouble3(int n, double alpha, void* devMultiVecX, double beta, void* devMultiVecY, int idxY)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchY = devVecY->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  spgpuDaxpby(handle, (double*) devVecY->v_ + (pitchY * idxY), n, beta, (double*) devVecY->v_ + (pitchY * idxY), alpha, (double*) devVecX->v_);
  
  return SPGPU_SUCCESS;
}

int axpbyMultiVecDeviceDouble4(int n, double alpha, void* devMultiVecX, int idxX, double beta, void* devMultiVecY, int idxY)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  int pitchY = devVecY->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  spgpuDaxpby(handle, (double*) devVecY->v_ + (pitchY * idxY), n, beta, (double*) devVecY->v_ + (pitchY * idxY), alpha, (double*) devVecX->v_ + (pitchX * idxX));
  
  return SPGPU_SUCCESS;
}

int axpbyMultiVecDeviceDouble5(int n, double alpha, void* devMultiVecX, double beta, void* devMultiVecY, void* devMultiVecZ)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  int pitchY = devVecY->pitch_;
  int pitchZ = devVecZ->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  if((devVecX->count_ != devVecZ->count_) || (devVecX->count_ != devVecZ->count_))
    return SPGPU_UNSUPPORTED;
  
  for(int j = 0; j < devVecZ->count_; ++j)
    spgpuDaxpby(handle, (double*) devVecZ->v_ + (pitchZ * j), n, beta, (double*) devVecY->v_ + (pitchY * j), alpha, (double*) devVecX->v_ + (pitchX * j));
  
  return SPGPU_SUCCESS;
}

int axpbyczMultiVecDeviceDouble1(int n, double alpha, void* devMultiVecX, double beta, void* devMultiVecY, double gamma, void* devMultiVecZ, int idxZ)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchZ = devVecZ->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  spgpuDaxpbycz(handle, (double*) devVecZ->v_ + (pitchZ * idxZ), n, gamma, (double*) devVecZ->v_ + (pitchZ * idxZ), beta, (double*) devVecY->v_, alpha, (double*) devVecX->v_);
  
  return SPGPU_SUCCESS;
}

int axpbyczMultiVecDeviceDouble2(int n, double alpha, void* devMultiVecX, double beta, void* devMultiVecY, int idxY, double gamma, void* devMultiVecZ, int idxZ)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchY = devVecY->pitch_;
  int pitchZ = devVecZ->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  spgpuDaxpbycz(handle, (double*) devVecZ->v_ + (pitchZ * idxZ), n, gamma, (double*) devVecZ->v_ + (pitchZ * idxZ), beta, (double*) devVecY->v_ + (pitchY * idxY), alpha, (double*) devVecX->v_);
  
  return SPGPU_SUCCESS;
}

int axpbyczMultiVecDeviceDouble3(int n, double alpha, void* devMultiVecX, int idxX, double beta, void* devMultiVecY, int idxY, double gamma, void* devMultiVecZ, int idxZ)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  int pitchY = devVecY->pitch_;
  int pitchZ = devVecZ->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  spgpuDaxpbycz(handle, (double*) devVecZ->v_ + (pitchZ * idxZ), n, gamma, (double*) devVecZ->v_ + (pitchZ * idxZ), beta, (double*) devVecY->v_ + (pitchY * idxY), alpha, (double*) devVecX->v_ + (pitchX * idxX));
  
  return SPGPU_SUCCESS;
}

int axpbyczMultiVecDeviceDouble4(int n, double alpha, void* devMultiVecX, double beta, void* devMultiVecY, double gamma, void* devMultiVecZ)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  int pitchY = devVecY->pitch_;
  int pitchZ = devVecZ->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  if((devVecX->count_ != devVecZ->count_) || (devVecX->count_ != devVecZ->count_))
    return SPGPU_UNSUPPORTED;

  for(int j = 0; j < devVecZ->count_; ++j)
    spgpuDaxpbycz(handle, (double*) devVecZ->v_ + (pitchZ * j), n, gamma, (double*) devVecZ->v_ + (pitchZ * j), beta, (double*) devVecY->v_ + (pitchY * j), alpha, (double*) devVecX->v_ + (pitchX * j));
  
  return SPGPU_SUCCESS;
}

int axpbyczMultiVecDeviceDouble5(int n, double alpha, void* devMultiVecX, int idxX, double beta, void* devMultiVecY, int idxY, double gamma, void* devMultiVecZ, int idxZ, void* devMultiVecW, int idxW)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  struct MultiVectDevice *devVecW = (struct MultiVectDevice *) devMultiVecW;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitchX = devVecX->pitch_;
  int pitchY = devVecY->pitch_;
  int pitchZ = devVecZ->pitch_;
  int pitchW = devVecW->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  
  if((devVecX->count_ != devVecW->count_) || (devVecX->count_ != devVecW->count_) || (devVecZ->count_ != devVecW->count_))
    return SPGPU_UNSUPPORTED;

  for(int j = 0; j < devVecZ->count_; ++j)
    spgpuDaxpbycz(handle, (double*) devVecW->v_ + (pitchW * j), n, gamma, (double*) devVecZ->v_ + (pitchZ * j), beta, (double*) devVecY->v_ + (pitchY * j), alpha, (double*) devVecX->v_ + (pitchX * j));
  
  return SPGPU_SUCCESS;
}

int upd_xyzMultiVecDeviceDouble(int n, double alpha, double beta, double gamma, double delta, void* devMultiVecX, void* devMultiVecY, void* devMultiVecZ)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitch = devVecY->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuDupd_xyz(handle, n, alpha, beta, gamma, delta, (double*) devVecX->v_, (double*) devVecY->v_, (double*) devVecZ->v_);
  return SPGPU_SUCCESS;
}

int xyzwMultiVecDeviceDouble(int n, double a, double b, double c, double d, double e, double f, void* devMultiVecX, void* devMultiVecY, void* devMultiVecZ, void* devMultiVecW)
{
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) devMultiVecZ;
  struct MultiVectDevice *devVecW = (struct MultiVectDevice *) devMultiVecW;
  spgpuHandle_t handle = psb_cudaGetHandle();
  int pitch = devVecY->pitch_;
  if((n > devVecY->size_) || (n > devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuDxyzw(handle, n, a, b, c, d, e, f, (double*) devVecX->v_, (double*) devVecY->v_, (double*) devVecZ->v_, (double*) devVecW->v_);
  return SPGPU_SUCCESS;
}

//Scaling and norms
int scalMultiVecDeviceDouble(double alpha, void* devMultiVecA)
{
  spgpuHandle_t handle = psb_cudaGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  // Note: inner kernel can handle aliased input/output
  spgpuDscal(handle, (double*) devVecA->v_, devVecA->pitch_, alpha, (double*) devVecA->v_);
  return SPGPU_SUCCESS;
}

int nrm2MultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{
  spgpuHandle_t handle = psb_cudaGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;

  spgpuDmnrm2(handle, y_res, n, (double*) devVecA->v_, devVecA->count_, devVecA->pitch_);
  return SPGPU_SUCCESS;
}

int amaxMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{
  spgpuHandle_t handle = psb_cudaGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;

  spgpuDmamax(handle, y_res, n, (double*) devVecA->v_, devVecA->count_, devVecA->pitch_);
  return SPGPU_SUCCESS;
}

int asumMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{
  spgpuHandle_t handle = psb_cudaGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;

  spgpuDmasum(handle, y_res, n, (double*) devVecA->v_, devVecA->count_, devVecA->pitch_);
  return SPGPU_SUCCESS;
}

//DOT-like operations
int dotMultiVecDeviceDoubleS(double* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
  spgpuHandle_t handle = psb_cudaGetHandle();

  spgpuDmdot(handle, y_res, n, (double*) devVecA->v_, (double*) devVecB->v_, devVecA->count_, devVecB->pitch_);
  return SPGPU_SUCCESS;
}

int dotMultiVecDeviceDoubleV(double* y_res, int n, int mA, void* devMultiVecA, void* devMultiVecB)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
  spgpuHandle_t handle = psb_cudaGetHandle();

  spgpuDmvdot(handle, y_res, n, (double*) devVecA->v_, (double*) devVecB->v_, devVecA->count_, devVecA->pitch_);
  return SPGPU_SUCCESS;
}

int dotMultiVecDeviceDoubleM(double* y_res, int n, int mA, int mB, void* devMultiVecA, void* devMultiVecB)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
  spgpuHandle_t handle = psb_cudaGetHandle();

  spgpuDmmdot(handle, y_res, n, (double*) devVecA->v_, (double*) devVecB->v_, devVecA->count_, devVecA->pitch_, devVecB->count_, devVecB->pitch_);
  return SPGPU_SUCCESS;
  
  // int opMask = ((devVecA->count_ > 1) << 1) | (devVecB->count_ > 1);
  // switch(opMask)
  // {
  //   case 0: /*vv*/ *y_res = spgpuDdot(handle, n, (double*) devVecA->v_, (double*) devVecB->v_);                                                                      break;
  //   case 1: /*vm*/ spgpuDmvdot(handle, y_res, n, (double*) devVecB->v_, (double*) devVecA->v_, devVecB->count_, devVecB->pitch_);                                    break;
  //   case 2: /*mv*/ spgpuDmvdot(handle, y_res, n, (double*) devVecA->v_, (double*) devVecB->v_, devVecA->count_, devVecA->pitch_);                                    break;
  //   case 3: /*mm*/ spgpuDmmdot(handle, y_res, n, (double*) devVecA->v_, (double*) devVecB->v_, devVecA->count_, devVecA->pitch_, devVecB->count_, devVecB->pitch_);  break;
  // }
}

//Elementwise operations
int axyMultiVecDeviceDouble(int n, double alpha, void *deviceVecA, void *deviceVecB)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle = psb_cudaGetHandle();
  if((n > devVecA->size_) || (n > devVecB->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuDmaxy(handle, (double*) devVecB->v_, n, alpha, (double*) devVecA->v_, (double*) devVecB->v_, devVecA->count_, devVecA->pitch_);

  return SPGPU_SUCCESS;
}

int axybzMultiVecDeviceDouble(int n, double alpha, void *deviceVecA, void *deviceVecB, double beta, void *deviceVecZ)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
  spgpuHandle_t handle = psb_cudaGetHandle();
  if((n > devVecA->size_) || (n > devVecB->size_ ) || (n > devVecZ->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuDmaxypbz(handle, (double*) devVecZ->v_, n, beta, (double*) devVecZ->v_, alpha, (double*) devVecA->v_, (double*) devVecB->v_, devVecB->count_, devVecB->pitch_);
  return SPGPU_SUCCESS;
}

int absMultiVecDeviceDouble2(int n, double alpha, void *deviceVecA, void *deviceVecB)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle = psb_cudaGetHandle();

  if((n > devVecA->size_) || (n > devVecB->size_ ))
    return SPGPU_UNSUPPORTED;

  spgpuDabs(handle, (double*) devVecB->v_, n, alpha, (double*) devVecA->v_);
  return SPGPU_SUCCESS;
}
 
int absMultiVecDeviceDouble(int n, double alpha, void *deviceVecA)
{
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  spgpuHandle_t handle = psb_cudaGetHandle();
  if(n > devVecA->size_)
    return SPGPU_UNSUPPORTED;

  spgpuDabs(handle, (double*)devVecA->v_, n, alpha, (double*)devVecA->v_);
  return SPGPU_SUCCESS;
}