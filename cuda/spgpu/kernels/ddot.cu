/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
 *     Davide Barbieri - University of Rome Tor Vergata
 *
 */

#include "stdio.h"
#include "cudalang.h"
#include "cudadebug.h"
#include "core.h"

extern "C"
{
	#include "vector.h"
}

#define BLOCK_SIZE 512

// #define USE_CUBLAS
// #define ASSUME_LOCK_SYNC_PARALLELISM

static __device__ double ddotReductionResult[128];

__global__ void spgpuDdot_kern(int n, double* x, double* y)
{
	__shared__ double sSum[BLOCK_SIZE];

	double res = 0;
	double* lastX = x + n;

	x += threadIdx.x + blockIdx.x * BLOCK_SIZE;
	y += threadIdx.x + blockIdx.x * BLOCK_SIZE;

	int blockOffset = gridDim.x * BLOCK_SIZE;

	while (x < lastX)
    {
		res = PREC_DADD(res, PREC_DMUL(x[0], y[0]));
		
		x += blockOffset;
		y += blockOffset;
	}

	if(threadIdx.x >= 32)
		sSum[threadIdx.x] = res;

	__syncthreads();

	// Start reduction!
	if(threadIdx.x < 32) 
	{
		for(int i = 1; i < BLOCK_SIZE/32; ++i)
			res += sSum[i*32 + threadIdx.x];

	//useless (because inter-warp)
#ifndef	ASSUME_LOCK_SYNC_PARALLELISM
	}
	__syncthreads(); 

	if(threadIdx.x < 32) 
	{
#endif	

#ifdef ASSUME_LOCK_SYNC_PARALLELISM
		volatile double* vsSum = sSum;
		vsSum[threadIdx.x] = res;

		if(threadIdx.x < 16) vsSum[threadIdx.x] += vsSum[threadIdx.x + 16];
		if(threadIdx.x < 8) vsSum[threadIdx.x] += vsSum[threadIdx.x + 8];
		if(threadIdx.x < 4) vsSum[threadIdx.x] += vsSum[threadIdx.x + 4];
		if(threadIdx.x < 2) vsSum[threadIdx.x] += vsSum[threadIdx.x + 2];
		if(threadIdx.x == 0)
			ddotReductionResult[blockIdx.x] = vsSum[0] + vsSum[1];
#else
		double* vsSum = sSum;
		vsSum[threadIdx.x] = res;

		if(threadIdx.x < 16) vsSum[threadIdx.x] += vsSum[threadIdx.x + 16];
		__syncthreads();
		if(threadIdx.x < 8) vsSum[threadIdx.x] += vsSum[threadIdx.x + 8];
		__syncthreads();
		if(threadIdx.x < 4) vsSum[threadIdx.x] += vsSum[threadIdx.x + 4];
		__syncthreads();
		if(threadIdx.x < 2) vsSum[threadIdx.x] += vsSum[threadIdx.x + 2];
		__syncthreads();
		if(threadIdx.x == 0)
		ddotReductionResult[blockIdx.x] = vsSum[0] + vsSum[1];
#endif
	}
}

double spgpuDdot(spgpuHandle_t handle, int n, __device double* a, __device double* b)
{
	double res = 0;
	int device;
	cudaGetDevice(&device);
#if 0
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);	

	int blocks = min(128, min(prop.multiProcessorCount, (n + BLOCK_SIZE - 1) / BLOCK_SIZE));
#else
	int blocks = min(128, min(handle->multiProcessorCount, (n + BLOCK_SIZE - 1) / BLOCK_SIZE));
#endif
	
	double tRes[128];

	spgpuDdot_kern<<<blocks, BLOCK_SIZE, 0, handle->currentStream>>>(n, a, b);
	cudaMemcpyFromSymbol(tRes, ddotReductionResult, blocks * sizeof(double));

	for(int i = 0; i < blocks; ++i)
		res += tRes[i];

	cudaCheckError("CUDA error on ddot");
	return res;
}

void spgpuDmdot(spgpuHandle_t handle, double* y, int n, __device double* a, __device double* b, int count, int pitch)
{
	for(int i = 0; i < count; ++i)
	{
		y[i] = spgpuDdot(handle, n, a, b);
		a += pitch;
		b += pitch;
	}
}

void spgpuDmsdot(spgpuHandle_t handle, double* y, int n, __device double* a, __device double* b)
{
	#ifdef USE_CUBLAS
		cublasDdot(handle->cublasHandle, n, a, 1, b, 1, y);
		cudaDeviceSynchronize();
	#else
		y[0] = spgpuDdot(handle, n, a, b);
	#endif
}

void spgpuDmvdot(spgpuHandle_t handle, double* y, int n, __device double* a, __device double* b, int countA, int pitchA)
{
	#ifdef USE_CUBLAS
		double res[countA];
		cublasDgemv(handle->cublasHandle, CUBLAS_OP_T, n, countA, &c_done, a, pitchA, b, &c_done, &c_dzero, res, 1);
		cudaDeviceSynchronize();
		for(int i = 0; i < countA; ++i)
			y[i] = res[i];
	#else
		//TO DO: optimize with custom gemv kernel?
		for(int i = 0; i < countA; ++i)
			y[i] = spgpuDdot(handle, n, a + (i * pitchA), b);
	#endif
}

void spgpuDmvdot_CFI(spgpuHandle_t handle, CFI_cdesc_t* y, int n, __device double* a, __device double* b, int countA, int pitchA)
{
	#ifdef USE_CUBLAS
		double res[countA];
		cublasDgemv(handle->cublasHandle, CUBLAS_OP_T, n, countA, &c_done, a, pitchA, b, &c_done, &c_dzero, res, 1);
		cudaDeviceSynchronize();
		for(int i = 0; i < countA; ++i)
			CFI_AT1(double, y, i) = res[i];
	#else
		//TO DO: optimize with custom gemv kernel?
		for(int i = 0; i < countA; ++i)
			CFI_AT1(double, y, i) = spgpuDdot(handle, n, a + (i * pitchA), b);
	#endif
}

void spgpuDmmdot(spgpuHandle_t handle, double* y, int n, __device double* a, __device double* b, int countA, int pitchA, int countB, int pitchB)
{
	#ifdef USE_CUBLAS
		double res[countA*countB];
		cublasDgemm(handle->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, countA, countB, n, &c_done, a, pitchA, b, &c_done, &c_zero, res, countA);
		cudaDeviceSynchronize();
		for(int i = 0; i < countB; ++i)
			for(int j = 0; j < countA; ++j)
				y[i*countA + j] = res[i*countA + j];
	#else
		//TO DO: optimize with custom gemm kernel?
		for(int i = 0; i < countB; ++i)
			for(int j = 0; j < countA; ++j)
				y[i*countA + j] = spgpuDdot(handle, n, a + (i * pitchA), b + (j * pitchB));
	#endif
}

void spgpuDmmdot_CFI(spgpuHandle_t handle, CFI_cdesc_t* y, int n, __device double* a, __device double* b, int countA, int pitchA, int countB, int pitchB)
{
	#ifdef USE_CUBLAS
		double res[countA*countB];
		cublasDgemv(handle->cublasHandle, CUBLAS_OP_T, n, countA, &c_done, a, pitchA, b, &c_done, &c_dzero, res, 1);
		cudaDeviceSynchronize();
		for(int i = 0; i < countA; ++i)
			for(int j = 0; j < countB; ++j)
				CFI_AT2(double, y, i, j) = res[i*countA + j];
	#else
		//TO DO: optimize with custom gemm kernel?
		for(int i = 0; i < countA; ++i)
			for(int j = 0; j < countB; ++j)
				CFI_AT2(double, y, i, j) = spgpuDdot(handle, n, a + (i * pitchA), b + (j * pitchB));
	#endif
}