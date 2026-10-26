/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
 *     Davide Barbieri - University of Rome Tor Vergata
 *
 */

#include "cudadebug.h"
#include "cudalang.h"
#include <cuda_runtime.h>
#include "core.h"
#include "cublas_v2.h"

extern "C"
{
	#include "vector.h"
	//#include "cuda_util.h"
	int getGPUMultiProcessors();
	int getGPUMaxThreadsPerMP();
	cublasHandle_t psb_cudaGetCublasHandle();
	
	static double cuDone = 1.0;
	static double cuDzero = 0.0;
	static CFIBuffer coeffBuffer;
	CFIBuffer* getColspanBuffer() {return &coeffBuffer;};
}

#include "debug.h"

#define BLOCK_SIZE 512

#define USE_CUBLAS 1

__global__ void spgpuDcolspan1D_krn(int n, double* x, int pitchX, int countX, const double* coeffPtr, double* y, bool updFlag)
{
	size_t id = threadIdx.x + BLOCK_SIZE * blockIdx.x;
	const size_t gridSize = blockDim.x * gridDim.x;

	for(; id < n; id += gridSize)
	{
		double acc = updFlag ? y[id] : 0.0;
		for(size_t j = 0; j < countX; ++j)
			acc = PREC_DADD(PREC_DMUL(coeffPtr[j], x[id + (j*pitchX)]), acc);
		y[id] = acc;
	}
}

__global__ void spgpuDcolspan2D_krn(int n, double* x, int countX, int pitchX, const double* coeffPtr, double* y, int countY, int pitchY, bool updFlag)
{
	size_t id = threadIdx.x + BLOCK_SIZE * blockIdx.x;
	const size_t gridSize = blockDim.x * gridDim.x;

	for(; id < n; id += gridSize)
		for(size_t k = 0; k < countY; ++k)
		{
			double acc = updFlag ? y[id + (k * pitchY)] : 0.0;
			for(size_t j = 0; j < countX; ++j)
				acc = PREC_DADD(PREC_DMUL(coeffPtr[j + k * countX], x[id + (j * pitchX)]), acc);
			y[id + (k * pitchY)] = acc;
		}
}

void spgpuDcolspan(spgpuHandle_t handle, int n, __device double* x, int countX, int pitchX, CFI_cdesc_t* coeff, __device double* y, int countY, int pitchY, bool updFlag)
{
    CFIBufferLoad(handle, coeff, &coeffBuffer);

	#if USE_CUBLAS
		cublasHandle_t cublasHandle = psb_cudaGetCublasHandle();
		double* beta = updFlag ? (&cuDone) : (&cuDzero);
		if(coeff->rank == 1) cublasDgemv(cublasHandle, CUBLAS_OP_N, n, countX, &cuDone, x, pitchX, (double*) coeffBuffer.devBuffer, 1, beta, y, 1);
		if(coeff->rank == 2) cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, countY, countX, &cuDone, x, pitchX, (double*) coeffBuffer.devBuffer, countX, beta, y, pitchY);
	#else
		int num_mp         = getGPUMultiProcessors();
		int max_threads_mp = getGPUMaxThreadsPerMP();
		int num_blocks_mp  = max_threads_mp/BLOCK_SIZE;
		int num_blocks     = num_blocks_mp * num_mp;
		dim3 grid(num_blocks);
		dim3 block(BLOCK_SIZE);
		if(coeff->rank == 1) spgpuDcolspan1D_krn<<<grid, block, 0, handle->currentStream>>>(n, x, countX, pitchX, (double*) coeffBuffer.devBuffer, y, updFlag);
		if(coeff->rank == 2) spgpuDcolspan2D_krn<<<grid, block, 0, handle->currentStream>>>(n, x, countX, pitchX, (double*) coeffBuffer.devBuffer, y, countY, pitchY, updFlag);
	#endif

}