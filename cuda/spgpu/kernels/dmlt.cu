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
}

#include "debug.h"

#define BLOCK_SIZE 512
#define USE_CUBLAS 1

__global__ void spgpuDmltpwise_krn(int n, double* z, double alpha, double* x, double* y, double beta, int count, int pitchX, int pitchY, int pitchZ, bool conjgX, bool conjgY)
{
	const size_t gridSize = blockDim.x * gridDim.x;
	size_t id = threadIdx.x + BLOCK_SIZE * blockIdx.x;

	//TODO: check beta = 0
	for(; id < n; id += gridSize)
	{
		size_t baseX = 0, baseY = 0, baseZ = 0;
		for(int idx = 0; idx < count; ++idx)
		{
			//z[baseZ + id] = alpha * x[baseX + id] * y[baseY + id] + beta * z[baseZ + id]
			z[baseZ + id] = PREC_DADD(PREC_DMUL(alpha, PREC_DMUL(x[baseX + id], y[baseY + id])), PREC_DMUL(beta, z[baseZ + id]));
			baseX += pitchX;
			baseY += pitchY;
			baseZ += pitchZ;
		}
	}
}

__global__ void spgpuDmltdscal_krn(int n, double* z, double alpha, double* x, double* y, double beta, int count, int pitchY, int pitchZ, bool conjgX, bool conjgY)
{
	const size_t gridSize = blockDim.x * gridDim.x;
	size_t id = threadIdx.x + BLOCK_SIZE * blockIdx.x;

	//TODO: check beta = 0
	for(; id < n; id += gridSize)
	{
		double ax = PREC_DMUL(alpha, x[id]);
		size_t baseY = 0, baseZ = 0;
		for(int idx = 0; idx < count; ++idx)
		{
			//z[baseZ + id] = alpha * x[baseX + id] * y[baseY + id] + beta * z[baseZ + id]
			z[baseZ + id] = PREC_DADD(PREC_DMUL(ax, y[baseY + id]), PREC_DMUL(beta, z[baseZ + id]));
			baseY += pitchY;
			baseZ += pitchZ;
		}
	}
}

__global__ void spgpuDmltbcast_krn(int n, double* z, double alpha, double* x, double* y, double beta, int count, int pitchZ, bool conjgX, bool conjgY)
{
	const size_t gridSize = blockDim.x * gridDim.x;
	size_t id = threadIdx.x + BLOCK_SIZE * blockIdx.x;

	//TODO: check beta = 0
	for(; id < n; id += gridSize)
	{
		double axy = PREC_DMUL(alpha, PREC_DMUL(x[id], y[id]));
		size_t baseZ = 0;
		for(int idx = 0; idx < count; ++idx)
		{
			//z[baseZ + id] = alpha * x[baseX + id] * y[baseY + id] + beta * z[baseZ + id]
			z[baseZ + id] = PREC_DADD(axy, PREC_DMUL(beta, z[baseZ + id]));
			baseZ += pitchZ;
		}
	}
}

void spgpuDmlt(spgpuHandle_t handle,  __device double* z, int countZ, int pitchZ, int n, double alpha, __device double* x, int countX, int pitchX, 
										__device double* y, int countY, int pitchY, double beta, bool conjgX, bool conjgY)
{
	dim3 block(BLOCK_SIZE);
	int num_mp         = getGPUMultiProcessors();
	int max_threads_mp = getGPUMaxThreadsPerMP();
	int num_blocks_mp  = max_threads_mp/BLOCK_SIZE;
	int num_blocks     = num_blocks_mp * num_mp;
	dim3 grid(num_blocks);

	//TODO: if alpha = 0 call dscal

	if((countZ == countY) && (countY == countX))
		spgpuDmltpwise_krn<<<grid, block, 0, handle->currentStream>>>(n, z, alpha, x, y, beta, countZ, pitchX, pitchY, pitchZ, conjgX, conjgY);
	else if((countZ == countY) && (countY != countX))
		spgpuDmltdscal_krn<<<grid, block, 0, handle->currentStream>>>(n, z, alpha, x, y, beta, countZ, pitchY, pitchZ, conjgX, conjgY);
	else if((countZ != countY) && (countY == countX))
		spgpuDmltbcast_krn<<<grid, block, 0, handle->currentStream>>>(n, z, alpha, x, y, beta, countZ, pitchZ, conjgX, conjgY);
	else
		return;

}