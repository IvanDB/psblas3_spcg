/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
 *     Davide Barbieri - University of Rome Tor Vergata
 *
 */

#include "core.h"

extern "C"
{
	static CFIBuffer dotBuffer;
	CFIBuffer* getDotBuffer() {return &dotBuffer;};
}

#define VALUE_TYPE cuDoubleComplex
#define TYPE_SYMBOL Z
#undef USE_CUBLAS
#include "dot.cuh"