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
	static CFIBuffer coeffBuffer;
	CFIBuffer* getColspanBuffer() {return &coeffBuffer;};
}

#define VALUE_TYPE float
#define TYPE_SYMBOL S
#include "colspan.cuh"