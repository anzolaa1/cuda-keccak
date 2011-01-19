#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <stdio.h>

#include "kernel.h"


#define THREADS_PER_BLOCK 64


unsigned long long *buffer_d;
unsigned long long *buffer1_d;
unsigned long long *buffer2_d;
unsigned long long *state_d;

unsigned int threads_number;
size_t size;

/*
 *
 */
__global__ void kernel(unsigned long long *messages_d, unsigned long long *state_d)
{
	// Squeeze

	// 23


	int i = 25 * (threadIdx.x + blockIdx.x * blockDim.x);
	int offset = 0;

	for(; offset < 25; offset++)
	{
		state_d[i+offset] = messages_d[i+offset] + 1;
	}
}


/*
 *
 */
//extern "C"
void launch_kernel(unsigned long long *messages_h, unsigned int token_number)
{
	dim3 threads_per_block(threads_number);
	int num_blocks = threads_number/THREADS_PER_BLOCK + threads_number%THREADS_PER_BLOCK>0?1:0;

	if(token_number%2 == 0)
		buffer_d = buffer1_d;
	else
		buffer_d = buffer2_d;

		// Copy messages_h into buffer_d
		cutilSafeCall( cudaMemcpy(buffer_d, messages_h, size,cudaMemcpyHostToDevice) );

		// Wait old kernel termination
		cudaThreadSynchronize();

		// launch new kernel
		kernel<<<num_blocks, threads_per_block>>>(buffer_d, state_d);
}


/*
 *
 */
//extern "C"
int init_cuda(unsigned int t)
{
	threads_number = t;
	size = 25*t*sizeof(unsigned long long); 

	return 0;
}


/*
 * Allocate and zero initialize GPU memory
 */
//extern "C"
int alloc_memory()
{	
	// Allocate GPU memory buffer 1
	cutilSafeCall( cudaMalloc((void**) &buffer1_d, size) );

	// Allocate GPU memory buffer 2
	cutilSafeCall( cudaMalloc((void**) &buffer2_d, size) );

	// Allocate GPU memory state
	cutilSafeCall( cudaMalloc((void**) &state_d, size) );


	// Zero init
	cutilSafeCall ( cudaMemset(buffer1_d, 0, size) );
	cutilSafeCall ( cudaMemset(buffer2_d, 0, size) );
	cutilSafeCall ( cudaMemset(state_d, 0, size) );

	return 0;
}


/*
 *
 */
//extern "C"
int free_memory()
{
	// Deallocate GPU memory buffer 1
	// Deallocate GPU memory buffer 2
	// Deallocate GPU memory state

	return 0;

}


/*
 *
 */
//extern "C" 
int get_state(unsigned long long *state_h)
{
	// Check kernel termination
	cudaThreadSynchronize();

	// State retrival
	cutilSafeCall( cudaMemcpy(state_h, state_d, size, cudaMemcpyDeviceToHost) );

	return 0;
}
