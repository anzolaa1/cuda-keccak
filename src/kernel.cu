#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <stdio.h>

#include "kernel.h"


#define __DEBUG_MODE_ON__
#define __BENCHMARK_MODE_ON__

#define THREADS_PER_BLOCK 256
#define ROUNDS_NUMBER 24
#define WORDS_NUMBER 25

#define index(x, y) (((x)%5)+5*((y)%5))

// NVCC Bug
//#define ROL64(a, offset) ((offset != 0) ? ((((UINT64)a) << offset) ^ (((UINT64)a) >> (64-offset))) : a)
__device__ inline UINT64 ROL64(UINT64 a, unsigned int offset)
{
	const int _offset = offset;
	return ((offset != 0) ? ((a << _offset) ^ (a >> (64-offset))) : a);
}


UINT64 *buffer_d;
UINT64 *buffer1_d;
UINT64 *buffer2_d;
UINT64 *state_d;

unsigned int threads_number;
size_t size;
size_t size_actual;

cudaEvent_t startEvent;
cudaEvent_t stopEvent;


__constant__ UINT64 KeccakRoundConstants[ROUNDS_NUMBER];
__constant__ unsigned int KeccakRhoOffsets[WORDS_NUMBER];


/*
 *
 */
__global__ void kernel(UINT64 *messages_d, UINT64 *state_d)
{
	int offset = WORDS_NUMBER * (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int i, x, y, round_number;
	UINT64 A[WORDS_NUMBER], tempA[WORDS_NUMBER], C[5], D[5];
	
	// Absorbing
	for(i = 0; i < WORDS_NUMBER; i++)
        	A[i] = state_d[offset + i] ^ messages_d[offset + i];

    	for(round_number = 0; round_number < ROUNDS_NUMBER; round_number++) {
		// Theta
		for(x=0; x<5; x++) {
			C[x] = 0; 
			for(y=0; y<5; y++) 
				C[x] ^= A[index(x, y)];
			D[x] = ROL64(C[x], 1);
		}
		for(x=0; x<5; x++)
			for(y=0; y<5; y++)
				A[index(x, y)] ^= D[(x+1)%5] ^ C[(x+4)%5];

        	// Rho
		for(x=0; x<5; x++) 
			for(y=0; y<5; y++)
				A[index(x, y)] = ROL64(A[index(x, y)], KeccakRhoOffsets[index(x, y)]);

		// Pi
        	for(x=0; x<5; x++) for(y=0; y<5; y++)
			tempA[index(x, y)] = A[index(x, y)];
		for(x=0; x<5; x++) for(y=0; y<5; y++)
			A[index(0*x+1*y, 2*x+3*y)] = tempA[index(x, y)];
		
        	// Chi
        	for(y=0; y<5; y++) { 
			for(x=0; x<5; x++)
				C[x] = A[index(x, y)] ^ ((~A[index(x+1, y)]) & A[index(x+2, y)]);
			for(x=0; x<5; x++)
				A[index(x, y)] = C[x];
		}
		
        	// Iota
		A[index(0, 0)] ^= KeccakRoundConstants[round_number];
    }
    
    for(i = 0; i < WORDS_NUMBER; i++)
        state_d[offset + i] = A[i];
}


/*
 *
 */
__global__ void kernel_optimixed(UINT64 *messages_d, UINT64 *state_d)
{
	int offset = WORDS_NUMBER * (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int x, y, round_number;
	UINT64 A[WORDS_NUMBER], tempA[WORDS_NUMBER], C[5], D[5];
	
	// Absorbing
	for(x = 0; x < WORDS_NUMBER; x++)
        	A[x] = state_d[offset + x] ^ messages_d[offset + x];

    	for(round_number = 0; round_number < ROUNDS_NUMBER; round_number++) {
		// Theta
		for(x=0; x<5; x++) {
			C[x] = 0; 
			for(y=0; y<5; y++) 
				C[x] = C[x] ^ A[index(x, y)];
			D[x] = ROL64(C[x], 1);
		}
		for(x=0; x<5; x++)
			for(y=0; y<5; y++)
				A[index(x, y)] = A[index(x,y)] ^ D[(x+1)%5] ^ C[(x+4)%5];

        	// Rho
		for(x=0; x<5; x++) 
			for(y=0; y<5; y++)
				A[index(x, y)] = ROL64(A[index(x, y)], KeccakRhoOffsets[index(x, y)]);

		// Pi
        	for(x=0; x<5; x++) for(y=0; y<5; y++)
			tempA[index(x, y)] = A[index(x, y)];
		for(x=0; x<5; x++) for(y=0; y<5; y++)
			A[index(0*x+1*y, 2*x+3*y)] = tempA[index(x, y)];
		
        	// Chi
        	for(y=0; y<5; y++) { 
			for(x=0; x<5; x++)
				C[x] = A[index(x, y)] ^ ((~A[index(x+1, y)]) & A[index(x+2, y)]);
			for(x=0; x<5; x++)
				A[index(x, y)] = C[x];
		}
		
        	// Iota
		A[index(0, 0)] = A[index(0,0)] ^ KeccakRoundConstants[round_number];
    }
    
    for(x = 0; x < WORDS_NUMBER; x++)
        state_d[offset + x] = A[x];
}







/*
 *
 */
void launch_kernel(unsigned long long *messages_h, unsigned int token_number)
{
	dim3 threads_per_block(THREADS_PER_BLOCK);
	int num_blocks = threads_number/THREADS_PER_BLOCK;

	if(token_number%2 == 0)
		buffer_d = buffer1_d;
	else
		buffer_d = buffer2_d;

	// Copy messages_h into buffer_d
	cutilSafeCall( cudaMemcpy(buffer_d, messages_h, size_actual, cudaMemcpyHostToDevice) );

	// Wait old kernel termination
	cudaThreadSynchronize();

	// Launch timer
	if(token_number == 0)
	{
		cutilSafeCall(cudaEventRecord(startEvent, 0));
	}

	// Launch new kernel
	kernel_optimixed<<<num_blocks, threads_per_block>>>(buffer_d, state_d);
}


/*
 *
 */
int init_cuda(unsigned int t, UINT64 *krc, unsigned int *kro)
{
	int dev_ID;
	cudaDeviceProp device_prop;

	// Get best device properties
	dev_ID = cutGetMaxGflopsDeviceId();
	cudaGetDeviceProperties(&device_prop, dev_ID);
	#ifdef __DEBUG_MODE_ON__
	printf("*\nMax Gflops Device: \"%s\"\n", device_prop.name);
	printf("\tCUDA Capability:                               %d.%d\n", device_prop.major, device_prop.minor);
	printf("\tTotal amount of Global Memory:                 %llu bytes\n", (UINT64) device_prop.totalGlobalMem);
	//printf("\tMultiprocessor x Cores/MP = Cores:             %d (MP) x %d (Cores/MP) = %d (Cores)\n", device_prop.multiProcessorCount, ConvertSMVer2Cores(deviceProp.major, deviceProp.minor), ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
	printf("\tTotal number of registers available per block: %d\n", device_prop.regsPerBlock);
	printf("\tMaximum number of threads per block:           %d\n", device_prop.maxThreadsPerBlock);
	printf("*\n");
	#endif

	// Set device
        cudaSetDevice(dev_ID);

	// Set the number of actual threads
	// In order to avoid control instructions inside the kernel, the number of threads is chooses...
	threads_number = ((t%THREADS_PER_BLOCK == 0) ? (t) : (t/THREADS_PER_BLOCK + 1)*THREADS_PER_BLOCK);

	// Meaningfull part of the memory
	size_actual = 25*t*sizeof(UINT64); 

	// Whole memory
	size = 25*threads_number*sizeof(UINT64); 
	
	// Initialize round constants
	cutilSafeCall( cudaMemcpyToSymbol("KeccakRoundConstants", krc, ROUNDS_NUMBER*sizeof(UINT64), 0, cudaMemcpyHostToDevice) );
	
	// Initialize rho offsets
	cutilSafeCall( cudaMemcpyToSymbol("KeccakRhoOffsets", kro, WORDS_NUMBER*sizeof(unsigned int), 0, cudaMemcpyHostToDevice) );

	// Create timers
	cutilSafeCall( cudaEventCreate(&startEvent) );
      	cutilSafeCall( cudaEventCreate(&stopEvent) );
	
	return 0;
}



/*
 * Allocate and zero initialize GPU memory
 */
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
int free_memory()
{
	// Deallocate GPU memory buffer 1
	cutilSafeCall(cudaFree(buffer1_d));

	// Deallocate GPU memory buffer 2
	cutilSafeCall(cudaFree(buffer2_d));

	// Deallocate GPU memory state
	cutilSafeCall(cudaFree(state_d));

	return 0;

}


/*
 *
 */
int get_state(UINT64 *state_h)
{
	float milliseconds;

	// Check kernel termination
	cutilSafeCall(cudaThreadSynchronize());

	// Stop timer
	cutilSafeCall(cudaEventRecord(stopEvent, 0));
	cutilSafeCall(cudaEventSynchronize(stopEvent));
	cutilSafeCall( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

	#ifdef __BENCHMARK_MODE_ON__
	printf("*\nGPU time: %.3f ms\n*\n", milliseconds);
	#endif

	// State retrival
	cutilSafeCall( cudaMemcpy(state_h, state_d, size_actual, cudaMemcpyDeviceToHost) );

	return 0;
}
