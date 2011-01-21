#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <stdio.h>

#include "kernel.h"


#define THREADS_PER_BLOCK 64
#define ROUNDS_NUMBER 24
#define WORDS_NUMBER 25

#define index(x, y) (((x)%5)+5*((y)%5))
#define ROL64(a, offset) ((offset != 0) ? ((((UINT64)a) >> (0-offset)) ^ (((UINT64)a) >> (64-offset))) : a)





UINT64 *buffer_d;
UINT64 *buffer1_d;
UINT64 *buffer2_d;
UINT64 *state_d;

unsigned int threads_number;
size_t size;


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
	for(i = 0; i < WORDS_NUMBER * 8; i++)
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
    
    for(i = 0; i < WORDS_NUMBER * 8; i++)
        state_d[offset + i] = A[i];
}


/*
 *
 */
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
int init_cuda(unsigned int t, UINT64 *krc, unsigned int *kro)
{
	threads_number = t;
	size = 25*t*sizeof(unsigned long long); 
	
	// Initialize round constants
	cutilSafeCall( cudaMemcpyToSymbol("KeccakRoundConstants", krc, ROUNDS_NUMBER*sizeof(UINT64), 0, cudaMemcpyHostToDevice) );
	
	// Initialize rho offsets
	cutilSafeCall( cudaMemcpyToSymbol("KeccakRhoOffsets", kro, WORDS_NUMBER*sizeof(unsigned int), 0, cudaMemcpyHostToDevice) );
	
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
	// Deallocate GPU memory buffer 2
	// Deallocate GPU memory state

	return 0;

}


/*
 *
 */
int get_state(unsigned long long *state_h)
{
	// Check kernel termination
	cudaThreadSynchronize();

	// State retrival
	cutilSafeCall( cudaMemcpy(state_h, state_d, size, cudaMemcpyDeviceToHost) );

	return 0;
}