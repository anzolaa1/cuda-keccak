#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <stdio.h>

#include "kernel.h"


//#define __DEBUG_MODE_ON__
#define __BENCHMARK_MODE_ON__

#define THREADS_PER_BLOCK 32
#define ROUNDS_NUMBER 24
#define WORDS_NUMBER 25

//#define index(x, y) (((x)%5)+5*((y)%5))

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



__global__ void kernel(UINT64 *messages_d, UINT64 *state_d)
{
	
	int offset = WORDS_NUMBER * (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int i, x, y, round_number;
	UINT64 A[25], tempA[25], C[5], D[5];
	

	// Absorbing
	for(i = 0; i < WORDS_NUMBER; i++)
        	A[i] = state_d[offset + i] ^ messages_d[offset + i];
	/*
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
		A[0] ^= KeccakRoundConstants[round_number];
    }

    for(i = 0; i < WORDS_NUMBER; i++)
        state_d[offset + i] = A[i];
*/
}


/*
 *
 */
__global__ void kernel_optimixed(UINT64 *messages_d, UINT64 *state_d)
{
	int offset = WORDS_NUMBER * (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int x;
	UINT64 A[25], tempA[25], C[5], D[5];

	// Absorbing
	/*
	for(x = 0; x < WORDS_NUMBER; x++)
        	A[x] = state_d[offset + x] ^ messages_d[offset + x];
	*/
	A[0] = state_d[offset + 0] ^ messages_d[offset + 0];
	A[1] = state_d[offset + 1] ^ messages_d[offset + 1];
	A[2] = state_d[offset + 2] ^ messages_d[offset + 2];
	A[3] = state_d[offset + 3] ^ messages_d[offset + 3];
	A[4] = state_d[offset + 4] ^ messages_d[offset + 4];
	A[5] = state_d[offset + 5] ^ messages_d[offset + 5];
	A[6] = state_d[offset + 6] ^ messages_d[offset + 6];
	A[7] = state_d[offset + 7] ^ messages_d[offset + 7];
	A[8] = state_d[offset + 8] ^ messages_d[offset + 8];
	A[9] = state_d[offset + 9] ^ messages_d[offset + 9];
	A[10] = state_d[offset + 10] ^ messages_d[offset + 10];
	A[11] = state_d[offset + 11] ^ messages_d[offset + 11];
	A[12] = state_d[offset + 12] ^ messages_d[offset + 12];
	A[13] = state_d[offset + 13] ^ messages_d[offset + 13];
	A[14] = state_d[offset + 14] ^ messages_d[offset + 14];
	A[15] = state_d[offset + 15] ^ messages_d[offset + 15];
	A[16] = state_d[offset + 16] ^ messages_d[offset + 16];
	A[17] = state_d[offset + 17] ^ messages_d[offset + 17];
	A[18] = state_d[offset + 18] ^ messages_d[offset + 18];
	A[19] = state_d[offset + 19] ^ messages_d[offset + 19];
	A[20] = state_d[offset + 20] ^ messages_d[offset + 20];
	A[21] = state_d[offset + 21] ^ messages_d[offset + 21];
	A[22] = state_d[offset + 22] ^ messages_d[offset + 22];
	A[23] = state_d[offset + 23] ^ messages_d[offset + 23];
	A[24] = state_d[offset + 24] ^ messages_d[offset + 24];

    	for(x = 0; x < ROUNDS_NUMBER; x++) {
		// Theta
		/*for(x=0; x<5; x++) {
			C[x] = 0; 
			for(y=0; y<5; y++) 
				C[x] = C[x] ^ A[index(x, y)];
			D[x] = ROL64(C[x], 1);
		}
		for(x=0; x<5; x++)
			for(y=0; y<5; y++)
				A[index(x, y)] = A[index(x,y)] ^ D[(x+1)%5] ^ C[(x+4)%5];
		*/
		
		//THETA unrolled
		//x = 0
		C[0] = 0;
		C[0] = C[0] ^ A[0];
		C[0] = C[0] ^ A[5];
		C[0] = C[0] ^ A[10];
		C[0] = C[0] ^ A[15];
		C[0] = C[0] ^ A[20];
		D[0] = ROL64(C[0], 1);
		//x = 1
		C[1] = 0;
		C[1] = C[1] ^ A[1];
		C[1] = C[1] ^ A[6];
		C[1] = C[1] ^ A[11];
		C[1] = C[1] ^ A[16];
		C[1] = C[1] ^ A[21];
		D[1] = ROL64(C[1], 1);
		// x = 2
		C[2] = 0;
		C[2] = C[2] ^ A[2];
		C[2] = C[2] ^ A[7];
		C[2] = C[2] ^ A[12];
		C[2] = C[2] ^ A[17];
		C[2] = C[2] ^ A[22];
		D[2] = ROL64(C[2], 1);
		// x = 3
		C[3] = 0;
		C[3] = C[3] ^ A[3];
		C[3] = C[3] ^ A[8];
		C[3] = C[3] ^ A[13];
		C[3] = C[3] ^ A[18];
		C[3] = C[3] ^ A[23];
		D[3] = ROL64(C[3], 1);
		// x = 4
		C[4] = 0;
		C[4] = C[4] ^ A[4];
		C[4] = C[4] ^ A[9];
		C[4] = C[4] ^ A[14];
		C[4] = C[4] ^ A[19];
		C[4] = C[4] ^ A[24];
		D[4] = ROL64(C[4], 1);
		
		A[0] = A[0] ^ D[1] ^ C[4];
		A[5] = A[5] ^ D[1] ^ C[4];
		A[10] = A[10] ^ D[1] ^ C[4];
		A[15] = A[15] ^ D[1] ^ C[4];
		A[20] = A[20] ^ D[1] ^ C[4];
		
		A[1] = A[1] ^ D[2] ^ C[0];
		A[6] = A[6] ^ D[2] ^ C[0];
		A[11] = A[11] ^ D[2] ^ C[0];
		A[16] = A[16] ^ D[2] ^ C[0];
		A[21] = A[21] ^ D[2] ^ C[0];
		
		A[2] = A[2] ^ D[3] ^ C[1];
		A[7] = A[7] ^ D[3] ^ C[1];
		A[12] = A[12] ^ D[3] ^ C[1];
		A[17] = A[17] ^ D[3] ^ C[1];
		A[22] = A[22] ^ D[3] ^ C[1];
		
		A[3] = A[3] ^ D[4] ^ C[2];
		A[8] = A[8] ^ D[4] ^ C[2];
		A[13] = A[13] ^ D[4] ^ C[2];
		A[18] = A[18] ^ D[4] ^ C[2];
		A[23] = A[23] ^ D[4] ^ C[2];
		
		A[4] = A[4] ^ D[0] ^ C[3];
		A[9] = A[9] ^ D[0] ^ C[3];
		A[14] = A[14] ^ D[0] ^ C[3];
		A[19] = A[19] ^ D[0] ^ C[3];
		A[24] = A[24] ^ D[0] ^ C[3];
        	// Rho
		/*for(x=0; x<5; x++) 
			for(y=0; y<5; y++)
				A[index(x, y)] = ROL64(A[index(x, y)], KeccakRhoOffsets[index(x, y)]);
		*/
		
		
		//RHO UNROLLED		
		A[0] = ROL64(A[0], KeccakRhoOffsets[0]);
		A[5] = ROL64(A[5], KeccakRhoOffsets[5]);
		A[10] = ROL64(A[10], KeccakRhoOffsets[10]);
		A[15] = ROL64(A[15], KeccakRhoOffsets[15]);
		A[20] = ROL64(A[20], KeccakRhoOffsets[20]);
		
		A[1] = ROL64(A[1], KeccakRhoOffsets[1]);
		A[6] = ROL64(A[6], KeccakRhoOffsets[6]);
		A[11] = ROL64(A[11], KeccakRhoOffsets[11]);
		A[16] = ROL64(A[16], KeccakRhoOffsets[16]);
		A[21] = ROL64(A[21], KeccakRhoOffsets[21]);

		A[2] = ROL64(A[2], KeccakRhoOffsets[2]);
		A[7] = ROL64(A[7], KeccakRhoOffsets[7]);
		A[12] = ROL64(A[12], KeccakRhoOffsets[12]);
		A[17] = ROL64(A[17], KeccakRhoOffsets[17]);
		A[22] = ROL64(A[22], KeccakRhoOffsets[22]);
		
		A[3] = ROL64(A[3], KeccakRhoOffsets[3]);
		A[8] = ROL64(A[8], KeccakRhoOffsets[8]);
		A[13] = ROL64(A[13], KeccakRhoOffsets[13]);
		A[18] = ROL64(A[18], KeccakRhoOffsets[18]);
		A[23] = ROL64(A[23], KeccakRhoOffsets[23]);
		
		A[4] = ROL64(A[4], KeccakRhoOffsets[4]);
		A[9] = ROL64(A[9], KeccakRhoOffsets[9]);
		A[14] = ROL64(A[14], KeccakRhoOffsets[14]);
		A[19] = ROL64(A[19], KeccakRhoOffsets[19]);
		A[24] = ROL64(A[24], KeccakRhoOffsets[24]);
		
		/*
		// Pi
        for(x=0; x<5; x++)
        	for(y=0; y<5; y++)
				tempA[index(x, y)] = A[index(x, y)];
		for(x=0; x<5; x++)
			for(y=0; y<5; y++)
				A[index(0*x+1*y, 2*x+3*y)] = tempA[index(x, y)];
		*/
		//UNROLLED PI
		tempA[0] = A[0];
		tempA[5] = A[5];
		tempA[10] = A[10];
		tempA[15] = A[15];
		tempA[20] = A[20];
		
		tempA[1] = A[1];
		tempA[6] = A[6];
		tempA[11] = A[11];
		tempA[16] = A[16];
		tempA[21] = A[21];
		
		tempA[2] = A[2];
		tempA[7] = A[7];
		tempA[12] = A[12];
		tempA[17] = A[17];
		tempA[22] = A[22];
		
		tempA[3] = A[3];
		tempA[8] = A[8];
		tempA[13] = A[13];
		tempA[18] = A[18];
		tempA[23] = A[23];
		
		tempA[4] = A[4];
		tempA[9] = A[9];
		tempA[14] = A[14];
		tempA[19] = A[19];
		tempA[24] = A[24];
		
		A[0] = tempA[0];
		A[16] = tempA[5];
		A[7] = tempA[10];
		A[23] = tempA[15];
		A[14] = tempA[20];
		
		A[10] = tempA[1];
		A[1] = tempA[6];
		A[17] = tempA[11];
		A[8] = tempA[16];
		A[24] = tempA[21];
		
		A[20] = tempA[2];
		A[11] = tempA[7];
		A[2] = tempA[12];
		A[18] = tempA[17];
		A[9] = tempA[22];
		
		A[5] = tempA[3];
		A[21] = tempA[8];
		A[12] = tempA[13];
		A[3] = tempA[18];
		A[19] = tempA[23];
		
		A[15] = tempA[4];
		A[6] = tempA[9];
		A[22] = tempA[14];
		A[13] = tempA[19];
		A[4] = tempA[24];
		
		
        // Chi
        /*for(y=0; y<5; y++) { 
			for(x=0; x<5; x++)
				C[x] = A[index(x, y)] ^ ((~A[index(x+1, y)]) & A[index(x+2, y)]);
			for(x=0; x<5; x++)
				A[index(x, y)] = C[x];
		}*/
		
		C[0] = A[0] ^ ((~A[1]) & A[2]);
		C[1] = A[1] ^ ((~A[2]) & A[3]);
		C[2] = A[2] ^ ((~A[3]) & A[4]);
		C[3] = A[3] ^ ((~A[4]) & A[0]);
		C[4] = A[4] ^ ((~A[0]) & A[1]);
		A[0] = C[0];
		A[1] = C[1];
		A[2] = C[2];
		A[3] = C[3];
		A[4] = C[4];
		
		C[0] = A[5] ^ ((~A[6]) & A[7]);
		C[1] = A[6] ^ ((~A[7]) & A[8]);
		C[2] = A[7] ^ ((~A[8]) & A[9]);
		C[3] = A[8] ^ ((~A[9]) & A[5]);
		C[4] = A[9] ^ ((~A[5]) & A[6]);
		A[5] = C[0];
		A[6] = C[1];
		A[7] = C[2];
		A[8] = C[3];
		A[9] = C[4];
		
		C[0] = A[10] ^ ((~A[11]) & A[12]);
		C[1] = A[11] ^ ((~A[12]) & A[13]);
		C[2] = A[12] ^ ((~A[13]) & A[14]);
		C[3] = A[13] ^ ((~A[14]) & A[10]);
		C[4] = A[14] ^ ((~A[10]) & A[11]);
		A[10] = C[0];
		A[11] = C[1];
		A[12] = C[2];
		A[13] = C[3];
		A[14] = C[4];
		
		C[0] = A[15] ^ ((~A[16]) & A[17]);
		C[1] = A[16] ^ ((~A[17]) & A[18]);
		C[2] = A[17] ^ ((~A[18]) & A[19]);
		C[3] = A[18] ^ ((~A[19]) & A[15]);
		C[4] = A[19] ^ ((~A[15]) & A[16]);
		A[15] = C[0];
		A[16] = C[1];
		A[17] = C[2];
		A[18] = C[3];
		A[19] = C[4];
		
		C[0] = A[20] ^ ((~A[21]) & A[22]);
		C[1] = A[21] ^ ((~A[22]) & A[23]);
		C[2] = A[22] ^ ((~A[23]) & A[24]);
		C[3] = A[23] ^ ((~A[24]) & A[20]);
		C[4] = A[24] ^ ((~A[20]) & A[21]);
		A[20] = C[0];
		A[21] = C[1];
		A[22] = C[2];
		A[23] = C[3];
		A[24] = C[4];
		
        // Iota
		A[0] = A[0] ^ KeccakRoundConstants[x];
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
	//kernel<<<num_blocks, threads_per_block>>>(buffer_d, state_d);
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
