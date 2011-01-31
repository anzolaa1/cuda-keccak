#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <stdio.h>

#include "kernel.h"


//#define __DEBUG_MODE_ON__
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
		A[index(0, 0)] ^= KeccakRoundConstants[round_number];
    }
    */
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
		
		A[index(1, 0)] = A[index(1,0)] ^ D[2] ^ C[0];
		A[index(1, 1)] = A[index(1,1)] ^ D[2] ^ C[0];
		A[index(1, 2)] = A[index(1,2)] ^ D[2] ^ C[0];
		A[index(1, 3)] = A[index(1,3)] ^ D[2] ^ C[0];
		A[index(1, 4)] = A[index(1,4)] ^ D[2] ^ C[0];
		
		A[index(2, 0)] = A[index(2,0)] ^ D[3] ^ C[1];
		A[index(2, 1)] = A[index(2,1)] ^ D[3] ^ C[1];
		A[index(2, 2)] = A[index(2,2)] ^ D[3] ^ C[1];
		A[index(2, 3)] = A[index(2,3)] ^ D[3] ^ C[1];
		A[index(2, 4)] = A[index(2,4)] ^ D[3] ^ C[1];
		
		A[index(3, 0)] = A[index(3,0)] ^ D[4] ^ C[2];
		A[index(3, 1)] = A[index(3,1)] ^ D[4] ^ C[2];
		A[index(3, 2)] = A[index(3,2)] ^ D[4] ^ C[2];
		A[index(3, 3)] = A[index(3,3)] ^ D[4] ^ C[2];
		A[index(3, 4)] = A[index(3,4)] ^ D[4] ^ C[2];
		
		A[index(4, 0)] = A[index(4,0)] ^ D[0] ^ C[3];
		A[index(4, 1)] = A[index(4,1)] ^ D[0] ^ C[3];
		A[index(4, 2)] = A[index(4,2)] ^ D[0] ^ C[3];
		A[index(4, 3)] = A[index(4,3)] ^ D[0] ^ C[3];
		A[index(4, 4)] = A[index(4,4)] ^ D[0] ^ C[3];
        	// Rho
		/*for(x=0; x<5; x++) 
			for(y=0; y<5; y++)
				A[index(x, y)] = ROL64(A[index(x, y)], KeccakRhoOffsets[index(x, y)]);
		*/
		
		
		//RHO UNROLLED		
		A[index(0, 0)] = ROL64(A[index(0, 0)], KeccakRhoOffsets[index(0, 0)]);
		A[index(0, 1)] = ROL64(A[index(0, 1)], KeccakRhoOffsets[index(0, 1)]);
		A[index(0, 2)] = ROL64(A[index(0, 2)], KeccakRhoOffsets[index(0, 2)]);
		A[index(0, 3)] = ROL64(A[index(0, 3)], KeccakRhoOffsets[index(0, 3)]);
		A[index(0, 4)] = ROL64(A[index(0, 4)], KeccakRhoOffsets[index(0, 4)]);
		
		A[index(1, 0)] = ROL64(A[index(1, 0)], KeccakRhoOffsets[index(1, 0)]);
		A[index(1, 1)] = ROL64(A[index(1, 1)], KeccakRhoOffsets[index(1, 1)]);
		A[index(1, 2)] = ROL64(A[index(1, 2)], KeccakRhoOffsets[index(1, 2)]);
		A[index(1, 3)] = ROL64(A[index(1, 3)], KeccakRhoOffsets[index(1, 3)]);
		A[index(1, 4)] = ROL64(A[index(1, 4)], KeccakRhoOffsets[index(1, 4)]);

		A[index(2, 0)] = ROL64(A[index(2, 0)], KeccakRhoOffsets[index(2, 0)]);
		A[index(2, 1)] = ROL64(A[index(2, 1)], KeccakRhoOffsets[index(2, 1)]);
		A[index(2, 2)] = ROL64(A[index(2, 2)], KeccakRhoOffsets[index(2, 2)]);
		A[index(2, 3)] = ROL64(A[index(2, 3)], KeccakRhoOffsets[index(2, 3)]);
		A[index(2, 4)] = ROL64(A[index(2, 4)], KeccakRhoOffsets[index(2, 4)]);
		
		A[index(3, 0)] = ROL64(A[index(3, 0)], KeccakRhoOffsets[index(3, 0)]);
		A[index(3, 1)] = ROL64(A[index(3, 1)], KeccakRhoOffsets[index(3, 1)]);
		A[index(3, 2)] = ROL64(A[index(3, 2)], KeccakRhoOffsets[index(3, 2)]);
		A[index(3, 3)] = ROL64(A[index(3, 3)], KeccakRhoOffsets[index(3, 3)]);
		A[index(3, 4)] = ROL64(A[index(3, 4)], KeccakRhoOffsets[index(3, 4)]);
		
		A[index(4, 0)] = ROL64(A[index(4, 0)], KeccakRhoOffsets[index(4, 0)]);
		A[index(4, 1)] = ROL64(A[index(4, 1)], KeccakRhoOffsets[index(4, 1)]);
		A[index(4, 2)] = ROL64(A[index(4, 2)], KeccakRhoOffsets[index(4, 2)]);
		A[index(4, 3)] = ROL64(A[index(4, 3)], KeccakRhoOffsets[index(4, 3)]);
		A[index(4, 4)] = ROL64(A[index(4, 4)], KeccakRhoOffsets[index(4, 4)]);
		
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
		tempA[index(0, 0)] = A[index(0, 0)];
		tempA[index(0, 1)] = A[index(0, 1)];
		tempA[index(0, 2)] = A[index(0, 2)];
		tempA[index(0, 3)] = A[index(0, 3)];
		tempA[index(0, 4)] = A[index(0, 4)];
		
		tempA[index(1, 0)] = A[index(1, 0)];
		tempA[index(1, 1)] = A[index(1, 1)];
		tempA[index(1, 2)] = A[index(1, 2)];
		tempA[index(1, 3)] = A[index(1, 3)];
		tempA[index(1, 4)] = A[index(1, 4)];
		
		tempA[index(2, 0)] = A[index(2, 0)];
		tempA[index(2, 1)] = A[index(2, 1)];
		tempA[index(2, 2)] = A[index(2, 2)];
		tempA[index(2, 3)] = A[index(2, 3)];
		tempA[index(2, 4)] = A[index(2, 4)];
		
		tempA[index(3, 0)] = A[index(3, 0)];
		tempA[index(3, 1)] = A[index(3, 1)];
		tempA[index(3, 2)] = A[index(3, 2)];
		tempA[index(3, 3)] = A[index(3, 3)];
		tempA[index(3, 4)] = A[index(3, 4)];
		
		tempA[index(4, 0)] = A[index(4, 0)];
		tempA[index(4, 1)] = A[index(4, 1)];
		tempA[index(4, 2)] = A[index(4, 2)];
		tempA[index(4, 3)] = A[index(4, 3)];
		tempA[index(4, 4)] = A[index(4, 4)];
		
		A[index(0, 2*0+3*0)] = tempA[index(0, 0)];
		A[index(1, 2*0+3*1)] = tempA[index(0, 1)];
		A[index(2, 2*0+3*2)] = tempA[index(0, 2)];
		A[index(3, 2*0+3*3)] = tempA[index(0, 3)];
		A[index(4, 2*0+3*4)] = tempA[index(0, 4)];
		
		A[index(0, 2*1+3*0)] = tempA[index(1, 0)];
		A[index(1, 2*1+3*1)] = tempA[index(1, 1)];
		A[index(2, 2*1+3*2)] = tempA[index(1, 2)];
		A[index(3, 2*1+3*3)] = tempA[index(1, 3)];
		A[index(4, 2*1+3*4)] = tempA[index(1, 4)];
		
		A[index(0, 2*2+3*0)] = tempA[index(2, 0)];
		A[index(1, 2*2+3*1)] = tempA[index(2, 1)];
		A[index(2, 2*2+3*2)] = tempA[index(2, 2)];
		A[index(3, 2*2+3*3)] = tempA[index(2, 3)];
		A[index(4, 2*2+3*4)] = tempA[index(2, 4)];
		
		A[index(0, 2*3+3*0)] = tempA[index(3, 0)];
		A[index(1, 2*3+3*1)] = tempA[index(3, 1)];
		A[index(2, 2*3+3*2)] = tempA[index(3, 2)];
		A[index(3, 2*3+3*3)] = tempA[index(3, 3)];
		A[index(4, 2*3+3*4)] = tempA[index(3, 4)];
		
		A[index(0, 2*4+3*0)] = tempA[index(4, 0)];
		A[index(1, 2*4+3*1)] = tempA[index(4, 1)];
		A[index(2, 2*4+3*2)] = tempA[index(4, 2)];
		A[index(3, 2*4+3*3)] = tempA[index(4, 3)];
		A[index(4, 2*4+3*4)] = tempA[index(4, 4)];
		
		
        // Chi
        /*for(y=0; y<5; y++) { 
			for(x=0; x<5; x++)
				C[x] = A[index(x, y)] ^ ((~A[index(x+1, y)]) & A[index(x+2, y)]);
			for(x=0; x<5; x++)
				A[index(x, y)] = C[x];
		}*/
		
		C[0] = A[index(0, 0)] ^ ((~A[index(0+1, 0)]) & A[index(0+2, 0)]);
		C[1] = A[index(1, 0)] ^ ((~A[index(1+1, 0)]) & A[index(1+2, 0)]);
		C[2] = A[index(2, 0)] ^ ((~A[index(2+1, 0)]) & A[index(2+2, 0)]);
		C[3] = A[index(3, 0)] ^ ((~A[index(3+1, 0)]) & A[index(3+2, 0)]);
		C[4] = A[index(4, 0)] ^ ((~A[index(4+1, 0)]) & A[index(4+2, 0)]);
		A[index(0, 0)] = C[0];
		A[index(1, 0)] = C[1];
		A[index(2, 0)] = C[2];
		A[index(3, 0)] = C[3];
		A[index(4, 0)] = C[4];
		
		C[0] = A[index(0, 1)] ^ ((~A[index(0+1, 1)]) & A[index(0+2, 1)]);
		C[1] = A[index(1, 1)] ^ ((~A[index(1+1, 1)]) & A[index(1+2, 1)]);
		C[2] = A[index(2, 1)] ^ ((~A[index(2+1, 1)]) & A[index(2+2, 1)]);
		C[3] = A[index(3, 1)] ^ ((~A[index(3+1, 1)]) & A[index(3+2, 1)]);
		C[4] = A[index(4, 1)] ^ ((~A[index(4+1, 1)]) & A[index(4+2, 1)]);
		A[index(0, 1)] = C[0];
		A[index(1, 1)] = C[1];
		A[index(2, 1)] = C[2];
		A[index(3, 1)] = C[3];
		A[index(4, 1)] = C[4];
		
		C[0] = A[index(0, 2)] ^ ((~A[index(0+1, 2)]) & A[index(0+2, 2)]);
		C[1] = A[index(1, 2)] ^ ((~A[index(1+1, 2)]) & A[index(1+2, 2)]);
		C[2] = A[index(2, 2)] ^ ((~A[index(2+1, 2)]) & A[index(2+2, 2)]);
		C[3] = A[index(3, 2)] ^ ((~A[index(3+1, 2)]) & A[index(3+2, 2)]);
		C[4] = A[index(4, 2)] ^ ((~A[index(4+1, 2)]) & A[index(4+2, 2)]);
		A[index(0, 2)] = C[0];
		A[index(1, 2)] = C[1];
		A[index(2, 2)] = C[2];
		A[index(3, 2)] = C[3];
		A[index(4, 2)] = C[4];
		
		C[0] = A[index(0, 3)] ^ ((~A[index(0+1, 3)]) & A[index(0+2, 3)]);
		C[1] = A[index(1, 3)] ^ ((~A[index(1+1, 3)]) & A[index(1+2, 3)]);
		C[2] = A[index(2, 3)] ^ ((~A[index(2+1, 3)]) & A[index(2+2, 3)]);
		C[3] = A[index(3, 3)] ^ ((~A[index(3+1, 3)]) & A[index(3+2, 3)]);
		C[4] = A[index(4, 3)] ^ ((~A[index(4+1, 3)]) & A[index(4+2, 3)]);
		A[index(0, 3)] = C[0];
		A[index(1, 3)] = C[1];
		A[index(2, 3)] = C[2];
		A[index(3, 3)] = C[3];
		A[index(4, 3)] = C[4];
		
		C[0] = A[index(0, 4)] ^ ((~A[index(0+1, 4)]) & A[index(0+2, 4)]);
		C[1] = A[index(1, 4)] ^ ((~A[index(1+1, 4)]) & A[index(1+2, 4)]);
		C[2] = A[index(2, 4)] ^ ((~A[index(2+1, 4)]) & A[index(2+2, 4)]);
		C[3] = A[index(3, 4)] ^ ((~A[index(3+1, 4)]) & A[index(3+2, 4)]);
		C[4] = A[index(4, 4)] ^ ((~A[index(4+1, 4)]) & A[index(4+2, 4)]);
		A[index(0, 4)] = C[0];
		A[index(1, 4)] = C[1];
		A[index(2, 4)] = C[2];
		A[index(3, 4)] = C[3];
		A[index(4, 4)] = C[4];
		
        // Iota
		A[0] = A[0] ^ KeccakRoundConstants[round_number];
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
	//kernel_optimixed<<<num_blocks, threads_per_block>>>(buffer_d, state_d);
	kernel<<<num_blocks, threads_per_block>>>(buffer_d, state_d);
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
