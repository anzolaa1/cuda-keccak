#ifndef __KERNEL_H__
#define __KERNEL_H__

typedef unsigned long long int UINT64;

void launch_kernel(unsigned long long *messages_h, unsigned int token_number);

int init_cuda(unsigned int t, UINT64 *krc, unsigned int *kro);

int alloc_memory();

int free_memory();

int get_state(unsigned long long *state_h);

#endif
