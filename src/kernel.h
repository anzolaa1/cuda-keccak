
void launch_kernel(unsigned long long *messages_h, unsigned int token_number);


int init_cuda(unsigned int t);

int alloc_memory();

int free_memory();

int get_state(unsigned long long *state_h);

