CC=g++

CUDAROOT=$(CUDA_HOME)
CUDASDKROOT=$(CUDA_SDK_HOME)
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS= -G -g -I $(CUDASDKROOT)/C/common/inc \
	  -I $(CUDASDKROOT)/include

CFLAGS= -g -L$(CUDAROOT)/lib  \
	-L$(CUDAROOT)/lib64  \
	-L$(CUDASDKROOT)/C/lib


OBJECTS =\
      kernel.o \
      main.o 
	
cuda-keccak: $(OBJECTS)
	$(CC) $(CFLAGS)  -o $@ $^ -lcuda -lcutil_x86_64 -lcudart

clean:
	rm -f *.o *.linkinfo cuda-keccak

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

