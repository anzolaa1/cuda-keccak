CC=gcc
CPP=g++

CUDAROOT=$(CUDA_HOME)
CUDASDKROOT=$(CUDA_SDK_HOME)
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS= -g -G -I $(CUDASDKROOT)/C/common/inc \
	  -I $(CUDASDKROOT)/include

CFLAGS= -g -O2 -L$(CUDAROOT)/lib  \
	-L$(CUDAROOT)/lib64  \
	-L$(CUDASDKROOT)/C/lib


OBJECTS =\
      main.o \
      kernel.o \
      KeccakF1600reference.o \
      displayIntermediateValues.o
      
	
cuda-keccak: $(OBJECTS)
	$(CPP) $(CFLAGS)  -o $@ $^ -lcuda -lcutil_x86_64 -lcudart

clean:
	rm -f *.o *.linkinfo cuda-keccak

%.o: %.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

