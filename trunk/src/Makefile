CC=gcc
CPP=g++

CUDAROOT=$(CUDA_HOME)
CUDASDKROOT=$(CUDA_SDK_HOME)
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS= -g -G \
          -O2 \
	  --ptxas-options=-v \
	  -arch=sm_11 \
	  -I $(CUDASDKROOT)/C/common/inc \
	  -I $(CUDASDKROOT)/include

CFLAGS= -g -O2 -L$(CUDAROOT)/lib  \
	-L$(CUDAROOT)/lib64  \
	-L$(CUDASDKROOT)/C/lib \
	-I $(CUDASDKROOT)/C/common/inc \
	-I $(CUDA_HOME)/include


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

