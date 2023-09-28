# DIRECTORIES
maindir=beampower
srcdir=$(maindir)/src
libdir=$(maindir)/lib

CC=gcc
NVCC=nvcc

all: $(libdir)/beamform_cpu.so $(libdir)/beamform_gpu.so
python_CPU: $(libdir)/beamform_cpu.so
python_GPU: $(libdir)/beamform_gpu.so
.SUFFIXES: .c .cu

# CPU flags
COPTIMFLAGS_CPU=-O3
CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native -std=c99
LDFLAGS_CPU=-shared

# GPU FLAGS
COPTIMFLAGS_GPU=-O3
CFLAGS_GPU=-Xcompiler "-fopenmp -fPIC -march=native -ftree-vectorize" -Xlinker -lgomp
ARCHFLAG=-gencode arch=compute_70,code=sm_70\
         -gencode arch=compute_75,code=sm_75\
         -gencode arch=compute_80,code=sm_80\
         #-gencode arch=compute_35,code=sm_35\

LDFLAGS_GPU=--shared

# build
$(libdir)/beamform_cpu.so: $(srcdir)/beamform.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

$(libdir)/beamform_gpu.so: $(srcdir)/beamform.cu
	$(NVCC) $(COPTIMFLAGS_GPU) $(CFLAGS_GPU) $(ARCHFLAG) $(LDFLAGS_GPU) $< -o $@

# clean
clean:
	rm $(libdir)/*.so
