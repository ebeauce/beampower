# DIRECTORIES
maindir=beampower
srcdir=$(maindir)/src
libdir=$(maindir)/lib

# Automatic platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Define compilers
NVCC=nvcc

# Auto-detect C compiler based on platform
ifeq ($(UNAME_S),Darwin)
    # macOS: prefer clang for ARM64, gcc for Intel
    ifeq ($(UNAME_M),arm64)
        CC=clang
    else
        CC=gcc
    endif
else
    # Linux and others: use gcc
    CC=gcc
endif

# Define commands
all: $(libdir)/beamform_cpu.so $(libdir)/beamform_gpu.so
python_CPU: $(libdir)/beamform_cpu.so
python_GPU: $(libdir)/beamform_gpu.so
.SUFFIXES: .c .cu

# -----------------------------------------------
#                CPU FLAGS
# -----------------------------------------------
COPTIMFLAGS_CPU=-O3

# Auto-detect CPU flags based on platform
ifeq ($(UNAME_S),Darwin)
    # macOS (both Intel and ARM64)
    CFLAGS_CPU=-fopenmp=libomp -fPIC -ftree-vectorize -march=native -std=c99
    LDFLAGS_CPU=-shared -fuse-ld=lld
else
    # Linux and Unix-like systems
    CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native -std=c99
    LDFLAGS_CPU=-shared
endif


# -----------------------------------------------
#              GPU FLAGS
# -----------------------------------------------
COPTIMFLAGS_GPU=-O3
CFLAGS_GPU=-Xcompiler "-fopenmp -fPIC -march=native -ftree-vectorize" -Xlinker -lgomp
ARCHFLAG=-gencode arch=compute_70,code=sm_70\
         -gencode arch=compute_75,code=sm_75\
         -gencode arch=compute_80,code=sm_80

LDFLAGS_GPU=--shared

# build for python
$(libdir)/beamform_cpu.so: $(srcdir)/beamform.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

$(libdir)/beamform_gpu.so: $(srcdir)/beamform.cu
	$(NVCC) $(COPTIMFLAGS_GPU) $(CFLAGS_GPU) $(ARCHFLAG) $(LDFLAGS_GPU) $< -o $@

clean:
	rm $(libdir)/*.so

