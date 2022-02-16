# DIRECTORIES
maindir=beamnetresponse
srcdir=$(maindir)/src
libdir=$(maindir)/lib

CC=gcc

all: $(libdir)/beamformed_nr_CPU.so
python_CPU: $(libdir)/beamformed_nr_CPU.so
.SUFFIXES: .c .cu

# CPU flags
COPTIMFLAGS_CPU=-O3
CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native -std=c99
LDFLAGS_CPU=-shared

# build
$(libdir)/beamformed_nr_CPU.so: $(srcdir)/beamformed_nr.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

# clean
clean:
	rm $(libdir)/*.so
