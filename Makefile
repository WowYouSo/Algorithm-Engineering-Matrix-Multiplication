# Matrix multiplication benchmark (macOS / Apple Silicon)
#
# Targets:
#   make naive        - CPU-only build (no Accelerate, no Metal)
#   make accel        - CPU + Accelerate/BLAS
#   make metal        - GPU via Metal Performance Shaders (float32) + Accelerate
#
# Optional knobs:
#   FAST_MATH=1       - add -ffast-math (faster, less precise)
#   DEBUG=1           - build with -O0 -g

CC  ?= clang
CXX ?= clang++

CFLAGS_COMMON := -march=armv8.6-a+fp+simd -funroll-loops -Wall -Wextra -I. -fblocks

ifeq ($(DEBUG),1)
  CFLAGS_COMMON += -O0 -g
else
  CFLAGS_COMMON += -O3
endif

ifeq ($(FAST_MATH),1)
  CFLAGS_COMMON += -ffast-math
endif

SRC_C := bench.c algos.c alpha48.c
HDR   := bench.h

all: naive

naive: $(SRC_C) $(HDR)
	$(CC) $(CFLAGS_COMMON) -std=c11 -DSKIP_METAL $(SRC_C) -o bench_naive $(LDFLAGS)

accel: $(SRC_C) $(HDR)
	$(CC) $(CFLAGS_COMMON) -std=c11 -DUSE_ACCELERATE -DSKIP_METAL $(SRC_C) -o bench_accel $(LDFLAGS) -framework Accelerate

metal: bench.o algos.o alpha48.o metal_mps.o
	$(CXX) -O3 bench.o algos.o alpha48.o metal_mps.o -o bench_metal \
		-framework Accelerate -framework Metal -framework MetalPerformanceShaders -lobjc

bench.o: bench.c $(HDR)
	$(CC) $(CFLAGS_COMMON) -std=c11 -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -c bench.c -o bench.o

algos.o: algos.c $(HDR)
	$(CC) $(CFLAGS_COMMON) -std=c11 -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -c algos.c -o algos.o

alpha48.o: alpha48.c $(HDR)
	$(CC) $(CFLAGS_COMMON) -std=c11 -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -c alpha48.c -o alpha48.o

metal_mps.o: metal_mps.mm
	$(CXX) $(CFLAGS_COMMON) -std=c++17 -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -fobjc-arc -c metal_mps.mm -o metal_mps.o

clean:
	rm -f bench_naive bench_accel bench_metal *.o

.PHONY: all clean naive accel metal
