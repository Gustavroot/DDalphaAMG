# --- COMPILER ----------------------------------------

# if using -std different than gnu11, some changes are needed
CC = mpicc -std=gnu11 -Wall -pedantic
MPI_INCLUDE = /usr/include/
MPI_LIB = /usr/lib/

CPP = cpp
MAKEDEP = $(CPP) -MM

# if using -std different than c++11, some changes are needed
NVCC = nvcc
CUDA_INCLUDE = /opt/cuda/include/
CUDA_LIB = /opt/cuda/lib64/
# NVTX is used to annotate profiling reports. It can be disabled by setting this variable to -DNVTX_DISABLE .
NVTX_DISABLE = # -DNVTX_DISABLE


# --- CUDA Support --------------------------------------
# This flag must be set to -DCUDA_OPT in order to compile dd_alpha_amg with CUDA acceleration.
# Note that some functionality is not yet or no longer available in the CUDA version of
# DD Alpha AMG.
CUDA_ENABLER = # -DCUDA_OPT

# --- SSE Support --------------------------------------
# This flag must be set to -DSSE -msse4.2 in order to compile dd_alpha_amg with SSE acceleration.
# Note that some functionality is not yet or no longer available in the SSE version of
# DD Alpha AMG.
SSE_ENABLER = -DSSE -msse4.2
