# --- COMPILER ----------------------------------------

# if using -std different than gnu11, some changes are needed
CC = mpicc

# You may specify MPI_INCLUDE and MPI_LIB if those are not automatically found by your system.
# MPI_INCLUDE = /usr/include/
# MPI_LIB = /usr/lib/

# NVCC if using CUDA acceleration. Otherwise ignored.
NVCC = nvcc

# You may specify CUDA_INCLUDE and CUDA_LIB if those are not automatically found by your system.
# CUDA_INCLUDE = /opt/cuda/include/
# CUDA_LIB = /opt/cuda/lib64/
# CUDA architecture for which PTX code will be generated
CUDA_ARCH = compute_80
# (optional) CUDA architecture for which a binary image will be compiled
CUDA_CODE = compute_80
# NVTX is used to annotate profiling reports. It can be disabled by setting this variable to -DNVTX_DISABLE .
NVTX_DISABLE = -DNVTX_DISABLE

# --- CUDA Support --------------------------------------
# This flag must be set to "yes" in order to compile dd_alpha_amg with CUDA acceleration.
# Note that some functionality is not yet or no longer available in the CUDA version of
# DD Alpha AMG.
CUDA_ENABLER = no

# --- SSE Support --------------------------------------
# This flag must be set to "yes" in order to compile dd_alpha_amg with SSE acceleration.
# Note that some functionality is not yet or no longer available in the SSE version of
# DD Alpha AMG.
SSE_ENABLER = yes

# --- AVX Support --------------------------------------
# This flag must be set to "yes" in order to compile dd_alpha_amg with AVX2 acceleration.
# Note that some functionality is not yet or no longer available in the AVX2 version of
# DD Alpha AMG.
AVX_ENABLER = yes

AVX512_ENABLER = no

# --- Unit Testing -------------------------------------
# If you want to run the unit tests, the gtest library and rapidcheck are required.
# GTEST_INCLUDE = /usr/include
# GTEST_LIB = /usr/lib
# RAPIDCHECK_INCLUDE = /usr/include
# RAPIDCHECK_LIB = /usr/lib
